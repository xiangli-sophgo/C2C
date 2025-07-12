"""
CrossRing专用IP接口实现。

基于C2C仓库的现有结构，结合CrossRing仓库的IP接口实现，
为CrossRing拓扑提供专用的IP接口管理，包括时钟域转换、资源管理和STI协议处理。
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Deque
from collections import deque, defaultdict
import logging

from .flit import CrossRingFlit, create_crossring_flit
from .config import CrossRingConfig
from src.noc.utils.types import NodeId
from src.noc.base.ip_interface import BaseIPInterface, PipelinedFIFO


class CrossRingIPInterface(BaseIPInterface):
    """
    CrossRing专用IP接口，集成资源管理和STI协议处理。

    该类负责：
    1. 时钟域转换（1GHz ↔ 2GHz）
    2. RN/SN资源管理（tracker, databuffer）
    3. STI三通道协议处理（REQ/RSP/DAT）
    4. 请求重试机制
    """

    def __init__(self, config: CrossRingConfig, ip_type: str, node_id: NodeId, model: Any):
        """
        初始化CrossRing IP接口

        Args:
            config: CrossRing配置
            ip_type: IP类型 ("gdma", "sdma", "cdma", "ddr", "l2m")
            node_id: 节点ID
            model: 主模型实例（用于注册和全局访问）
        """
        # 调用父类构造函数
        super().__init__(ip_type, node_id, config, model, CrossRingFlit)

        # CrossRing特有的配置
        self.config = config

        # ========== RN资源管理 ==========
        # RN Tracker
        self.rn_tracker = {"read": [], "write": []}
        self.rn_tracker_count = {"read": config.tracker_config.rn_r_tracker_ostd, "write": config.tracker_config.rn_w_tracker_ostd}
        self.rn_tracker_pointer = {"read": 0, "write": 0}

        # RN Data Buffer
        self.rn_rdb = {}  # 读数据缓冲 {packet_id: [flits]}
        self.rn_rdb_count = config.rn_rdb_size
        self.rn_rdb_reserve = 0  # 预留数量用于重试

        self.rn_wdb = {}  # 写数据缓冲 {packet_id: [flits]}
        self.rn_wdb_count = config.rn_wdb_size

        # ========== SN资源管理 ==========
        self.sn_tracker = []

        # 根据IP类型设置SN tracker数量
        if ip_type.startswith("ddr"):
            self.sn_tracker_count = {"ro": config.tracker_config.sn_ddr_r_tracker_ostd, "share": config.tracker_config.sn_ddr_w_tracker_ostd}  # 读专用  # 写共享
            self.sn_wdb_count = config.sn_ddr_wdb_size
        elif ip_type.startswith("l2m"):
            self.sn_tracker_count = {"ro": config.tracker_config.sn_l2m_r_tracker_ostd, "share": config.tracker_config.sn_l2m_w_tracker_ostd}
            self.sn_wdb_count = config.sn_l2m_wdb_size
        else:
            # DMA类IP通常不作为SN
            self.sn_tracker_count = {"ro": 0, "share": 0}
            self.sn_wdb_count = 0

        self.sn_wdb = {}  # SN写数据缓冲

        # 等待队列（资源不足时的请求队列）
        self.sn_req_wait = {"read": [], "write": []}

        # SN tracker延迟释放
        self.sn_tracker_release_time = defaultdict(list)

        # ========== 统计信息 ==========
        self.read_retry_num_stat = 0
        self.write_retry_num_stat = 0

        # 等待周期统计
        self.req_wait_cycles_h = 0
        self.req_wait_cycles_v = 0
        self.rsp_wait_cycles_h = 0
        self.rsp_wait_cycles_v = 0
        self.data_wait_cycles_h = 0
        self.data_wait_cycles_v = 0

        # 环路完成统计
        self.req_cir_h_num = 0
        self.req_cir_v_num = 0
        self.rsp_cir_h_num = 0
        self.rsp_cir_v_num = 0
        self.data_cir_h_num = 0
        self.data_cir_v_num = 0

        # 创建分通道的pending队列，替代inject_fifos和父类pending_requests
        self.pending_by_channel = {"req": deque(), "rsp": deque(), "data": deque()}

    def _check_and_reserve_resources(self, flit) -> bool:
        """检查并预占RN端资源"""
        if flit.req_type == "read":
            # 检查是否已经在tracker中（避免重复添加）
            for existing_req in self.rn_tracker["read"]:
                if hasattr(existing_req, "packet_id") and hasattr(flit, "packet_id") and existing_req.packet_id == flit.packet_id:
                    return True  # 已经预占过资源，直接返回成功

            # 检查读资源：tracker + rdb + reserve
            rdb_available = self.rn_rdb_count >= flit.burst_length
            tracker_available = self.rn_tracker_count["read"] > 0
            reserve_ok = self.rn_rdb_count > self.rn_rdb_reserve * flit.burst_length

            if not (rdb_available and tracker_available and reserve_ok):
                return False

            # 预占资源
            self.rn_rdb_count -= flit.burst_length
            self.rn_tracker_count["read"] -= 1
            self.rn_rdb[flit.packet_id] = []
            self.rn_tracker["read"].append(flit)
            self.rn_tracker_pointer["read"] += 1

        elif flit.req_type == "write":
            # 检查是否已经在tracker中（避免重复添加）
            for existing_req in self.rn_tracker["write"]:
                if hasattr(existing_req, "packet_id") and hasattr(flit, "packet_id") and existing_req.packet_id == flit.packet_id:
                    return True  # 已经预占过资源，直接返回成功

            # 检查写资源：tracker + wdb
            wdb_available = self.rn_wdb_count >= flit.burst_length
            tracker_available = self.rn_tracker_count["write"] > 0

            if not (wdb_available and tracker_available):
                return False

            # 预占资源
            self.rn_wdb_count -= flit.burst_length
            self.rn_tracker_count["write"] -= 1
            self.rn_wdb[flit.packet_id] = []
            self.rn_tracker["write"].append(flit)
            self.rn_tracker_pointer["write"] += 1

        return True

    def _inject_to_topology_network(self, flit, channel: str) -> bool:
        """
        注入到CrossRing网络

        Returns:
            是否成功注入
        """
        # 获取对应的节点
        if self.node_id in self.model.crossring_nodes:
            node = self.model.crossring_nodes[self.node_id]

            # 注入到节点的对应IP的channel buffer
            ip_key = f"{self.ip_type}_node{self.node_id}"
            if ip_key in node.ip_inject_channel_buffers:
                channel_buffer = node.ip_inject_channel_buffers[ip_key][channel]
                if channel_buffer.can_accept_input():
                    if channel_buffer.write_input(flit):
                        # 更新flit位置信息
                        flit.flit_position = "IQ_CH"
                        flit.current_node_id = self.node_id

                        self.logger.debug(f"IP {self.ip_type} 成功注入flit到节点{self.node_id}的channel buffer")

                        # 更新时间戳
                        if channel == "req" and hasattr(flit, "req_attr") and flit.req_attr == "new":
                            flit.cmd_entry_noc_from_cake0_cycle = self.current_cycle
                        elif channel == "rsp":
                            flit.cmd_entry_noc_from_cake1_cycle = self.current_cycle
                        elif channel == "data":
                            if flit.req_type == "read":
                                flit.data_entry_noc_from_cake1_cycle = self.current_cycle
                            elif flit.req_type == "write":
                                flit.data_entry_noc_from_cake0_cycle = self.current_cycle

                        # 更新RequestTracker状态：flit成功注入到网络
                        if hasattr(self.model, "request_tracker") and hasattr(flit, "packet_id"):
                            if channel == "req":
                                self.model.request_tracker.mark_request_injected(flit.packet_id, self.current_cycle)
                                self.model.request_tracker.add_request_flit(flit.packet_id, flit)
                            elif channel == "rsp":
                                self.model.request_tracker.add_response_flit(flit.packet_id, flit)
                            elif channel == "data":
                                self.model.request_tracker.add_data_flit(flit.packet_id, flit)

                        return True
                    else:
                        self.logger.warning(f"IP {self.ip_type} 无法注入flit到节点{self.node_id}的channel buffer - 写入失败")
                        return False
                else:
                    # Channel buffer满，无法注入
                    return False
            else:
                self.logger.error(f"节点{self.node_id}没有IP {ip_key}的channel buffer")
                return False
        else:
            self.logger.error(f"节点{self.node_id}不存在于CrossRing网络中")
            return False

    def _eject_from_topology_network(self, channel: str):
        """从CrossRing网络弹出"""
        # 获取对应的节点
        if self.node_id in self.model.crossring_nodes:
            node = self.model.crossring_nodes[self.node_id]

            # 从节点的对应IP的eject channel buffer获取flit
            ip_key = f"{self.ip_type}_node{self.node_id}"

            if ip_key in node.ip_eject_channel_buffers:
                eject_buffer = node.ip_eject_channel_buffers[ip_key][channel]
                if eject_buffer.valid_signal():
                    flit = eject_buffer.read_output()
                    if flit:
                        # 更新flit状态，从EQ_CH转移到H2L处理
                        flit.flit_position = "H2L"
                        self.logger.debug(f"IP {self.ip_type} 成功从节点{self.node_id}的eject buffer获取flit")
                    return flit
            else:
                print(f"❌ IP接口 {self.ip_type} 在节点{self.node_id}找不到eject buffer key: {ip_key}")
        else:
            self.logger.error(f"节点{self.node_id}不存在于CrossRing网络中")

        return None

    def _process_delayed_resource_release(self) -> None:
        """处理延迟释放的资源（重写父类方法）"""
        # 处理SN tracker延迟释放
        self._process_sn_tracker_release()

    def _handle_received_request(self, req: CrossRingFlit) -> None:
        """
        处理收到的请求（SN端）

        Args:
            req: 收到的请求flit
        """
        # 首先打印调试信息

        # 只有SN端IP类型才能处理请求
        if not (self.ip_type.startswith("ddr") or self.ip_type.startswith("l2m")):
            return

        req.cmd_received_by_cake1_cycle = self.current_cycle

        # 统计等待周期和环路数
        self.req_wait_cycles_h += req.wait_cycle_h
        self.req_wait_cycles_v += req.wait_cycle_v
        self.req_cir_h_num += req.circuits_completed_h
        self.req_cir_v_num += req.circuits_completed_v

        if req.req_type == "read":
            if req.req_attr == "new":
                # 新读请求：检查SN资源
                if self.sn_tracker_count["ro"] > 0:
                    req.sn_tracker_type = "ro"
                    self.sn_tracker.append(req)
                    self.sn_tracker_count["ro"] -= 1

                    self.logger.info(f"🎯 SN端开始处理读请求 {req.packet_id}: 直接生成数据")

                    self._create_read_packet(req)
                    self._release_completed_sn_tracker(req)

                    self._notify_request_arrived(req)
                else:
                    # 资源不足，发送negative响应
                    print(f"❌ SN端 {self.ip_type} 资源不足，发送negative响应给 {req.packet_id}")
                    self._create_negative_response(req)
                    self.sn_req_wait["read"].append(req)
            else:
                # 重试读请求：直接生成数据
                self._create_read_packet(req)
                self._release_completed_sn_tracker(req)

                # **重要修复：通知RequestTracker请求已到达**
                self._notify_request_arrived(req)

        elif req.req_type == "write":
            if req.req_attr == "new":
                # 新写请求：检查SN资源（tracker + wdb）
                if self.sn_tracker_count["share"] > 0 and self.sn_wdb_count >= req.burst_length:
                    req.sn_tracker_type = "share"
                    self.sn_tracker.append(req)
                    self.sn_tracker_count["share"] -= 1
                    self.sn_wdb[req.packet_id] = []
                    self.sn_wdb_count -= req.burst_length
                    self._create_datasend_response(req)
                else:
                    # 资源不足，发送negative响应
                    self._create_negative_response(req)
                    self.sn_req_wait["write"].append(req)
            else:
                # 重试写请求：直接发送datasend
                self._create_datasend_response(req)

    def _handle_received_response(self, rsp: CrossRingFlit) -> None:
        """
        处理收到的响应（RN端）

        Args:
            rsp: 收到的响应flit
        """
        rsp.cmd_received_by_cake0_cycle = self.current_cycle

        # 统计等待周期和环路数
        self.rsp_wait_cycles_h += rsp.wait_cycle_h
        self.rsp_wait_cycles_v += rsp.wait_cycle_v
        self.rsp_cir_h_num += rsp.circuits_completed_h
        self.rsp_cir_v_num += rsp.circuits_completed_v

        # 更新重试统计
        if rsp.rsp_type == "negative":
            if rsp.req_type == "read":
                self.read_retry_num_stat += 1
            elif rsp.req_type == "write":
                self.write_retry_num_stat += 1

        # 查找对应的请求
        req = self._find_matching_request(rsp)
        if not req:
            # 对于datasend类型的响应，即使找不到匹配的请求也要处理，因为这是正常的write流程
            if hasattr(rsp, "rsp_type") and rsp.rsp_type == "datasend":
                print(f"🎯 处理datasend响应: packet_id={rsp.packet_id}, req_type={rsp.req_type}")
                self.logger.debug(f"收到datasend响应 {rsp.packet_id}，请求可能已移出tracker，继续处理")
                # 直接处理datasend响应，req可以为None
                if rsp.req_type == "write":
                    self._handle_write_response(rsp, req)
                return
            else:
                logging.warning(f"RSP {rsp} do not have matching REQ")
                return

        # 同步延迟记录
        req.sync_latency_record(rsp)

        # 处理不同类型的响应
        if rsp.req_type == "read":
            self._handle_read_response(rsp, req)
        elif rsp.req_type == "write":
            self._handle_write_response(rsp, req)

    def _handle_received_data(self, flit: CrossRingFlit) -> None:
        """
        处理收到的数据

        Args:
            flit: 收到的数据flit
        """
        flit.arrival_cycle = self.current_cycle

        # 统计等待周期和环路数
        self.data_wait_cycles_h += flit.wait_cycle_h
        self.data_wait_cycles_v += flit.wait_cycle_v
        self.data_cir_h_num += flit.circuits_completed_h
        self.data_cir_v_num += flit.circuits_completed_v

        if flit.req_type == "read":
            # 读数据到达RN端
            self.rn_rdb[flit.packet_id].append(flit)

            # 检查是否收集完整个burst
            if len(self.rn_rdb[flit.packet_id]) == flit.burst_length:
                req = self._find_rn_tracker_by_packet_id(flit.packet_id, "read")
                if req:
                    # 释放RN tracker和资源
                    self.rn_tracker["read"].remove(req)
                    self.rn_tracker_count["read"] += 1
                    self.rn_tracker_pointer["read"] -= 1
                    self.rn_rdb_count += req.burst_length

                    # 设置完成时间戳
                    for f in self.rn_rdb[flit.packet_id]:
                        f.leave_db_cycle = self.current_cycle
                        f.sync_latency_record(req)
                        f.data_received_complete_cycle = self.current_cycle

                    # 计算延迟
                    first_flit = self.rn_rdb[flit.packet_id][0]
                    for f in self.rn_rdb[flit.packet_id]:
                        f.cmd_latency = f.cmd_received_by_cake1_cycle - f.cmd_entry_noc_from_cake0_cycle
                        f.data_latency = f.data_received_complete_cycle - first_flit.data_entry_noc_from_cake1_cycle
                        f.transaction_latency = f.data_received_complete_cycle - f.cmd_entry_cake0_cycle

                    # **关键修复：通知RequestTracker读请求已完成（RN收到全部数据）**
                    self._notify_request_completion(req)

                    # 清理数据缓冲
                    del self.rn_rdb[flit.packet_id]

        elif flit.req_type == "write":
            # 写数据到达SN端
            self.sn_wdb[flit.packet_id].append(flit)

            # 检查是否收集完整个burst
            if len(self.sn_wdb[flit.packet_id]) == flit.burst_length:
                req = self._find_sn_tracker_by_packet_id(flit.packet_id)
                if req:
                    # 设置延迟释放时间
                    release_time = self.current_cycle + self.config.tracker_config.sn_tracker_release_latency

                    # 设置完成时间戳
                    first_flit = self.sn_wdb[flit.packet_id][0]
                    for f in self.sn_wdb[flit.packet_id]:
                        f.leave_db_cycle = release_time
                        f.sync_latency_record(req)
                        f.data_received_complete_cycle = self.current_cycle
                        f.cmd_latency = f.cmd_received_by_cake0_cycle - f.cmd_entry_noc_from_cake0_cycle
                        f.data_latency = f.data_received_complete_cycle - first_flit.data_entry_noc_from_cake0_cycle
                        f.transaction_latency = f.data_received_complete_cycle + self.config.tracker_config.sn_tracker_release_latency - f.cmd_entry_cake0_cycle

                    # **关键修复：通知RequestTracker写请求已完成（SN收到全部数据）**
                    self._notify_request_completion(req)

                    # 清理数据缓冲
                    del self.sn_wdb[flit.packet_id]

                    # 添加到延迟释放队列
                    self.sn_tracker_release_time[release_time].append(req)

    def _find_matching_request(self, rsp: CrossRingFlit) -> Optional[CrossRingFlit]:
        """根据响应查找匹配的请求"""
        for req in self.rn_tracker[rsp.req_type]:
            if req.packet_id == rsp.packet_id:
                return req
        return None

    def _find_rn_tracker_by_packet_id(self, packet_id: str, req_type: str) -> Optional[CrossRingFlit]:
        """根据包ID查找RN tracker"""
        for req in self.rn_tracker[req_type]:
            if req.packet_id == packet_id:
                return req
        return None

    def _find_sn_tracker_by_packet_id(self, packet_id: str) -> Optional[CrossRingFlit]:
        """根据包ID查找SN tracker"""
        for req in self.sn_tracker:
            if req.packet_id == packet_id:
                return req
        return None

    def _handle_read_response(self, rsp: CrossRingFlit, req: CrossRingFlit) -> None:
        """处理读响应（只处理negative响应，读请求成功时不发送响应）"""
        if rsp.rsp_type == "negative":
            # 读重试逻辑
            if req.req_attr == "old":
                return  # 已经在重试中

            req.reset_for_retry()
            self.rn_rdb_count += req.burst_length
            if req.packet_id in self.rn_rdb:
                del self.rn_rdb[req.packet_id]
            self.rn_rdb_reserve += 1

            # 重新放入请求队列
            req.req_state = "valid"
            req.req_attr = "old"
            req.is_injected = False
            req.path_index = 0
            req.is_new_on_network = True
            req.is_arrive = False

            # 重新入队到队首（高优先级重试）
            self.pending_by_channel["req"].appendleft(req)
            self.rn_rdb_reserve -= 1
        else:
            # 读请求不应该收到positive或其他类型的响应
            self.logger.warning(f"读请求 {req.packet_id} 收到了意外的响应类型: {rsp.rsp_type}")

    def _handle_write_response(self, rsp: CrossRingFlit, req: CrossRingFlit) -> None:
        """处理写响应"""
        if rsp.rsp_type == "negative":
            # 写重试逻辑
            if req.req_attr == "old":
                return
            req.reset_for_retry()

        elif rsp.rsp_type == "positive":
            # 写重试：重新注入
            req.req_state = "valid"
            req.req_attr = "old"
            req.is_injected = False
            req.path_index = 0
            req.is_new_on_network = True
            req.is_arrive = False
            # 重新入队到队首（高优先级重试）
            self.pending_by_channel["req"].appendleft(req)

        elif rsp.rsp_type == "datasend":
            # ✅ 修复：收到datasend响应后才创建并发送写数据
            print(f"🎯 处理datasend响应: packet_id={rsp.packet_id}")

            # 确保WDB条目存在
            if rsp.packet_id not in self.rn_wdb:
                self.logger.error(f"⚠️ 没有{rsp.packet_id}对应的请求")

            # 创建写数据flits
            self._create_write_data_flits(req)

            # 发送写数据
            data_flits = self.rn_wdb.get(rsp.packet_id, [])
            self.logger.info(f"🔶 准备发送 {len(data_flits)} 个DATA flit for packet {rsp.packet_id}")

            for flit in data_flits:
                self.pending_by_channel["data"].append(flit)
                # 添加到RequestTracker
                if hasattr(self.model, "request_tracker"):
                    self.model.request_tracker.add_data_flit(flit.packet_id, flit)

            # 释放RN write tracker（如果存在）
            for tracker_req in self.rn_tracker["write"][:]:  # 使用副本避免修改迭代列表
                if tracker_req.packet_id == rsp.packet_id:
                    self.rn_tracker["write"].remove(tracker_req)
                    self.rn_wdb_count += tracker_req.burst_length
                    self.rn_tracker_count["write"] += 1
                    self.rn_tracker_pointer["write"] -= 1

    def _create_write_data_flits(self, req: CrossRingFlit) -> None:
        """创建写数据flits"""
        self.logger.info(f"🔧 开始创建写数据flits: packet_id={req.packet_id}, burst_length={req.burst_length}")
        for i in range(req.burst_length):
            # 计算发送延迟
            if req.destination_type and req.destination_type.startswith("ddr"):
                latency = self.config.latency_config.ddr_w_latency
            else:
                latency = self.config.latency_config.l2m_w_latency

            data_flit = create_crossring_flit(
                source=req.source,
                destination=req.destination,
                req_type=req.req_type,
                packet_id=req.packet_id,
                flit_id=i,
                burst_length=req.burst_length,
                channel="data",
                flit_type="data",
                departure_cycle=self.current_cycle + latency + i * self.clock_ratio,
            )

            data_flit.sync_latency_record(req)
            data_flit.source_type = req.source_type
            data_flit.destination_type = req.destination_type
            data_flit.is_last_flit = i == req.burst_length - 1

            self.rn_wdb[req.packet_id].append(data_flit)

    def _create_read_packet(self, req: CrossRingFlit) -> None:
        """创建读数据包，使用现有的pending_by_channel机制"""
        for i in range(req.burst_length):
            # 计算发送延迟
            if req.destination_type and req.destination_type.startswith("ddr"):
                latency = self.config.latency_config.ddr_r_latency
            else:
                latency = self.config.latency_config.l2m_r_latency

            # 读数据从SN返回到RN
            data_flit = create_crossring_flit(
                source=req.destination,  # SN位置
                destination=req.source,  # RN位置
                req_type=req.req_type,
                packet_id=req.packet_id,
                flit_id=i,
                burst_length=req.burst_length,
                channel="data",
                flit_type="data",
                departure_cycle=self.current_cycle + latency + i * self.clock_ratio,
            )

            data_flit.sync_latency_record(req)
            data_flit.source_type = req.destination_type
            data_flit.destination_type = req.source_type
            data_flit.is_last_flit = i == req.burst_length - 1
            data_flit.flit_position = "L2H"
            data_flit.current_node_id = self.node_id

            # 使用分通道的pending队列
            self.pending_by_channel["data"].append(data_flit)

            self.logger.info(f"🔶 SN端生成数据flit: {data_flit.packet_id}.{i} -> {data_flit.destination}, departure={data_flit.departure_cycle}")

    def _create_negative_response(self, req: CrossRingFlit) -> None:
        """创建negative响应"""
        rsp = create_crossring_flit(
            source=req.destination,
            destination=req.source,
            req_type=req.req_type,
            packet_id=req.packet_id,
            channel="rsp",
            flit_type="rsp",
            rsp_type="negative",
            departure_cycle=self.current_cycle + self.clock_ratio,
        )

        rsp.sync_latency_record(req)
        rsp.source_type = req.destination_type
        rsp.destination_type = req.source_type
        rsp.sn_rsp_generate_cycle = self.current_cycle

        self.pending_by_channel["rsp"].append(rsp)

    def _create_datasend_response(self, req: CrossRingFlit) -> None:
        """创建datasend响应"""
        self.logger.info(f"🏭 SN端创建datasend响应: packet_id={req.packet_id}, 从节点{req.destination}发送到节点{req.source}")
        rsp = create_crossring_flit(
            source=req.destination,
            destination=req.source,
            req_type=req.req_type,
            packet_id=req.packet_id,
            channel="rsp",
            flit_type="rsp",
            rsp_type="datasend",
            departure_cycle=self.current_cycle + self.clock_ratio,
        )

        rsp.sync_latency_record(req)
        rsp.source_type = req.destination_type
        rsp.destination_type = req.source_type
        rsp.sn_rsp_generate_cycle = self.current_cycle

        self.pending_by_channel["rsp"].append(rsp)

    def _release_completed_sn_tracker(self, req: CrossRingFlit) -> None:
        """释放完成的SN tracker"""
        if req in self.sn_tracker:
            self.sn_tracker.remove(req)
            self.sn_tracker_count[req.sn_tracker_type] += 1

        # 对于写请求，释放WDB
        if req.req_type == "write":
            self.sn_wdb_count += req.burst_length

        # 尝试处理等待队列
        self._process_waiting_requests(req.req_type, req.sn_tracker_type)

    def _process_waiting_requests(self, req_type: str, tracker_type: str) -> None:
        """处理等待队列中的请求"""
        wait_list = self.sn_req_wait[req_type]
        if not wait_list:
            return

        if req_type == "write":
            # 检查tracker和wdb资源
            if self.sn_tracker_count[tracker_type] > 0 and self.sn_wdb_count > 0:
                new_req = wait_list.pop(0)
                new_req.sn_tracker_type = tracker_type

                # 分配资源
                self.sn_tracker.append(new_req)
                self.sn_tracker_count[tracker_type] -= 1
                self.sn_wdb_count -= new_req.burst_length

                # 发送datasend响应
                self._create_datasend_response(new_req)

        elif req_type == "read":
            # 检查tracker资源
            if self.sn_tracker_count[tracker_type] > 0:
                new_req = wait_list.pop(0)
                new_req.sn_tracker_type = tracker_type

                # 分配tracker
                self.sn_tracker.append(new_req)
                self.sn_tracker_count[tracker_type] -= 1

                # 直接生成读数据包
                self._create_read_packet(new_req)

    def _process_sn_tracker_release(self) -> None:
        """处理SN tracker的延迟释放"""
        if self.current_cycle in self.sn_tracker_release_time:
            for req in self.sn_tracker_release_time[self.current_cycle]:
                self._release_completed_sn_tracker(req)
            del self.sn_tracker_release_time[self.current_cycle]

    def get_status(self) -> Dict[str, Any]:
        """获取IP接口状态"""
        return {
            "ip_type": self.ip_type,
            "node_id": self.node_id,
            "current_cycle": self.current_cycle,
            "rn_resources": {
                "read_tracker_active": len(self.rn_tracker["read"]),
                "read_tracker_available": self.rn_tracker_count["read"],
                "write_tracker_active": len(self.rn_tracker["write"]),
                "write_tracker_available": self.rn_tracker_count["write"],
                "rdb_available": self.rn_rdb_count,
                "wdb_available": self.rn_wdb_count,
                "rdb_reserve": self.rn_rdb_reserve,
            },
            "sn_resources": {
                "tracker_active": len(self.sn_tracker),
                "tracker_ro_available": self.sn_tracker_count["ro"],
                "tracker_share_available": self.sn_tracker_count["share"],
                "wdb_available": self.sn_wdb_count,
                "req_wait_read": len(self.sn_req_wait["read"]),
                "req_wait_write": len(self.sn_req_wait["write"]),
            },
            "statistics": {
                "read_retries": self.read_retry_num_stat,
                "write_retries": self.write_retry_num_stat,
                "req_wait_cycles_h": self.req_wait_cycles_h,
                "req_wait_cycles_v": self.req_wait_cycles_v,
                "req_circuits_h": self.req_cir_h_num,
                "req_circuits_v": self.req_cir_v_num,
            },
            "fifo_status": {
                channel: {
                    "pending": len(self.pending_by_channel[channel]),
                    "l2h": len(self.l2h_fifos[channel]),
                    "l2h_valid": self.l2h_fifos[channel].valid_signal(),
                    "l2h_ready": self.l2h_fifos[channel].ready_signal(),
                    "h2l": len(self.h2l_fifos[channel]),
                    "h2l_valid": self.h2l_fifos[channel].valid_signal(),
                    "h2l_ready": self.h2l_fifos[channel].ready_signal(),
                }
                for channel in ["req", "rsp", "data"]
            },
        }

    # ========== 实现抽象方法 ==========

    def _can_handle_new_read_request(self, source: NodeId, destination: NodeId, burst_length: int) -> bool:
        """检查是否可以处理新的读请求"""
        # 检查RN读tracker是否有空间
        if self.rn_tracker_count["read"] <= 0:
            return False

        # 检查RN读数据库是否有空间
        if self.rn_rdb_count < burst_length:
            return False

        return True

    def _can_handle_new_write_request(self, source: NodeId, destination: NodeId, burst_length: int) -> bool:
        """检查是否可以处理新的写请求"""
        # 检查RN写tracker是否有空间
        if self.rn_tracker_count["write"] <= 0:
            return False

        # 检查RN写数据库是否有空间
        if self.rn_wdb_count < burst_length:
            return False

        return True

    def _process_read_request(self, source: NodeId, destination: NodeId, burst_length: int, packet_id: str) -> bool:
        """处理读请求"""
        try:
            # 分配RN资源
            if not self._allocate_rn_resources("read", burst_length):
                return False

            # 创建读请求flit
            req_flit = create_crossring_flit(source, destination, [source, destination])
            req_flit.packet_id = packet_id
            req_flit.req_type = "read"
            req_flit.burst_length = burst_length
            req_flit.channel = "req"
            req_flit.req_attr = "new"

            # 添加到RN tracker
            self.rn_tracker["read"].append(req_flit)

            # 注入到网络
            return self._inject_to_network(req_flit)

        except Exception as e:
            self.logger.error(f"处理读请求失败: {e}")
            return False

    def _process_write_request(self, source: NodeId, destination: NodeId, burst_length: int, packet_id: str) -> bool:
        """处理写请求"""
        try:
            # 分配RN资源
            if not self._allocate_rn_resources("write", burst_length):
                return False

            # 创建写请求flit
            req_flit = create_crossring_flit(source, destination, [source, destination])
            req_flit.packet_id = packet_id
            req_flit.req_type = "write"
            req_flit.burst_length = burst_length
            req_flit.channel = "req"
            req_flit.req_attr = "new"

            # 添加到RN tracker
            self.rn_tracker["write"].append(req_flit)

            # 创建写数据flits
            self._create_write_data_flits(req_flit)

            # 注入到网络
            return self._inject_to_network(req_flit)

        except Exception as e:
            self.logger.error(f"处理写请求失败: {e}")
            return False

    def _allocate_rn_resources(self, req_type: str, burst_length: int) -> bool:
        """
        分配RN资源（整合读写资源分配）

        Args:
            req_type: 请求类型 ("read" 或 "write")
            burst_length: 突发长度

        Returns:
            是否分配成功
        """
        if req_type == "read":
            # 检查读资源
            if self.rn_tracker_count["read"] <= 0 or self.rn_rdb_count < burst_length:
                return False
            # 分配读资源
            self.rn_tracker_count["read"] -= 1
            self.rn_rdb_count -= burst_length
            self.rn_rdb_reserve += 1
        elif req_type == "write":
            # 检查写资源
            if self.rn_tracker_count["write"] <= 0 or self.rn_wdb_count < burst_length:
                return False
            # 分配写资源
            self.rn_tracker_count["write"] -= 1
            self.rn_wdb_count -= burst_length
        else:
            return False

        return True

    def _create_write_data_flits(self, req_flit: CrossRingFlit) -> None:
        """创建写数据flits"""
        for i in range(req_flit.burst_length):
            data_flit = create_crossring_flit(req_flit.source, req_flit.destination, req_flit.path)
            data_flit.packet_id = f"{req_flit.packet_id}_data_{i}"
            data_flit.channel = "data"
            data_flit.flit_id = i
            data_flit.is_last_flit = i == req_flit.burst_length - 1

            # 添加到写数据库
            if req_flit.packet_id not in self.rn_wdb:
                self.rn_wdb[req_flit.packet_id] = []
            self.rn_wdb[req_flit.packet_id].append(data_flit)

    def _inject_to_network(self, flit: CrossRingFlit) -> bool:
        """将flit注入到网络"""
        try:
            # 添加到注入FIFO
            if len(self.inject_fifos[flit.channel]) < self.config.inject_buffer_depth:
                self.inject_fifos[flit.channel].append(flit)
                flit.departure_cycle = self.current_cycle
                return True
            else:
                return False
        except Exception as e:
            self.logger.error(f"注入网络失败: {e}")
            return False

    def inject_request(
        self, source: NodeId, destination: NodeId, req_type: str, burst_length: int = 4, packet_id: str = None, source_type: str = None, destination_type: str = None, **kwargs
    ) -> bool:
        """
        注入请求到IP接口，保证请求永不丢失

        数据流： inject_request -> pending_by_channel -> L2H -> Node channel_buffer

        Args:
            source: 源节点ID
            destination: 目标节点ID
            req_type: 请求类型 ("read" | "write")
            burst_length: 突发长度
            packet_id: 包ID
            source_type: 源IP类型（从traffic文件获取）
            destination_type: 目标IP类型（从traffic文件获取）
            **kwargs: 其他参数

        Returns:
            总是返回True（请求被添加到pending_by_channel队列）
        """
        if not packet_id:
            packet_id = f"{req_type}_{source}_{destination}_{self.current_cycle}"

        try:
            # 创建CrossRing Flit
            flit = create_crossring_flit(source=source, destination=destination, packet_id=packet_id, req_type=req_type, burst_length=burst_length, num_col=self.config.num_col)

            # 设置IP类型信息
            flit.source_type = source_type if source_type else self.ip_type
            flit.destination_type = destination_type if destination_type else "unknown"

            # 如果没有提供destination_type，记录警告
            if not destination_type:
                self.logger.warning(f"⚠️ 没有提供destination_type参数，使用'unknown'作为默认值。建议从traffic文件传入正确的destination_type。")
            flit.channel = "req"
            flit.inject_cycle = kwargs.get("inject_cycle", self.current_cycle)

            # 注册到请求追踪器
            if hasattr(self.model, "request_tracker"):
                self.model.request_tracker.start_request(
                    packet_id=packet_id, source=source, destination=destination, op_type=req_type, burst_size=burst_length, cycle=kwargs.get("inject_cycle", self.current_cycle)
                )

            # 设置flit位置信息
            flit.flit_position = "L2H"
            flit.current_node_id = self.node_id

            # 对于read请求，需要在RN端预占资源以接收返回的data
            if req_type == "read":
                if not self._check_and_reserve_resources(flit):
                    self.logger.warning(f"⚠️ RN端资源不足，read请求 {packet_id} 仍会发送但可能导致数据接收失败")
                    # 即使资源不足也要创建rn_rdb条目，避免KeyError
                    if flit.packet_id not in self.rn_rdb:
                        self.rn_rdb[flit.packet_id] = []

            # 对于write请求，需要在RN端预占WDB资源以存储待发送的data
            elif req_type == "write":
                if not self._check_and_reserve_resources(flit):
                    self.logger.warning(f"⚠️ RN端资源不足，write请求 {packet_id} 仍会发送但可能导致数据发送失败")
                # 确保创建rn_wdb条目，避免KeyError
                if flit.packet_id not in self.rn_wdb:
                    self.rn_wdb[flit.packet_id] = []
                print(f"📝 Write请求{packet_id}创建WDB条目: {self.rn_wdb[flit.packet_id]}")

            # 添加到pending_by_channel队列（无限大，永不失败）
            self.pending_by_channel["req"].append(flit)

            # 添加到活跃请求追踪
            self.active_requests[packet_id] = {
                "flit": flit,
                "source": source,
                "destination": destination,
                "req_type": req_type,
                "burst_length": burst_length,
                "source_type": source_type if source_type else self.ip_type,
                "destination_type": destination_type if destination_type else "unknown",
                "inject_cycle": kwargs.get("inject_cycle", self.current_cycle),
                "created_cycle": self.current_cycle,
                "stage": "pending",
            }

            self.logger.debug(f"请求已添加到pending_by_channel: {packet_id} ({req_type}: {source}->{destination})")
            return True

        except Exception as e:
            self.logger.error(f"添加请求到pending_by_channel失败: {e}")
            return False

    def step(self, current_cycle: int) -> None:
        """
        IP接口周期步进，处理pending_by_channel -> L2H -> Node的数据流

        Args:
            current_cycle: 当前周期
        """
        self.current_cycle = current_cycle

        # 执行计算阶段
        self.step_compute_phase(current_cycle)

        # 执行更新阶段
        self.step_update_phase(current_cycle)

    def step_compute_phase(self, current_cycle: int) -> None:
        """计算阶段：计算传输决策但不执行"""

        # 初始化传输决策存储
        self._transfer_decisions = {
            "pending_to_l2h": {"channel": None, "flit": None},
            "l2h_to_node": {"channel": None, "flit": None},
            "network_to_h2l": {"channel": None, "flit": None},
            "h2l_to_completion": {"channel": None, "flit": None},
        }

        # 1. 计算pending到l2h的传输决策
        self._compute_pending_to_l2h_decision(current_cycle)

        # 2. 计算l2h到node的传输决策
        self._compute_l2h_to_node_decision(current_cycle)

        # 3. 计算network到h2l的传输决策
        self._compute_network_to_h2l_decision(current_cycle)

        # 4. 计算h2l到completion的传输决策
        self._compute_h2l_to_completion_decision(current_cycle)

        # 更新所有FIFO的计算阶段
        for channel in ["req", "rsp", "data"]:
            self.l2h_fifos[channel].step_compute_phase()
            self.h2l_fifos[channel].step_compute_phase()

    def _compute_pending_to_l2h_decision(self, current_cycle: int) -> None:
        """计算pending到l2h的传输决策"""
        # 按优先级顺序检查：req > rsp > data
        for channel in ["req", "rsp", "data"]:
            if self.pending_by_channel[channel] and self.l2h_fifos[channel].ready_signal():
                flit = self.pending_by_channel[channel][0]
                if flit.departure_cycle <= current_cycle:
                    self._transfer_decisions["pending_to_l2h"]["channel"] = channel
                    self._transfer_decisions["pending_to_l2h"]["flit"] = flit
                    return

    def _compute_l2h_to_node_decision(self, current_cycle: int) -> None:
        """计算l2h到node的传输决策"""
        # # 只有当pending到l2h没有传输时才考虑l2h到node
        # if self._transfer_decisions["pending_to_l2h"]["channel"] is not None:
        #     return

        # 按优先级顺序检查：req > rsp > data
        for channel in ["req", "rsp", "data"]:
            if self.l2h_fifos[channel].valid_signal():
                flit = self.l2h_fifos[channel].peek_output()
                if flit and self._can_inject_to_node(flit, channel):
                    self._transfer_decisions["l2h_to_node"]["channel"] = channel
                    self._transfer_decisions["l2h_to_node"]["flit"] = flit
                    return

    def _compute_network_to_h2l_decision(self, current_cycle: int) -> None:
        """计算network到h2l的传输决策"""
        # 按优先级顺序检查：req > rsp > data
        for channel in ["req", "rsp", "data"]:
            if self.h2l_fifos[channel].ready_signal():
                flit = self._peek_from_topology_network(channel)
                if flit:
                    self._transfer_decisions["network_to_h2l"]["channel"] = channel
                    self._transfer_decisions["network_to_h2l"]["flit"] = flit
                    return

    def _compute_h2l_to_completion_decision(self, current_cycle: int) -> None:
        """计算h2l到completion的传输决策"""
        # # 只有当network到h2l没有传输时才考虑h2l到completion
        # if self._transfer_decisions["network_to_h2l"]["channel"] is not None:
        #     return

        # 按优先级顺序检查：req > rsp > data
        for channel in ["req", "rsp", "data"]:
            if self.h2l_fifos[channel].valid_signal():
                flit = self.h2l_fifos[channel].peek_output()
                if flit:
                    self._transfer_decisions["h2l_to_completion"]["channel"] = channel
                    self._transfer_decisions["h2l_to_completion"]["flit"] = flit
                    return

    def _can_inject_to_node(self, flit, channel: str) -> bool:
        """检查是否可以注入到node"""
        # 获取对应的节点
        if self.node_id in self.model.crossring_nodes:
            node = self.model.crossring_nodes[self.node_id]
            ip_key = f"{self.ip_type}_node{self.node_id}"

            if ip_key in node.ip_inject_channel_buffers:
                inject_buffer = node.ip_inject_channel_buffers[ip_key][channel]
                return inject_buffer.ready_signal()
        return False

    def _peek_from_topology_network(self, channel: str):
        """查看network中是否有可eject的flit"""
        # 获取对应的节点
        if self.node_id in self.model.crossring_nodes:
            node = self.model.crossring_nodes[self.node_id]
            ip_key = f"{self.ip_type}_node{self.node_id}"

            if ip_key in node.ip_eject_channel_buffers:
                eject_buffer = node.ip_eject_channel_buffers[ip_key][channel]
                if eject_buffer.valid_signal():
                    return eject_buffer.peek_output()
        return None

    def _process_pending_to_l2h(self, current_cycle: int) -> bool:
        """处理所有pending队列到L2H FIFO的传输"""
        # 按优先级顺序处理：req > rsp > data
        channels = ["req", "rsp", "data"]

        for channel in channels:
            # 统一使用pending_by_channel处理所有通道
            if self.pending_by_channel[channel] and self.l2h_fifos[channel].can_accept_input():
                flit = self.pending_by_channel[channel][0]  # 查看队首

                # 检查departure_cycle延迟
                if hasattr(flit, "departure_cycle") and current_cycle < flit.departure_cycle:
                    continue  # 还没到发送时间

                # 尝试写入L2H
                if self.l2h_fifos[channel].write_input(flit):
                    # 成功写入，从pending队列移除
                    self.pending_by_channel[channel].popleft()

                    # 更新flit位置信息
                    flit.flit_position = "L2H"

                    # 更新请求状态（仅对req通道）
                    if channel == "req" and hasattr(flit, "packet_id") and flit.packet_id in self.active_requests:
                        self.active_requests[flit.packet_id]["stage"] = "l2h_fifo"

                    return True

        return False

    def step_update_phase(self, current_cycle: int) -> None:
        """更新阶段：执行compute阶段的传输决策"""
        # 执行compute阶段的传输决策
        self._execute_transfer_decisions(current_cycle)

        # 更新所有FIFO的时序状态
        for channel in ["req", "rsp", "data"]:
            self.l2h_fifos[channel].step_update_phase()
            self.h2l_fifos[channel].step_update_phase()

    def _execute_transfer_decisions(self, current_cycle: int) -> None:
        """执行compute阶段计算的传输决策"""
        # 1. 执行pending到l2h的传输
        if self._transfer_decisions["pending_to_l2h"]["channel"]:
            channel = self._transfer_decisions["pending_to_l2h"]["channel"]
            flit = self._transfer_decisions["pending_to_l2h"]["flit"]

            # 从pending队列移除并写入l2h FIFO
            self.pending_by_channel[channel].popleft()
            flit.flit_position = "L2H"
            self.l2h_fifos[channel].write_input(flit)

            # 更新请求状态
            if channel == "req" and hasattr(flit, "packet_id") and flit.packet_id in self.active_requests:
                self.active_requests[flit.packet_id]["stage"] = "l2h_fifo"

        # 2. 执行l2h到node的传输
        if self._transfer_decisions["l2h_to_node"]["channel"]:
            channel = self._transfer_decisions["l2h_to_node"]["channel"]
            flit = self._transfer_decisions["l2h_to_node"]["flit"]

            # 从l2h FIFO读取并注入到node
            self.l2h_fifos[channel].read_output()
            self._inject_to_node(flit, channel)

        # 3. 执行network到h2l的传输
        if self._transfer_decisions["network_to_h2l"]["channel"]:
            channel = self._transfer_decisions["network_to_h2l"]["channel"]
            flit = self._transfer_decisions["network_to_h2l"]["flit"]

            # 从network eject并写入h2l FIFO
            self._eject_from_topology_network(channel)  # 这会执行实际的read
            flit.flit_position = "H2L"
            self.h2l_fifos[channel].write_input(flit)

        # 4. 执行h2l到completion的传输
        if self._transfer_decisions["h2l_to_completion"]["channel"]:
            channel = self._transfer_decisions["h2l_to_completion"]["channel"]
            flit = self._transfer_decisions["h2l_to_completion"]["flit"]

            # 从h2l FIFO读取并处理completion
            self.h2l_fifos[channel].read_output()
            flit.set_ejection_time(current_cycle)

            # 根据通道类型处理
            if channel == "req":
                self._handle_received_request(flit)
            elif channel == "rsp":
                self._handle_received_response(flit)
            elif channel == "data":
                self._handle_received_data(flit)

    def _inject_to_node(self, flit, channel: str) -> bool:
        """将flit注入到node"""
        # 获取对应的节点
        if self.node_id in self.model.crossring_nodes:
            node = self.model.crossring_nodes[self.node_id]
            ip_key = f"{self.ip_type}_node{self.node_id}"

            if ip_key in node.ip_inject_channel_buffers:
                inject_buffer = node.ip_inject_channel_buffers[ip_key][channel]
                if inject_buffer.write_input(flit):
                    # 更新flit位置
                    flit.flit_position = "IQ_CH"
                    flit.current_node_id = self.node_id

                    # 更新请求状态
                    if channel == "req" and hasattr(flit, "packet_id") and flit.packet_id in self.active_requests:
                        self.active_requests[flit.packet_id]["stage"] = "node_inject"

                        # ✅ 修复：添加flit到RequestTracker
                        if hasattr(self.model, "request_tracker"):
                            self.model.request_tracker.mark_request_injected(flit.packet_id, self.current_cycle)
                            self.model.request_tracker.add_request_flit(flit.packet_id, flit)

                    # 对于RSP和DATA也要追踪
                    elif hasattr(flit, "packet_id") and hasattr(self.model, "request_tracker"):
                        if channel == "rsp":
                            self.model.request_tracker.add_response_flit(flit.packet_id, flit)
                        elif channel == "data":
                            self.model.request_tracker.add_data_flit(flit.packet_id, flit)

                    self.logger.debug(f"IP {self.ip_type} 成功注入flit到节点{self.node_id}")
                    return True
        return False

    def _process_l2h_to_node(self, current_cycle: int) -> bool:
        """处理L2H到Node的传输

        Returns:
            bool: 是否执行了传输
        """
        # 按优先级顺序处理：req > rsp > data
        channels = ["req", "rsp", "data"]

        for channel in channels:
            if self.l2h_fifos[channel].valid_signal():
                flit = self.l2h_fifos[channel].peek_output()
                if flit:
                    # 尝试注入到拓扑网络（实际上是注入到对应的channel_buffer）
                    if self._inject_to_topology_network(flit, channel):
                        # 成功注入，从L2H移除
                        self.l2h_fifos[channel].read_output()

                        # 更新flit位置信息
                        flit.flit_position = "IQ_CH"
                        flit.current_node_id = self.node_id

                        # 更新请求状态
                        if hasattr(flit, "packet_id") and flit.packet_id in self.active_requests:
                            self.active_requests[flit.packet_id]["stage"] = "in_network"

                        # 每个IP每个周期只传输一个flit，传输完成后退出
                        return True

        return False

    def _process_network_to_h2l(self, current_cycle: int) -> bool:
        """处理网络到H2L的传输（eject）

        Returns:
            bool: 是否执行了传输
        """
        # 按优先级顺序处理：req > rsp > data
        channels = ["req", "rsp", "data"]

        for channel in channels:
            # 尝试从网络中eject flit
            flit = self._eject_from_topology_network(channel)
            if flit:
                # 成功获取到flit，写入h2l FIFO
                if self.h2l_fifos[channel].ready_signal():
                    flit.flit_position = "H2L"
                    self.h2l_fifos[channel].write_input(flit)

                    # 每个IP每个周期只传输一个flit
                    return True
                else:
                    # h2l FIFO满了，需要等待
                    self.logger.warning(f"⚠️ 周期{current_cycle}: h2l FIFO[{channel}]满，无法eject flit {flit.packet_id}")
                    return False

        return False

    def _process_h2l_to_completion(self, current_cycle: int) -> bool:
        """处理H2L到IP完成的传输

        Returns:
            bool: 是否执行了传输
        """
        # 按优先级顺序处理：req > rsp > data
        channels = ["req", "rsp", "data"]

        for channel in channels:
            if self.h2l_fifos[channel].valid_signal():
                flit = self.h2l_fifos[channel].read_output()
                if flit:
                    # 设置ejection时间
                    flit.set_ejection_time(current_cycle)

                    # 根据通道类型处理
                    if channel == "req":
                        self._handle_received_request(flit)
                    elif channel == "rsp":
                        self._handle_received_response(flit)
                    elif channel == "data":
                        self._handle_received_data(flit)

                    # 每个IP每个周期只完成一个flit
                    return True

        return False

    def _notify_request_arrived(self, req: CrossRingFlit) -> None:
        """通知RequestTracker请求已到达目标

        Args:
            req: 到达的请求flit
        """
        if hasattr(self.model, "request_tracker") and hasattr(req, "packet_id"):
            from src.noc.debug import RequestState

            try:
                self.model.request_tracker.update_request_state(req.packet_id, RequestState.ARRIVED, self.current_cycle)
                self.logger.debug(f"✅ 通知RequestTracker: 请求{req.packet_id}已到达")
            except Exception as e:
                self.logger.warning(f"⚠️ 通知RequestTracker失败: {e}")

    def _notify_request_completion(self, req: CrossRingFlit) -> None:
        """通知RequestTracker请求已完成

        Args:
            req: 完成的请求flit
        """
        if hasattr(self.model, "request_tracker") and hasattr(req, "packet_id"):
            from src.noc.debug import RequestState

            try:
                self.model.request_tracker.update_request_state(req.packet_id, RequestState.COMPLETED, self.current_cycle)
                self.logger.debug(f"✅ 通知RequestTracker: 请求{req.packet_id}已完成")
            except Exception as e:
                self.logger.warning(f"⚠️ 通知RequestTracker失败: {e}")
