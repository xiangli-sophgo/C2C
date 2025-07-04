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


class CrossRingIPInterface:
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
        self.config = config
        self.ip_type = ip_type
        self.node_id = node_id
        self.model = model
        self.current_cycle = 0

        # ========== 时钟域转换（1GHz ↔ 2GHz） ==========
        self.clock_ratio = config.basic_config.network_frequency
        self.l2h_fifos = {
            "req": deque(maxlen=getattr(config, "IP_L2H_FIFO_DEPTH", 4)),
            "rsp": deque(maxlen=getattr(config, "IP_L2H_FIFO_DEPTH", 4)),
            "data": deque(maxlen=getattr(config, "IP_L2H_FIFO_DEPTH", 4)),
        }
        self.h2l_fifos = {
            "req": deque(maxlen=getattr(config, "IP_H2L_FIFO_DEPTH", 4)),
            "rsp": deque(maxlen=getattr(config, "IP_H2L_FIFO_DEPTH", 4)),
            "data": deque(maxlen=getattr(config, "IP_H2L_FIFO_DEPTH", 4)),
        }

        # L2H和H2L的预缓冲区
        self.l2h_pre_buffers = {"req": None, "rsp": None, "data": None}
        self.h2l_pre_buffers = {"req": None, "rsp": None, "data": None}

        # IP inject/eject FIFO（1GHz域）
        self.inject_fifos = {"req": deque(), "rsp": deque(), "data": deque()}

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

        # 注册到模型
        if hasattr(model, "register_ip_interface"):
            model.register_ip_interface(self)

    def step(self, cycle: int) -> None:
        """
        执行一个周期（包含时钟域处理）

        Args:
            cycle: 当前仿真周期
        """
        self.current_cycle = cycle

        # 处理SN tracker延迟释放
        self._process_sn_tracker_release()

        # 1GHz操作（IP频率）
        if cycle % self.clock_ratio == 0:
            self._step_1ghz()

        # 2GHz操作（网络频率）
        self._step_2ghz()

        # 预缓冲区移动（每周期都执行）
        self._move_pre_to_fifo()

    def _step_1ghz(self) -> None:
        """1GHz域操作"""
        # IP发起新请求（实际应用中由外部调用enqueue_request）
        # self._ip_generate_requests()

        # H2L FIFO → IP完成
        self._h2l_to_ip_completion()

    def _step_2ghz(self) -> None:
        """2GHz域操作"""
        # L2H FIFO → 网络注入
        self._l2h_to_network()

        # 网络 → H2L FIFO接收
        self._network_to_h2l()

    def _move_pre_to_fifo(self) -> None:
        """预缓冲区到正式FIFO的移动"""
        for channel in ["req", "rsp", "data"]:
            # L2H预缓冲区 → L2H FIFO
            if self.l2h_pre_buffers[channel] is not None and len(self.l2h_fifos[channel]) < self.l2h_fifos[channel].maxlen:
                self.l2h_fifos[channel].append(self.l2h_pre_buffers[channel])
                self.l2h_pre_buffers[channel] = None

            # H2L预缓冲区 → H2L FIFO
            if self.h2l_pre_buffers[channel] is not None and len(self.h2l_fifos[channel]) < self.h2l_fifos[channel].maxlen:
                self.h2l_fifos[channel].append(self.h2l_pre_buffers[channel])
                self.h2l_pre_buffers[channel] = None

    def enqueue_request(self, source: NodeId, destination: NodeId, req_type: str, burst_length: int = 4, packet_id: str = None, **kwargs) -> bool:
        """
        将新请求加入inject FIFO

        Args:
            source: 源节点
            destination: 目标节点
            req_type: 请求类型 ("read" | "write")
            burst_length: 突发长度
            packet_id: 包ID
            **kwargs: 其他参数

        Returns:
            是否成功入队
        """
        if packet_id is None:
            packet_id = f"{self.ip_type}_{self.node_id}_{self.current_cycle}_{len(self.inject_fifos['req'])}"

        # 创建请求flit
        flit = create_crossring_flit(
            source=source,
            destination=destination,
            req_type=req_type,
            burst_length=burst_length,
            packet_id=packet_id,
            channel="req",
            flit_type="req",
            cmd_entry_cake0_cycle=self.current_cycle,
            **kwargs,
        )

        # 设置IP类型信息
        flit.source_type = self.ip_type
        flit.destination_type = self._get_destination_type(destination)
        flit.original_source_type = flit.source_type
        flit.original_destination_type = flit.destination_type

        self.inject_fifos["req"].append(flit)
        return True

    def _get_destination_type(self, destination: NodeId) -> str:
        """根据目标节点ID推断目标IP类型"""
        # 这里需要根据实际的节点-IP映射来实现
        # 简化实现：假设前半部分是ddr，后半部分是l2m
        if destination < self.config.num_nodes // 2:
            return "ddr"
        else:
            return "l2m"

    def _inject_to_l2h_pre(self, channel: str) -> None:
        """
        1GHz: inject_fifo → l2h_pre_buffer

        Args:
            channel: 通道名称
        """
        if not self.inject_fifos[channel] or len(self.l2h_fifos[channel]) >= self.l2h_fifos[channel].maxlen or self.l2h_pre_buffers[channel] is not None:
            return

        flit = self.inject_fifos[channel][0]

        # 根据通道类型进行不同的处理
        if channel == "req":
            # 检查并预占资源
            if flit.req_attr == "new" and not self._check_and_reserve_rn_resources(flit):
                return  # 资源不足，保持在inject_fifo中

            flit.flit_position = "L2H_FIFO"
            flit.start_inject = True
            self.l2h_pre_buffers[channel] = self.inject_fifos[channel].popleft()

        elif channel == "rsp":
            # 响应直接移动
            flit.flit_position = "L2H_FIFO"
            flit.start_inject = True
            self.l2h_pre_buffers[channel] = self.inject_fifos[channel].popleft()

        elif channel == "data":
            # 检查发送时间
            if hasattr(flit, "departure_cycle") and flit.departure_cycle > self.current_cycle:
                return

            flit.flit_position = "L2H_FIFO"
            flit.start_inject = True
            self.l2h_pre_buffers[channel] = self.inject_fifos[channel].popleft()

    def _l2h_to_network(self) -> None:
        """2GHz: l2h_fifo → 网络IQ"""
        for channel in ["req", "rsp", "data"]:
            if not self.l2h_fifos[channel]:
                continue

            # 模拟注入到网络（实际实现中需要与网络模块对接）
            flit = self.l2h_fifos[channel].popleft()
            flit.flit_position = "IQ_CH"

            # 更新时间戳
            if channel == "req" and flit.req_attr == "new":
                flit.cmd_entry_noc_from_cake0_cycle = self.current_cycle
            elif channel == "rsp":
                flit.cmd_entry_noc_from_cake1_cycle = self.current_cycle
            elif channel == "data":
                if flit.req_type == "read":
                    flit.data_entry_noc_from_cake1_cycle = self.current_cycle
                elif flit.req_type == "write":
                    flit.data_entry_noc_from_cake0_cycle = self.current_cycle

            # 这里应该调用网络的inject方法
            # network.inject_flit(flit, channel)

    def _network_to_h2l(self) -> None:
        """2GHz: 网络EQ → h2l_pre_buffer"""
        for channel in ["req", "rsp", "data"]:
            if self.h2l_pre_buffers[channel] is not None:
                continue

            # 这里应该从网络的EQ获取flit
            # flit = network.eject_flit(self.ip_type, self.node_id, channel)
            # if flit:
            #     flit.is_arrive = True
            #     flit.flit_position = "H2L_FIFO"
            #     self.h2l_pre_buffers[channel] = flit

    def _h2l_to_ip_completion(self) -> None:
        """1GHz: h2l_fifo → IP处理完成"""
        for channel in ["req", "rsp", "data"]:
            if not self.h2l_fifos[channel]:
                continue

            flit = self.h2l_fifos[channel].popleft()
            flit.flit_position = "IP_eject"
            flit.is_finish = True

            # 根据通道类型进行处理
            if channel == "req":
                self._handle_received_request(flit)
            elif channel == "rsp":
                self._handle_received_response(flit)
            elif channel == "data":
                self._handle_received_data(flit)

    def _check_and_reserve_rn_resources(self, req: CrossRingFlit) -> bool:
        """
        检查并预占RN端资源

        Args:
            req: 请求flit

        Returns:
            是否成功预占资源
        """
        if req.req_type == "read":
            # 检查读资源：tracker + rdb + reserve
            rdb_available = self.rn_rdb_count >= req.burst_length
            tracker_available = self.rn_tracker_count["read"] > 0
            reserve_ok = self.rn_rdb_count > self.rn_rdb_reserve * req.burst_length

            if not (rdb_available and tracker_available and reserve_ok):
                return False

            # 预占资源
            self.rn_rdb_count -= req.burst_length
            self.rn_tracker_count["read"] -= 1
            self.rn_rdb[req.packet_id] = []
            self.rn_tracker["read"].append(req)
            self.rn_tracker_pointer["read"] += 1

        elif req.req_type == "write":
            # 检查写资源：tracker + wdb
            wdb_available = self.rn_wdb_count >= req.burst_length
            tracker_available = self.rn_tracker_count["write"] > 0

            if not (wdb_available and tracker_available):
                return False

            # 预占资源
            self.rn_wdb_count -= req.burst_length
            self.rn_tracker_count["write"] -= 1
            self.rn_wdb[req.packet_id] = []
            self.rn_tracker["write"].append(req)
            self.rn_tracker_pointer["write"] += 1

            # 创建写数据包
            self._create_write_packet(req)

        return True

    def _handle_received_request(self, req: CrossRingFlit) -> None:
        """
        处理收到的请求（SN端）

        Args:
            req: 收到的请求flit
        """
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
                    self._create_read_packet(req)
                    self._release_completed_sn_tracker(req)
                else:
                    # 资源不足，发送negative响应
                    self._create_negative_response(req)
                    self.sn_req_wait["read"].append(req)
            else:
                # 重试读请求：直接处理
                self._create_read_packet(req)
                self._release_completed_sn_tracker(req)

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
        """处理读响应"""
        if rsp.rsp_type == "negative":
            # 读重试逻辑
            if req.req_attr == "old":
                return  # 已经在重试中

            req.reset_for_retry()
            self.rn_rdb_count += req.burst_length
            if req.packet_id in self.rn_rdb:
                del self.rn_rdb[req.packet_id]
            self.rn_rdb_reserve += 1

        elif rsp.rsp_type == "positive":
            # 处理positive响应：准备重试
            if req.req_attr == "new":
                self.rn_rdb_count += req.burst_length
                if req.packet_id in self.rn_rdb:
                    del self.rn_rdb[req.packet_id]
                self.rn_rdb_reserve += 1

            req.req_state = "valid"
            req.req_attr = "old"
            req.is_injected = False
            req.path_index = 0
            req.is_new_on_network = True
            req.is_arrive = False

            # 重新放入请求队列
            self.inject_fifos["req"].appendleft(req)
            self.rn_rdb_reserve -= 1

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
            self.inject_fifos["req"].appendleft(req)

        elif rsp.rsp_type == "datasend":
            # 发送写数据
            for flit in self.rn_wdb[rsp.packet_id]:
                self.inject_fifos["data"].append(flit)

            # 释放RN write tracker
            if req in self.rn_tracker["write"]:
                self.rn_tracker["write"].remove(req)
                self.rn_wdb_count += req.burst_length
                self.rn_tracker_count["write"] += 1
                self.rn_tracker_pointer["write"] -= 1

            # 清理写缓冲
            if rsp.packet_id in self.rn_wdb:
                del self.rn_wdb[rsp.packet_id]

    def _create_write_packet(self, req: CrossRingFlit) -> None:
        """创建写数据包"""
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
            data_flit.original_source_type = req.original_source_type
            data_flit.original_destination_type = req.original_destination_type
            data_flit.is_last_flit = i == req.burst_length - 1

            self.rn_wdb[req.packet_id].append(data_flit)

    def _create_read_packet(self, req: CrossRingFlit) -> None:
        """创建读数据包"""
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
            data_flit.original_source_type = req.original_destination_type
            data_flit.original_destination_type = req.original_source_type
            data_flit.is_last_flit = i == req.burst_length - 1

            self.inject_fifos["data"].append(data_flit)

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

        self.inject_fifos["rsp"].append(rsp)

    def _create_datasend_response(self, req: CrossRingFlit) -> None:
        """创建datasend响应"""
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

        self.inject_fifos["rsp"].append(rsp)

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

                # 发送positive响应
                self._create_positive_response(new_req)

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

    def _create_positive_response(self, req: CrossRingFlit) -> None:
        """创建positive响应"""
        rsp = create_crossring_flit(
            source=req.destination,
            destination=req.source,
            req_type=req.req_type,
            packet_id=req.packet_id,
            channel="rsp",
            flit_type="rsp",
            rsp_type="positive",
            departure_cycle=self.current_cycle + self.clock_ratio,
        )

        rsp.sync_latency_record(req)
        rsp.source_type = req.destination_type
        rsp.destination_type = req.source_type
        rsp.sn_rsp_generate_cycle = self.current_cycle

        self.inject_fifos["rsp"].append(rsp)

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
                    "inject": len(self.inject_fifos[channel]),
                    "l2h": len(self.l2h_fifos[channel]),
                    "h2l": len(self.h2l_fifos[channel]),
                    "l2h_pre": self.l2h_pre_buffers[channel] is not None,
                    "h2l_pre": self.h2l_pre_buffers[channel] is not None,
                }
                for channel in ["req", "rsp", "data"]
            },
        }
