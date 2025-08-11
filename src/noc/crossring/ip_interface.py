"""
CrossRing专用IP接口实现。

基于C2C仓库的现有结构，结合CrossRing仓库的IP接口实现，
为CrossRing拓扑提供专用的IP接口管理，包括时钟域转换、资源管理和STI协议处理。
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Deque
from collections import deque, defaultdict
import logging
import numpy as np

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
        self.rn_tracker_count = {"read": config.tracker_config.RN_R_TRACKER_OSTD, "write": config.tracker_config.RN_W_TRACKER_OSTD}
        self.rn_tracker_pointer = {"read": 0, "write": 0}

        # RN Data Buffer
        self.rn_rdb = {}  # 读数据缓冲 {packet_id: [flits]}
        self.rn_rdb_count = config.RN_RDB_SIZE
        self.rn_rdb_reserve = 0  # 预留数量用于重试

        self.rn_wdb = {}  # 写数据缓冲 {packet_id: [flits]}
        self.rn_wdb_count = config.RN_WDB_SIZE

        # ========== SN资源管理 ==========
        self.sn_tracker = []

        # 根据IP类型设置SN tracker数量
        if ip_type.startswith("ddr"):
            self.sn_tracker_count = {"ro": config.tracker_config.SN_DDR_R_TRACKER_OSTD, "share": config.tracker_config.SN_DDR_W_TRACKER_OSTD}  # 读专用  # 写共享
            self.sn_wdb_count = config.SN_DDR_WDB_SIZE
        elif ip_type.startswith("l2m"):
            self.sn_tracker_count = {"ro": config.tracker_config.SN_L2M_R_TRACKER_OSTD, "share": config.tracker_config.SN_L2M_W_TRACKER_OSTD}
            self.sn_wdb_count = config.SN_L2M_WDB_SIZE
        else:
            # DMA类IP通常不作为SN
            self.sn_tracker_count = {"ro": 0, "share": 0}
            self.sn_wdb_count = 0

        self.sn_wdb = {}  # SN写数据缓冲

        # ✅ 修复：简化为FIFO等待队列结构
        self.sn_req_wait = {"read": deque(), "write": deque()}

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

        # 创建分通道的pending队列，替代父类pending_requests
        self.pending_by_channel = {"req": deque(), "rsp": deque(), "data": deque()}

        # ========== 初始化带宽限制 ==========
        self._initialize_token_bucket()

    def _initialize_token_bucket(self) -> None:
        """根据IP类型初始化令牌桶"""
        # 获取FLIT_SIZE配置（字节）
        flit_size = self.config.basic_config.FLIT_SIZE

        # 获取网络频率（GHz）
        network_freq_ghz = self.config.basic_config.NETWORK_FREQUENCY

        # 根据IP类型设置带宽限制
        if self.ip_type.startswith("ddr"):
            # DDR通道限速
            bw_limit_gbps = self.config.ip_config.DDR_BW_LIMIT  # GB/s
            # 转换为每网络周期的flit数
            # 网络频率是 GHz，即每秒 10^9 个周期
            # 每周期的字节数 = bw_limit_gbps * 10^9 字节/秒 / (network_freq_ghz * 10^9 周期/秒)
            #                = bw_limit_gbps / network_freq_ghz 字节/周期
            bytes_per_cycle = bw_limit_gbps / network_freq_ghz  # 字节/周期
            rate = bytes_per_cycle / flit_size  # flits/周期
            self._setup_dual_token_buckets(rate=rate, bucket_size=bw_limit_gbps)

        elif self.ip_type.startswith("l2m"):
            # L2M通道限速
            bw_limit_gbps = self.config.ip_config.L2M_BW_LIMIT
            bytes_per_cycle = bw_limit_gbps / network_freq_ghz  # GB/周期
            rate = bytes_per_cycle * 1e9 / flit_size  # flits/周期
            self._setup_dual_token_buckets(rate=rate, bucket_size=bw_limit_gbps)

        elif self.ip_type.startswith("gdma"):
            # GDMA通道限速
            bw_limit_gbps = self.config.ip_config.GDMA_BW_LIMIT
            bytes_per_cycle = bw_limit_gbps / network_freq_ghz  # GB/周期
            rate = bytes_per_cycle * 1e9 / flit_size  # flits/周期
            self._setup_dual_token_buckets(rate=rate, bucket_size=bw_limit_gbps)

        elif self.ip_type.startswith("sdma"):
            # SDMA通道限速
            bw_limit_gbps = self.config.ip_config.SDMA_BW_LIMIT
            bytes_per_cycle = bw_limit_gbps / network_freq_ghz  # GB/周期
            rate = bytes_per_cycle * 1e9 / flit_size  # flits/周期
            self._setup_dual_token_buckets(rate=rate, bucket_size=bw_limit_gbps)

        elif self.ip_type.startswith("cdma"):
            # CDMA通道限速
            bw_limit_gbps = self.config.ip_config.CDMA_BW_LIMIT
            bytes_per_cycle = bw_limit_gbps / network_freq_ghz  # GB/周期
            rate = bytes_per_cycle * 1e9 / flit_size  # flits/周期
            self._setup_dual_token_buckets(rate=rate, bucket_size=bw_limit_gbps)

        else:
            # 默认不限速
            self.tx_token_bucket = None
            self.rx_token_bucket = None

    def _setup_dual_token_buckets(self, rate: float, bucket_size: float) -> None:
        """
        设置双令牌桶用于TX和RX独立带宽限制

        Args:
            rate: 每周期生成的令牌数
            bucket_size: 桶的最大容量
        """
        from src.noc.utils.token_bucket import TokenBucket

        self.tx_token_bucket = TokenBucket(rate=rate, bucket_size=bucket_size)
        self.rx_token_bucket = TokenBucket(rate=rate, bucket_size=bucket_size)

    def _check_and_reserve_resources(self, flit) -> bool:
        """检查并预占RN端资源"""
        if flit.req_type == "read":
            # 检查是否已经在tracker中（避免重复添加）
            for existing_req in self.rn_tracker["read"]:
                if hasattr(existing_req, "packet_id") and hasattr(flit, "packet_id") and existing_req.packet_id == flit.packet_id:
                    return True  # 已经预占过资源，直接返回成功

            # 检查读资源：tracker + rdb（包含预留空间）
            rdb_available = self.rn_rdb_count >= flit.burst_length + self.rn_rdb_reserve
            tracker_available = self.rn_tracker_count["read"] > 0

            if not (rdb_available and tracker_available):
                return False

            # 预占资源
            self.rn_rdb_count -= flit.burst_length
            self.rn_tracker_count["read"] -= 1

            # ✅ 修复：检查RDB条目是否已存在（retry场景下可能已有数据）
            if flit.packet_id not in self.rn_rdb:
                self.rn_rdb[flit.packet_id] = []

            # 记录cmd_entry_cake0_cycle
            if hasattr(flit, "cmd_entry_cake0_cycle") and (flit.cmd_entry_cake0_cycle == np.inf or flit.cmd_entry_cake0_cycle == -1):
                flit.cmd_entry_cake0_cycle = self.current_cycle

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
            self.rn_wdb[flit.packet_id] = []  # 只创建空的WDB条目
            # 记录cmd_entry_cake0_cycle
            if hasattr(flit, "cmd_entry_cake0_cycle") and (flit.cmd_entry_cake0_cycle == np.inf or flit.cmd_entry_cake0_cycle == -1):
                flit.cmd_entry_cake0_cycle = self.current_cycle
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
        if self.node_id in self.model.nodes:
            node = self.model.nodes[self.node_id]

            # 注入到节点的对应IP的channel buffer
            ip_key = self.ip_type
            if ip_key in node.ip_inject_channel_buffers:
                channel_buffer = node.ip_inject_channel_buffers[ip_key][channel]
                if channel_buffer.can_accept_input():
                    if channel_buffer.write_input(flit):
                        # 更新flit位置信息
                        flit.flit_position = "IQ_CH"
                        flit.current_node_id = self.node_id

                        # 更新时间戳
                        if channel == "req" and hasattr(flit, "req_attr") and flit.req_attr == "new":
                            # 设置命令从源IP进入NoC的时间（进入channel buffer）
                            flit.cmd_entry_noc_from_cake0_cycle = self.current_cycle
                        elif channel == "rsp":
                            # 响应从目标IP进入NoC的时间
                            flit.cmd_entry_noc_from_cake1_cycle = self.current_cycle
                        elif channel == "data":
                            # 只有第一个data flit设置entry时间戳
                            if flit.flit_id == 0:
                                if flit.req_type == "read":
                                    # 读数据从目标IP进入NoC的时间
                                    flit.data_entry_noc_from_cake1_cycle = self.current_cycle
                                elif flit.req_type == "write":
                                    # 写数据从源IP进入NoC的时间
                                    flit.data_entry_noc_from_cake0_cycle = self.current_cycle

                        # 更新RequestTracker状态：flit成功注入到网络
                        if hasattr(self.model, "request_tracker") and hasattr(flit, "packet_id"):
                            if channel == "req":
                                self.model.request_tracker.mark_request_injected(flit.packet_id, self.current_cycle)
                                # request flit已经在inject_request时添加，这里不重复添加
                            # response和data flit已经在创建时添加，这里不重复添加

                        # ✅ 写数据完成检查：如果是写请求的最后一个数据flit，释放RN tracker
                        # 注意：现在改为在datasend响应处理后，数据实际发送完成时才释放
                        if channel == "data" and hasattr(flit, "req_type") and flit.req_type == "write" and hasattr(flit, "is_last_flit") and flit.is_last_flit:
                            self._release_write_tracker_on_completion(flit)

                        return True
                    else:
                        return False
                else:
                    # Channel buffer满，无法注入
                    return False
            else:
                return False
        else:
            return False

    def _eject_from_topology_network(self, channel: str):
        """从CrossRing网络弹出"""
        # 获取对应的节点
        if self.node_id in self.model.nodes:
            node = self.model.nodes[self.node_id]

            # 从节点的对应IP的eject channel buffer获取flit
            ip_key = self.ip_type

            if ip_key in node.ip_eject_channel_buffers:
                eject_buffer = node.ip_eject_channel_buffers[ip_key][channel]
                if eject_buffer.valid_signal():
                    flit = eject_buffer.read_output()
                    if flit:
                        # 更新flit状态，从EQ_CH转移到H2L处理
                        flit.flit_position = "H2L"
                    return flit

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
        # 只有SN端IP类型才能处理请求
        if not (self.ip_type.startswith("ddr") or self.ip_type.startswith("l2m")):
            return

        # ✅ 防重复处理：检查请求是否已经成功处理过
        if hasattr(req, "_request_processed") and req._request_processed:
            return

        # 注意：不在这里标记为已处理，而是在成功分配资源后标记

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

                    # ✅ 成功分配资源，标记为已处理
                    req._request_processed = True

                    self._create_read_packet(req)
                    self._release_completed_sn_tracker(req)

                    self._notify_request_arrived(req)
                else:
                    # 资源不足，发送negative响应
                    self._create_response(req, "negative")

                    # ✅ 修复：使用简单FIFO等待队列
                    self.sn_req_wait["read"].append(req)
            else:
                # 重试读请求：直接生成数据
                # ✅ 标记为已处理（retry请求也算成功处理）
                req._request_processed = True
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

                    # ✅ 成功分配资源，标记为已处理
                    req._request_processed = True

                    self._create_response(req, "datasend")
                else:
                    # 资源不足，发送negative响应
                    self._create_response(req, "negative")

                    # ✅ 修复：使用简单FIFO等待队列
                    self.sn_req_wait["write"].append(req)
            else:
                # 重试写请求：应该已经有资源分配（通过positive响应），直接发送datasend
                # 检查请求是否在SN tracker中（positive响应发送时应该已经分配了资源）
                existing_req = self._find_sn_tracker_by_packet_id(req.packet_id)
                if existing_req:
                    # ✅ 标记为已处理（retry请求也算成功处理）
                    req._request_processed = True
                    # 使用tracker中的请求对象（有正确的sn_tracker_type）
                    self._create_response(existing_req, "datasend")

    def _handle_received_response(self, rsp: CrossRingFlit) -> None:
        """
        处理收到的响应（RN端）

        Args:
            rsp: 收到的响应flit
        """
        # ✅ 增强防重复处理保护：检查响应是否已经处理过
        response_id = f"{rsp.packet_id}_{rsp.rsp_type}_{rsp.channel}"
        if hasattr(rsp, "_response_processed") and rsp._response_processed:
            return

        # 标记响应为已处理
        rsp._response_processed = True

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
            # 对于datasend类型的响应，即使找不到匹配的请求也要处理
            if hasattr(rsp, "rsp_type") and rsp.rsp_type == "datasend":
                # 直接处理datasend响应，req可以为None
                if rsp.req_type == "write":
                    self._handle_write_response(rsp, req)
                return
            else:
                # 对于其他类型的响应，如果找不到匹配请求，直接返回
                print(f"⚠️ 警告: 找不到响应 {rsp.packet_id} 对应的请求 (rsp_type: {getattr(rsp, 'rsp_type', 'unknown')})")
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
            # 确保RDB条目存在（retry场景下数据可能在请求重新注入前到达）
            if flit.packet_id not in self.rn_rdb:
                self.rn_rdb[flit.packet_id] = []

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

                    # 设置完成时间戳并同步第一个flit的entry时间
                    # 找到真正的第一个data flit (flit_id=0)
                    first_data_flit = next((f for f in self.rn_rdb[flit.packet_id] if f.flit_id == 0), None)
                    for f in self.rn_rdb[flit.packet_id]:
                        f.leave_db_cycle = self.current_cycle
                        f.sync_latency_record(req)
                        if first_data_flit:
                            f.sync_latency_record(first_data_flit)  # 同步第一个data flit的entry时间
                        f.data_received_complete_cycle = self.current_cycle

                    # 计算延迟
                    for f in self.rn_rdb[flit.packet_id]:
                        f.cmd_latency = f.cmd_received_by_cake1_cycle - f.cmd_entry_noc_from_cake0_cycle
                        if first_data_flit:
                            f.data_latency = f.data_received_complete_cycle - first_data_flit.data_entry_noc_from_cake1_cycle
                        f.transaction_latency = f.data_received_complete_cycle - f.cmd_entry_cake0_cycle

                    # **关键修复：通知RequestTracker读请求已完成（RN收到全部数据）**
                    self._notify_request_completion(req)

                    # 清理数据缓冲
                    del self.rn_rdb[flit.packet_id]

        elif flit.req_type == "write":
            # 写数据到达SN端
            # 确保sn_wdb中有对应的列表
            if flit.packet_id not in self.sn_wdb:
                self.sn_wdb[flit.packet_id] = []
            self.sn_wdb[flit.packet_id].append(flit)

            # 检查是否收集完整个burst
            if len(self.sn_wdb[flit.packet_id]) == flit.burst_length:
                req = self._find_sn_tracker_by_packet_id(flit.packet_id)
                if req:
                    # 设置延迟释放时间 (将ns转换为cycles)
                    network_freq_ghz = self.config.basic_config.NETWORK_FREQUENCY
                    sn_tracker_release_latency_cycles = int(self.config.tracker_config.SN_TRACKER_RELEASE_LATENCY * network_freq_ghz)
                    release_time = self.current_cycle + sn_tracker_release_latency_cycles

                    # 设置完成时间戳并同步第一个flit的entry时间
                    # 找到真正的第一个data flit (flit_id=0)
                    first_data_flit = next((f for f in self.sn_wdb[flit.packet_id] if f.flit_id == 0), None)
                    for f in self.sn_wdb[flit.packet_id]:
                        f.leave_db_cycle = release_time
                        f.sync_latency_record(req)
                        if first_data_flit:
                            f.sync_latency_record(first_data_flit)  # 同步第一个data flit的entry时间
                        f.data_received_complete_cycle = self.current_cycle
                        f.cmd_latency = f.cmd_received_by_cake0_cycle - f.cmd_entry_noc_from_cake0_cycle
                        if first_data_flit:
                            f.data_latency = f.data_received_complete_cycle - first_data_flit.data_entry_noc_from_cake0_cycle
                        f.transaction_latency = f.data_received_complete_cycle + sn_tracker_release_latency_cycles - f.cmd_entry_cake0_cycle

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
        """处理读响应（negative和positive响应）"""
        if rsp.rsp_type == "negative":
            # ✅ 新逻辑：如果请求已经发出去了，就不需要再处理
            if req.req_attr == "old":
                return  # 请求已经在retry中，不需要再处理

            # 标记为retry状态但不立即重发（等待positive响应）
            req.req_attr = "old"
            req.req_state = "invalid"  # 等待positive响应

            # 为retry预留RDB空间
            self.rn_rdb_reserve += req.burst_length

        elif rsp.rsp_type == "positive":
            # ✅ 新逻辑：如果请求不是retry状态，将请求标为retry并重新发送
            if req.req_attr != "old":
                req.req_attr = "old"
                req.req_state = "valid"

            # 重置请求状态并重新发送
            req.reset_for_retry()
            req.is_injected = False
            req.path_index = 0
            req.is_new_on_network = True
            req.is_arrive = False

            # 清除重复处理标记
            if hasattr(req, "_request_processed"):
                delattr(req, "_request_processed")

            # 释放为retry预留的RDB空间
            if self.rn_rdb_reserve >= req.burst_length:
                self.rn_rdb_reserve -= req.burst_length

            # 重新注入到队首
            self.pending_by_channel["req"].appendleft(req)

    def _handle_write_response(self, rsp: CrossRingFlit, req: CrossRingFlit) -> None:
        """处理写响应"""
        if rsp.rsp_type == "negative":
            # ✅ 新逻辑：如果请求已经发出去了，就不需要再处理
            if req.req_attr == "old":
                return  # 请求已经在retry中，不需要再处理

            # 标记为retry状态但不立即重发（等待positive响应）
            req.req_attr = "old"
            req.req_state = "invalid"  # 等待positive响应

        elif rsp.rsp_type == "positive":
            # ✅ 新逻辑：如果请求不是retry状态，将请求标为retry并重新发送
            if req.req_attr != "old":
                req.req_attr = "old"
                req.req_state = "valid"

            # 重置请求状态并重新发送
            req.reset_for_retry()
            req.is_injected = False
            req.path_index = 0
            req.is_new_on_network = True
            req.is_arrive = False

            # 清除重复处理标记
            if hasattr(req, "_request_processed"):
                delattr(req, "_request_processed")

            # 重新注入到队首
            self.pending_by_channel["req"].appendleft(req)

        elif rsp.rsp_type == "datasend":
            # ✅ 正确逻辑：收到datasend响应时才创建写数据（正确计算写延迟）

            # 检查是否已经有数据（可能是重复的datasend响应）
            data_flits = self.rn_wdb.get(rsp.packet_id, [])
            if not data_flits:
                # 数据不存在，现在创建（这样可以正确计算从datasend响应开始的写延迟）
                if rsp.packet_id not in self.rn_wdb:
                    # WDB条目不存在，可能是重复响应或时序问题
                    if req:
                        self.rn_wdb[rsp.packet_id] = []
                        self._create_write_data_flits(req)
                    else:
                        return
                else:
                    # WDB条目存在但是空的，正常情况，创建数据
                    if req:
                        self._create_write_data_flits(req)
                    else:
                        return

                data_flits = self.rn_wdb.get(rsp.packet_id, [])

            # 发送数据
            if data_flits:
                for flit in data_flits:
                    if flit not in self.pending_by_channel["data"]:
                        self.pending_by_channel["data"].append(flit)

    def _create_write_data_flits(self, req: CrossRingFlit) -> None:
        """创建写数据flits"""
        for i in range(req.burst_length):
            # 计算发送延迟 (将ns转换为cycles)
            network_freq_ghz = self.config.basic_config.NETWORK_FREQUENCY
            if req.destination_type and req.destination_type.startswith("ddr"):
                latency_ns = self.config.latency_config.DDR_W_LATENCY
                latency = int(latency_ns * network_freq_ghz)  # ns * GHz = cycles
            else:
                latency_ns = self.config.latency_config.L2M_W_LATENCY
                latency = int(latency_ns * network_freq_ghz)  # ns * GHz = cycles

            # 计算完整路径
            path = self.model.topology.calculate_shortest_path(req.source, req.destination)

            data_flit = create_crossring_flit(
                source=req.source,
                destination=req.destination,
                path=path,
                req_type=req.req_type,
                packet_id=req.packet_id,
                flit_id=i,
                burst_length=req.burst_length,
                channel="data",
                flit_type="data",
                departure_cycle=self.current_cycle + latency + i * self.clock_ratio,
                num_col=self.config.NUM_COL,
                num_row=self.config.NUM_ROW,
            )

            data_flit.sync_latency_record(req)
            data_flit.source_type = req.source_type
            data_flit.destination_type = req.destination_type
            data_flit.is_last_flit = i == req.burst_length - 1

            self.rn_wdb[req.packet_id].append(data_flit)

    def _create_read_packet(self, req: CrossRingFlit) -> None:
        """创建读数据包，使用现有的pending_by_channel机制"""
        for i in range(req.burst_length):
            # 计算发送延迟 (将ns转换为cycles)
            network_freq_ghz = self.config.basic_config.NETWORK_FREQUENCY
            if req.destination_type and req.destination_type.startswith("ddr"):
                latency_ns = self.config.latency_config.DDR_R_LATENCY
                latency = int(latency_ns * network_freq_ghz)  # ns * GHz = cycles
            else:
                latency_ns = self.config.latency_config.L2M_R_LATENCY
                latency = int(latency_ns * network_freq_ghz)  # ns * GHz = cycles

            # 计算完整路径（SN到RN）
            path = self.model.topology.calculate_shortest_path(req.destination, req.source)

            # 读数据从SN返回到RN
            data_flit = create_crossring_flit(
                source=req.destination,  # SN位置
                destination=req.source,  # RN位置
                path=path,
                req_type=req.req_type,
                packet_id=req.packet_id,
                flit_id=i,
                burst_length=req.burst_length,
                channel="data",
                flit_type="data",
                departure_cycle=self.current_cycle + latency + i * self.clock_ratio,
                num_col=self.config.NUM_COL,
                num_row=self.config.NUM_ROW,
            )

            data_flit.sync_latency_record(req)
            data_flit.source_type = req.destination_type
            data_flit.destination_type = req.source_type
            data_flit.is_last_flit = i == req.burst_length - 1
            data_flit.flit_position = "IP_inject"  # 在源IP准备注入
            data_flit.current_node_id = self.node_id

            # 使用分通道的pending队列
            self.pending_by_channel["data"].append(data_flit)

            # 数据flit会在进入L2H时被添加到RequestTracker

    def _create_response(self, req: CrossRingFlit, rsp_type: str) -> None:
        """创建响应（统一的响应创建函数）

        Args:
            req: 请求flit
            rsp_type: 响应类型 ("negative", "datasend", "positive")
        """

        # 计算完整路径（SN到RN）
        path = self.model.topology.calculate_shortest_path(req.destination, req.source)

        rsp = create_crossring_flit(
            source=req.destination,
            destination=req.source,
            path=path,
            req_type=req.req_type,
            packet_id=req.packet_id,
            channel="rsp",
            flit_type="rsp",
            rsp_type=rsp_type,
            departure_cycle=self.current_cycle + self.clock_ratio,
            num_col=self.config.NUM_COL,
            num_row=self.config.NUM_ROW,
        )

        rsp.sync_latency_record(req)
        rsp.source_type = req.destination_type
        rsp.destination_type = req.source_type
        rsp.flit_position = "IP_inject"  # 在源IP准备注入

        self.pending_by_channel["rsp"].append(rsp)

        # 响应flit会在进入L2H时被添加到RequestTracker

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
        """处理等待队列中的请求 - 发送positive响应通知RN端资源可用"""
        # ✅ 修复：使用简单FIFO等待队列
        wait_queue = self.sn_req_wait[req_type]

        if not wait_queue:
            return

        # 从队列头部取出最早的等待请求
        waiting_req = wait_queue[0]  # peek，不移除

        if req_type == "write":
            # 检查tracker和wdb资源
            if self.sn_tracker_count[tracker_type] > 0 and self.sn_wdb_count >= waiting_req.burst_length:
                # 从等待队列中移除请求
                wait_queue.popleft()
                waiting_req.sn_tracker_type = tracker_type

                # ✅ 关键修复：为等待的请求分配资源
                self.sn_tracker.append(waiting_req)
                self.sn_tracker_count[tracker_type] -= 1
                self.sn_wdb_count -= waiting_req.burst_length

                # ✅ 关键修复：发送positive响应，通知RN端资源已分配
                self._create_response(waiting_req, "positive")

        elif req_type == "read":
            # 检查tracker资源
            if self.sn_tracker_count[tracker_type] > 0:
                # 从等待队列中移除请求
                wait_queue.popleft()
                waiting_req.sn_tracker_type = tracker_type

                # ✅ 关键修复：为等待的请求分配资源
                self.sn_tracker.append(waiting_req)
                self.sn_tracker_count[tracker_type] -= 1

                # ✅ 关键修复：发送positive响应，通知RN端资源已分配
                self._create_response(waiting_req, "positive")

    def _count_waiting_requests(self, req_type: str) -> int:
        """计算指定类型的等待请求总数"""
        return len(self.sn_req_wait[req_type])

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
                "req_wait_read": self._count_waiting_requests("read"),
                "req_wait_write": self._count_waiting_requests("write"),
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
                    "h2l_h": len(self.h2l_h_fifos[channel]),
                    "h2l_h_valid": self.h2l_h_fifos[channel].valid_signal(),
                    "h2l_h_ready": self.h2l_h_fifos[channel].ready_signal(),
                    "h2l_l": len(self.h2l_l_fifos[channel]),
                    "h2l_l_valid": self.h2l_l_fifos[channel].valid_signal(),
                    "h2l_l_ready": self.h2l_l_fifos[channel].ready_signal(),
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

            # 计算完整路径
            path = self.model.topology.calculate_shortest_path(source, destination)

            # 创建读请求flit
            req_flit = create_crossring_flit(source, destination, path, num_col=self.config.NUM_COL, num_row=self.config.NUM_ROW)
            req_flit.packet_id = packet_id
            req_flit.req_type = "read"
            req_flit.burst_length = burst_length
            req_flit.channel = "req"
            req_flit.req_attr = "new"

            # 添加到RN tracker
            self.rn_tracker["read"].append(req_flit)

            # ✅ 修复：使用pending_by_channel队列而不是inject_fifos
            self.pending_by_channel["req"].append(req_flit)
            return True

        except Exception as e:
            return False

    def _process_write_request(self, source: NodeId, destination: NodeId, burst_length: int, packet_id: str) -> bool:
        """处理写请求"""
        try:
            # 分配RN资源
            if not self._allocate_rn_resources("write", burst_length):
                return False

            # 计算完整路径
            path = self.model.topology.calculate_shortest_path(source, destination)

            # 创建写请求flit
            req_flit = create_crossring_flit(source, destination, path, num_col=self.config.NUM_COL, num_row=self.config.NUM_ROW)
            req_flit.packet_id = packet_id
            req_flit.req_type = "write"
            req_flit.burst_length = burst_length
            req_flit.channel = "req"
            req_flit.req_attr = "new"

            # 添加到RN tracker
            self.rn_tracker["write"].append(req_flit)

            # 注意：写数据flit在收到datasend响应后才创建
            # 这里先预留WDB空间
            self.rn_wdb[packet_id] = []  # 预留空的数据缓冲区

            # ✅ 修复：使用pending_by_channel队列而不是inject_fifos
            self.pending_by_channel["req"].append(req_flit)
            return True

        except Exception as e:
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

    def _inject_retry_to_front(self, retry_flit: CrossRingFlit, channel: str) -> bool:
        """
        将retry请求直接插入到node的inject_fifo队首，实现优先级处理

        Args:
            retry_flit: retry请求flit
            channel: 通道类型 ("req", "rsp", "data")

        Returns:
            是否成功插入
        """
        try:
            # 获取对应的节点
            if self.node_id not in self.model.nodes:
                return False

            node = self.model.nodes[self.node_id]
            ip_key = self.ip_type

            # 检查node的inject channel buffer是否存在
            if ip_key not in node.ip_inject_channel_buffers:
                return False

            inject_buffer = node.ip_inject_channel_buffers[ip_key][channel]

            # ✅ 关键修复：检查是否是支持队首插入的数据结构
            if hasattr(inject_buffer, "appendleft"):
                # 使用队首插入实现retry优先级
                inject_buffer.appendleft(retry_flit)
            elif hasattr(inject_buffer, "write_input"):
                # 如果是PipelinedFIFO，需要特殊处理
                # 这里可能需要特殊的优先级插入机制
                return inject_buffer.write_input(retry_flit)
            else:
                # 其他数据结构，尝试普通插入
                inject_buffer.append(retry_flit)

            # 更新flit状态和时间戳
            retry_flit.flit_position = "IQ_CH"
            retry_flit.current_node_id = self.node_id
            retry_flit.cmd_entry_noc_from_cake0_cycle = self.current_cycle

            return True

        except Exception as e:
            return False

    def inject_request(self, source: NodeId, destination: NodeId, req_type: str, burst_length: int = 4, packet_id: str = None, source_type: str = None, destination_type: str = None, **kwargs) -> bool:
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
            # 计算完整路径
            path = self.model.topology.calculate_shortest_path(source, destination)

            # 创建CrossRing Flit
            flit = create_crossring_flit(
                source=source,
                destination=destination,
                path=path,
                packet_id=packet_id,
                req_type=req_type,
                burst_length=burst_length,
                num_col=self.config.NUM_COL,
                num_row=self.config.NUM_ROW,
            )

            # 设置IP类型信息
            flit.source_type = source_type if source_type else self.ip_type
            flit.destination_type = destination_type if destination_type else "unknown"

            # 如果没有提供destination_type，记录警告
            flit.channel = "req"

            # 设置命令进入源IP的时间戳
            flit.cmd_entry_cake0_cycle = self.current_cycle
            flit.inject_cycle = kwargs.get("inject_cycle", self.current_cycle)

            # 注册到请求追踪器
            if hasattr(self.model, "request_tracker"):
                self.model.request_tracker.start_request(
                    packet_id=packet_id, source=source, destination=destination, op_type=req_type, burst_size=burst_length, cycle=kwargs.get("inject_cycle", self.current_cycle)
                )
                # flit会在进入L2H时被添加到追踪器

            # 设置flit位置信息
            flit.flit_position = "IP_inject"  # 在源IP准备注入
            flit.current_node_id = self.node_id

            # 请求总是添加到pending队列，资源检查在传输到L2H时进行
            # 这里只是标记请求类型，实际的资源检查在step()中的传输阶段进行

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

            return True

        except Exception as e:
            import traceback

            traceback.print_exc()
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

        # 1. 首先处理延迟资源释放（移到最前面）
        self._process_delayed_resource_release()

        # 2. 初始化传输决策存储
        self._transfer_decisions = {
            "pending_to_l2h": {"req": None, "rsp": None, "data": None},  # 每个通道独立决策
            "l2h_to_node": {"req": None, "rsp": None, "data": None},  # 每个通道独立决策
            "network_to_h2l_h": {"req": None, "rsp": None, "data": None},  # 每个通道独立决策
            "h2l_h_to_h2l_l": {"req": None, "rsp": None, "data": None},  # 每个通道独立决策
            "h2l_l_to_completion": {"req": None, "rsp": None, "data": None},  # 每个通道独立决策
        }

        # 3. 刷新令牌桶
        if self.tx_token_bucket:
            self.tx_token_bucket.refill(current_cycle)
        if self.rx_token_bucket:
            self.rx_token_bucket.refill(current_cycle)

        # 4. 计算各阶段传输决策
        self._compute_pending_to_l2h_decision(current_cycle)
        self._compute_l2h_to_node_decision(current_cycle)
        self._compute_network_to_h2l_h_decision(current_cycle)
        self._compute_h2l_h_to_h2l_l_decision(current_cycle)
        self._compute_h2l_l_to_completion_decision(current_cycle)

        # 5. 更新所有FIFO的计算阶段
        for channel in ["req", "rsp", "data"]:
            self.l2h_fifos[channel].step_compute_phase(current_cycle)
            self.h2l_h_fifos[channel].step_compute_phase(current_cycle)
            self.h2l_l_fifos[channel].step_compute_phase(current_cycle)

    def _compute_pending_to_l2h_decision(self, current_cycle: int) -> None:
        """计算pending到l2h的传输决策"""
        # IP内部处理频率是1GHz，只有在偶数周期才能处理
        if current_cycle % self.clock_ratio != 0:
            return
        # 每个通道独立处理：req, rsp, data
        for channel in ["req", "rsp", "data"]:
            if self.pending_by_channel[channel]:
                l2h_ready = self.l2h_fifos[channel].ready_signal()
                flit = self.pending_by_channel[channel][0]

                if l2h_ready:
                    if flit.departure_cycle <= current_cycle:
                        # 检查带宽限制（仅针对data通道）
                        if self.tx_token_bucket and channel == "data":
                            # 数据传输每个flit消耗1个令牌
                            tokens_needed = 1

                            # 尝试消耗令牌
                            if not self.tx_token_bucket.consume(tokens_needed):
                                continue  # 令牌不足时跳过此flit

                        # 对于req通道，检查RN端资源是否足够处理响应
                        if channel == "req":
                            if not self._check_and_reserve_resources(flit):
                                continue  # 资源不足时跳过此请求，检查下一个

                        self._transfer_decisions["pending_to_l2h"][channel] = flit
                        # 不要return，继续检查其他通道

    def _compute_l2h_to_node_decision(self, current_cycle: int) -> None:
        """计算l2h到node的传输决策"""
        # 每个通道独立处理：req, rsp, data
        for channel in ["req", "rsp", "data"]:
            if self.l2h_fifos[channel].valid_signal():
                flit = self.l2h_fifos[channel].peek_output()
                if flit:
                    can_inject = self._can_inject_to_node(flit, channel)
                    if can_inject:
                        self._transfer_decisions["l2h_to_node"][channel] = flit
                        # 不要return，继续检查其他通道

    def _compute_network_to_h2l_h_decision(self, current_cycle: int) -> None:
        """计算network到h2l_h的传输决策"""
        # 每个通道独立处理：req, rsp, data
        for channel in ["req", "rsp", "data"]:
            if self.h2l_h_fifos[channel].ready_signal():
                flit = self._peek_from_topology_network(channel)
                if flit:
                    self._transfer_decisions["network_to_h2l_h"][channel] = flit
                    # 不要return，继续检查其他通道

    def _compute_h2l_h_to_h2l_l_decision(self, current_cycle: int) -> None:
        """计算h2l_h到h2l_l的传输决策（网络频率）"""
        # 每个通道独立处理：req, rsp, data
        for channel in ["req", "rsp", "data"]:
            if self.h2l_h_fifos[channel].valid_signal() and self.h2l_l_fifos[channel].ready_signal():
                flit = self.h2l_h_fifos[channel].peek_output()
                if flit:
                    self._transfer_decisions["h2l_h_to_h2l_l"][channel] = flit

    def _compute_h2l_l_to_completion_decision(self, current_cycle: int) -> None:
        """计算h2l_l到completion的传输决策（IP频率）"""
        # IP内部处理频率是1GHz，只有在偶数周期才能处理
        if current_cycle % self.clock_ratio != 0:
            return

        # 每个通道独立处理：req, rsp, data
        for channel in ["req", "rsp", "data"]:
            if self.h2l_l_fifos[channel].valid_signal():
                flit = self.h2l_l_fifos[channel].peek_output()
                if flit:
                    # 带宽限制检查
                    if self.rx_token_bucket and channel == "data":
                        if not self.rx_token_bucket.consume(1):
                            continue

                    self._transfer_decisions["h2l_l_to_completion"][channel] = flit

    def _can_inject_to_node(self, flit, channel: str) -> bool:
        """检查是否可以注入到node"""
        # 获取对应的节点
        if self.node_id in self.model.nodes:
            node = self.model.nodes[self.node_id]
            ip_key = self.ip_type

            if ip_key in node.ip_inject_channel_buffers:
                inject_buffer = node.ip_inject_channel_buffers[ip_key][channel]
                return inject_buffer.ready_signal()
        return False

    def _peek_from_topology_network(self, channel: str):
        """查看network中是否有可eject的flit"""
        # 获取对应的节点
        if self.node_id in self.model.nodes:
            node = self.model.nodes[self.node_id]
            ip_key = self.ip_type

            if ip_key in node.ip_eject_channel_buffers:
                eject_buffer = node.ip_eject_channel_buffers[ip_key][channel]
                if eject_buffer.valid_signal():
                    return eject_buffer.peek_output()
        return None

    def step_update_phase(self, current_cycle: int) -> None:
        """更新阶段：执行compute阶段的传输决策"""
        # 执行compute阶段的传输决策
        self._execute_transfer_decisions(current_cycle)

        # 更新所有FIFO的时序状态
        for channel in ["req", "rsp", "data"]:
            self.l2h_fifos[channel].step_update_phase()
            self.h2l_h_fifos[channel].step_update_phase()
            self.h2l_l_fifos[channel].step_update_phase()

    def _execute_transfer_decisions(self, current_cycle: int) -> None:
        """执行compute阶段计算的传输决策"""
        self.current_cycle = current_cycle

        # 1. 执行pending到l2h的传输（每个通道独立）
        for channel in ["req", "rsp", "data"]:
            if self._transfer_decisions["pending_to_l2h"][channel]:
                flit = self._transfer_decisions["pending_to_l2h"][channel]
                self.pending_by_channel[channel].popleft()
                flit.flit_position = "L2H"
                self.l2h_fifos[channel].write_input(flit)

                # 在进入L2H时添加到RequestTracker
                if hasattr(self.model, "request_tracker") and hasattr(flit, "packet_id"):
                    if channel == "req":
                        self.model.request_tracker.add_request_flit(flit.packet_id, flit)
                    elif channel == "rsp":
                        self.model.request_tracker.add_response_flit(flit.packet_id, flit)
                    elif channel == "data":
                        self.model.request_tracker.add_data_flit(flit.packet_id, flit)

                # 更新请求状态
                if channel == "req" and hasattr(flit, "packet_id") and flit.packet_id in self.active_requests:
                    self.active_requests[flit.packet_id]["stage"] = "l2h_fifo"

        # 2. 执行l2h到node的传输（每个通道独立）
        for channel in ["req", "rsp", "data"]:
            if self._transfer_decisions["l2h_to_node"][channel]:
                flit = self._transfer_decisions["l2h_to_node"][channel]
                self.l2h_fifos[channel].read_output()
                self._inject_to_topology_network(flit, channel)

        # 3. 执行network到h2l_h的传输（每个通道独立）
        for channel in ["req", "rsp", "data"]:
            if self._transfer_decisions["network_to_h2l_h"][channel]:
                ejected_flit = self._eject_from_topology_network(channel)
                if ejected_flit:
                    ejected_flit.flit_position = "H2L_H"
                    self.h2l_h_fifos[channel].write_input(ejected_flit)

        # 4. 执行h2l_h到h2l_l的传输（每个通道独立）
        for channel in ["req", "rsp", "data"]:
            if self._transfer_decisions["h2l_h_to_h2l_l"][channel]:
                flit = self.h2l_h_fifos[channel].read_output()
                if flit:
                    flit.flit_position = "H2L_L"
                    self.h2l_l_fifos[channel].write_input(flit)

        # 5. 执行h2l_l到completion的传输（每个通道独立）
        for channel in ["req", "rsp", "data"]:
            if self._transfer_decisions["h2l_l_to_completion"][channel]:
                flit_before_read = len(self.h2l_l_fifos[channel])
                flit = self.h2l_l_fifos[channel].read_output()
                flit_after_read = len(self.h2l_l_fifos[channel])

                if flit:
                    flit.flit_position = "IP_eject"

                    if channel == "req":
                        self._handle_received_request(flit)
                    elif channel == "rsp":
                        self._handle_received_response(flit)
                    elif channel == "data":
                        self._handle_received_data(flit)

    def _inject_to_node(self, flit, channel: str) -> bool:
        """将flit注入到node"""
        # 获取对应的节点
        if self.node_id in self.model.nodes:
            node = self.model.nodes[self.node_id]
            ip_key = self.ip_type

            if ip_key in node.ip_inject_channel_buffers:
                inject_buffer = node.ip_inject_channel_buffers[ip_key][channel]
                if inject_buffer.write_input(flit):
                    # 更新flit位置
                    flit.flit_position = "IQ_CH"
                    flit.current_node_id = self.node_id

                    # 更新时间戳
                    if channel == "req" and hasattr(flit, "req_attr") and flit.req_attr == "new":
                        # 设置命令从源IP进入NoC的时间（进入channel buffer）
                        flit.cmd_entry_noc_from_cake0_cycle = self.current_cycle
                    elif channel == "rsp":
                        # 响应从目标IP进入NoC的时间
                        flit.cmd_entry_noc_from_cake1_cycle = self.current_cycle
                    elif channel == "data":
                        if flit.req_type == "read":
                            # 读数据从目标IP进入NoC的时间
                            flit.data_entry_noc_from_cake1_cycle = self.current_cycle
                        elif flit.req_type == "write":
                            # 写数据从源IP进入NoC的时间
                            flit.data_entry_noc_from_cake0_cycle = self.current_cycle

                    # 更新请求状态
                    if channel == "req" and hasattr(flit, "packet_id") and flit.packet_id in self.active_requests:
                        self.active_requests[flit.packet_id]["stage"] = "node_inject"

                        # ✅ 修复：更新RequestTracker状态
                        if hasattr(self.model, "request_tracker"):
                            self.model.request_tracker.mark_request_injected(flit.packet_id, self.current_cycle)
                            # request flit已经在inject_request时添加，这里不重复添加

                    # response和data flit已经在创建时添加到RequestTracker，这里不重复添加

                    return True
        return False

    def _notify_request_arrived(self, req: CrossRingFlit) -> None:
        """通知RequestTracker请求已到达目标

        Args:
            req: 到达的请求flit
        """
        if hasattr(self.model, "request_tracker") and hasattr(req, "packet_id"):
            from src.noc.debug import RequestState

            self.model.request_tracker.update_request_state(req.packet_id, RequestState.ARRIVED, self.current_cycle)

    def _notify_request_completion(self, req: CrossRingFlit) -> None:
        """通知RequestTracker请求已完成

        Args:
            req: 完成的请求flit
        """
        if hasattr(self.model, "request_tracker") and hasattr(req, "packet_id"):
            from src.noc.debug import RequestState

            self.model.request_tracker.update_request_state(req.packet_id, RequestState.COMPLETED, self.current_cycle)

    def _release_write_tracker_on_completion(self, flit: CrossRingFlit) -> None:
        """写数据发送完成时释放RN write tracker

        Args:
            flit: 最后一个写数据flit
        """
        # 查找对应的RN write tracker
        req = self._find_rn_tracker_by_packet_id(flit.packet_id, "write")
        if not req:
            return

        # 释放RN write tracker
        self.rn_tracker["write"].remove(req)
        self.rn_tracker_count["write"] += 1
        self.rn_tracker_pointer["write"] -= 1

        # 释放WDB资源
        self.rn_wdb_count += req.burst_length

        # 清理WDB条目
        if flit.packet_id in self.rn_wdb:
            del self.rn_wdb[flit.packet_id]

        # 注意：不在这里标记请求完成，因为写请求的完成应该在SN端收到数据时标记
        # 这里只是释放RN端的tracker资源
