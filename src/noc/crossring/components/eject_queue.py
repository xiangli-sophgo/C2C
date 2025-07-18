"""
CrossRing弹出队列管理。

负责处理：
- 弹出输入FIFO管理
- IP弹出缓冲区管理
- 弹出仲裁逻辑
- IP分发决策
"""

from typing import Dict, List, Optional, Tuple
import logging

from src.noc.base.ip_interface import PipelinedFIFO
from ..flit import CrossRingFlit
from ..config import CrossRingConfig, RoutingStrategy


class EjectQueue:
    """弹出队列管理类。"""
    
    def __init__(self, node_id: int, coordinates: Tuple[int, int], config: CrossRingConfig, logger: logging.Logger):
        """
        初始化弹出队列管理器。
        
        Args:
            node_id: 节点ID
            coordinates: 节点坐标
            config: CrossRing配置
            logger: 日志记录器
        """
        self.node_id = node_id
        self.coordinates = coordinates
        self.config = config
        self.logger = logger
        
        # 获取FIFO配置
        self.eq_in_depth = getattr(config, "EQ_IN_DEPTH", 16)
        self.eq_ch_depth = getattr(config, "EQ_CH_DEPTH", 10)
        
        # 连接的IP列表
        self.connected_ips = []
        
        # 每个IP的eject channel_buffer
        self.ip_eject_channel_buffers = {}
        
        # eject输入FIFO
        self.eject_input_fifos = self._create_eject_input_fifos()
        
        # Eject轮询仲裁器状态
        self.eject_arbitration_state = {
            "req": {"current_source": 0, "current_ip": 0, "sources": [], "last_served_source": {}, "last_served_ip": {}},
            "rsp": {"current_source": 0, "current_ip": 0, "sources": [], "last_served_source": {}, "last_served_ip": {}},
            "data": {"current_source": 0, "current_ip": 0, "sources": [], "last_served_source": {}, "last_served_ip": {}},
        }
        
        # 性能统计
        self.stats = {"ejected_flits": {"req": 0, "rsp": 0, "data": 0}}
        
    def _create_eject_input_fifos(self) -> Dict[str, Dict[str, PipelinedFIFO]]:
        """创建eject输入FIFO集合。"""
        return {
            channel: {
                direction: PipelinedFIFO(f"eject_in_{channel}_{direction}_{self.node_id}", depth=self.eq_in_depth)
                for direction in ["TU", "TD", "TR", "TL"]
            }
            for channel in ["req", "rsp", "data"]
        }
        
    def connect_ip(self, ip_id: str) -> bool:
        """
        连接一个IP到当前节点。
        
        Args:
            ip_id: IP标识符
            
        Returns:
            是否成功连接
        """
        if ip_id not in self.connected_ips:
            self.connected_ips.append(ip_id)
            
            # 为这个IP创建eject channel_buffer
            self.ip_eject_channel_buffers[ip_id] = {
                "req": PipelinedFIFO(f"ip_eject_channel_req_{ip_id}_{self.node_id}", depth=self.eq_ch_depth),
                "rsp": PipelinedFIFO(f"ip_eject_channel_rsp_{ip_id}_{self.node_id}", depth=self.eq_ch_depth),
                "data": PipelinedFIFO(f"ip_eject_channel_data_{ip_id}_{self.node_id}", depth=self.eq_ch_depth),
            }
            
            # 更新eject仲裁状态中的IP列表
            self._update_eject_arbitration_ips()
            self.logger.debug(f"节点{self.node_id}成功连接IP {ip_id}")
            return True
        else:
            self.logger.warning(f"IP {ip_id}已经连接到节点{self.node_id}")
            return False
            
    def disconnect_ip(self, ip_id: str) -> None:
        """断开IP连接。"""
        if ip_id in self.connected_ips:
            self.connected_ips.remove(ip_id)
            del self.ip_eject_channel_buffers[ip_id]
            self._update_eject_arbitration_ips()
            self.logger.debug(f"节点{self.node_id}断开IP {ip_id}连接")
            
    def _update_eject_arbitration_ips(self) -> None:
        """更新eject仲裁状态中的IP列表。"""
        for channel in ["req", "rsp", "data"]:
            arb_state = self.eject_arbitration_state[channel]
            arb_state["current_ip"] = 0
            arb_state["last_served_ip"] = {ip_id: 0 for ip_id in self.connected_ips}
            
    def _get_active_eject_sources(self) -> List[str]:
        """根据路由策略获取活跃的eject输入源。"""
        routing_strategy = getattr(self.config, "ROUTING_STRATEGY", "XY")
        if hasattr(routing_strategy, "value"):
            routing_strategy = routing_strategy.value
            
        # 这两个源总是存在
        sources = ["IQ_EQ", "ring_bridge_EQ"]
        
        if routing_strategy == "XY":
            sources.extend(["TU", "TD"])
        elif routing_strategy == "YX":
            sources.extend(["TR", "TL"])
        else:  # ADAPTIVE 或其他
            sources.extend(["TU", "TD", "TR", "TL"])
            
        return sources
        
    def _initialize_eject_arbitration_sources(self) -> None:
        """初始化eject仲裁的源列表。"""
        active_sources = self._get_active_eject_sources()
        
        for channel in ["req", "rsp", "data"]:
            arb_state = self.eject_arbitration_state[channel]
            arb_state["sources"] = active_sources.copy()
            arb_state["last_served_source"] = {source: 0 for source in active_sources}
            
    def compute_arbitration(self, cycle: int, inject_direction_fifos: Dict, ring_bridge: 'RingBridge') -> None:
        """
        计算阶段：确定要传输的flit但不执行传输。
        
        Args:
            cycle: 当前周期
            inject_direction_fifos: 注入方向FIFO
            ring_bridge: RingBridge实例
        """
        # 首先初始化源列表（如果还没有初始化）
        if not self.eject_arbitration_state["req"]["sources"]:
            self._initialize_eject_arbitration_sources()
            
        # 存储传输计划
        self._eject_transfer_plan = []
        
        # 为每个通道计算eject仲裁
        for channel in ["req", "rsp", "data"]:
            self._compute_channel_eject_arbitration(channel, cycle, inject_direction_fifos, ring_bridge)
            
    def _compute_channel_eject_arbitration(self, channel: str, cycle: int, inject_direction_fifos: Dict, ring_bridge: 'RingBridge') -> None:
        """计算单个通道的eject仲裁。"""
        if not self.connected_ips:
            return
            
        arb_state = self.eject_arbitration_state[channel]
        sources = arb_state["sources"]
        
        # 轮询所有输入源
        for source_attempt in range(len(sources)):
            current_source_idx = arb_state["current_source"]
            source = sources[current_source_idx]
            
            # 获取来自当前源的flit
            flit = self._get_flit_from_eject_source(source, channel, inject_direction_fifos, ring_bridge)
            if flit is not None:
                # 找到flit，现在确定分配给哪个IP
                target_ip = self._find_target_ip_for_flit(flit, channel, cycle)
                if target_ip:
                    # 保存传输计划
                    self._eject_transfer_plan.append((source, channel, flit, target_ip))
                    arb_state["last_served_source"][source] = cycle
                    break
                    
            # 移动到下一个源
            arb_state["current_source"] = (current_source_idx + 1) % len(sources)
            
    def execute_arbitration(self, cycle: int, inject_direction_fifos: Dict, ring_bridge: 'RingBridge') -> None:
        """
        执行阶段：基于compute阶段的计算执行实际传输。
        
        Args:
            cycle: 当前周期
            inject_direction_fifos: 注入方向FIFO
            ring_bridge: RingBridge实例
        """
        if not hasattr(self, '_eject_transfer_plan'):
            return
            
        # 临时修复：重置ring_bridge FIFO的读取状态，确保可以读取
        if ring_bridge:
            for ch in ["req", "rsp", "data"]:
                eq_fifo = ring_bridge.ring_bridge_output_fifos[ch]["EQ"]
                eq_fifo.read_this_cycle = False
            
        # 执行所有计划的传输
        for source, channel, flit, target_ip in self._eject_transfer_plan:
            # 从源获取flit（实际取出）
            actual_flit = self._get_flit_from_eject_source(source, channel, inject_direction_fifos, ring_bridge)
            if actual_flit and self._assign_flit_to_ip(actual_flit, target_ip, channel, cycle):
                # 成功传输，更新统计
                self.stats["ejected_flits"][channel] += 1
                
    def _get_flit_from_eject_source(self, source: str, channel: str, inject_direction_fifos: Dict, ring_bridge: 'RingBridge') -> Optional[CrossRingFlit]:
        """从指定的eject源获取flit。"""
        if source == "IQ_EQ":
            # 直接从inject_direction_fifos的EQ获取
            eq_fifo = inject_direction_fifos[channel]["EQ"]
            if eq_fifo.valid_signal():
                return eq_fifo.read_output()
                
        elif source == "ring_bridge_EQ":
            # 从ring_bridge的EQ输出获取
            return ring_bridge.get_eq_output_flit(channel)
            
        elif source in ["TU", "TD", "TR", "TL"]:
            # 从eject_input_fifos获取
            input_fifo = self.eject_input_fifos[channel][source]
            if input_fifo.valid_signal():
                return input_fifo.read_output()
                
        return None
        
    def _find_target_ip_for_flit(self, flit: CrossRingFlit, channel: str, cycle: int) -> Optional[str]:
        """为flit找到目标IP。"""
        if not self.connected_ips:
            return None
            
        # 首先尝试根据flit的destination_type匹配对应的IP
        if hasattr(flit, "destination_type") and flit.destination_type:
            for ip_id in self.connected_ips:
                # 从IP ID中提取IP类型
                ip_type = "_".join(ip_id.split("_")[:-1])
                ip_base_type = ip_type.split("_")[0]
                
                # 从destination_type中提取基础类型
                dest_base_type = flit.destination_type.split("_")[0]
                
                # 匹配逻辑
                if ip_type == flit.destination_type or ip_base_type == dest_base_type:
                    eject_buffer = self.ip_eject_channel_buffers[ip_id][channel]
                    if eject_buffer.ready_signal():
                        return ip_id
                        
        # 如果没有匹配的IP，使用round-robin逻辑
        arb_state = self.eject_arbitration_state[channel]
        for ip_attempt in range(len(self.connected_ips)):
            current_ip_idx = arb_state["current_ip"]
            ip_id = self.connected_ips[current_ip_idx]
            
            eject_buffer = self.ip_eject_channel_buffers[ip_id][channel]
            if eject_buffer.ready_signal():
                arb_state["current_ip"] = (current_ip_idx + 1) % len(self.connected_ips)
                return ip_id
                
            arb_state["current_ip"] = (current_ip_idx + 1) % len(self.connected_ips)
            
        return None
        
    def _assign_flit_to_ip(self, flit: CrossRingFlit, ip_id: str, channel: str, cycle: int) -> bool:
        """将flit分配给指定IP。"""
        eject_buffer = self.ip_eject_channel_buffers[ip_id][channel]
        if eject_buffer.write_input(flit):
            # 更新flit状态
            flit.flit_position = "EQ_CH"
            
            # 更新IP仲裁状态
            arb_state = self.eject_arbitration_state[channel]
            arb_state["last_served_ip"][ip_id] = cycle
            
            self.logger.debug(f"节点{self.node_id}成功将{channel}通道flit分配给IP {ip_id}")
            return True
        return False
        
    def get_eject_flit(self, ip_id: str, channel: str) -> Optional[CrossRingFlit]:
        """IP从其eject channel buffer获取flit。"""
        if ip_id not in self.connected_ips:
            self.logger.error(f"IP {ip_id}未连接到节点{self.node_id}")
            return None
            
        eject_buffer = self.ip_eject_channel_buffers[ip_id][channel]
        if eject_buffer.valid_signal():
            return eject_buffer.read_output()
        return None
        
    def step_compute_phase(self, cycle: int) -> None:
        """FIFO组合逻辑更新。"""
        # 更新IP eject channel buffers
        for ip_id in self.connected_ips:
            for channel in ["req", "rsp", "data"]:
                self.ip_eject_channel_buffers[ip_id][channel].step_compute_phase(cycle)
                
        # 更新eject input FIFOs
        for channel in ["req", "rsp", "data"]:
            for direction in ["TU", "TD", "TR", "TL"]:
                self.eject_input_fifos[channel][direction].step_compute_phase(cycle)
                
    def step_update_phase(self) -> None:
        """FIFO时序逻辑更新。"""
        # 更新IP eject channel buffers
        for ip_id in self.connected_ips:
            for channel in ["req", "rsp", "data"]:
                self.ip_eject_channel_buffers[ip_id][channel].step_update_phase()
                
        # 更新eject input FIFOs
        for channel in ["req", "rsp", "data"]:
            for direction in ["TU", "TD", "TR", "TL"]:
                self.eject_input_fifos[channel][direction].step_update_phase()
                
    def get_connected_ips(self) -> List[str]:
        """获取连接的IP列表。"""
        return self.connected_ips.copy()
        
    def get_stats(self) -> Dict:
        """获取统计信息。"""
        return {
            "ejected_flits": dict(self.stats["ejected_flits"]),
            "buffer_occupancy": {
                "ip_eject_channel_buffers": {
                    ip_id: {k: len(v) for k, v in channels.items()}
                    for ip_id, channels in self.ip_eject_channel_buffers.items()
                },
                "eject_input_fifos": {
                    k: {d: len(v) for d, v in vv.items()}
                    for k, vv in self.eject_input_fifos.items()
                },
            }
        }