"""
简化的请求追踪器 - 只保留核心功能
"""

from typing import Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum


class RequestState(Enum):
    """请求状态"""
    CREATED = "created"
    INJECTED = "injected"
    IN_NETWORK = "in_network"
    ARRIVED = "arrived"
    COMPLETED = "completed"


class FlitType(Enum):
    """Flit类型"""
    REQUEST = "req"
    RESPONSE = "rsp"
    DATA = "data"


@dataclass
class RequestLifecycle:
    """请求生存周期记录"""
    packet_id: str
    source: int
    destination: int
    op_type: str
    burst_size: int
    
    # 时间戳
    created_cycle: int = 0
    injected_cycle: int = 0
    completed_cycle: int = 0
    current_state: RequestState = RequestState.CREATED
    
    # Flit列表 (核心功能)
    request_flits: List[Any] = field(default_factory=list)
    response_flits: List[Any] = field(default_factory=list)
    data_flits: List[Any] = field(default_factory=list)
    
    # 调试控制
    debug_started: bool = False


class RequestTracker:
    """简化的请求追踪器"""
    
    def __init__(self, network_frequency: int = 1):
        self.active_requests: Dict[str, RequestLifecycle] = {}
        self.completed_requests: Dict[str, RequestLifecycle] = {}
    
    def start_request(self, packet_id: str, source: int, destination: int, op_type: str, burst_size: int, cycle: int):
        """开始追踪请求"""
        lifecycle = RequestLifecycle(
            packet_id=packet_id,
            source=source,
            destination=destination,
            op_type=op_type,
            burst_size=burst_size,
            created_cycle=cycle
        )
        self.active_requests[packet_id] = lifecycle
    
    def update_request_state(self, packet_id: str, new_state: RequestState, cycle: int):
        """更新请求状态"""
        if packet_id in self.active_requests:
            lifecycle = self.active_requests[packet_id]
            lifecycle.current_state = new_state
            
            if new_state == RequestState.INJECTED:
                lifecycle.injected_cycle = cycle
            elif new_state == RequestState.COMPLETED:
                lifecycle.completed_cycle = cycle
                # 移动到完成列表
                self.completed_requests[packet_id] = lifecycle
                del self.active_requests[packet_id]
    
    def add_request_flit(self, packet_id: str, flit: Any):
        """添加请求flit"""
        if packet_id in self.active_requests:
            self.active_requests[packet_id].request_flits.append(flit)
    
    def add_response_flit(self, packet_id: str, flit: Any):
        """添加响应flit"""
        if packet_id in self.active_requests:
            self.active_requests[packet_id].response_flits.append(flit)
        elif packet_id in self.completed_requests:
            self.completed_requests[packet_id].response_flits.append(flit)
    
    def add_data_flit(self, packet_id: str, flit: Any):
        """添加数据flit"""
        if packet_id in self.active_requests:
            self.active_requests[packet_id].data_flits.append(flit)
        elif packet_id in self.completed_requests:
            self.completed_requests[packet_id].data_flits.append(flit)
    
    def should_print_debug(self, packet_id: str) -> bool:
        """判断是否应该打印调试信息"""
        lifecycle = self.active_requests.get(packet_id)
        if not lifecycle:
            lifecycle = self.completed_requests.get(packet_id)
        if not lifecycle:
            return False
            
        # 已完成则停止打印
        if lifecycle.current_state == RequestState.COMPLETED:
            return False
            
        # 检查是否应该开始打印
        if not lifecycle.debug_started:
            # 如果请求已经注入且有flit，就开始打印
            if lifecycle.current_state == RequestState.INJECTED and len(lifecycle.request_flits) > 0:
                lifecycle.debug_started = True
                return True
            # 或者检查flit是否在网络中传输
            all_flits = lifecycle.request_flits + lifecycle.response_flits + lifecycle.data_flits
            for flit in all_flits:
                if hasattr(flit, 'flit_position') and flit.flit_position:
                    # 任何有位置信息的flit都表示已经在网络中
                    if flit.flit_position in ["channel", "l2h_fifo", "h2l_fifo", "pending"]:
                        lifecycle.debug_started = True
                        return True
                    elif "IP_CH" in flit.flit_position or "N" in flit.flit_position:
                        lifecycle.debug_started = True
                        return True
            return False
        
        return True
    
    def get_active_tracked_requests(self) -> Dict[str, RequestLifecycle]:
        """获取活跃的追踪请求"""
        return self.active_requests
    
    def mark_request_injected(self, packet_id: str, cycle: int):
        """标记请求已注入"""
        self.update_request_state(packet_id, RequestState.INJECTED, cycle)