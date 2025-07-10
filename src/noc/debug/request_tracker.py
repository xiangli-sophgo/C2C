"""
请求生存周期追踪器
用于追踪和验证NoC中每个请求的完整生命周期，包括请求、响应和数据的正确性
"""

from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict, deque
from dataclasses import dataclass, field
import logging
from enum import Enum


class RequestState(Enum):
    """请求状态"""

    CREATED = "created"
    INJECTED = "injected"
    IN_NETWORK = "in_network"
    ARRIVED = "arrived"
    RESPONSE_SENT = "response_sent"
    DATA_SENDING = "data_sending"
    COMPLETED = "completed"
    ERROR = "error"


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
    op_type: str  # R/W
    burst_size: int

    # 时间戳记录
    created_cycle: int = 0
    injected_cycle: int = 0
    arrived_cycle: int = 0
    response_sent_cycle: int = 0
    data_start_cycle: int = 0
    completed_cycle: int = 0

    # 状态记录
    current_state: RequestState = RequestState.CREATED
    error_message: str = ""

    # Flit追踪
    request_flits: List[Any] = field(default_factory=list)
    response_flits: List[Any] = field(default_factory=list)
    data_flits: List[Any] = field(default_factory=list)

    # 位置追踪
    request_path: List[Tuple[int, int]] = field(default_factory=list)  # [(node_id, cycle), ...]
    data_path: List[Tuple[int, int]] = field(default_factory=list)

    # 验证标志
    request_valid: bool = False
    response_valid: bool = False
    data_valid: bool = False
    data_integrity_ok: bool = False

    # 报告标志
    reported: bool = False

    def get_total_latency(self) -> int:
        """获取总延迟"""
        if self.completed_cycle > 0:
            return self.completed_cycle - self.injected_cycle
        return 0

    def get_request_latency(self) -> int:
        """获取请求延迟"""
        if self.arrived_cycle > 0:
            return self.arrived_cycle - self.injected_cycle
        return 0

    def get_data_latency(self) -> int:
        """获取数据延迟"""
        if self.completed_cycle > 0 and self.data_start_cycle > 0:
            return self.completed_cycle - self.data_start_cycle
        return 0


class RequestTracker:
    """请求追踪器"""

    def __init__(self, network_frequency: int = 1):
        self.network_frequency = network_frequency
        self.logger = logging.getLogger(f"RequestTracker_{id(self)}")

        # 追踪数据结构
        self.active_requests: Dict[str, RequestLifecycle] = {}
        self.completed_requests: Dict[str, RequestLifecycle] = {}
        self.tracked_packet_ids: Set[str] = set()

        # 网络状态追踪
        self.network_state: Dict[str, Dict[int, List[Any]]] = {"req": defaultdict(list), "rsp": defaultdict(list), "data": defaultdict(list)}

        # 调试配置
        self.debug_enabled: bool = False
        self.verbose_level: int = 0
        self.trace_specific_packets: Set[str] = set()
        self.print_cycle_interval: int = 100

        # 验证配置
        self.validate_responses: bool = True
        self.validate_data_integrity: bool = True
        self.expected_data_pattern: str = "incremental"  # "incremental", "random", "custom"

        # 统计信息
        self.stats = {
            "total_requests": 0,
            "completed_requests": 0,
            "failed_requests": 0,
            "avg_latency": 0.0,
            "max_latency": 0,
            "min_latency": float("inf"),
            "data_errors": 0,
            "response_errors": 0,
        }

    def enable_debug(self, level: int = 1, specific_packets: List[str] = None):
        """启用调试模式"""
        self.debug_enabled = True
        self.verbose_level = level
        if specific_packets:
            self.trace_specific_packets.update(specific_packets)
        self.logger.info(f"调试模式已启用，级别: {level}")

    def track_packet(self, packet_id: str):
        """添加要追踪的特定包ID"""
        self.trace_specific_packets.add(packet_id)
        self.logger.info(f"开始追踪包: {packet_id}")

    def start_request(self, packet_id: str, source: int, destination: int, op_type: str, burst_size: int, cycle: int):
        """开始追踪一个新请求"""
        lifecycle = RequestLifecycle(packet_id=packet_id, source=source, destination=destination, op_type=op_type, burst_size=burst_size, created_cycle=cycle)

        self.active_requests[packet_id] = lifecycle
        self.stats["total_requests"] += 1

        if self.debug_enabled and (not self.trace_specific_packets or packet_id in self.trace_specific_packets):
            self.logger.info(f"周期 {cycle}: 创建请求 {packet_id} - {source} -> {destination}, {op_type}, burst={burst_size}")

    def update_request_state(self, packet_id: str, new_state: RequestState, cycle: int, **kwargs):
        """更新请求状态"""
        if packet_id not in self.active_requests:
            return

        lifecycle = self.active_requests[packet_id]
        old_state = lifecycle.current_state
        lifecycle.current_state = new_state

        # 更新对应的时间戳
        if new_state == RequestState.INJECTED:
            lifecycle.injected_cycle = cycle
        elif new_state == RequestState.ARRIVED:
            lifecycle.arrived_cycle = cycle
        elif new_state == RequestState.RESPONSE_SENT:
            lifecycle.response_sent_cycle = cycle
        elif new_state == RequestState.DATA_SENDING:
            lifecycle.data_start_cycle = cycle
        elif new_state == RequestState.COMPLETED:
            lifecycle.completed_cycle = cycle
            self._complete_request(packet_id)
        elif new_state == RequestState.ERROR:
            lifecycle.error_message = kwargs.get("error", "未知错误")
            self.stats["failed_requests"] += 1

        if self.debug_enabled and (not self.trace_specific_packets or packet_id in self.trace_specific_packets):
            self.logger.info(f"周期 {cycle}: 包 {packet_id} 状态变化: {old_state.value} -> {new_state.value}")

    def track_flit_position(self, packet_id: str, flit_type: FlitType, node_id: int, cycle: int, flit_data: Any = None):
        """追踪flit在网络中的位置"""
        if packet_id not in self.active_requests:
            return

        lifecycle = self.active_requests[packet_id]

        if flit_type == FlitType.REQUEST:
            lifecycle.request_path.append((node_id, cycle))
            if flit_data:
                lifecycle.request_flits.append(flit_data)
        elif flit_type == FlitType.DATA:
            lifecycle.data_path.append((node_id, cycle))
            if flit_data:
                lifecycle.data_flits.append(flit_data)
        elif flit_type == FlitType.RESPONSE:
            if flit_data:
                lifecycle.response_flits.append(flit_data)

        # 更新网络状态记录
        self.network_state[flit_type.value][cycle].append({"packet_id": packet_id, "node_id": node_id, "flit_data": flit_data})

        if self.debug_enabled and (not self.trace_specific_packets or packet_id in self.trace_specific_packets):
            self.logger.debug(f"周期 {cycle}: 包 {packet_id} 的 {flit_type.value} flit 在节点 {node_id}")

    def validate_request_response(self, packet_id: str, response_data: Any) -> bool:
        """验证响应的正确性"""
        if packet_id not in self.active_requests:
            return False

        lifecycle = self.active_requests[packet_id]

        # 基本验证：响应应该来自目标节点
        if hasattr(response_data, "source") and response_data.source == lifecycle.destination:
            lifecycle.response_valid = True
            return True
        else:
            lifecycle.response_valid = False
            self.stats["response_errors"] += 1
            self.logger.error(f"包 {packet_id} 响应验证失败")
            return False

    def validate_data_integrity(self, packet_id: str, data_flits: List[Any]) -> bool:
        """验证数据完整性"""
        if packet_id not in self.active_requests:
            return False

        lifecycle = self.active_requests[packet_id]

        # 检查数据flit数量是否匹配burst_size
        if len(data_flits) != lifecycle.burst_size:
            self.logger.error(f"包 {packet_id} 数据flit数量错误: 期望 {lifecycle.burst_size}, 实际 {len(data_flits)}")
            self.stats["data_errors"] += 1
            return False

        # 验证数据模式
        if self.expected_data_pattern == "incremental":
            for i, flit in enumerate(data_flits):
                expected_value = i
                if hasattr(flit, "data") and flit.data != expected_value:
                    self.logger.error(f"包 {packet_id} 数据flit {i} 值错误: 期望 {expected_value}, 实际 {flit.data}")
                    self.stats["data_errors"] += 1
                    return False

        lifecycle.data_valid = True
        lifecycle.data_integrity_ok = True
        return True

    def _complete_request(self, packet_id: str):
        """完成请求的处理"""
        if packet_id not in self.active_requests:
            return

        lifecycle = self.active_requests[packet_id]

        # 移动到已完成列表
        self.completed_requests[packet_id] = lifecycle
        del self.active_requests[packet_id]

        # 更新统计信息
        self.stats["completed_requests"] += 1
        latency = lifecycle.get_total_latency()

        if latency > 0:
            # 更新延迟统计
            if latency > self.stats["max_latency"]:
                self.stats["max_latency"] = latency
            if latency < self.stats["min_latency"]:
                self.stats["min_latency"] = latency

            # 计算平均延迟
            total_completed = self.stats["completed_requests"]
            self.stats["avg_latency"] = (self.stats["avg_latency"] * (total_completed - 1) + latency) / total_completed

        if self.debug_enabled and (not self.trace_specific_packets or packet_id in self.trace_specific_packets):
            self.logger.info(f"包 {packet_id} 完成: 总延迟 {latency} 周期")
            self._print_request_summary(lifecycle)

    def _print_request_summary(self, lifecycle: RequestLifecycle):
        """打印请求摘要信息"""
        print(f"\n=== 请求 {lifecycle.packet_id} 生存周期摘要 ===")
        print(f"源节点: {lifecycle.source} -> 目标节点: {lifecycle.destination}")
        print(f"操作类型: {lifecycle.op_type}, 突发长度: {lifecycle.burst_size}")
        print(f"创建周期: {lifecycle.created_cycle}")
        print(f"注入周期: {lifecycle.injected_cycle}")
        print(f"到达周期: {lifecycle.arrived_cycle}")
        print(f"响应发送周期: {lifecycle.response_sent_cycle}")
        print(f"数据开始周期: {lifecycle.data_start_cycle}")
        print(f"完成周期: {lifecycle.completed_cycle}")
        print(f"总延迟: {lifecycle.get_total_latency()} 周期")
        print(f"请求延迟: {lifecycle.get_request_latency()} 周期")
        print(f"数据延迟: {lifecycle.get_data_latency()} 周期")
        print(f"响应有效: {lifecycle.response_valid}")
        print(f"数据有效: {lifecycle.data_valid}")
        print(f"数据完整性: {lifecycle.data_integrity_ok}")

        if lifecycle.request_path:
            print(f"请求路径: {lifecycle.request_path[:5]}{'...' if len(lifecycle.request_path) > 5 else ''}")
        if lifecycle.data_path:
            print(f"数据路径: {lifecycle.data_path[:5]}{'...' if len(lifecycle.data_path) > 5 else ''}")
        print("=" * 50)

    def print_network_state(self, cycle: int):
        """打印当前周期的网络状态"""
        if not self.debug_enabled or cycle % self.print_cycle_interval != 0:
            return

        print(f"\n=== 周期 {cycle} 网络状态 ===")

        for flit_type in ["req", "rsp", "data"]:
            if cycle in self.network_state[flit_type]:
                flits = self.network_state[flit_type][cycle]
                if flits:
                    print(f"{flit_type.upper()} 网络:")
                    for flit_info in flits:
                        print(f"  节点 {flit_info['node_id']}: 包 {flit_info['packet_id']}")

        print(f"活跃请求: {len(self.active_requests)}")
        print(f"已完成请求: {len(self.completed_requests)}")
        print("-" * 30)

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()

    def get_request_status(self, packet_id: str) -> Optional[RequestLifecycle]:
        """获取特定请求的状态"""
        if packet_id in self.active_requests:
            return self.active_requests[packet_id]
        elif packet_id in self.completed_requests:
            return self.completed_requests[packet_id]
        return None

    def print_final_report(self):
        """打印最终报告"""
        print("\n" + "=" * 60)
        print("请求追踪最终报告")
        print("=" * 60)
        print(f"总请求数: {self.stats['total_requests']}")
        print(f"已完成请求: {self.stats['completed_requests']}")
        print(f"失败请求: {self.stats['failed_requests']}")
        print(f"完成率: {self.stats['completed_requests']/max(1, self.stats['total_requests'])*100:.1f}%")
        print(f"平均延迟: {self.stats['avg_latency']:.1f} 周期")
        print(f"最大延迟: {self.stats['max_latency']} 周期")
        print(f"最小延迟: {self.stats['min_latency'] if self.stats['min_latency'] != float('inf') else 0} 周期")
        print(f"响应错误: {self.stats['response_errors']}")
        print(f"数据错误: {self.stats['data_errors']}")

        # 打印部分请求详情
        if self.trace_specific_packets:
            print(f"\n追踪的特定请求:")
            for packet_id in self.trace_specific_packets:
                lifecycle = self.get_request_status(packet_id)
                if lifecycle:
                    print(f"  {packet_id}: {lifecycle.current_state.value}, 延迟: {lifecycle.get_total_latency()}")

        print("=" * 60)

    def reset(self):
        """重置追踪器"""
        self.active_requests.clear()
        self.completed_requests.clear()
        self.tracked_packet_ids.clear()
        self.network_state = {"req": defaultdict(list), "rsp": defaultdict(list), "data": defaultdict(list)}
        self.stats = {
            "total_requests": 0,
            "completed_requests": 0,
            "failed_requests": 0,
            "avg_latency": 0.0,
            "max_latency": 0,
            "min_latency": float("inf"),
            "data_errors": 0,
            "response_errors": 0,
        }

    # ========== IP接口兼容性方法 ==========
    
    def mark_request_injected(self, packet_id: str, cycle: int):
        """标记请求已注入（IP接口兼容性方法）"""
        self.update_request_state(packet_id, RequestState.INJECTED, cycle)
        
    def add_request_flit(self, packet_id: str, flit: Any):
        """添加请求flit（IP接口兼容性方法）"""
        if packet_id in self.active_requests:
            self.active_requests[packet_id].request_flits.append(flit)
            # 如果flit有位置信息，同时追踪位置
            if hasattr(flit, 'current_node') and hasattr(flit, 'current_cycle'):
                self.track_flit_position(packet_id, FlitType.REQUEST, flit.current_node, flit.current_cycle, flit)
    
    def add_response_flit(self, packet_id: str, flit: Any):
        """添加响应flit（IP接口兼容性方法）"""
        if packet_id in self.active_requests:
            self.active_requests[packet_id].response_flits.append(flit)
            # 如果flit有位置信息，同时追踪位置
            if hasattr(flit, 'current_node') and hasattr(flit, 'current_cycle'):
                self.track_flit_position(packet_id, FlitType.RESPONSE, flit.current_node, flit.current_cycle, flit)
    
    def add_data_flit(self, packet_id: str, flit: Any):
        """添加数据flit（IP接口兼容性方法）"""
        if packet_id in self.active_requests:
            self.active_requests[packet_id].data_flits.append(flit)
            # 如果flit有位置信息，同时追踪位置
            if hasattr(flit, 'current_node') and hasattr(flit, 'current_cycle'):
                self.track_flit_position(packet_id, FlitType.DATA, flit.current_node, flit.current_cycle, flit)
