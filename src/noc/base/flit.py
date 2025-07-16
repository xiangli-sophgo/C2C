"""
通用NoC Flit基础类。

提供所有NoC拓扑共用的基础Flit功能，包括基础字段、时间戳记录、
路径管理等。各拓扑可以继承并扩展特有功能。
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any, Union
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections import deque
import threading

from src.noc.utils.types import NodeId, Priority


@dataclass
class BaseFlit(ABC):
    """
    NoC基础Flit类。

    包含所有NoC拓扑共用的基础字段和方法，
    各拓扑可以继承并添加特有的字段和功能。
    """

    # ========== 基础标识字段 ==========
    packet_id: str = ""
    flit_id: int = 0
    source: NodeId = 0
    destination: NodeId = 0

    # ========== 请求类型和数据 ==========
    req_type: str = "R"  # "R: read" | "W: write" | "C: control"
    burst_length: int = 1
    flit_size: int = 128  # 数据大小（bytes）
    priority: Priority = Priority.MEDIUM
    source_type: str = "gdma_0"
    destination_type: str = "ddr_0"

    # ========== STI三通道协议（NoC公有） ==========
    channel: str = "req"  # "req" | "rsp" | "data"
    flit_type: str = "req"  # "req" | "rsp" | "data" | "control"

    # 请求属性
    req_attr: str = "new"  # "new" | "old" (重试标识)
    req_state: str = "valid"  # "valid" | "invalid"

    # 响应类型
    rsp_type: Optional[str] = None  # "ack" | "nack" | "data_ready" | "completion"

    # ========== 重试机制（NoC公有） ==========
    retry_count: int = 0
    max_retries: int = 1
    original_req_time: float = np.inf
    retry_reason: str = ""
    is_retry: bool = False

    # ========== 路径和路由信息 ==========
    path: List[NodeId] = field(default_factory=list)
    path_index: int = 0
    current_position: NodeId = -1
    next_hop: NodeId = -1
    routing_info: Dict[str, Any] = field(default_factory=dict)

    # ========== 通用状态字段 ==========
    is_injected: bool = False
    is_ejected: bool = False
    is_arrive: bool = False
    is_finish: bool = False
    is_head_flit: bool = True
    is_tail_flit: bool = True

    # ========== 网络状态 ==========
    is_new_on_network: bool = True
    hop_count: int = 0

    # ========== 时间戳记录 ==========
    # 基础时间戳
    creation_time: float = 0.0
    injection_time: float = np.inf
    ejection_time: float = np.inf
    completion_time: float = np.inf

    # 网络传输时间戳
    network_entry_time: float = np.inf
    network_exit_time: float = np.inf
    first_hop_time: float = np.inf
    last_hop_time: float = np.inf

    # 延迟计算
    cmd_latency: float = np.inf
    data_latency: float = np.inf
    transaction_latency: float = np.inf

    cmd_entry_cake0_cycle: float = np.inf  # cmd进入Cake0
    cmd_entry_noc_from_cake0_cycle: float = np.inf  # cmd从Cake0进入NoC
    cmd_entry_noc_from_cake1_cycle: float = np.inf  # cmd从Cake1进入NoC
    cmd_received_by_cake0_cycle: float = np.inf  # cmd从NoC进入Cake0
    cmd_received_by_cake1_cycle: float = np.inf  # cmd从NoC进入Cake1
    data_entry_noc_from_cake0_cycle: float = np.inf  # data从Cake0进入NoC(写)
    data_entry_noc_from_cake1_cycle: float = np.inf  # data从Cake1进入NoC(读)
    data_received_complete_cycle: float = np.inf  # data接收完成

    # ========== 位置和链路状态 ==========
    flit_position: str = "created"  # "created", "inject_queue", "network", "eject_queue", "completed"
    current_buffer: Optional[str] = None  # 当前缓冲区

    # ========== 流控和拥塞 ==========
    flow_control_info: Dict[str, Any] = field(default_factory=dict)
    congestion_info: Dict[str, Any] = field(default_factory=dict)

    # ========== IP和协议信息 ==========
    protocol_info: Dict[str, Any] = field(default_factory=dict)

    # ========== 调试和统计 ==========
    debug_info: Dict[str, Any] = field(default_factory=dict)
    custom_fields: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """初始化后处理"""
        if not self.path and self.source != -1 and self.destination != -1:
            self.path = [self.source, self.destination]
        if self.current_position == -1 and self.source != -1:
            self.current_position = self.source
        if self.creation_time == 0.0:
            import time

            self.creation_time = time.time()

    # ========== 抽象方法（由子类实现） ==========

    @abstractmethod
    def get_routing_info(self) -> Dict[str, Any]:
        """获取路由信息（拓扑特定）"""
        pass

    @abstractmethod
    def calculate_expected_hops(self) -> int:
        """计算预期跳数（拓扑特定）"""
        pass

    @abstractmethod
    def is_valid_next_hop(self, next_node: NodeId) -> bool:
        """检查下一跳是否有效（拓扑特定）"""
        pass

    # ========== 通用方法 ==========

    def advance_path(self) -> bool:
        """
        沿路径前进一步

        Returns:
            是否成功前进
        """
        if self.path_index + 1 >= len(self.path):
            return False

        self.path_index += 1
        self.current_position = self.path[self.path_index]
        self.hop_count += 1

        # 检查是否到达目的地
        if self.current_position == self.destination:
            self.is_arrive = True

        return True

    # ========== STI协议通用方法 ==========

    def create_response(self, rsp_type: str, **kwargs) -> "BaseFlit":
        """
        创建响应Flit

        Args:
            rsp_type: 响应类型
            **kwargs: 其他参数

        Returns:
            响应Flit
        """
        response = self.__class__(
            source=self.destination,  # 响应从目标返回
            destination=self.source,  # 到达原始源
            packet_id=self.packet_id,
            flit_id=0,  # 响应通常是单flit
            channel="rsp",
            flit_type="rsp",
            rsp_type=rsp_type,
            req_type=self.req_type,
            **kwargs,
        )

        # 同步相关信息
        response.original_req_time = self.original_req_time if self.original_req_time < np.inf else self.creation_time
        response.custom_fields.update(self.custom_fields)

        return response

    def create_data_flit(self, flit_id: int = 0, **kwargs) -> "BaseFlit":
        """
        创建数据Flit

        Args:
            flit_id: Flit ID
            **kwargs: 其他参数

        Returns:
            数据Flit
        """
        # 对于读操作，数据从目标返回到源
        # 对于写操作，数据从源发送到目标
        if self.req_type == "read":
            src, dst = self.destination, self.source
        else:
            src, dst = self.source, self.destination

        data_flit = self.__class__(
            source=src, destination=dst, packet_id=self.packet_id, flit_id=flit_id, channel="data", flit_type="data", req_type=self.req_type, burst_length=self.burst_length, **kwargs
        )

        # 设置头尾标识
        data_flit.is_head_flit = flit_id == 0
        data_flit.is_tail_flit = flit_id == self.burst_length - 1

        # 同步相关信息
        data_flit.original_req_time = self.original_req_time if self.original_req_time < np.inf else self.creation_time
        data_flit.custom_fields.update(self.custom_fields)

        return data_flit

    # ========== 重试机制通用方法 ==========

    def can_retry(self) -> bool:
        """检查是否可以重试"""
        return self.retry_count < self.max_retries

    def prepare_for_retry(self, reason: str = "") -> None:
        """准备重试"""
        if not self.can_retry():
            raise ValueError(f"Already reached max retries ({self.max_retries})")

        self.retry_count += 1
        self.retry_reason = reason
        self.is_retry = True
        self.req_attr = "old"
        self.req_state = "invalid"

        # 重置网络状态
        self.is_injected = False
        self.is_arrive = False
        self.is_finish = False
        self.path_index = 0
        self.current_position = self.source
        self.hop_count = 0
        self.is_new_on_network = True

        # 保持原始请求时间
        if self.original_req_time == np.inf:
            self.original_req_time = self.creation_time

    def mark_retry_success(self) -> None:
        """标记重试成功"""
        self.req_attr = "new"
        self.req_state = "valid"
        # 注意：不重置retry_count，保留重试历史

    def get_retry_info(self) -> Dict[str, Any]:
        """获取重试信息"""
        return {
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "is_retry": self.is_retry,
            "retry_reason": self.retry_reason,
            "can_retry": self.can_retry(),
            "original_req_time": self.original_req_time,
        }

    # ========== 延迟记录同步（NoC公有功能） ==========

    def sync_latency_record(self, other_flit: "BaseFlit") -> None:
        """
        同步延迟记录（用于请求-响应-数据之间的时间同步）

        Args:
            other_flit: 要同步的flit
        """
        # 同步基础时间戳
        if other_flit.creation_time > 0 and self.creation_time > other_flit.creation_time:
            self.creation_time = other_flit.creation_time

        if other_flit.injection_time < self.injection_time:
            self.injection_time = other_flit.injection_time

        if other_flit.original_req_time < self.original_req_time:
            self.original_req_time = other_flit.original_req_time

        # 同步自定义字段
        for key, value in other_flit.custom_fields.items():
            if key not in self.custom_fields:
                self.custom_fields[key] = value

        # 同步时间戳
        if other_flit.req_type == "read":
            self.cmd_entry_cake0_cycle = min(other_flit.cmd_entry_cake0_cycle, self.cmd_entry_cake0_cycle)
            self.cmd_entry_noc_from_cake0_cycle = min(other_flit.cmd_entry_noc_from_cake0_cycle, self.cmd_entry_noc_from_cake0_cycle)
            self.cmd_received_by_cake1_cycle = min(other_flit.cmd_received_by_cake1_cycle, self.cmd_received_by_cake1_cycle)
            self.data_entry_noc_from_cake1_cycle = min(other_flit.data_entry_noc_from_cake1_cycle, self.data_entry_noc_from_cake1_cycle)
            self.data_received_complete_cycle = min(other_flit.data_received_complete_cycle, self.data_received_complete_cycle)
        elif other_flit.req_type == "write":
            self.cmd_entry_cake0_cycle = min(other_flit.cmd_entry_cake0_cycle, self.cmd_entry_cake0_cycle)
            self.cmd_entry_noc_from_cake0_cycle = min(other_flit.cmd_entry_noc_from_cake0_cycle, self.cmd_entry_noc_from_cake0_cycle)
            self.cmd_received_by_cake1_cycle = min(other_flit.cmd_received_by_cake1_cycle, self.cmd_received_by_cake1_cycle)
            self.cmd_entry_noc_from_cake1_cycle = min(other_flit.cmd_entry_noc_from_cake1_cycle, self.cmd_entry_noc_from_cake1_cycle)
            self.cmd_received_by_cake0_cycle = min(other_flit.cmd_received_by_cake0_cycle, self.cmd_received_by_cake0_cycle)
            self.data_entry_noc_from_cake0_cycle = min(other_flit.data_entry_noc_from_cake0_cycle, self.data_entry_noc_from_cake0_cycle)
            self.data_received_complete_cycle = min(other_flit.data_received_complete_cycle, self.data_received_complete_cycle)

    def calculate_latencies(self) -> Dict[str, float]:
        """计算延迟指标"""
        latencies = {}

        # 命令延迟
        if self.cmd_entry_noc_from_cake0_cycle < np.inf and self.cmd_received_by_cake1_cycle < np.inf:
            latencies["cmd_latency"] = self.cmd_latency = self.cmd_received_by_cake1_cycle - self.cmd_entry_noc_from_cake0_cycle

        # 数据延迟
        if self.req_type == "read":
            if self.data_entry_noc_from_cake1_cycle < np.inf and self.data_received_complete_cycle < np.inf:
                latencies["data_latency"] = self.data_latency = self.data_received_complete_cycle - self.data_entry_noc_from_cake1_cycle
        elif self.req_type == "write":
            if self.data_entry_noc_from_cake0_cycle < np.inf and self.data_received_complete_cycle < np.inf:
                latencies["data_latency"] = self.data_latency = self.data_received_complete_cycle - self.data_entry_noc_from_cake0_cycle

        # 事务延迟
        if self.cmd_entry_cake0_cycle < np.inf and self.data_received_complete_cycle < np.inf:
            latencies["transaction_latency"] = self.transaction_latency = self.data_received_complete_cycle - self.cmd_entry_cake0_cycle

        return latencies

    def get_coordinates(self, grid_width: int) -> tuple[int, int]:
        """
        获取节点在网格中的坐标（适用于网格类拓扑）

        Args:
            grid_width: 网格宽度

        Returns:
            (x, y)坐标
        """
        x = self.current_position % grid_width
        y = self.current_position // grid_width
        return x, y

    def distance_to_destination(self, grid_width: int) -> int:
        """
        计算到目的地的曼哈顿距离（适用于网格类拓扑）

        Args:
            grid_width: 网格宽度

        Returns:
            曼哈顿距离
        """
        src_x, src_y = self.get_coordinates(grid_width)
        dst_x = self.destination % grid_width
        dst_y = self.destination // grid_width
        return abs(dst_x - src_x) + abs(dst_y - src_y)

    def get_status_summary(self) -> Dict[str, Any]:
        """获取状态摘要"""
        return {
            "packet_id": self.packet_id,
            "flit_id": self.flit_id,
            "source": self.source,
            "destination": self.destination,
            "current_position": self.current_position,
            "path_progress": f"{self.path_index}/{len(self.path)}",
            "hop_count": self.hop_count,
            "is_injected": self.is_injected,
            "is_arrive": self.is_arrive,
            "is_finish": self.is_finish,
            "position": self.flit_position,
            "cmd_latency": self.cmd_latency,
            "data_latency": self.data_latency,
            "transaction_latency": self.transaction_latency,
        }

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式用于序列化"""
        return {
            "packet_id": self.packet_id,
            "flit_id": self.flit_id,
            "source": self.source,
            "destination": self.destination,
            "req_type": self.req_type,
            "burst_length": self.burst_length,
            "path": self.path,
            "current_position": self.current_position,
            "hop_count": self.hop_count,
            "is_injected": self.is_injected,
            "is_arrive": self.is_arrive,
            "is_finish": self.is_finish,
            "cmd_latency": self.cmd_latency,
            "data_latency": self.data_latency,
            "transaction_latency": self.transaction_latency,
            "custom_fields": self.custom_fields,
        }

    def __repr__(self) -> str:
        """字符串表示"""
        status = []
        if self.rsp_type:
            status.append(self.rsp_type[:3])
        if self.is_finish:
            status.append("F")

        status_str = "".join(status) if status else "N"

        return (
            f"{self.packet_id}.{self.flit_id}, "
            f"{self.source}{self.source_type[0]}{self.source_type[-1]}->{self.destination}{self.destination_type[0]}{self.destination_type[-1]}@{self.current_position}, "
            f"{self.flit_type}, {status_str}"
        )


class FlitPool:
    """通用Flit对象池"""

    def __init__(self, flit_class: type, initial_size: int = 1000):
        self.flit_class = flit_class
        self._pool = deque()
        self._lock = threading.Lock()
        self._created_count = 0

        # 预先填充池
        for _ in range(initial_size):
            self._pool.append(self._create_new_flit())

    def _create_new_flit(self):
        """创建新的Flit实例"""
        self._created_count += 1
        return self.flit_class.__new__(self.flit_class)

    def get_flit(self, **kwargs):
        """从池中获取Flit"""
        with self._lock:
            if self._pool:
                flit = self._pool.popleft()
            else:
                flit = self._create_new_flit()

        # 重新初始化
        flit.__init__(**kwargs)
        return flit

    def return_flit(self, flit):
        """将Flit返回到池中"""
        if flit is None:
            return

        # 重置状态
        flit._reset_for_reuse()

        with self._lock:
            if len(self._pool) < 2000:  # 限制池大小
                self._pool.append(flit)

    def get_stats(self) -> Dict[str, int]:
        """获取池统计信息"""
        with self._lock:
            return {"pool_size": len(self._pool), "created_count": self._created_count, "flit_type": self.flit_class.__name__}


# 为BaseFlit添加重置方法
def _reset_for_reuse(self):
    """重置Flit以供重用"""
    # ========== 基础标识字段 ==========
    self.packet_id = ""
    self.flit_id = 0
    self.source = 0
    self.destination = 0

    # ========== 请求类型和数据 ==========
    self.req_type = "R"
    self.burst_length = 1
    self.flit_size = 128
    self.priority = Priority.MEDIUM
    self.source_type = "gdma_0"
    self.destination_type = "ddr_0"

    # ========== STI三通道协议 ==========
    self.channel = "req"
    self.flit_type = "req"
    self.req_attr = "new"
    self.req_state = "valid"
    self.rsp_type = None

    # ========== 重试机制 ==========
    self.retry_count = 0
    self.max_retries = 1
    self.original_req_time = np.inf
    self.retry_reason = ""
    self.is_retry = False

    # ========== 路径和路由信息 ==========
    self.path = []
    self.path_index = 0
    self.current_position = -1
    self.next_hop = -1

    # ========== 通用状态字段 ==========
    self.is_injected = False
    self.is_ejected = False
    self.is_arrive = False
    self.is_finish = False
    self.is_head_flit = True
    self.is_tail_flit = True

    # ========== 网络状态 ==========
    self.is_new_on_network = True
    self.hop_count = 0

    # ========== 时间戳记录 ==========
    self.creation_time = 0.0
    self.injection_time = np.inf
    self.ejection_time = np.inf
    self.completion_time = np.inf
    self.network_entry_time = np.inf
    self.network_exit_time = np.inf

    # ========== 延迟计算 ==========
    self.cmd_latency = np.inf
    self.data_latency = np.inf
    self.transaction_latency = np.inf

    # ========== 详细时间戳 ==========
    self.cmd_entry_cake0_cycle = np.inf
    self.cmd_entry_noc_from_cake0_cycle = np.inf
    self.cmd_entry_noc_from_cake1_cycle = np.inf
    self.cmd_received_by_cake0_cycle = np.inf
    self.cmd_received_by_cake1_cycle = np.inf
    self.data_entry_noc_from_cake0_cycle = np.inf
    self.data_entry_noc_from_cake1_cycle = np.inf
    self.data_received_complete_cycle = np.inf

    # ========== 位置和链路状态 ==========
    self.flit_position = "created"
    self.current_buffer = None

    # ========== 清空字典 ==========
    self.routing_info.clear()
    self.flow_control_info.clear()
    self.congestion_info.clear()
    self.protocol_info.clear()
    self.debug_info.clear()
    self.custom_fields.clear()


BaseFlit._reset_for_reuse = _reset_for_reuse


# 工厂函数
def create_flit(flit_class: type, source: NodeId, destination: NodeId, path: List[NodeId] = None, **kwargs) -> BaseFlit:
    """
    创建Flit的工厂函数

    Args:
        flit_class: Flit类型
        source: 源节点
        destination: 目标节点
        path: 路径（可选）
        **kwargs: 其他参数

    Returns:
        Flit实例
    """
    if path is None:
        path = [source, destination]

    return flit_class(source=source, destination=destination, path=path, **kwargs)
