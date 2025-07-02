"""
NoC节点抽象类。

本模块定义了NoC网络中节点的抽象接口，包括：
- 节点基本功能
- 缓冲区管理
- 路由决策
- 性能监控
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import deque, defaultdict
from enum import Enum

from ..types import (
    NodeId, Position, NodeType, Priority, NodeMetrics,
    MetricsDict, ConfigDict
)


class NodeState(Enum):
    """节点状态枚举。"""
    IDLE = "idle"           # 空闲
    PROCESSING = "processing"   # 处理中
    BLOCKED = "blocked"     # 阻塞
    FAILED = "failed"       # 故障


class BufferStatus(Enum):
    """缓冲区状态枚举。"""
    EMPTY = "empty"         # 空
    AVAILABLE = "available"  # 可用
    FULL = "full"           # 满
    OVERFLOW = "overflow"    # 溢出


class BaseNoCNode(ABC):
    """
    NoC节点抽象基类。
    
    定义了所有NoC节点必须实现的基本接口，包括数据处理、
    缓冲区管理、路由决策等功能。
    """
    
    def __init__(self, node_id: NodeId, position: Position, 
                 node_type: NodeType = NodeType.ROUTER):
        """
        初始化NoC节点。
        
        Args:
            node_id: 节点ID
            position: 节点位置坐标
            node_type: 节点类型
        """
        self.node_id = node_id
        self.position = position
        self.node_type = node_type
        
        # 邻居节点
        self.neighbors: List[NodeId] = []
        
        # 节点状态
        self.state = NodeState.IDLE
        self.is_active = True
        
        # 缓冲区配置
        self.input_buffer_size = 8
        self.output_buffer_size = 8
        self.virtual_channels = 2
        
        # 缓冲区存储
        self.input_buffers: Dict[str, deque] = {}
        self.output_buffers: Dict[str, deque] = {}
        self.virtual_channel_buffers: Dict[str, List[deque]] = {}
        
        # 缓冲区状态
        self.buffer_status: Dict[str, BufferStatus] = {}
        self.buffer_occupancy: Dict[str, int] = {}
        
        # 路由相关
        self.routing_table: Dict[NodeId, List[NodeId]] = {}
        self.next_hop_cache: Dict[NodeId, NodeId] = {}
        
        # 性能监控
        self.metrics = NodeMetrics()
        self.stats: Dict[str, Any] = defaultdict(int)
        
        # 流量控制
        self.credit_count: Dict[str, int] = {}
        self.max_credits: Dict[str, int] = {}
        
        # 优先级管理
        self.priority_queues: Dict[Priority, deque] = {
            Priority.LOW: deque(),
            Priority.MEDIUM: deque(),
            Priority.HIGH: deque(),
            Priority.CRITICAL: deque()
        }
        
        # 初始化节点
        self._initialize_buffers()
        self._initialize_flow_control()
    
    def _initialize_buffers(self) -> None:
        """初始化缓冲区。"""
        # 为每个方向初始化输入输出缓冲区
        directions = ['north', 'south', 'east', 'west', 'local']
        
        for direction in directions:
            # 输入缓冲区
            self.input_buffers[direction] = deque(maxlen=self.input_buffer_size)
            # 输出缓冲区
            self.output_buffers[direction] = deque(maxlen=self.output_buffer_size)
            
            # 虚拟通道缓冲区
            self.virtual_channel_buffers[direction] = [
                deque(maxlen=self.input_buffer_size // self.virtual_channels)
                for _ in range(self.virtual_channels)
            ]
            
            # 初始化缓冲区状态
            self.buffer_status[direction] = BufferStatus.EMPTY
            self.buffer_occupancy[direction] = 0
    
    def _initialize_flow_control(self) -> None:
        """初始化流控制。"""
        directions = ['north', 'south', 'east', 'west', 'local']
        
        for direction in directions:
            self.credit_count[direction] = self.output_buffer_size
            self.max_credits[direction] = self.output_buffer_size
    
    # ========== 抽象方法 - 必须被子类实现 ==========
    
    @abstractmethod
    def process_flit(self, flit: Any, input_port: str) -> bool:
        """
        处理接收到的flit。
        
        Args:
            flit: 要处理的flit对象
            input_port: 输入端口名称
            
        Returns:
            是否成功处理
        """
        pass
    
    @abstractmethod
    def route_flit(self, flit: Any) -> Optional[str]:
        """
        为flit进行路由决策。
        
        Args:
            flit: 要路由的flit对象
            
        Returns:
            输出端口名称，如果无法路由则返回None
        """
        pass
    
    @abstractmethod
    def can_accept_flit(self, input_port: str, priority: Priority = Priority.MEDIUM) -> bool:
        """
        检查是否可以接收新的flit。
        
        Args:
            input_port: 输入端口名称
            priority: flit优先级
            
        Returns:
            是否可以接收
        """
        pass
    
    # ========== 邻居管理方法 ==========
    
    def add_neighbor(self, neighbor_id: NodeId, direction: Optional[str] = None) -> None:
        """
        添加邻居节点。
        
        Args:
            neighbor_id: 邻居节点ID
            direction: 连接方向（可选）
        """
        if neighbor_id not in self.neighbors:
            self.neighbors.append(neighbor_id)
        
        # 如果指定了方向，更新路由表
        if direction:
            self.routing_table[neighbor_id] = [neighbor_id]
    
    def remove_neighbor(self, neighbor_id: NodeId) -> None:
        """
        移除邻居节点。
        
        Args:
            neighbor_id: 要移除的邻居节点ID
        """
        if neighbor_id in self.neighbors:
            self.neighbors.remove(neighbor_id)
        
        # 清理路由表
        if neighbor_id in self.routing_table:
            del self.routing_table[neighbor_id]
        if neighbor_id in self.next_hop_cache:
            del self.next_hop_cache[neighbor_id]
    
    def get_neighbors(self) -> List[NodeId]:
        """
        获取邻居节点列表。
        
        Returns:
            邻居节点ID列表
        """
        return self.neighbors.copy()
    
    # ========== 缓冲区管理方法 ==========
    
    def get_buffer_occupancy(self, buffer_name: str) -> int:
        """
        获取缓冲区占用情况。
        
        Args:
            buffer_name: 缓冲区名称
            
        Returns:
            缓冲区占用的flit数量
        """
        if buffer_name in self.input_buffers:
            return len(self.input_buffers[buffer_name])
        elif buffer_name in self.output_buffers:
            return len(self.output_buffers[buffer_name])
        return 0
    
    def get_buffer_status(self, buffer_name: str) -> BufferStatus:
        """
        获取缓冲区状态。
        
        Args:
            buffer_name: 缓冲区名称
            
        Returns:
            缓冲区状态
        """
        occupancy = self.get_buffer_occupancy(buffer_name)
        max_size = self._get_buffer_max_size(buffer_name)
        
        if occupancy == 0:
            return BufferStatus.EMPTY
        elif occupancy >= max_size:
            return BufferStatus.FULL
        else:
            return BufferStatus.AVAILABLE
    
    def _get_buffer_max_size(self, buffer_name: str) -> int:
        """
        获取缓冲区最大容量。
        
        Args:
            buffer_name: 缓冲区名称
            
        Returns:
            最大容量
        """
        if buffer_name in self.input_buffers:
            return self.input_buffer_size
        elif buffer_name in self.output_buffers:
            return self.output_buffer_size
        return 0
    
    def is_buffer_full(self, buffer_name: str) -> bool:
        """
        检查缓冲区是否已满。
        
        Args:
            buffer_name: 缓冲区名称
            
        Returns:
            是否已满
        """
        return self.get_buffer_status(buffer_name) == BufferStatus.FULL
    
    def is_buffer_empty(self, buffer_name: str) -> bool:
        """
        检查缓冲区是否为空。
        
        Args:
            buffer_name: 缓冲区名称
            
        Returns:
            是否为空
        """
        return self.get_buffer_status(buffer_name) == BufferStatus.EMPTY
    
    def get_available_space(self, buffer_name: str) -> int:
        """
        获取缓冲区可用空间。
        
        Args:
            buffer_name: 缓冲区名称
            
        Returns:
            可用空间大小
        """
        max_size = self._get_buffer_max_size(buffer_name)
        occupancy = self.get_buffer_occupancy(buffer_name)
        return max_size - occupancy
    
    # ========== 虚拟通道管理 ==========
    
    def get_virtual_channel(self, direction: str, vc_id: int) -> Optional[deque]:
        """
        获取虚拟通道。
        
        Args:
            direction: 方向
            vc_id: 虚拟通道ID
            
        Returns:
            虚拟通道队列，如果不存在则返回None
        """
        if direction in self.virtual_channel_buffers:
            if 0 <= vc_id < len(self.virtual_channel_buffers[direction]):
                return self.virtual_channel_buffers[direction][vc_id]
        return None
    
    def allocate_virtual_channel(self, direction: str, priority: Priority = Priority.MEDIUM) -> Optional[int]:
        """
        分配虚拟通道。
        
        Args:
            direction: 方向
            priority: 优先级
            
        Returns:
            分配的虚拟通道ID，如果无法分配则返回None
        """
        if direction not in self.virtual_channel_buffers:
            return None
        
        # 根据优先级选择虚拟通道
        vc_buffers = self.virtual_channel_buffers[direction]
        
        # 优先分配给高优先级
        if priority in [Priority.HIGH, Priority.CRITICAL]:
            for i in range(len(vc_buffers)):
                if len(vc_buffers[i]) < vc_buffers[i].maxlen:
                    return i
        
        # 为普通优先级找最空的通道
        min_occupancy = float('inf')
        best_vc = None
        
        for i, vc_buffer in enumerate(vc_buffers):
            occupancy = len(vc_buffer)
            if occupancy < vc_buffer.maxlen and occupancy < min_occupancy:
                min_occupancy = occupancy
                best_vc = i
        
        return best_vc
    
    # ========== 流控制方法 ==========
    
    def has_credit(self, output_port: str) -> bool:
        """
        检查是否有可用的信用。
        
        Args:
            output_port: 输出端口
            
        Returns:
            是否有可用信用
        """
        return self.credit_count.get(output_port, 0) > 0
    
    def consume_credit(self, output_port: str) -> bool:
        """
        消费一个信用。
        
        Args:
            output_port: 输出端口
            
        Returns:
            是否成功消费
        """
        if self.has_credit(output_port):
            self.credit_count[output_port] -= 1
            return True
        return False
    
    def return_credit(self, input_port: str) -> None:
        """
        返回一个信用。
        
        Args:
            input_port: 输入端口
        """
        if input_port in self.credit_count:
            max_credits = self.max_credits.get(input_port, self.input_buffer_size)
            if self.credit_count[input_port] < max_credits:
                self.credit_count[input_port] += 1
    
    # ========== 优先级管理 ==========
    
    def enqueue_by_priority(self, flit: Any, priority: Priority) -> bool:
        """
        按优先级将flit加入队列。
        
        Args:
            flit: flit对象
            priority: 优先级
            
        Returns:
            是否成功加入队列
        """
        if priority in self.priority_queues:
            self.priority_queues[priority].append(flit)
            return True
        return False
    
    def dequeue_by_priority(self) -> Optional[Tuple[Any, Priority]]:
        """
        按优先级从队列中取出flit。
        
        Returns:
            (flit, priority)元组，如果队列为空则返回None
        """
        # 按优先级顺序检查队列
        priority_order = [Priority.CRITICAL, Priority.HIGH, Priority.MEDIUM, Priority.LOW]
        
        for priority in priority_order:
            if self.priority_queues[priority]:
                flit = self.priority_queues[priority].popleft()
                return flit, priority
        
        return None
    
    def get_priority_queue_status(self) -> Dict[Priority, int]:
        """
        获取各优先级队列的状态。
        
        Returns:
            各优先级队列的长度
        """
        return {priority: len(queue) for priority, queue in self.priority_queues.items()}
    
    # ========== 路由相关方法 ==========
    
    def update_routing_table(self, routing_table: Dict[NodeId, List[NodeId]]) -> None:
        """
        更新路由表。
        
        Args:
            routing_table: 新的路由表
        """
        self.routing_table.update(routing_table)
        # 清除缓存
        self.next_hop_cache.clear()
    
    def get_next_hop(self, destination: NodeId) -> Optional[NodeId]:
        """
        获取到目标节点的下一跳。
        
        Args:
            destination: 目标节点ID
            
        Returns:
            下一跳节点ID，如果不存在路由则返回None
        """
        # 检查缓存
        if destination in self.next_hop_cache:
            return self.next_hop_cache[destination]
        
        # 查找路由表
        if destination in self.routing_table:
            path = self.routing_table[destination]
            if len(path) > 1:
                next_hop = path[1]  # 第一个是当前节点，第二个是下一跳
                self.next_hop_cache[destination] = next_hop
                return next_hop
        
        return None
    
    def set_route(self, destination: NodeId, path: List[NodeId]) -> None:
        """
        设置到目标节点的路由。
        
        Args:
            destination: 目标节点ID
            path: 路径（节点ID列表）
        """
        self.routing_table[destination] = path
        # 清除相关缓存
        if destination in self.next_hop_cache:
            del self.next_hop_cache[destination]
    
    # ========== 性能监控方法 ==========
    
    def update_metrics(self, metrics_update: Dict[str, Any]) -> None:
        """
        更新节点性能指标。
        
        Args:
            metrics_update: 指标更新字典
        """
        for key, value in metrics_update.items():
            if hasattr(self.metrics, key):
                setattr(self.metrics, key, value)
            else:
                self.stats[key] = value
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        获取性能统计。
        
        Returns:
            性能统计字典
        """
        stats = {
            'node_id': self.node_id,
            'node_type': self.node_type.value,
            'state': self.state.value,
            'packets_processed': self.metrics.packets_processed,
            'packets_generated': self.metrics.packets_generated,
            'packets_received': self.metrics.packets_received,
            'packet_drops': self.metrics.packet_drops,
            'average_latency': self.metrics.average_packet_latency,
            'power_consumption': self.metrics.power_consumption,
        }
        
        # 添加缓冲区统计
        for direction in self.input_buffers:
            stats[f'input_buffer_{direction}_occupancy'] = self.get_buffer_occupancy(direction)
            stats[f'output_buffer_{direction}_occupancy'] = self.get_buffer_occupancy(f'output_{direction}')
        
        # 添加优先级队列统计
        priority_stats = self.get_priority_queue_status()
        for priority, count in priority_stats.items():
            stats[f'priority_queue_{priority.name.lower()}'] = count
        
        # 添加自定义统计
        stats.update(self.stats)
        
        return stats
    
    def reset_stats(self) -> None:
        """重置统计数据。"""
        self.metrics = NodeMetrics()
        self.stats.clear()
    
    # ========== 故障处理方法 ==========
    
    def set_failed(self, reason: str = "未知故障") -> None:
        """
        设置节点为故障状态。
        
        Args:
            reason: 故障原因
        """
        self.state = NodeState.FAILED
        self.is_active = False
        self.stats['failure_reason'] = reason
        self.stats['failure_time'] = self.stats.get('current_cycle', 0)
    
    def recover(self) -> None:
        """从故障中恢复。"""
        if self.state == NodeState.FAILED:
            self.state = NodeState.IDLE
            self.is_active = True
            self.stats['recovery_time'] = self.stats.get('current_cycle', 0)
    
    def is_operational(self) -> bool:
        """
        检查节点是否正常运行。
        
        Returns:
            是否正常运行
        """
        return self.is_active and self.state != NodeState.FAILED
    
    # ========== 配置方法 ==========
    
    def configure(self, config: ConfigDict) -> None:
        """
        配置节点参数。
        
        Args:
            config: 配置字典
        """
        # 更新缓冲区大小
        if 'input_buffer_size' in config:
            self.input_buffer_size = config['input_buffer_size']
        if 'output_buffer_size' in config:
            self.output_buffer_size = config['output_buffer_size']
        if 'virtual_channels' in config:
            self.virtual_channels = config['virtual_channels']
        
        # 重新初始化缓冲区（如果大小发生变化）
        if any(key in config for key in ['input_buffer_size', 'output_buffer_size', 'virtual_channels']):
            self._initialize_buffers()
            self._initialize_flow_control()
    
    def get_config(self) -> ConfigDict:
        """
        获取节点配置。
        
        Returns:
            配置字典
        """
        return {
            'node_id': self.node_id,
            'position': self.position,
            'node_type': self.node_type.value,
            'input_buffer_size': self.input_buffer_size,
            'output_buffer_size': self.output_buffer_size,
            'virtual_channels': self.virtual_channels,
            'neighbors': self.neighbors.copy()
        }
    
    # ========== 调试和可视化方法 ==========
    
    def get_status_summary(self) -> str:
        """
        获取节点状态摘要。
        
        Returns:
            状态摘要字符串
        """
        lines = [f"节点 {self.node_id} ({self.node_type.value})"]
        lines.append(f"位置: {self.position}")
        lines.append(f"状态: {self.state.value}")
        lines.append(f"活跃: {'是' if self.is_active else '否'}")
        lines.append(f"邻居: {self.neighbors}")
        
        lines.append("缓冲区状态:")
        for direction in self.input_buffers:
            input_occ = self.get_buffer_occupancy(direction)
            output_occ = self.get_buffer_occupancy(f'output_{direction}')
            lines.append(f"  {direction}: 输入={input_occ}, 输出={output_occ}")
        
        return "\n".join(lines)
    
    def __str__(self) -> str:
        """字符串表示。"""
        return f"Node({self.node_id}, {self.position}, {self.node_type.value})"
    
    def __repr__(self) -> str:
        """详细字符串表示。"""
        return f"Node(id={self.node_id}, pos={self.position}, type={self.node_type.value}, state={self.state.value})"