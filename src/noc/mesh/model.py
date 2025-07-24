"""
Mesh NoC模型实现。

本模块实现完整的Mesh NoC仿真模型，包括：
- 节点和路由器模型
- 包注入和传输
- 性能统计
- 仿真控制
"""

import time
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque

from ..base.model import BaseNoCModel
from .config import MeshConfig
from .topology import MeshTopology
from src.noc.utils.types import NodeId

# 定义缺失的类型
PacketId = int
FlitId = int


class MeshPacket:
    """Mesh网络包"""

    def __init__(self, packet_id: PacketId, src: NodeId, dst: NodeId, size: int, injection_time: int, packet_type: str = "data"):
        self.packet_id = packet_id
        self.src = src
        self.dst = dst
        self.size = size  # 包大小（flits）
        self.packet_type = packet_type
        self.injection_time = injection_time
        self.completion_time = -1
        self.path = []
        self.current_hop = 0
        self.hops = 0


class MeshRouter:
    """Mesh路由器模型"""

    def __init__(self, router_id: NodeId, config: MeshConfig):
        self.router_id = router_id
        self.config = config

        # 缓冲区
        self.input_buffers = defaultdict(deque)  # {direction: deque}
        self.output_buffers = defaultdict(deque)

        # 统计信息
        self.packets_routed = 0
        self.flits_routed = 0
        self.buffer_occupancy = defaultdict(int)

    def can_accept_packet(self, direction: str) -> bool:
        """检查是否可以接受包"""
        buffer_depth = self.config.mesh_config.INPUT_BUFFER_DEPTH
        return len(self.input_buffers[direction]) < buffer_depth

    def inject_packet(self, packet: MeshPacket, direction: str = "local") -> bool:
        """注入包到路由器"""
        if self.can_accept_packet(direction):
            self.input_buffers[direction].append(packet)
            return True
        return False

    def route_packets(self, topology: MeshTopology, current_cycle: int) -> List[Tuple[MeshPacket, str]]:
        """路由包处理"""
        routed_packets = []

        # 处理所有输入缓冲区的包
        for direction, buffer in self.input_buffers.items():
            if buffer:
                packet = buffer.popleft()

                # 如果到达目标，完成传输
                if packet.dst == self.router_id:
                    packet.completion_time = current_cycle
                    packet.hops = len(packet.path)
                    routed_packets.append((packet, "completed"))
                else:
                    # 计算下一跳
                    next_hop = self._get_next_hop(packet, topology)
                    if next_hop is not None:
                        packet.path.append(self.router_id)
                        routed_packets.append((packet, f"next_hop_{next_hop}"))

                self.packets_routed += 1
                self.flits_routed += packet.size

        return routed_packets

    def _get_next_hop(self, packet: MeshPacket, topology: MeshTopology) -> Optional[NodeId]:
        """获取下一跳节点"""
        # 使用拓扑的路由策略
        route = topology.get_route(self.router_id, packet.dst)

        if len(route) > 1:
            # 找到当前节点在路径中的位置
            try:
                current_index = route.index(self.router_id)
                if current_index + 1 < len(route):
                    return route[current_index + 1]
            except ValueError:
                # 如果当前节点不在路径中，重新计算
                pass

        return None


class MeshModel(BaseNoCModel):
    """
    Mesh NoC仿真模型。

    实现完整的Mesh NoC仿真，包括包注入、路由、传输和统计。
    """

    def __init__(self, config: MeshConfig):
        """
        初始化Mesh模型。

        Args:
            config: Mesh配置对象
        """
        super().__init__(config)
        self.mesh_config = config
        self.topology = MeshTopology(config)

        # 路由器
        self.routers = {}
        for node_id in range(config.num_nodes):
            self.routers[node_id] = MeshRouter(node_id, config)

        # 仿真状态
        self.current_cycle = 0
        self.next_packet_id = 0

        # 包管理
        self.active_packets = {}  # {packet_id: packet}
        self.completed_packets = []
        self.injected_packets = []

        # 统计信息
        self.stats = {
            "total_packets_injected": 0,
            "total_packets_completed": 0,
            "total_flits_injected": 0,
            "total_flits_completed": 0,
            "total_latency": 0,
            "total_hops": 0,
            "cycle_counts": defaultdict(int),
        }

        # 性能监控
        self.link_utilization = defaultdict(float)
        self.buffer_utilization = defaultdict(float)

        # 验证拓扑
        is_valid, error_msg = self.topology.validate_topology()
        if not is_valid:
            raise ValueError(f"Mesh拓扑验证失败: {error_msg}")

    def initialize_network(self) -> None:
        """初始化网络"""
        # 重置所有状态
        self.current_cycle = 0
        self.next_packet_id = 0
        self.active_packets.clear()
        self.completed_packets.clear()
        self.injected_packets.clear()

        # 重置统计
        for key in self.stats:
            if isinstance(self.stats[key], dict):
                self.stats[key].clear()
            else:
                self.stats[key] = 0

        print(f"Mesh网络初始化完成: {self.mesh_config.rows}x{self.mesh_config.cols}")

    def inject_packet(self, src_node: NodeId, dst_node: NodeId, op_type: str = "R", burst_size: int = 4, cycle: int = None) -> bool:
        """
        注入包到网络。

        Args:
            src_node: 源节点ID
            dst_node: 目标节点ID
            op_type: 操作类型
            burst_size: 包大小（flits）
            cycle: 注入周期

        Returns:
            是否成功注入
        """
        if cycle is None:
            cycle = self.current_cycle

        # 验证节点ID
        if not (0 <= src_node < self.mesh_config.num_nodes):
            return False
        if not (0 <= dst_node < self.mesh_config.num_nodes):
            return False

        # 创建包
        packet = MeshPacket(packet_id=self.next_packet_id, src=src_node, dst=dst_node, size=burst_size, injection_time=cycle, packet_type=op_type)

        # 尝试注入到源节点路由器
        router = self.routers[src_node]
        if router.inject_packet(packet, "local"):
            self.active_packets[packet.packet_id] = packet
            self.injected_packets.append(packet)
            self.next_packet_id += 1

            # 更新统计
            self.stats["total_packets_injected"] += 1
            self.stats["total_flits_injected"] += burst_size

            return True

        return False

    def advance_cycle(self) -> None:
        """推进一个仿真周期"""
        self.current_cycle += 1

        # 处理所有路由器的包
        completed_in_cycle = []
        packets_to_forward = []

        for router_id, router in self.routers.items():
            routed_results = router.route_packets(self.topology, self.current_cycle)

            for packet, action in routed_results:
                if action == "completed":
                    completed_in_cycle.append(packet)
                elif action.startswith("next_hop_"):
                    next_hop = int(action.split("_")[-1])
                    packets_to_forward.append((packet, next_hop))

        # 处理完成的包
        for packet in completed_in_cycle:
            self._complete_packet(packet)

        # 转发包到下一跳
        for packet, next_hop in packets_to_forward:
            next_router = self.routers[next_hop]
            if next_router.inject_packet(packet, "network"):
                # 如果成功转发，从当前活跃包中继续跟踪
                pass
            else:
                # 如果转发失败，需要处理阻塞
                # 简化实现：暂时丢弃（实际应该处理背压）
                pass

        # 更新统计
        self.stats["cycle_counts"][self.current_cycle] = len(completed_in_cycle)

    def _complete_packet(self, packet: MeshPacket) -> None:
        """完成包传输"""
        if packet.packet_id in self.active_packets:
            del self.active_packets[packet.packet_id]

        self.completed_packets.append(packet)

        # 更新统计
        self.stats["total_packets_completed"] += 1
        self.stats["total_flits_completed"] += packet.size

        latency = packet.completion_time - packet.injection_time
        self.stats["total_latency"] += latency
        self.stats["total_hops"] += packet.hops

    def get_completed_packets(self) -> List[Dict[str, Any]]:
        """获取已完成的包信息"""
        completed_info = []

        for packet in self.completed_packets:
            packet_info = {
                "packet_id": packet.packet_id,
                "src": packet.src,
                "dst": packet.dst,
                "flit_count": packet.size,
                "injection_time": packet.injection_time,
                "completion_time": packet.completion_time,
                "latency": packet.completion_time - packet.injection_time,
                "hops": packet.hops,
                "path": packet.path,
                "traffic_id": f"mesh_packet_{packet.packet_id}",
            }
            completed_info.append(packet_info)

        # 清空已处理的包
        self.completed_packets.clear()

        return completed_info

    def get_network_statistics(self) -> Dict[str, Any]:
        """获取网络统计信息"""
        total_packets = self.stats["total_packets_completed"]

        # 计算平均值
        avg_latency = self.stats["total_latency"] / total_packets if total_packets > 0 else 0
        avg_hops = self.stats["total_hops"] / total_packets if total_packets > 0 else 0

        # 计算利用率（简化）
        total_possible_flits = self.current_cycle * self.mesh_config.num_nodes
        utilization = (self.stats["total_flits_completed"] / total_possible_flits * 100) if total_possible_flits > 0 else 0

        # 吞吐量
        throughput = self.stats["total_packets_completed"] / self.current_cycle if self.current_cycle > 0 else 0

        return {
            "cycle": self.current_cycle,
            "total_packets_injected": self.stats["total_packets_injected"],
            "total_packets_completed": self.stats["total_packets_completed"],
            "total_flits_injected": self.stats["total_flits_injected"],
            "total_flits_completed": self.stats["total_flits_completed"],
            "active_packets": len(self.active_packets),
            "avg_latency": avg_latency,
            "avg_hops": avg_hops,
            "utilization": utilization,
            "throughput": throughput,
            "network_diameter": self.topology.get_diameter(),
            "average_degree": self.topology.get_average_degree(),
        }

    def get_node_count(self) -> int:
        """获取节点数量"""
        return self.mesh_config.num_nodes

    def get_topology_info(self) -> Dict[str, Any]:
        """获取拓扑信息"""
        return self.topology.get_topology_info()

    def get_router_statistics(self, router_id: NodeId) -> Dict[str, Any]:
        """获取路由器统计信息"""
        if router_id not in self.routers:
            return {}

        router = self.routers[router_id]

        return {
            "router_id": router_id,
            "packets_routed": router.packets_routed,
            "flits_routed": router.flits_routed,
            "input_buffer_occupancy": {direction: len(buffer) for direction, buffer in router.input_buffers.items()},
            "output_buffer_occupancy": {direction: len(buffer) for direction, buffer in router.output_buffers.items()},
            "position": self.topology.get_node_position(router_id),
            "neighbors": self.topology.get_neighbors(router_id),
            "degree": self.topology.get_node_degree(router_id),
        }

    def get_all_router_statistics(self) -> Dict[NodeId, Dict[str, Any]]:
        """获取所有路由器的统计信息"""
        return {router_id: self.get_router_statistics(router_id) for router_id in self.routers}

    def reset_statistics(self) -> None:
        """重置统计信息"""
        self.stats = {
            "total_packets_injected": 0,
            "total_packets_completed": 0,
            "total_flits_injected": 0,
            "total_flits_completed": 0,
            "total_latency": 0,
            "total_hops": 0,
            "cycle_counts": defaultdict(int),
        }

        # 重置路由器统计
        for router in self.routers.values():
            router.packets_routed = 0
            router.flits_routed = 0
            router.buffer_occupancy.clear()

    def is_simulation_complete(self) -> bool:
        """检查仿真是否完成（所有包都已传输完成）"""
        return len(self.active_packets) == 0

    # ========== 实现BaseNoCModel的抽象方法 ==========

    def _setup_topology_network(self) -> None:
        """设置拓扑网络（实现抽象方法）"""
        # 在__init__中已经设置了拓扑和路由器
        pass

    def _step_topology_network(self) -> None:
        """拓扑网络步进（实现抽象方法）"""
        # advance_cycle已经处理了网络步进
        pass

    def _get_topology_info(self) -> Dict[str, Any]:
        """获取拓扑信息（实现抽象方法）"""
        return self.get_topology_info()

    def _calculate_path(self, source: NodeId, destination: NodeId) -> List[NodeId]:
        """计算路径（实现抽象方法）"""
        return self.topology.get_route(source, destination)

    def get_simulation_summary(self) -> Dict[str, Any]:
        """获取仿真总结"""
        stats = self.get_network_statistics()
        topo_info = self.get_topology_info()

        return {
            "simulation_info": {
                "model_type": "Mesh",
                "topology": f"{self.mesh_config.rows}x{self.mesh_config.cols}",
                "total_cycles": self.current_cycle,
                "routing_strategy": self.topology.routing_strategy,
            },
            "network_statistics": stats,
            "topology_info": topo_info,
            "configuration": self.mesh_config.get_topology_params(),
        }
