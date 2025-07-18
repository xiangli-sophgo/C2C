"""
CrossRing节点重构版本 - 组件化设计。

使用独立的组件类：
- InjectQueue: 注入队列管理 (from components.inject_queue)
- EjectQueue: 弹出队列管理 (from components.eject_queue)
- RingBridge: 环形桥接管理 (from components.ring_bridge)
- CrossRingNode: 主节点类（协调组件）
"""

from typing import Dict, List, Any, Tuple, Optional
import logging

from src.noc.crossring.flit import CrossRingFlit
from src.noc.crossring.config import CrossRingConfig, RoutingStrategy
from src.noc.crossring.components.cross_point import CrossRingCrossPoint, CrossPointDirection
from src.noc.crossring.components.inject_queue import InjectQueue
from src.noc.crossring.components.eject_queue import EjectQueue
from src.noc.crossring.components.ring_bridge import RingBridge


# 删除内嵌的组件类，使用导入的组件


class CrossRingNode:
    """
    CrossRing节点类（重构版本）。

    简化的组件化设计，主要负责协调各个组件。
    """

    def __init__(self, node_id: int, coordinates: Tuple[int, int], config: CrossRingConfig, logger: logging.Logger, topology=None):
        """初始化CrossRing节点。"""
        self.node_id = node_id
        self.coordinates = coordinates
        self.config = config
        self.logger = logger
        self.topology = topology

        # 初始化组件
        self.inject_queue = InjectQueue(node_id, coordinates, config, logger, topology=topology)
        self.eject_queue = EjectQueue(node_id, coordinates, config, logger)
        self.ring_bridge = RingBridge(node_id, coordinates, config, logger, topology=topology)

        # 初始化CrossPoint实例
        self.horizontal_crosspoint = CrossRingCrossPoint(
            crosspoint_id=f"node_{node_id}_horizontal",
            node_id=node_id,
            direction=CrossPointDirection.HORIZONTAL,
            config=config,
            coordinates=coordinates,
            parent_node=self,
            logger=logger,
        )

        self.vertical_crosspoint = CrossRingCrossPoint(
            crosspoint_id=f"node_{node_id}_vertical",
            node_id=node_id,
            direction=CrossPointDirection.VERTICAL,
            config=config,
            coordinates=coordinates,
            parent_node=self,
            logger=logger,
        )

        # 性能统计
        self.stats = {
            "injected_flits": {"req": 0, "rsp": 0, "data": 0},
            "transferred_flits": {"horizontal": 0, "vertical": 0},
            "congestion_events": 0,
        }

        self.logger.debug(f"CrossRing节点初始化: ID={node_id}, 坐标={coordinates}")

    @property
    def ip_eject_channel_buffers(self):
        """提供对eject队列中IP缓冲区的访问，保持接口兼容性。"""
        return self.eject_queue.ip_eject_channel_buffers

    @property
    def ip_inject_channel_buffers(self):
        """提供对inject队列中IP缓冲区的访问，保持接口兼容性。"""
        return self.inject_queue.ip_inject_channel_buffers

    @property
    def inject_direction_fifos(self):
        """提供对inject方向FIFO的访问，保持接口兼容性。"""
        return self.inject_queue.inject_direction_fifos

    def connect_ip(self, ip_id: str) -> bool:
        """连接IP到当前节点。"""
        inject_success = self.inject_queue.connect_ip(ip_id)
        eject_success = self.eject_queue.connect_ip(ip_id)
        return inject_success and eject_success

    def get_connected_ips(self) -> List[str]:
        """获取连接的IP列表。"""
        return self.inject_queue.connected_ips.copy()

    def add_to_inject_queue(self, flit: CrossRingFlit, channel: str, ip_id: str) -> bool:
        """IP注入flit。"""
        success = self.inject_queue.add_to_inject_queue(flit, channel, ip_id)
        if success:
            self.stats["injected_flits"][channel] += 1
        return success

    def get_eject_flit(self, ip_id: str, channel: str) -> Optional[CrossRingFlit]:
        """IP获取flit。"""
        return self.eject_queue.get_eject_flit(ip_id, channel)

    def get_crosspoint(self, direction: str) -> Optional[CrossRingCrossPoint]:
        """获取CrossPoint。"""
        if direction == "horizontal":
            return self.horizontal_crosspoint
        elif direction == "vertical":
            return self.vertical_crosspoint
        return None

    def add_to_ring_bridge_input(self, flit: CrossRingFlit, direction: str, channel: str) -> bool:
        """添加flit到ring_bridge输入。"""
        return self.ring_bridge.add_to_ring_bridge_input(flit, direction, channel)

    def _calculate_routing_direction(self, flit: CrossRingFlit) -> str:
        """计算路由方向，代理给inject_queue的路由逻辑。"""
        return self.inject_queue._calculate_routing_direction(flit)

    def get_ring_bridge_output_flit(self, direction: str, channel: str) -> Optional[CrossRingFlit]:
        """从ring_bridge输出获取flit。"""
        return self.ring_bridge.get_output_flit(direction, channel)

    def step_compute_phase(self, cycle: int) -> None:
        """
        计算阶段：正确的两阶段执行顺序。
        """
        # 1. 更新所有FIFO的组合逻辑阶段
        self.inject_queue.step_compute_phase(cycle)
        self.eject_queue.step_compute_phase(cycle)
        self.ring_bridge.step_compute_phase(cycle)

        # 2. 计算所有仲裁决策（但不执行）
        self.inject_queue.compute_arbitration(cycle)
        self.eject_queue.compute_arbitration(cycle, self.inject_queue.inject_direction_fifos, self.ring_bridge)
        self.ring_bridge.compute_arbitration(cycle, self.inject_queue.inject_direction_fifos)

        # 3. 处理CrossPoint的计算阶段
        if hasattr(self.horizontal_crosspoint, "step_compute_phase"):
            self.horizontal_crosspoint.step_compute_phase(cycle, self.inject_queue.inject_direction_fifos, self.eject_queue.eject_input_fifos)
        if hasattr(self.vertical_crosspoint, "step_compute_phase"):
            self.vertical_crosspoint.step_compute_phase(cycle, self.inject_queue.inject_direction_fifos, self.eject_queue.eject_input_fifos)

    def step_update_phase(self, cycle: int) -> None:
        """
        更新阶段：正确的两阶段执行顺序。
        """
        # 1. 执行注入仲裁（优先执行）
        self.inject_queue.execute_arbitration(cycle)

        # 2. CrossPoint执行
        if hasattr(self.horizontal_crosspoint, "step_update_phase"):
            self.horizontal_crosspoint.step_update_phase(cycle, self.inject_queue.inject_direction_fifos, self.eject_queue.eject_input_fifos)
        else:
            self.horizontal_crosspoint.step(cycle, self.inject_queue.inject_direction_fifos, self.eject_queue.eject_input_fifos)

        if hasattr(self.vertical_crosspoint, "step_update_phase"):
            self.vertical_crosspoint.step_update_phase(cycle, self.inject_queue.inject_direction_fifos, self.eject_queue.eject_input_fifos)
        else:
            self.vertical_crosspoint.step(cycle, self.inject_queue.inject_direction_fifos, self.eject_queue.eject_input_fifos)

        # 3. 执行ring_bridge仲裁
        self.ring_bridge.execute_arbitration(cycle, self.inject_queue.inject_direction_fifos)

        # 4. 执行eject仲裁
        self.eject_queue.execute_arbitration(cycle, self.inject_queue.inject_direction_fifos, self.ring_bridge)

        # 5. 更新所有FIFO的时序逻辑阶段
        self.inject_queue.step_update_phase()
        self.eject_queue.step_update_phase()
        self.ring_bridge.step_update_phase()

    def get_inject_direction_status(self) -> Dict[str, Any]:
        """获取注入方向队列的状态。"""
        status = {}
        for channel in ["req", "rsp", "data"]:
            status[channel] = {}
            for direction in ["TR", "TL", "TU", "TD", "EQ"]:
                fifo = self.inject_queue.inject_direction_fifos[channel][direction]
                status[channel][direction] = {
                    "occupancy": len(fifo),
                    "ready": fifo.ready_signal(),
                    "valid": fifo.valid_signal(),
                }
        return status

    def get_crosspoint_status(self) -> Dict[str, Any]:
        """获取CrossPoint状态信息。"""
        return {"horizontal": self.horizontal_crosspoint.get_crosspoint_status(), "vertical": self.vertical_crosspoint.get_crosspoint_status()}

    def get_stats(self) -> Dict[str, Any]:
        """获取节点统计信息。"""
        return {
            "node_id": self.node_id,
            "coordinates": self.coordinates,
            "injected_flits": dict(self.stats["injected_flits"]),
            "transferred_flits": dict(self.stats["transferred_flits"]),
            "congestion_events": self.stats["congestion_events"],
            "connected_ips": len(self.get_connected_ips()),
            "inject_direction_status": self.get_inject_direction_status(),
            "crosspoint_status": self.get_crosspoint_status(),
        }

    def __str__(self) -> str:
        return f"CrossRingNode({self.node_id}, {self.coordinates})"

    def __repr__(self) -> str:
        return f"CrossRingNode(id={self.node_id}, pos={self.coordinates}, ips={len(self.get_connected_ips())})"
