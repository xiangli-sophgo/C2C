"""
CrossRing节点实现。

提供CrossRing网络中节点的详细实现，包括：
- 注入/提取队列管理
- 环形缓冲区管理
- 拥塞控制机制
- 仲裁逻辑
"""

from typing import Dict, List, Any, Tuple, Optional
import logging

from src.noc.base.node import BaseNoCNode
from .flit import CrossRingFlit
from .config import CrossRingConfig


class CrossRingNode:
    """
    CrossRing节点类。

    实现CrossRing节点的内部结构和逻辑，包括：
    1. 注入/提取队列管理
    2. 环形缓冲区管理
    3. ETag/ITag拥塞控制
    4. 仲裁逻辑
    """

    def __init__(self, node_id: int, coordinates: Tuple[int, int], config: CrossRingConfig, logger: logging.Logger):
        """
        初始化CrossRing节点

        Args:
            node_id: 节点ID
            coordinates: 节点坐标 (x, y)
            config: CrossRing配置
            logger: 日志记录器
        """
        self.node_id = node_id
        self.coordinates = coordinates
        self.config = config
        self.logger = logger

        # 注入/提取队列
        self.inject_queues = {"req": [], "rsp": [], "data": []}
        self.eject_queues = {"req": [], "rsp": [], "data": []}

        # 环形缓冲区
        self.ring_buffers = {
            "horizontal": {
                "req": {"TR": [], "TL": [], "TD": [], "TU": []},
                "rsp": {"TR": [], "TL": [], "TD": [], "TU": []},
                "data": {"TR": [], "TL": [], "TD": [], "TU": []},
            },
            "vertical": {
                "req": {"TR": [], "TL": [], "TD": [], "TU": []},
                "rsp": {"TR": [], "TL": [], "TD": [], "TU": []},
                "data": {"TR": [], "TL": [], "TD": [], "TU": []},
            },
        }

        # 拥塞控制状态
        self.etag_status = {
            "horizontal": {"req": False, "rsp": False, "data": False},
            "vertical": {"req": False, "rsp": False, "data": False},
        }
        self.itag_status = {
            "horizontal": {"req": False, "rsp": False, "data": False},
            "vertical": {"req": False, "rsp": False, "data": False},
        }

        # 仲裁状态 - 使用更准确的方向优先级
        self.arbitration_state = {
            "horizontal_priority": "inject",  # inject, ring_tr, ring_tl
            "vertical_priority": "inject",  # inject, ring_td, ring_tu
            "last_arbitration": {"horizontal": 0, "vertical": 0},
        }

        # 性能统计
        self.stats = {
            "injected_flits": {"req": 0, "rsp": 0, "data": 0},
            "ejected_flits": {"req": 0, "rsp": 0, "data": 0},
            "transferred_flits": {"horizontal": 0, "vertical": 0},
            "congestion_events": 0,
        }

        self.logger.debug(f"CrossRing节点初始化: ID={node_id}, 坐标={coordinates}")

    def update_state(self, cycle: int) -> None:
        """
        更新节点状态

        Args:
            cycle: 当前周期
        """
        # 更新仲裁状态
        self._update_arbitration_state(cycle)

        # 更新拥塞控制状态
        self._update_congestion_state()

    def _update_arbitration_state(self, cycle: int) -> None:
        """
        更新仲裁状态

        Args:
            cycle: 当前周期
        """
        # 检查是否需要重置仲裁优先级
        for direction in ["horizontal", "vertical"]:
            last_arbitration = self.arbitration_state["last_arbitration"][direction]
            if cycle - last_arbitration > self.config.arbitration_timeout:
                # 重置为默认优先级
                self.arbitration_state[f"{direction}_priority"] = "inject"
                self.logger.debug(f"节点{self.node_id}的{direction}仲裁状态重置为默认")

    def _update_congestion_state(self) -> None:
        """更新拥塞控制状态"""
        # 更新ETag状态
        for direction in ["horizontal", "vertical"]:
            for channel in ["req", "rsp", "data"]:
                # 检查eject队列的拥塞情况
                eject_queue = self.eject_queues[channel]
                eject_threshold = self.config.eject_buffer_depth * 0.8

                # 检查ring缓冲区的拥塞情况
                ring_buffers = self.ring_buffers[direction][channel]
                ring_congestion = False

                for ring_dir in ["TR", "TL", "TD", "TU"]:
                    buffer_occupancy = len(ring_buffers[ring_dir])
                    ring_threshold = self.config.ring_buffer_depth * 0.8
                    if buffer_occupancy >= ring_threshold:
                        ring_congestion = True
                        break

                # 设置ETag状态
                eject_congestion = len(eject_queue) >= eject_threshold
                old_status = self.etag_status[direction][channel]
                new_status = eject_congestion or ring_congestion

                if old_status != new_status:
                    self.etag_status[direction][channel] = new_status
                    if new_status:
                        self.stats["congestion_events"] += 1
                        self.logger.debug(f"节点{self.node_id}的{direction} {channel} ETag状态变为拥塞")
                    else:
                        self.logger.debug(f"节点{self.node_id}的{direction} {channel} ETag状态变为畅通")

    def can_inject_flit(self, channel: str, direction: str) -> bool:
        """
        检查是否可以注入flit

        Args:
            channel: 通道类型 ("req", "rsp", "data")
            direction: 注入方向 ("horizontal", "vertical")

        Returns:
            是否可以注入
        """
        # 检查拥塞状态
        if self.etag_status[direction][channel]:
            return False

        # 检查仲裁状态
        if self.arbitration_state[f"{direction}_priority"] != "inject":
            return False

        return True

    def inject_flit(self, flit: CrossRingFlit, channel: str, direction: str, dir_code: str, cycle: int) -> bool:
        """
        注入flit到环形缓冲区

        Args:
            flit: 要注入的flit
            channel: 通道类型 ("req", "rsp", "data")
            direction: 注入方向 ("horizontal", "vertical")
            dir_code: 具体方向代码 ("TR", "TL", "TD", "TU")
            cycle: 当前周期

        Returns:
            是否成功注入
        """
        # 检查环形缓冲区是否有空间
        ring_buffer = self.ring_buffers[direction][channel][dir_code]
        if len(ring_buffer) >= self.config.ring_buffer_depth:
            return False

        # 注入flit
        ring_buffer.append(flit)
        flit.network_entry_cycle = cycle

        # 更新统计
        self.stats["injected_flits"][channel] += 1

        # 更新仲裁状态
        # 将方向代码映射到仲裁优先级
        dir_priority_map = {"TR": "ring_tr", "TL": "ring_tl", "TD": "ring_td", "TU": "ring_tu"}
        self.arbitration_state[f"{direction}_priority"] = dir_priority_map[dir_code]
        self.arbitration_state["last_arbitration"][direction] = cycle

        return True

    def can_transfer_flit(self, direction: str, dir_code: str, channel: str) -> bool:
        """
        检查是否可以传输flit

        Args:
            direction: 传输方向 ("horizontal", "vertical")
            dir_code: 具体方向代码 ("TR", "TL", "TD", "TU")
            channel: 通道类型 ("req", "rsp", "data")

        Returns:
            是否可以传输
        """
        # 检查环形缓冲区是否有flit
        ring_buffer = self.ring_buffers[direction][channel][dir_code]
        if not ring_buffer:
            return False

        # 检查仲裁状态
        # 将方向代码映射到仲裁优先级
        priority_map = {"TR": "ring_tr", "TL": "ring_tl", "TD": "ring_td", "TU": "ring_tu"}
        requester = priority_map[dir_code]

        if self.arbitration_state[f"{direction}_priority"] != requester:
            return False

        return True

    def transfer_flit(self, direction: str, dir_code: str, channel: str, cycle: int) -> Optional[CrossRingFlit]:
        """
        从环形缓冲区传输flit

        Args:
            direction: 传输方向 ("horizontal", "vertical")
            dir_code: 具体方向代码 ("TR", "TL", "TD", "TU")
            channel: 通道类型 ("req", "rsp", "data")
            cycle: 当前周期

        Returns:
            传输的flit，如果没有则返回None
        """
        # 检查环形缓冲区是否有flit
        ring_buffer = self.ring_buffers[direction][channel][dir_code]
        if not ring_buffer:
            return None

        # 传输flit
        flit = ring_buffer.pop(0)

        # 更新统计
        self.stats["transferred_flits"][direction] += 1

        # 更新仲裁状态 - 轮转优先级
        priority_map = {"ring_tr": "ring_tl", "ring_tl": "inject", "ring_td": "ring_tu", "ring_tu": "inject"}

        # 将方向代码映射到仲裁优先级
        dir_priority_map = {"TR": "ring_tr", "TL": "ring_tl", "TD": "ring_td", "TU": "ring_tu"}
        current_priority = dir_priority_map[dir_code]

        # 更新为下一个优先级
        self.arbitration_state[f"{direction}_priority"] = priority_map.get(current_priority, "inject")
        self.arbitration_state["last_arbitration"][direction] = cycle

        return flit

    def receive_flit(self, flit: CrossRingFlit, direction: str, dir_code: str, channel: str) -> bool:
        """
        接收flit到环形缓冲区

        Args:
            flit: 要接收的flit
            direction: 接收方向 ("horizontal", "vertical")
            dir_code: 具体方向代码 ("TR", "TL", "TD", "TU")
            channel: 通道类型 ("req", "rsp", "data")

        Returns:
            是否成功接收
        """
        # 检查环形缓冲区是否有空间
        ring_buffer = self.ring_buffers[direction][channel][dir_code]
        if len(ring_buffer) >= self.config.ring_buffer_depth:
            return False

        # 接收flit
        ring_buffer.append(flit)

        return True

    def eject_flit(self, flit: CrossRingFlit, channel: str, cycle: int) -> bool:
        """
        将flit弹出到eject队列

        Args:
            flit: 要弹出的flit
            channel: 通道类型 ("req", "rsp", "data")
            cycle: 当前周期

        Returns:
            是否成功弹出
        """
        # 检查eject队列是否有空间
        if len(self.eject_queues[channel]) >= self.config.eject_buffer_depth:
            return False

        # 弹出flit
        self.eject_queues[channel].append(flit)
        flit.is_arrive = True
        flit.arrival_network_cycle = cycle

        # 更新统计
        self.stats["ejected_flits"][channel] += 1

        return True

    def get_eject_flit(self, channel: str) -> Optional[CrossRingFlit]:
        """
        从eject队列获取flit

        Args:
            channel: 通道类型 ("req", "rsp", "data")

        Returns:
            获取的flit，如果没有则返回None
        """
        if not self.eject_queues[channel]:
            return None

        return self.eject_queues[channel].pop(0)

    def add_to_inject_queue(self, flit: CrossRingFlit, channel: str) -> bool:
        """
        添加flit到inject队列

        Args:
            flit: 要添加的flit
            channel: 通道类型 ("req", "rsp", "data")

        Returns:
            是否成功添加
        """
        if len(self.inject_queues[channel]) >= self.config.inject_buffer_depth:
            return False

        self.inject_queues[channel].append(flit)
        return True

    def get_stats(self) -> Dict[str, Any]:
        """
        获取节点统计信息

        Returns:
            统计信息字典
        """
        return {
            "node_id": self.node_id,
            "coordinates": self.coordinates,
            "injected_flits": dict(self.stats["injected_flits"]),
            "ejected_flits": dict(self.stats["ejected_flits"]),
            "transferred_flits": dict(self.stats["transferred_flits"]),
            "congestion_events": self.stats["congestion_events"],
            "buffer_occupancy": {
                "inject": {k: len(v) for k, v in self.inject_queues.items()},
                "eject": {k: len(v) for k, v in self.eject_queues.items()},
                "horizontal": {k: {d: len(v) for d, v in vv.items()} for k, vv in self.ring_buffers["horizontal"].items()},
                "vertical": {k: {d: len(v) for d, v in vv.items()} for k, vv in self.ring_buffers["vertical"].items()},
            },
            "congestion_status": {
                "etag": {
                    "horizontal": dict(self.etag_status["horizontal"]),
                    "vertical": dict(self.etag_status["vertical"]),
                },
                "itag": {
                    "horizontal": dict(self.itag_status["horizontal"]),
                    "vertical": dict(self.itag_status["vertical"]),
                },
            },
        }
