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
from src.noc.base.ip_interface import PipelinedFIFO, FlowControlledTransfer
from .flit import CrossRingFlit
from .config import CrossRingConfig, RoutingStrategy


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

        # 注入/提取队列 - 保持命名但使用PipelinedFIFO
        self.inject_queues = {
            "req": PipelinedFIFO(f"inject_req_{node_id}", depth=config.inject_buffer_depth),
            "rsp": PipelinedFIFO(f"inject_rsp_{node_id}", depth=config.inject_buffer_depth),
            "data": PipelinedFIFO(f"inject_data_{node_id}", depth=config.inject_buffer_depth),
        }
        self.eject_queues = {
            "req": PipelinedFIFO(f"eject_req_{node_id}", depth=config.eject_buffer_depth),
            "rsp": PipelinedFIFO(f"eject_rsp_{node_id}", depth=config.eject_buffer_depth),
            "data": PipelinedFIFO(f"eject_data_{node_id}", depth=config.eject_buffer_depth),
        }

        # 环形缓冲区 - 保持命名但使用PipelinedFIFO
        self.ring_buffers = {
            "horizontal": {
                "req": {
                    "TR": PipelinedFIFO(f"ring_h_req_TR_{node_id}", depth=config.ring_buffer_depth),
                    "TL": PipelinedFIFO(f"ring_h_req_TL_{node_id}", depth=config.ring_buffer_depth),
                    "TD": PipelinedFIFO(f"ring_h_req_TD_{node_id}", depth=config.ring_buffer_depth),
                    "TU": PipelinedFIFO(f"ring_h_req_TU_{node_id}", depth=config.ring_buffer_depth),
                },
                "rsp": {
                    "TR": PipelinedFIFO(f"ring_h_rsp_TR_{node_id}", depth=config.ring_buffer_depth),
                    "TL": PipelinedFIFO(f"ring_h_rsp_TL_{node_id}", depth=config.ring_buffer_depth),
                    "TD": PipelinedFIFO(f"ring_h_rsp_TD_{node_id}", depth=config.ring_buffer_depth),
                    "TU": PipelinedFIFO(f"ring_h_rsp_TU_{node_id}", depth=config.ring_buffer_depth),
                },
                "data": {
                    "TR": PipelinedFIFO(f"ring_h_data_TR_{node_id}", depth=config.ring_buffer_depth),
                    "TL": PipelinedFIFO(f"ring_h_data_TL_{node_id}", depth=config.ring_buffer_depth),
                    "TD": PipelinedFIFO(f"ring_h_data_TD_{node_id}", depth=config.ring_buffer_depth),
                    "TU": PipelinedFIFO(f"ring_h_data_TU_{node_id}", depth=config.ring_buffer_depth),
                },
            },
            "vertical": {
                "req": {
                    "TR": PipelinedFIFO(f"ring_v_req_TR_{node_id}", depth=config.ring_buffer_depth),
                    "TL": PipelinedFIFO(f"ring_v_req_TL_{node_id}", depth=config.ring_buffer_depth),
                    "TD": PipelinedFIFO(f"ring_v_req_TD_{node_id}", depth=config.ring_buffer_depth),
                    "TU": PipelinedFIFO(f"ring_v_req_TU_{node_id}", depth=config.ring_buffer_depth),
                },
                "rsp": {
                    "TR": PipelinedFIFO(f"ring_v_rsp_TR_{node_id}", depth=config.ring_buffer_depth),
                    "TL": PipelinedFIFO(f"ring_v_rsp_TL_{node_id}", depth=config.ring_buffer_depth),
                    "TD": PipelinedFIFO(f"ring_v_rsp_TD_{node_id}", depth=config.ring_buffer_depth),
                    "TU": PipelinedFIFO(f"ring_v_rsp_TU_{node_id}", depth=config.ring_buffer_depth),
                },
                "data": {
                    "TR": PipelinedFIFO(f"ring_v_data_TR_{node_id}", depth=config.ring_buffer_depth),
                    "TL": PipelinedFIFO(f"ring_v_data_TL_{node_id}", depth=config.ring_buffer_depth),
                    "TD": PipelinedFIFO(f"ring_v_data_TD_{node_id}", depth=config.ring_buffer_depth),
                    "TU": PipelinedFIFO(f"ring_v_data_TU_{node_id}", depth=config.ring_buffer_depth),
                },
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

    def set_routing_strategy_bias(self, routing_strategy: RoutingStrategy) -> None:
        """
        根据路由策略设置仲裁偏向

        Args:
            routing_strategy: 路由策略
        """
        if routing_strategy == RoutingStrategy.XY:
            # XY路由：稍微偏向水平方向
            self.routing_bias = {"horizontal": 1.2, "vertical": 1.0}
        elif routing_strategy == RoutingStrategy.YX:
            # YX路由：稍微偏向垂直方向
            self.routing_bias = {"horizontal": 1.0, "vertical": 1.2}
        else:
            # 其他策略：均衡
            self.routing_bias = {"horizontal": 1.0, "vertical": 1.0}

        self.logger.debug(f"节点{self.node_id}设置路由偏向: {routing_strategy.value} -> {self.routing_bias}")

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
        eject_fifo = self.eject_queues[channel]
        if not eject_fifo.valid_signal():
            return None

        return eject_fifo.read_output()

    def add_to_inject_queue(self, flit: CrossRingFlit, channel: str) -> bool:
        """
        添加flit到inject队列（保持兼容性接口）

        Args:
            flit: 要添加的flit
            channel: 通道类型 ("req", "rsp", "data")

        Returns:
            是否成功添加
        """
        inject_fifo = self.inject_queues[channel]
        if not inject_fifo.ready_signal():
            return False

        return inject_fifo.write_input(flit)

    def _compute_inject_to_ring_transfers(self, cycle: int) -> None:
        """计算从inject队列到ring缓冲区的传输可能性"""
        for direction in ["horizontal", "vertical"]:
            for channel in ["req", "rsp", "data"]:
                # 检查是否有inject队列中的flit可以传输
                inject_fifo = self.inject_queues[channel]

                # 检查仲裁状态
                if not self.can_inject_flit(channel, direction):
                    self._transfer_decisions["inject_to_ring"][direction][channel] = None
                    continue

                # 找到最合适的方向代码
                best_dir_code = self._find_best_direction_code(direction, channel)
                if best_dir_code:
                    ring_fifo = self.ring_buffers[direction][channel][best_dir_code]
                    if FlowControlledTransfer.can_transfer(inject_fifo, ring_fifo):
                        self._transfer_decisions["inject_to_ring"][direction][channel] = best_dir_code
                    else:
                        self._transfer_decisions["inject_to_ring"][direction][channel] = None
                else:
                    self._transfer_decisions["inject_to_ring"][direction][channel] = None

    def _compute_ring_to_ring_transfers(self, cycle: int) -> None:
        """计算环形缓冲区之间的传输可能性"""
        for direction in ["horizontal", "vertical"]:
            for channel in ["req", "rsp", "data"]:
                self._transfer_decisions["ring_to_ring"][direction][channel] = {}

                for dir_code in ["TR", "TL", "TD", "TU"]:
                    if self.can_transfer_flit(direction, dir_code, channel):
                        # 计算目标方向和节点
                        target_direction, target_dir_code = self._get_ring_transfer_target(direction, dir_code)
                        if target_direction and target_dir_code:
                            source_fifo = self.ring_buffers[direction][channel][dir_code]
                            # 这里假设目标是相邻节点，实际实现中需要获取相邻节点的FIFO
                            if source_fifo.valid_signal():
                                self._transfer_decisions["ring_to_ring"][direction][channel][dir_code] = (target_direction, target_dir_code)
                            else:
                                self._transfer_decisions["ring_to_ring"][direction][channel][dir_code] = None
                        else:
                            self._transfer_decisions["ring_to_ring"][direction][channel][dir_code] = None
                    else:
                        self._transfer_decisions["ring_to_ring"][direction][channel][dir_code] = None

    def _compute_ring_to_eject_transfers(self, cycle: int) -> None:
        """计算从ring缓冲区到eject队列的传输可能性"""
        for channel in ["req", "rsp", "data"]:
            # 检查是否有到达本节点的flit
            eject_fifo = self.eject_queues[channel]
            found_flit = False

            for direction in ["horizontal", "vertical"]:
                for dir_code in ["TR", "TL", "TD", "TU"]:
                    ring_fifo = self.ring_buffers[direction][channel][dir_code]
                    if ring_fifo.valid_signal():
                        flit = ring_fifo.peek_output()
                        if flit and self._should_eject_flit(flit):
                            if FlowControlledTransfer.can_transfer(ring_fifo, eject_fifo):
                                self._transfer_decisions["ring_to_eject"][channel] = (direction, dir_code)
                                found_flit = True
                                break
                if found_flit:
                    break

            if not found_flit:
                self._transfer_decisions["ring_to_eject"][channel] = None

    def _execute_inject_to_ring_transfers(self, cycle: int) -> None:
        """执行从inject队列到ring缓冲区的传输"""
        for direction in ["horizontal", "vertical"]:
            for channel in ["req", "rsp", "data"]:
                decision = self._transfer_decisions["inject_to_ring"][direction][channel]
                if decision:
                    dir_code = decision
                    inject_fifo = self.inject_queues[channel]
                    ring_fifo = self.ring_buffers[direction][channel][dir_code]

                    if FlowControlledTransfer.try_transfer(inject_fifo, ring_fifo):
                        # 更新传输的flit
                        flit = ring_fifo.peek_output()
                        if flit:
                            flit.network_entry_cycle = cycle
                            self.stats["injected_flits"][channel] += 1

                        # 更新仲裁状态
                        dir_priority_map = {"TR": "ring_tr", "TL": "ring_tl", "TD": "ring_td", "TU": "ring_tu"}
                        self.arbitration_state[f"{direction}_priority"] = dir_priority_map[dir_code]
                        self.arbitration_state["last_arbitration"][direction] = cycle

    def _execute_ring_to_ring_transfers(self, cycle: int) -> None:
        """执行环形缓冲区之间的传输"""
        for direction in ["horizontal", "vertical"]:
            for channel in ["req", "rsp", "data"]:
                decisions = self._transfer_decisions["ring_to_ring"][direction][channel]

                for dir_code, decision in decisions.items():
                    if decision:
                        target_direction, target_dir_code = decision
                        source_fifo = self.ring_buffers[direction][channel][dir_code]

                        # 执行读取操作
                        flit = source_fifo.read_output()
                        if flit:
                            self.stats["transferred_flits"][direction] += 1

                            # 更新仲裁状态
                            priority_map = {"ring_tr": "ring_tl", "ring_tl": "inject", "ring_td": "ring_tu", "ring_tu": "inject"}
                            dir_priority_map = {"TR": "ring_tr", "TL": "ring_tl", "TD": "ring_td", "TU": "ring_tu"}
                            current_priority = dir_priority_map[dir_code]
                            self.arbitration_state[f"{direction}_priority"] = priority_map.get(current_priority, "inject")
                            self.arbitration_state["last_arbitration"][direction] = cycle

    def _execute_ring_to_eject_transfers(self, cycle: int) -> None:
        """执行从ring缓冲区到eject队列的传输"""
        for channel in ["req", "rsp", "data"]:
            decision = self._transfer_decisions["ring_to_eject"][channel]
            if decision:
                direction, dir_code = decision
                ring_fifo = self.ring_buffers[direction][channel][dir_code]
                eject_fifo = self.eject_queues[channel]

                if FlowControlledTransfer.try_transfer(ring_fifo, eject_fifo):
                    # 更新ejected flit的状态
                    flit = eject_fifo.peek_output()
                    if flit:
                        flit.is_arrive = True
                        flit.arrival_network_cycle = cycle
                        self.stats["ejected_flits"][channel] += 1

    def _find_best_direction_code(self, direction: str, channel: str) -> Optional[str]:
        """找到最合适的方向代码进行inject"""
        # 简化实现：选择第一个可用的方向
        for dir_code in ["TR", "TL", "TD", "TU"]:
            ring_fifo = self.ring_buffers[direction][channel][dir_code]
            if ring_fifo.ready_signal():
                return dir_code
        return None

    def _get_ring_transfer_target(self, direction: str, dir_code: str) -> Tuple[Optional[str], Optional[str]]:
        """获取环形传输的目标方向和代码"""
        # 简化实现：返回相同的方向和代码
        # 实际实现需要根据拓扑结构计算
        return direction, dir_code

    def _should_eject_flit(self, flit: CrossRingFlit) -> bool:
        """检查是否应该弹出flit"""
        # 检查flit是否到达目标节点
        if hasattr(flit, "destination") and flit.destination == self.node_id:
            return True
        if hasattr(flit, "dest_node_id") and flit.dest_node_id == self.node_id:
            return True
        return False

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
