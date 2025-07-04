"""
RingNetwork class for Ring topology NoC simulation.
Extends the base Network class with Ring-specific functionality.
"""

from __future__ import annotations
import numpy as np
from collections import deque, defaultdict
from config.config import CrossRingConfig
from .flit import Flit
from .network import Network
import logging


class RingNetwork(Network):
    """
    Ring网络的Mixin类，为Network添加Ring特有的功能
    """

    def __init__(self, config, adjacency_matrix, name="RingNetwork"):
        super().__init__(config, adjacency_matrix, name)
        # 初始化环相关的数据结构
        self._init_ring_structure()
        # ---- UE 计数（方向 × 等级） ---------------------------------
        cfg = self.config
        self.ue_used = {}
        self.ue_cap = {}
        for node in range(self.config.RING_NUM_NODE):
            self.ue_used[node] = {("TL", "T0"): 0, ("TL", "T1"): 0, ("TL", "T2"): 0, ("TR", "T0"): 0, ("TR", "T1"): 0, ("TR", "T2"): 0}

            self.ue_cap[node] = {
                ("TL", "T0"): cfg.EQ_IN_FIFO_DEPTH - cfg.TL_Etag_T1_UE_MAX,
                ("TL", "T1"): cfg.TL_Etag_T1_UE_MAX - cfg.TL_Etag_T2_UE_MAX,
                ("TL", "T2"): cfg.TL_Etag_T2_UE_MAX,
                ("TR", "T0"): cfg.EQ_IN_FIFO_DEPTH - cfg.TL_Etag_T1_UE_MAX,
                ("TR", "T1"): cfg.TL_Etag_T1_UE_MAX - cfg.TL_Etag_T2_UE_MAX,
                ("TR", "T2"): cfg.TL_Etag_T2_UE_MAX,
            }
            self.itag_req_counter["TR"][node] = 0
            self.itag_req_counter["TL"][node] = 0
            self.tagged_counter["TR"][node] = 0
            self.tagged_counter["TL"][node] = 0
            self.remain_tag["TR"][node] = 0
            self.remain_tag["TL"][node] = 0

    def update_excess_ITag(self):
        """
        Ring topology specific ITag management.
        """
        # Ensure each direction has a count entry for every node
        for direction, node_dict in self.excess_ITag_to_remove.items():
            for node_id in range(self.num_nodes):
                node_dict.setdefault(node_id, 0)

        # Ring-specific ITag management
        for direction in ["TL", "TR"]:
            for node_id in range(self.num_nodes):
                if self.excess_ITag_to_remove[direction][node_id] > 0:
                    # Find ITags created by this node and release them
                    for link, tag_info in self.links_tag.items():
                        if tag_info[0] is not None and tag_info[0] == [node_id, direction] and link[0] == node_id:
                            # Release excess ITag
                            self.links_tag[link][0] = None
                            self.tagged_counter[direction][node_id] -= 1
                            self.remain_tag[direction][node_id] += 1
                            self.excess_ITag_to_remove[direction][node_id] -= 1
                            break  # Only release one at a time

    def can_move_to_next(self, flit, current, next_node):
        """Ring-specific movement validation with ITag management"""
        # Handle ring wraparound for direction calculation
        if next_node == (current + 1) % self.num_nodes:
            direction = "TR"  # Clockwise
        elif next_node == (current - 1) % self.num_nodes:
            direction = "TL"  # Counter-clockwise
        else:
            # Invalid next_node for ring topology
            return False

        link = (current, next_node)

        # Ring ITag processing
        if self.links[link][0] is not None:  # Link occupied
            # 增加等待周期统计 - flit被阻塞在链路入口
            flit.wait_cycle_h += 1

            # Check if ITag should be created (inline all check logic)
            if (
                self.links_tag[link][0] is None
                and flit.wait_cycle_h > self.config.ITag_TRIGGER_Th_H
                and self.tagged_counter[direction][current] < self.config.ITag_MAX_NUM_H
                and self.itag_req_counter[direction][current] > 0
                and self.remain_tag[direction][current] > 0
            ):

                # Create ITag mark (inline logic)
                self.remain_tag[direction][current] -= 1
                self.tagged_counter[direction][current] += 1
                self.links_tag[link][0] = [current, direction]
                flit.itag_h = True
            return False

        else:  # Link free
            if self.links_tag[link][0] is None:  # No reservation
                return True  # Direct entry to ring
            else:  # Has reservation
                if self.links_tag[link][0] == [current, direction]:  # Own reservation
                    # Use reservation (inline logic)
                    self.links_tag[link][0] = None
                    self.remain_tag[direction][current] += 1
                    self.tagged_counter[direction][current] -= 1
                    return True
        return False

    def _init_ring_structure(self):
        """初始化Ring特有的网络结构"""
        self.ring_mode = True
        self.num_nodes = self.config.RING_NUM_NODE

        # 为画图功能添加ring_nodes属性
        self.ring_nodes = list(range(self.num_nodes))

        # Ring特有的注入/弹出队列结构
        # 保持与CrossRing兼容的接口，但简化为双向结构
        self._init_ring_inject_queues()
        self._init_ring_eject_queues()
        self._init_ring_links()

        for ip_type in self.config.CH_NAME_LIST:
            # Pre-buffer: start empty for each node
            self.IQ_channel_buffer_pre.setdefault(ip_type, {})
            for node_id in range(self.num_nodes):
                self.IQ_channel_buffer_pre[ip_type][node_id] = None

            # Channel buffer: deque with configured depth
            self.IQ_channel_buffer.setdefault(ip_type, {})
            for node_id in range(self.num_nodes):
                self.IQ_channel_buffer[ip_type][node_id] = deque(maxlen=self.config.IQ_CH_FIFO_DEPTH)

        # Ring特有的统计信息
        self.ring_stats = {"cw_flits": 0, "ccw_flits": 0, "local_flits": 0, "etag_upgrades": 0, "itag_reservations": 0}

        # 路由策略（如果设置了）
        self.routing_strategy = getattr(self, "routing_strategy", None)

        logging.info(f"Ring network initialized: {self.num_nodes} nodes, name: {self.name}")

    def _init_ring_inject_queues(self):
        """初始化Ring注入队列"""
        # 为Ring拓扑重新组织注入队列
        # 使用TL(左/逆时针)和TR(右/顺时针)来表示Ring的两个方向

        for node_id in range(self.num_nodes):
            # 每个节点有两个方向的注入队列
            if node_id not in self.inject_queues["TL"]:
                self.inject_queues["TL"][node_id] = deque(maxlen=self.config.IQ_OUT_FIFO_DEPTH)
            if node_id not in self.inject_queues["TR"]:
                self.inject_queues["TR"][node_id] = deque(maxlen=self.config.IQ_OUT_FIFO_DEPTH)

            # Pre缓冲区
            if node_id not in self.inject_queues_pre["TL"]:
                self.inject_queues_pre["TL"][node_id] = None
            if node_id not in self.inject_queues_pre["TR"]:
                self.inject_queues_pre["TR"][node_id] = None

    def _init_ring_eject_queues(self):
        """初始化Ring弹出队列"""
        # Ring拓扑中，每个节点可以从两个方向弹出flit
        for direction in ["TL", "TR"]:
            # Ensure direction exists
            if direction not in self.eject_queues:
                self.eject_queues[direction] = {}
            if direction not in self.eject_queues_in_pre:
                self.eject_queues_in_pre[direction] = {}
            for node_id in range(self.num_nodes):
                # Main eject FIFO per node and direction
                self.eject_queues[direction][node_id] = deque(maxlen=self.config.EQ_IN_FIFO_DEPTH)
                # Pre-buffer for eject
                self.eject_queues_in_pre[direction][node_id] = None

    def _init_ring_links(self):
        """初始化Ring链路"""
        # Ring拓扑的链路结构：每个节点连接到相邻的两个节点
        for i in range(self.num_nodes):
            # 顺时针链路 (向右)
            next_node = (i + 1) % self.num_nodes
            link_key = (i, next_node)
            if link_key not in self.links:
                self.links[link_key] = [None] * self.config.SLICE_PER_LINK
                self.links_tag[link_key] = [None] * self.config.SLICE_PER_LINK
                self.links_flow_stat["read"][link_key] = 0
                self.links_flow_stat["write"][link_key] = 0

            # 逆时针链路 (向左)
            prev_node = (i - 1) % self.num_nodes
            link_key = (i, prev_node)
            if link_key not in self.links:
                self.links[link_key] = [None] * self.config.SLICE_PER_LINK
                self.links_tag[link_key] = [None] * self.config.SLICE_PER_LINK
                self.links_flow_stat["read"][link_key] = 0
                self.links_flow_stat["write"][link_key] = 0

    def plan_move(self, flit: Flit, cycle):
        """
        RingNetwork 专用 flit 调度：
        1) 在链路内部 seat++；
        2) 到达目标 EQ 节点时，按 ETag 优先级尝试三档 FIFO；
            · 先试自身等级
            · 该级满则降档尝试
            · 三档全满 → 立即"升级" (T2→T1→T0) 并继续沿环
        3) 未到 EQ 节点或占位失败，继续沿环前进
        """
        self.cycle = cycle
        if flit.is_new_on_network:
            current = flit.path[flit.path_index]
            next_node = flit.path[flit.path_index + 1]
            flit.current_position = current
            flit.is_new_on_network = False
            flit.flit_position = "Link"
            flit.is_arrive = False
            flit.is_on_station = False
            flit.current_link = (current, next_node)
            if flit.source == flit.destination:
                flit.is_arrive = True
            else:
                flit.current_seat_index = 0
            return

        # ----------------- 内部小工具 -----------------
        def fifo_has_space(curr_node, direction: str, level: str, flit) -> bool:
            """检查 dir×level FIFO 是否还有空位"""
            if level == "T0":
                return self.T0_Etag_Order_FIFO[0] == (curr_node, flit) and self.ue_used[curr_node][(direction, level)] < self.ue_cap[curr_node][(direction, level)]
            return self.ue_used[curr_node][(direction, level)] < self.ue_cap[curr_node][(direction, level)]

        def occupy_fifo(curr_node, direction: str, level: str) -> None:
            """占用一个 entry；计数器 +1"""
            self.ue_used[curr_node][(direction, level)] += 1

        # ----------------- 1. 链路内部推进 -----------------
        curr_link = self.links[flit.current_link]
        if flit.current_seat_index < len(curr_link) - 1:
            curr_link[flit.current_seat_index] = None
            flit.current_seat_index += 1
            return

        # ----------------- 2. 到达 EQ 目标节点 -----------------
        curr_node = flit.current_position = flit.current_link[1]
        if curr_node == flit.destination:
            dir_type = "TL" if (flit.current_link[0] - flit.current_link[1]) % self.config.RING_NUM_NODE == 1 else "TR"  # "TL" or "TR"
            priority_order = ["T0", "T1", "T2"]  # 高 → 低
            start_idx = priority_order.index(flit.ETag_priority)
            # 2-A. 依 flit 当前优先级向低档探测
            if flit.ETag_priority in ["T1", "T0"]:
                flit.circuits_completed_h += 1
            for p in priority_order[start_idx:]:
                if fifo_has_space(curr_node, dir_type, p, flit):
                    occupy_fifo(curr_node, dir_type, p)  # 占位
                    curr_link[flit.current_seat_index] = None
                    flit.used_entry_level = p

                    # 处理T0优先级的特殊逻辑（与Network类保持一致）
                    if flit.ETag_priority == "T0":
                        # T0 flit使用entry时需要从T0队列中移除
                        if hasattr(self, "T0_Etag_Order_FIFO"):
                            try:
                                self.T0_Etag_Order_FIFO.remove((curr_node, flit))
                            except ValueError:
                                pass  # flit不在队列中，忽略

                    flit.is_arrive = True
                    return

            # 2-B. 三档都满：升级ETag优先级（与Network类保持一致）
            if flit.ETag_priority == "T2":
                flit.ETag_priority = "T1"
            elif flit.ETag_priority == "T1":
                flit.ETag_priority = "T0"
                # T1升级到T0时，需要添加到T0队列中（与Network类保持一致）
                if hasattr(self, "T0_Etag_Order_FIFO"):
                    self.T0_Etag_Order_FIFO.append((curr_node, flit))
            # (若已是 T0 则保持不变)

            # 2-C. 升级后继续沿环前进
            self._move_forward_on_ring(flit)
            return

        # ----------------- 3. 非 EQ 节点：正常环向转发 -----------------
        self._move_forward_on_ring(flit)

    def _move_forward_on_ring(self, flit: Flit):
        """
        规划 flit 沿环前进一步：
            • 若下一条环链 seat0 空 → 把 flit.next_link / next_seat_index 设为 (next_link, 0)
            • 否则保持当前位置（即延迟一个 cycle）
        """
        # 计算下一节点（顺/逆时针）
        direction = "TL" if (flit.current_link[0] - flit.current_link[1] + self.config.RING_NUM_NODE) % self.config.RING_NUM_NODE == 1 else "TR"
        if direction == "TR":  # 顺时针
            nxt_node = (flit.current_position + 1) % self.config.RING_NUM_NODE
        else:  # "TL" 逆时针
            nxt_node = (flit.current_position - 1) % self.config.RING_NUM_NODE
        curr_link = self.links[flit.current_link]
        curr_link[flit.current_seat_index] = None
        next_link = (flit.current_position, nxt_node)

        flit.current_link = next_link
        flit.current_seat_index = 0

    def execute_moves(self, flit, cycle):
        """
        Override execute_moves for Ring topology:
        - Move flit along link slices using base logic.
        - When finishing last slice of final hop (based on path_index), eject flit.
        """
        if not flit.is_arrive:
            current, next_node = flit.current_link
            link = self.links.get(flit.current_link)
            self.set_link_slice(flit.current_link, flit.current_seat_index, flit, cycle)

            if (flit.current_seat_index == len(link) - 2 and len(link) > 2) or (flit.current_seat_index == 1 and len(link) == 2):
                self.links_flow_stat[flit.req_type][flit.current_link] += 1

            return False
        else:
            if flit.current_link is not None:
                current, next_node = flit.current_link
            flit.arrival_network_cycle = cycle
            queue_pre = None
            if (flit.current_link[0] - flit.current_link[1]) % self.config.RING_NUM_NODE == 1:
                queue = self.eject_queues["TL"]
                queue_pre = self.eject_queues_in_pre["TL"]
            else:
                queue = self.eject_queues["TR"]
                queue_pre = self.eject_queues_in_pre["TR"]

            # flit.flit_position = f"EQ_{direction}"
            if queue_pre[next_node]:
                return False
            else:
                queue_pre[next_node] = flit
                flit.itag_v = False
                return True
