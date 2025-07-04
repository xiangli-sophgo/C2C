"""
Network class for NoC simulation.
Contains the core network implementation with routing and flow control mechanisms.
"""
from __future__ import annotations
import numpy as np
from collections import deque, defaultdict
from config.config import CrossRingConfig
from .flit import Flit, TokenBucket
import logging
import inspect


class Network:
    def __init__(self, config: CrossRingConfig, adjacency_matrix, name="network"):
        self.config = config
        self.name = name
        
        # Pre-calculate frequently used position sets for performance
        self._all_ip_positions = None
        self._rn_positions = None
        self._sn_positions = None
        self._positions_cache_lock = None  # For thread safety if needed
        self.current_cycle = []
        self.flits_num = []
        self.schedules = {"sdma": None}
        self.inject_num = 0
        self.eject_num = 0
        self.inject_queues = {"TL": {}, "TR": {}, "TU": {}, "TD": {}, "EQ": {}}
        self.inject_queues_pre = {"TL": {}, "TR": {}, "TU": {}, "TD": {}, "EQ": {}}
        self.eject_queues = {"TU": {}, "TD": {}}
        self.eject_queues_in_pre = {"TU": {}, "TD": {}}
        self.arrive_node_pre = self.config._make_channels(("sdma", "gdma", "cdma", "ddr", "l2m"))
        self.IQ_channel_buffer = self.config._make_channels(("sdma", "gdma", "cdma", "ddr", "l2m"))
        self.EQ_channel_buffer = self.config._make_channels(("sdma", "gdma", "cdma", "ddr", "l2m"))
        self.IQ_channel_buffer_pre = self.config._make_channels(("sdma", "gdma", "cdma", "ddr", "l2m"))
        self.EQ_channel_buffer_pre = self.config._make_channels(("sdma", "gdma", "cdma", "ddr", "l2m"))
        self.links = {}
        self.cross_point = {"horizontal": defaultdict(lambda: defaultdict(list)), "vertical": defaultdict(lambda: defaultdict(list))}
        self.links_flow_stat = {"read": {}, "write": {}}
        # ITag setup
        self.links_tag = {}
        self.remain_tag = {"TL": {}, "TR": {}, "TU": {}, "TD": {}}
        self.tagged_counter = {"TL": {}, "TR": {}, "TU": {}, "TD": {}}  # 环上已标记ITag数
        self.itag_req_counter = {"TL": {}, "TR": {}, "TU": {}, "TD": {}}  # FIFO中ITag需求数
        self.excess_ITag_to_remove = {"TL": {}, "TR": {}, "TD": {}, "TU": {}}

        # 每个FIFO Entry的等待计数器
        self.fifo_counters = {"TL": {}, "TR": {}}
        self.ring_bridge = {"TL": {}, "TR": {}, "TU": {}, "TD": {}, "EQ": {}}
        self.ring_bridge_pre = {"TL": {}, "TR": {}, "TU": {}, "TD": {}, "EQ": {}}
        self.round_robin = {"IQ": defaultdict(lambda: defaultdict(dict)), "RB": defaultdict(lambda: defaultdict(dict)), "EQ": defaultdict(lambda: defaultdict(dict))}
        self.round_robin_counter = 0

        self.recv_flits_num = 0
        self.send_flits = defaultdict(list)
        self.arrive_flits = defaultdict(list)
        self.all_latency = []
        self.ject_latency = []
        self.network_latency = []
        self.predicted_recv_time = []
        self.inject_time = {}
        self.eject_time = {}
        self.avg_inject_time = {}
        self.avg_eject_time = {}
        self.predicted_avg_latency = None
        self.predicted_max_latency = None
        self.actual_avg_latency = None
        self.actual_max_latency = None
        self.actual_avg_ject_latency = None
        self.actual_max_ject_latency = None
        self.actual_avg_net_latency = None
        self.actual_max_net_latency = None
        #
        self.circuits_h = []
        self.circuits_v = []
        self.avg_circuits_h = None
        self.max_circuits_h = None
        self.avg_circuits_v = None
        self.max_circuits_v = None
        self.circuits_flit_h = self.config._make_channels(("sdma", "gdma", "cdma", "ddr", "l2m"))
        self.circuits_flit_v = self.config._make_channels(("sdma", "gdma", "cdma", "ddr", "l2m"))
        self.gdma_recv = 0
        self.gdma_remainder = 0
        self.gdma_count = 512
        self.l2m_recv = 0
        self.l2m_remainder = 0
        self.sdma_send = []
        self.num_send = self.config._make_channels(("sdma", "gdma", "cdma", "ddr", "l2m"))
        self.num_recv = self.config._make_channels(("sdma", "gdma", "cdma", "ddr", "l2m"))
        self.per_send_throughput = self.config._make_channels(("sdma", "gdma", "cdma", "ddr", "l2m"))
        self.per_recv_throughput = self.config._make_channels(("sdma", "gdma", "cdma", "ddr", "l2m"))
        self.send_throughput = self.config._make_channels(("sdma", "gdma", "cdma", "ddr", "l2m"))
        self.recv_throughput = self.config._make_channels(("sdma", "gdma", "cdma", "ddr", "l2m"))
        self.last_select = self.config._make_channels(("sdma", "gdma", "cdma", "ddr", "l2m"))
        self.throughput = self.config._make_channels(("sdma", "gdma", "cdma", "ddr", "l2m"))

        # # channel buffer setup

        self.ring_bridge_map = {
            0: ("TL", self.config.RB_IN_FIFO_DEPTH),
            1: ("TR", self.config.RB_IN_FIFO_DEPTH),
            -1: ("IQ_TU", self.config.IQ_OUT_FIFO_DEPTH),
            -2: ("IQ_TD", self.config.IQ_OUT_FIFO_DEPTH),
        }

        self.token_bucket = defaultdict(dict)
        self.flit_size_bytes = 128
        for ch_name in self.IQ_channel_buffer.keys():
            for ip_pos in set(self.config.DDR_SEND_POSITION_LIST + self.config.L2M_SEND_POSITION_LIST):

                if ch_name.startswith("ddr"):
                    self.token_bucket[ip_pos][ch_name] = TokenBucket(
                        rate=self.config.DDR_BW_LIMIT / self.config.NETWORK_FREQUENCY / self.flit_size_bytes,
                        bucket_size=self.config.DDR_BW_LIMIT,
                    )
                    self.token_bucket[ip_pos - self.config.NUM_COL][ch_name] = TokenBucket(
                        rate=self.config.DDR_BW_LIMIT / self.config.NETWORK_FREQUENCY / self.flit_size_bytes,
                        bucket_size=self.config.DDR_BW_LIMIT,
                    )
                elif ch_name.startswith("l2m"):
                    self.token_bucket[ip_pos][ch_name] = TokenBucket(
                        rate=self.config.L2M_BW_LIMIT / self.config.NETWORK_FREQUENCY / self.flit_size_bytes,
                        bucket_size=self.config.L2M_BW_LIMIT,
                    )

        # ETag setup
        self.T0_Etag_Order_FIFO = deque()  # 用于轮询选择 T0 Flit 的 Order FIFO
        self.RB_UE_Counters = {"TL": {}, "TR": {}}
        self.EQ_UE_Counters = {"TU": {}, "TD": {}}
        self.ETag_BOTHSIDE_UPGRADE = False

        for ip_pos in set(config.DDR_SEND_POSITION_LIST + config.SDMA_SEND_POSITION_LIST + config.CDMA_SEND_POSITION_LIST + config.L2M_SEND_POSITION_LIST + config.GDMA_SEND_POSITION_LIST):
            self.cross_point["horizontal"][ip_pos]["TL"] = [None] * 2
            self.cross_point["horizontal"][ip_pos]["TR"] = [None] * 2
            self.cross_point["vertical"][ip_pos]["TU"] = [None] * 2
            self.cross_point["vertical"][ip_pos]["TD"] = [None] * 2
            self.inject_queues["TL"][ip_pos] = deque(maxlen=config.IQ_OUT_FIFO_DEPTH)
            self.inject_queues["TR"][ip_pos] = deque(maxlen=config.IQ_OUT_FIFO_DEPTH)
            self.inject_queues["TU"][ip_pos] = deque(maxlen=config.IQ_OUT_FIFO_DEPTH)
            self.inject_queues["TD"][ip_pos] = deque(maxlen=config.IQ_OUT_FIFO_DEPTH)
            self.inject_queues["EQ"][ip_pos] = deque(maxlen=config.IQ_OUT_FIFO_DEPTH)
            self.inject_queues_pre["TL"][ip_pos] = None
            self.inject_queues_pre["TR"][ip_pos] = None
            self.inject_queues_pre["TU"][ip_pos] = None
            self.inject_queues_pre["TD"][ip_pos] = None
            self.inject_queues_pre["EQ"][ip_pos] = None
            for key in self.config.CH_NAME_LIST:
                self.IQ_channel_buffer_pre[key][ip_pos] = None
                self.EQ_channel_buffer_pre[key][ip_pos - config.NUM_COL] = None
            for key in self.arrive_node_pre:
                self.arrive_node_pre[key][ip_pos - config.NUM_COL] = None
            self.eject_queues["TU"][ip_pos - config.NUM_COL] = deque(maxlen=config.EQ_IN_FIFO_DEPTH)
            self.eject_queues["TD"][ip_pos - config.NUM_COL] = deque(maxlen=config.EQ_IN_FIFO_DEPTH)
            self.eject_queues_in_pre["TU"][ip_pos - config.NUM_COL] = None
            self.eject_queues_in_pre["TD"][ip_pos - config.NUM_COL] = None
            self.EQ_UE_Counters["TU"][ip_pos - config.NUM_COL] = {"T2": 0, "T1": 0, "T0": 0}
            self.EQ_UE_Counters["TD"][ip_pos - config.NUM_COL] = {"T2": 0, "T1": 0}

            for key in self.round_robin.keys():
                if key == "IQ":
                    for fifo_name in ["TR", "TL", "TU", "TD", "EQ"]:
                        self.round_robin[key][fifo_name][ip_pos] = deque()
                        self.round_robin[key][fifo_name][ip_pos - config.NUM_COL] = deque()
                        for ch_name in self.IQ_channel_buffer.keys():
                            self.round_robin[key][fifo_name][ip_pos].append(ch_name)
                            self.round_robin[key][fifo_name][ip_pos - config.NUM_COL].append(ch_name)
                elif key == "EQ":
                    for ch_name in self.IQ_channel_buffer.keys():
                        self.round_robin[key][ch_name][ip_pos] = deque([0, 1, 2, 3])
                        self.round_robin[key][ch_name][ip_pos - config.NUM_COL] = deque([0, 1, 2, 3])
                else:
                    for fifo_name in ["TU", "TD", "EQ"]:
                        self.round_robin[key][fifo_name][ip_pos] = deque([0, 1, 2, 3])
                        self.round_robin[key][fifo_name][ip_pos - config.NUM_COL] = deque([0, 1, 2, 3])

            self.inject_time[ip_pos] = []
            self.eject_time[ip_pos - config.NUM_COL] = []
            self.avg_inject_time[ip_pos] = 0
            self.avg_eject_time[ip_pos - config.NUM_COL] = 1

        for i in range(config.NUM_NODE):
            for j in range(config.NUM_NODE):
                if adjacency_matrix[i][j] == 1 and i - j != config.NUM_COL:
                    self.links[(i, j)] = [None] * config.SLICE_PER_LINK
                    self.links_flow_stat["read"][(i, j)] = 0
                    self.links_flow_stat["write"][(i, j)] = 0
                    self.links_tag[(i, j)] = [None] * config.SLICE_PER_LINK
            if i in range(0, config.NUM_COL):
                self.links[(i, i)] = [None] * 2
                self.links[(i + config.NUM_NODE - config.NUM_COL * 2, i + config.NUM_NODE - config.NUM_COL * 2)] = [None] * 2
                self.links_flow_stat["read"][(i, i)] = 0
                self.links_flow_stat["write"][(i, i)] = 0
                self.links_flow_stat["read"][(i + config.NUM_NODE - config.NUM_COL * 2, i + config.NUM_NODE - config.NUM_COL * 2)] = 0
                self.links_flow_stat["write"][(i + config.NUM_NODE - config.NUM_COL * 2, i + config.NUM_NODE - config.NUM_COL * 2)] = 0
                self.links_tag[(i, i)] = [None] * 2
                self.links_tag[(i + config.NUM_NODE - config.NUM_COL * 2, i + config.NUM_NODE - config.NUM_COL * 2)] = [None] * 2
            if i % config.NUM_COL == 0 and (i // config.NUM_COL) % 2 != 0:
                self.links[(i, i)] = [None] * 2
                self.links[(i + config.NUM_COL - 1, i + config.NUM_COL - 1)] = [None] * 2
                self.links_flow_stat["read"][(i, i)] = 0
                self.links_flow_stat["write"][(i, i)] = 0
                self.links_flow_stat["read"][(i + config.NUM_COL - 1, i + config.NUM_COL - 1)] = 0
                self.links_flow_stat["write"][(i + config.NUM_COL - 1, i + config.NUM_COL - 1)] = 0
                self.links_tag[(i, i)] = [None] * 2
                self.links_tag[(i + config.NUM_COL - 1, i + config.NUM_COL - 1)] = [None] * 2

        for row in range(1, config.NUM_ROW, 2):
            for col in range(config.NUM_COL):
                pos = row * config.NUM_COL + col
                next_pos = pos - config.NUM_COL
                self.ring_bridge["TL"][(pos, next_pos)] = deque(maxlen=config.RB_IN_FIFO_DEPTH)
                self.ring_bridge["TR"][(pos, next_pos)] = deque(maxlen=config.RB_IN_FIFO_DEPTH)
                self.ring_bridge["TU"][(pos, next_pos)] = deque(maxlen=config.RB_OUT_FIFO_DEPTH)
                self.ring_bridge["TD"][(pos, next_pos)] = deque(maxlen=config.RB_OUT_FIFO_DEPTH)
                self.ring_bridge["EQ"][(pos, next_pos)] = deque(maxlen=config.RB_OUT_FIFO_DEPTH)

                self.ring_bridge_pre["TL"][(pos, next_pos)] = None
                self.ring_bridge_pre["TR"][(pos, next_pos)] = None
                self.ring_bridge_pre["TU"][(pos, next_pos)] = None
                self.ring_bridge_pre["TD"][(pos, next_pos)] = None
                self.ring_bridge_pre["EQ"][(pos, next_pos)] = None

                self.RB_UE_Counters["TL"][(pos, next_pos)] = {"T2": 0, "T1": 0, "T0": 0}
                self.RB_UE_Counters["TR"][(pos, next_pos)] = {"T2": 0, "T1": 0}
                # self.round_robin["TU"][next_pos] = deque([0, 1, 2])
                # self.round_robin["TD"][next_pos] = deque([0, 1, 2])
                # self.round_robin["RB"][next_pos] = deque([0, 1, 2])
                for direction in ["TL", "TR"]:
                    self.remain_tag[direction][pos] = config.ITag_MAX_NUM_H
                    self.itag_req_counter[direction][pos] = 0
                    self.tagged_counter[direction][pos] = 0
                    self.excess_ITag_to_remove[direction][pos] = 0
                for direction in ["TU", "TD"]:
                    self.remain_tag[direction][pos] = config.ITag_MAX_NUM_V
                    self.itag_req_counter[direction][pos] = 0
                    self.tagged_counter[direction][pos] = 0
                    self.excess_ITag_to_remove[direction][pos] = 0

        for ip_type in self.num_recv.keys():
            source_positions = getattr(config, f"{ip_type[:-2].upper()}_SEND_POSITION_LIST")
            for source in source_positions:
                destination = source - config.NUM_COL
                self.num_send[ip_type][source] = 0
                self.num_recv[ip_type][destination] = 0
                self.per_send_throughput[ip_type][source] = 0
                self.per_recv_throughput[ip_type][destination] = 0

        for ip_type in self.IQ_channel_buffer.keys():
            for ip_index in getattr(config, f"{ip_type[:-2].upper()}_SEND_POSITION_LIST"):
                ip_recv_index = ip_index - config.NUM_COL
                self.IQ_channel_buffer[ip_type][ip_index] = deque(maxlen=config.IQ_CH_FIFO_DEPTH)
                self.EQ_channel_buffer[ip_type][ip_recv_index] = deque(maxlen=config.EQ_CH_FIFO_DEPTH)
                self.EQ_channel_buffer[ip_type][ip_index] = deque(maxlen=config.EQ_CH_FIFO_DEPTH)
        for ip_type in self.last_select.keys():
            for ip_index in getattr(config, f"{ip_type[:-2].upper()}_SEND_POSITION_LIST"):
                self.last_select[ip_type][ip_index] = "write"
        for ip_type in self.throughput.keys():
            for ip_index in getattr(config, f"{ip_type[:-2].upper()}_SEND_POSITION_LIST"):
                self.throughput[ip_type][ip_index] = [0, 0, 10000000, 0]

        self.RB_CAPACITY = {"TL": {}, "TR": {}}
        self.EQ_CAPACITY = {"TU": {}, "TD": {}}

        # TL capacity
        def _cap_tl(lvl):
            if lvl == "T2":
                return self.config.TL_Etag_T2_UE_MAX
            if lvl == "T1":
                return self.config.TL_Etag_T1_UE_MAX - self.config.TL_Etag_T2_UE_MAX
            if lvl == "T0":
                return self.config.RB_IN_FIFO_DEPTH - self.config.TL_Etag_T1_UE_MAX

        # TR capacity
        def _cap_tr(lvl):
            if lvl == "T2":
                return self.config.TR_Etag_T2_UE_MAX
            if lvl == "T1":
                return self.config.RB_IN_FIFO_DEPTH - self.config.TR_Etag_T2_UE_MAX
            return 0  # TR 无 T0

        # TU capacity
        def _cap_tu(lvl):
            if lvl == "T2":
                return self.config.TU_Etag_T2_UE_MAX
            if lvl == "T1":
                return self.config.TU_Etag_T1_UE_MAX - self.config.TU_Etag_T2_UE_MAX
            if lvl == "T0":
                return self.config.EQ_IN_FIFO_DEPTH - self.config.TU_Etag_T1_UE_MAX

        # TD capacity
        def _cap_td(lvl):
            if lvl == "T2":
                return self.config.TD_Etag_T2_UE_MAX
            if lvl == "T1":
                return self.config.EQ_IN_FIFO_DEPTH - self.config.TD_Etag_T2_UE_MAX
            return 0  # TD 无 T0

        for pair in self.RB_UE_Counters["TL"]:
            self.RB_CAPACITY["TL"][pair] = {lvl: _cap_tl(lvl) for lvl in ("T0", "T1", "T2")}
        for pair in self.RB_UE_Counters["TR"]:
            self.RB_CAPACITY["TR"][pair] = {lvl: _cap_tr(lvl) for lvl in ("T1", "T2")}

        for pos in self.EQ_UE_Counters["TU"]:
            self.EQ_CAPACITY["TU"][pos] = {lvl: _cap_tu(lvl) for lvl in ("T0", "T1", "T2")}
        for pos in self.EQ_UE_Counters["TD"]:
            self.EQ_CAPACITY["TD"][pos] = {lvl: _cap_td(lvl) for lvl in ("T1", "T2")}
            
    @property
    def all_ip_positions(self):
        """Cached property for all IP positions"""
        if self._all_ip_positions is None:
            self._all_ip_positions = list(set(
                self.config.DDR_SEND_POSITION_LIST + 
                self.config.SDMA_SEND_POSITION_LIST + 
                self.config.CDMA_SEND_POSITION_LIST + 
                self.config.L2M_SEND_POSITION_LIST + 
                self.config.GDMA_SEND_POSITION_LIST
            ))
        return self._all_ip_positions
    
    @property
    def rn_positions(self):
        """Cached property for RN positions"""
        if self._rn_positions is None:
            self._rn_positions = list(set(
                self.config.GDMA_SEND_POSITION_LIST + 
                self.config.SDMA_SEND_POSITION_LIST + 
                self.config.CDMA_SEND_POSITION_LIST
            ))
        return self._rn_positions
    
    @property  
    def sn_positions(self):
        """Cached property for SN positions"""
        if self._sn_positions is None:
            self._sn_positions = list(set(
                self.config.DDR_SEND_POSITION_LIST + 
                self.config.L2M_SEND_POSITION_LIST
            ))
        return self._sn_positions

    def _entry_available(self, dir_type, key, level):
        if dir_type in ("TL", "TR"):
            cap = self.RB_CAPACITY[dir_type][key][level]
            occ = self.RB_UE_Counters[dir_type][key][level]
        else:
            cap = self.EQ_CAPACITY[dir_type][key][level]
            occ = self.EQ_UE_Counters[dir_type][key][level]
        return occ < cap

    # ------------------------------------------------------------------
    def _occupy_entry(self, dir_type, key, level, flit):
        """
        统一处理占用计数器，并记录 flit.used_entry_level
        dir_type : "TL"|"TR"|"TU"|"TD"
        key      : (cur,next) for RB  or  dest_pos for EQ
        level    : "T0"/"T1"/"T2"
        """
        if dir_type in ("TL", "TR"):
            self.RB_UE_Counters[dir_type][key][level] += 1
            # flit.flit_position = f"RB_{dir_type}"
        else:
            self.EQ_UE_Counters[dir_type][key][level] += 1
            # flit.flit_position = f"EQ_{dir_type}"
        flit.used_entry_level = level

    def error_log(self, flit, target_id, flit_id):
        if flit and flit.packet_id == target_id and flit.flit_id == flit_id:
            print(inspect.currentframe().f_back.f_code.co_name, self.cycle, flit)

    def set_link_slice(self, link: tuple[int, int], slice_index: int, flit: "Flit", cycle, *, override: bool = False):
        """
        Safely assign a flit to a given slice on a link.

        Parameters
        ----------
        link : tuple[int, int]
            (u, v) node indices of the directed link.
        index : int
            Slice index on that link (0 == head).
        flit : Flit
            The flit object to place.
        override : bool, optional
            If True, forcibly override the existing flit (logging a warning).
            If False (default), raise RuntimeError when the slot is occupied.

        Raises
        ------
        RuntimeError
            When the target slice is already occupied and override == False.
        """
        try:
            current = self.links[link][slice_index]
        except KeyError as e:
            raise KeyError(f"Link {link} does not exist in Network '{self.name}'") from e
        except IndexError as e:
            raise IndexError(f"Slice index {slice_index} out of range for link {link}") from e

        if current is not None and not override:
            raise RuntimeError(f"[Cycle {cycle}] " f"Attempt to assign flit {flit} to occupied " f"link {link}[{slice_index}] already holding flit {current}")

        if current is not None and override:
            logging.warning(f"[Cycle {cycle}] " f"Overriding link {link}[{slice_index}] flit {current.packet_id}.{current.flit_id} " f"with {flit.packet_id}.{flit.flit_id}")

        self.links[link][slice_index] = flit

    def can_move_to_next(self, flit, current, next_node):
        # 1. flit不进入Cross Poing
        if flit.source - flit.destination == self.config.NUM_COL:
            return len(self.inject_queues["EQ"]) < self.config.IQ_OUT_FIFO_DEPTH
        elif current - next_node == self.config.NUM_COL:
            # 向 Ring Bridge 移动. v1.3 在IQ中分TU和TD两个FIFO
            if len(flit.path) > 2 and flit.path[2] - flit.path[1] == self.config.NUM_COL * 2:
                return len(self.inject_queues["TD"][current]) < self.config.IQ_OUT_FIFO_DEPTH
            elif len(flit.path) > 2 and flit.path[2] - flit.path[1] == -self.config.NUM_COL * 2:
                return len(self.inject_queues["TU"][current]) < self.config.IQ_OUT_FIFO_DEPTH
            else:
                raise Exception(f"Invalid path: {flit.path}")

        direction = "TR" if next_node == current + 1 else "TL"
        link = (current, next_node)

        # 横向环ITag处理
        if self.links[link][0] is not None:  # Link被占用
            # 检查是否需要标记ITag（内联所有检查逻辑）
            if (
                self.links_tag[link][0] is None
                and flit.wait_cycle_h > self.config.ITag_TRIGGER_Th_H
                and self.tagged_counter[direction][current] < self.config.ITag_MAX_NUM_H
                and self.itag_req_counter[direction][current] > 0
                and self.remain_tag[direction][current] > 0
            ):

                # 创建ITag标记（内联逻辑）
                self.remain_tag[direction][current] -= 1
                self.tagged_counter[direction][current] += 1
                self.links_tag[link][0] = [current, direction]
                flit.itag_h = True
            return False

        else:  # Link空闲
            if self.links_tag[link][0] is None:  # 无预约
                return True  # 直接上环
            else:  # 有预约
                if self.links_tag[link][0] == [current, direction]:  # 是自己的预约
                    # 使用预约（内联逻辑）
                    self.links_tag[link][0] = None
                    self.remain_tag[direction][current] += 1  # 修复：使用direction
                    self.tagged_counter[direction][current] -= 1
                    return True
        return False

    def update_excess_ITag(self):
        """在主循环中调用，处理多余ITag释放"""
        # 处理多余ITag释放（简化版）
        for direction in ["TL", "TR"]:
            for node_id in set(self.config.DDR_SEND_POSITION_LIST + self.config.SDMA_SEND_POSITION_LIST + self.config.L2M_SEND_POSITION_LIST + self.config.GDMA_SEND_POSITION_LIST):
                if self.excess_ITag_to_remove[direction][node_id] > 0:
                    # 寻找该节点创建的ITag并释放
                    for link, tag_info in self.links_tag.items():
                        if tag_info[0] is not None and tag_info[0] == [node_id, direction] and link[0] == node_id:  # ITag回到创建节点
                            # 释放多余ITag
                            self.links_tag[link][0] = None
                            self.tagged_counter[direction][node_id] -= 1
                            self.remain_tag[direction][node_id] += 1
                            self.excess_ITag_to_remove[direction][node_id] -= 1
                            break  # 一次只释放一个

    def update_cross_point(self):
        for ip_pos in set(self.config.DDR_SEND_POSITION_LIST + self.config.SDMA_SEND_POSITION_LIST + self.config.L2M_SEND_POSITION_LIST + self.config.GDMA_SEND_POSITION_LIST):
            left_pos = ip_pos - 1 if ip_pos % self.config.NUM_COL != 0 else ip_pos
            right_pos = ip_pos + 1 if ip_pos % self.config.NUM_COL != self.config.NUM_COL - 1 else ip_pos
            up_pos = ip_pos - self.config.NUM_COL * 3 if ip_pos // self.config.NUM_COL != 1 else ip_pos - self.config.NUM_COL
            down_pos = ip_pos + self.config.NUM_COL * 1 if ip_pos // self.config.NUM_COL != self.config.NUM_ROW - 1 else ip_pos - self.config.NUM_COL
            self.cross_point["horizontal"][ip_pos]["TR"] = [self.links[(left_pos, ip_pos)][-1], self.links[(ip_pos, right_pos)][0]]
            self.cross_point["horizontal"][ip_pos]["TL"] = [self.links[(ip_pos, left_pos)][0], self.links[(right_pos, ip_pos)][-1]]
            self.cross_point["vertical"][ip_pos]["TU"] = [self.links[(down_pos, ip_pos - self.config.NUM_COL)][-1], self.links[(ip_pos - self.config.NUM_COL, up_pos)][0]]
            self.cross_point["vertical"][ip_pos]["TD"] = [self.links[(ip_pos - self.config.NUM_COL, down_pos)][0], self.links[(up_pos, ip_pos - self.config.NUM_COL)][-1]]

    def plan_move(self, flit, cycle):
        self.cycle = cycle
        if flit.is_new_on_network:
            # current = flit.source
            current = flit.path[flit.path_index]
            next_node = flit.path[flit.path_index + 1]
            flit.current_position = current
            flit.is_new_on_network = False
            flit.flit_position = "Link"
            flit.is_arrive = False
            flit.is_on_station = False
            flit.current_link = (current, next_node)
            if flit.source - flit.destination == self.config.NUM_COL:
                flit.is_arrive = True
            elif current - next_node == self.config.NUM_COL:
                if len(flit.path) > 2 and flit.path[flit.path_index + 2] - next_node == 2 * self.config.NUM_COL:
                    flit.current_seat_index = -1
                elif len(flit.path) > 2 and flit.path[flit.path_index + 2] - next_node == -2 * self.config.NUM_COL:
                    flit.current_seat_index = -2
            else:
                flit.current_seat_index = 0

            return

        # 计算行和列的起始和结束点
        current, next_node = flit.current_link
        if current - next_node != self.config.NUM_COL:
            row_start = (current // self.config.NUM_COL) * self.config.NUM_COL
            row_start = row_start if (row_start // self.config.NUM_COL) % 2 != 0 else -1
            row_end = row_start + self.config.NUM_COL - 1 if row_start > 0 else -1
            col_start = current % (self.config.NUM_COL * 2)
            col_start = col_start if col_start < self.config.NUM_COL else -1
            col_end = col_start + self.config.NUM_NODE - self.config.NUM_COL * 2 if col_start >= 0 else -1

            link = self.links.get(flit.current_link)
            # self.error_log(flit, 6491, -1)

            # Plan non ring bridge moves
            # Handling delay flits
            if flit.is_delay:
                return self._handle_delay_flit(flit, link, current, next_node, row_start, row_end, col_start, col_end)
            # Handling regular flits
            else:
                return self._handle_regular_flit(flit, link, current, next_node, row_start, row_end, col_start, col_end)

    def _handle_delay_flit(self, flit: Flit, link, current, next_node, row_start, row_end, col_start, col_end):
        # 1. 非链路末端
        if flit.current_seat_index < len(link) - 1:
            link[flit.current_seat_index] = None
            flit.current_seat_index += 1
            return
        # self.error_log(flit, 144, -1)
        # 2. 到达链路末端，此时flit在next_node节点
        target_eject_node_id = flit.path[flit.path_index + 1] if flit.path_index + 1 < len(flit.path) else flit.path[flit.path_index]  # delay情况下path_index不更新
        # A. 处理横边界情况
        if current == next_node:
            # A1. 左边界情况
            if next_node == row_start:
                if next_node == flit.current_position:
                    # Flit已经绕横向环一圈
                    flit.circuits_completed_h += 1
                    link_station = self.ring_bridge["TR"].get((next_node, target_eject_node_id))
                    can_use_T1 = self._entry_available("TR", (next_node, target_eject_node_id), "T1")
                    can_use_T2 = self._entry_available("TR", (next_node, target_eject_node_id), "T2")
                    # TR方向尝试下环
                    if len(link_station) < self.config.RB_IN_FIFO_DEPTH and (
                        (flit.ETag_priority == "T1" and can_use_T1)
                        or (flit.ETag_priority == "T2" and can_use_T2)
                        or (flit.ETag_priority == "T1" and not can_use_T1 and can_use_T2)  # T1使用T2 entry
                        or (flit.ETag_priority == "T0" and can_use_T1)  # T0使用T1 entry
                        or (flit.ETag_priority == "T0" and not can_use_T1 and can_use_T2)  # T0使用T2 entry
                    ):
                        flit.is_delay = False
                        flit.current_link = (next_node, target_eject_node_id)
                        link[flit.current_seat_index] = None
                        flit.current_seat_index = 1
                        if flit.ETag_priority == "T0":
                            # 若升级到T0则需要从T0队列中移除flit
                            self.T0_Etag_Order_FIFO.remove((next_node, flit))
                            if can_use_T1:
                                self._occupy_entry("TR", (next_node, target_eject_node_id), "T1", flit)
                            else:
                                self._occupy_entry("TR", (next_node, target_eject_node_id), "T2", flit)
                        elif flit.ETag_priority == "T2":
                            self._occupy_entry("TR", (next_node, target_eject_node_id), "T2", flit)
                        elif flit.ETag_priority == "T1":
                            if can_use_T1:
                                self._occupy_entry("TR", (next_node, target_eject_node_id), "T1", flit)
                            else:
                                self._occupy_entry("TR", (next_node, target_eject_node_id), "T2", flit)
                    else:
                        # 无法下环,TR方向的flit不能升级T0
                        link[flit.current_seat_index] = None
                        next_pos = next_node + 1
                        flit.current_link = (next_node, next_pos)
                        flit.current_seat_index = 0
                        if self.ETag_BOTHSIDE_UPGRADE and flit.ETag_priority == "T2":
                            flit.ETag_priority = "T1"
                else:
                    # Flit未绕回下环点，向右绕环
                    link[flit.current_seat_index] = None
                    next_pos = next_node + 1
                    flit.current_link = (next_node, next_pos)
                    flit.current_seat_index = 0

            # A2. 右边界情况:
            elif next_node == row_end:
                if next_node == flit.current_position:
                    flit.circuits_completed_h += 1
                    link_station = self.ring_bridge["TL"].get((next_node, target_eject_node_id))
                    can_use_T0 = self._entry_available("TL", (next_node, target_eject_node_id), "T0")
                    can_use_T1 = self._entry_available("TL", (next_node, target_eject_node_id), "T1")
                    can_use_T2 = self._entry_available("TL", (next_node, target_eject_node_id), "T2")
                    # 尝试TL下环，非T0情况
                    if flit.ETag_priority in ["T1", "T2"]:
                        if len(link_station) < self.config.RB_IN_FIFO_DEPTH and (
                            (flit.ETag_priority == "T1" and can_use_T1)
                            or (flit.ETag_priority == "T2" and can_use_T2)
                            or (flit.ETag_priority == "T1" and not can_use_T1 and can_use_T2)  # T1使用T2 entry
                        ):
                            flit.is_delay = False
                            flit.current_link = (next_node, target_eject_node_id)
                            link[flit.current_seat_index] = None
                            flit.current_seat_index = 0
                            if flit.ETag_priority == "T2":
                                self._occupy_entry("TL", (next_node, target_eject_node_id), "T2", flit)
                            elif flit.ETag_priority == "T1":
                                if can_use_T1:
                                    # T1使用T1 entry
                                    self._occupy_entry("TL", (next_node, target_eject_node_id), "T1", flit)
                                else:
                                    # T1使用T2 entry
                                    self._occupy_entry("TL", (next_node, target_eject_node_id), "T2", flit)

                        else:
                            # 无法下环,升级ETag并记录
                            if flit.ETag_priority == "T2":
                                flit.ETag_priority = "T1"
                            elif flit.ETag_priority == "T1":
                                flit.ETag_priority = "T0"
                                self.T0_Etag_Order_FIFO.append((next_node, flit))
                            link[flit.current_seat_index] = None
                            next_pos = next_node - 1
                            flit.current_link = (next_node, next_pos)
                            flit.current_seat_index = 0
                    # 尝试TL以T0下环
                    elif flit.ETag_priority == "T0":
                        if len(link_station) < self.config.RB_IN_FIFO_DEPTH:
                            # 按优先级尝试: T0专用 > T1 > T2
                            if self.T0_Etag_Order_FIFO[0] == (next_node, flit) and can_use_T0:
                                # 使用T0专用entry
                                self._occupy_entry("TL", (next_node, target_eject_node_id), "T0", flit)
                                flit.is_delay = False
                                flit.current_link = (next_node, target_eject_node_id)
                                link[flit.current_seat_index] = None
                                flit.current_seat_index = 0
                                self.T0_Etag_Order_FIFO.popleft()
                            elif can_use_T1:
                                # 使用T1 entry
                                self._occupy_entry("TL", (next_node, target_eject_node_id), "T1", flit)
                                self.T0_Etag_Order_FIFO.remove((next_node, flit))
                                flit.is_delay = False
                                flit.current_link = (next_node, target_eject_node_id)
                                link[flit.current_seat_index] = None
                                flit.current_seat_index = 0
                            elif can_use_T2:
                                # 使用T2 entry
                                self._occupy_entry("TL", (next_node, target_eject_node_id), "T2", flit)
                                self.T0_Etag_Order_FIFO.remove((next_node, flit))
                                flit.is_delay = False
                                flit.current_link = (next_node, target_eject_node_id)
                                link[flit.current_seat_index] = None
                                flit.current_seat_index = 0
                            else:
                                # 无法下环，继续绕环
                                link[flit.current_seat_index] = None
                                next_pos = next_node - 1
                                flit.current_link = (next_node, next_pos)
                                flit.current_seat_index = 0
                        else:
                            # link_station满，无法下环
                            link[flit.current_seat_index] = None
                            next_pos = next_node - 1
                            flit.current_link = (next_node, next_pos)
                            flit.current_seat_index = 0
                # 未到下环节点，继续向左绕环
                else:
                    link[flit.current_seat_index] = None
                    next_pos = next_node - 1
                    flit.current_link = (next_node, next_pos)
                    flit.current_seat_index = 0
            # A3. 上边界情况:
            elif next_node == col_start:
                if next_node == flit.current_position:
                    flit.circuits_completed_v += 1
                    link_eject = self.eject_queues["TD"][next_node]
                    can_use_T1 = self._entry_available("TD", next_node, "T1")
                    can_use_T2 = self._entry_available("TD", next_node, "T2")

                    if len(link_eject) < self.config.EQ_IN_FIFO_DEPTH and (
                        (flit.ETag_priority == "T1" and can_use_T1) or (flit.ETag_priority == "T2" and can_use_T2) or (flit.ETag_priority == "T0" and (can_use_T1 or can_use_T2))
                    ):
                        flit.is_delay = False
                        flit.is_arrive = True
                        link[flit.current_seat_index] = None
                        flit.current_seat_index = 0

                        if flit.ETag_priority == "T0":
                            self.T0_Etag_Order_FIFO.remove((next_node, flit))
                            if can_use_T1:
                                # T0使用T1 entry
                                self._occupy_entry("TD", next_node, "T1", flit)
                            else:
                                # T0使用T2 entry
                                self._occupy_entry("TD", next_node, "T2", flit)
                        elif flit.ETag_priority == "T2":
                            self._occupy_entry("TD", next_node, "T2", flit)
                        elif flit.ETag_priority == "T1":
                            if can_use_T1:
                                # T1使用T1 entry
                                self._occupy_entry("TD", next_node, "T1", flit)
                            else:
                                # T1使用T2 entry
                                self._occupy_entry("TD", next_node, "T2", flit)
                    else:
                        # 无法下环,TD方向的flit不能升级T0
                        if self.ETag_BOTHSIDE_UPGRADE and flit.ETag_priority == "T2":
                            flit.ETag_priority = "T1"
                        link[flit.current_seat_index] = None
                        next_pos = next_node + self.config.NUM_COL * 2
                        flit.current_link = (next_node, next_pos)
                        flit.current_seat_index = 0
                else:
                    link[flit.current_seat_index] = None
                    next_pos = next_node + self.config.NUM_COL * 2
                    flit.current_link = (next_node, next_pos)
                    flit.current_seat_index = 0
            # A4. 下边界情况:
            elif next_node == col_end:
                if next_node == flit.current_position:
                    flit.circuits_completed_v += 1
                    link_eject = self.eject_queues["TU"][next_node]
                    can_use_T0 = self._entry_available("TU", next_node, "T0")
                    can_use_T1 = self._entry_available("TU", next_node, "T1")
                    can_use_T2 = self._entry_available("TU", next_node, "T2")

                    if flit.ETag_priority in ["T1", "T2"]:
                        if len(link_eject) < self.config.EQ_IN_FIFO_DEPTH and (
                            (flit.ETag_priority == "T1" and can_use_T1)
                            or (flit.ETag_priority == "T2" and can_use_T2)
                            or (flit.ETag_priority == "T1" and not can_use_T1 and can_use_T2)  # T1使用T2 entry
                        ):
                            flit.is_delay = False
                            flit.is_arrive = True
                            link[flit.current_seat_index] = None
                            flit.current_seat_index = 0

                            if flit.ETag_priority == "T2":
                                self._occupy_entry("TU", next_node, "T2", flit)
                            elif flit.ETag_priority == "T1":
                                if can_use_T1:
                                    # T1使用T1 entry
                                    self._occupy_entry("TU", next_node, "T1", flit)
                                else:
                                    # T1使用T2 entry
                                    self._occupy_entry("TU", next_node, "T2", flit)
                        else:
                            # 无法下环,升级ETag并记录
                            if flit.ETag_priority == "T2":
                                flit.ETag_priority = "T1"
                            elif flit.ETag_priority == "T1":
                                self.T0_Etag_Order_FIFO.append((next_node, flit))
                                flit.ETag_priority = "T0"
                            link[flit.current_seat_index] = None
                            next_pos = next_node - self.config.NUM_COL * 2
                            flit.current_link = (next_node, next_pos)
                            flit.current_seat_index = 0

                    elif flit.ETag_priority == "T0":
                        if len(link_eject) < self.config.EQ_IN_FIFO_DEPTH:
                            # 按优先级尝试: T0专用 > T1 > T2
                            if self.T0_Etag_Order_FIFO[0] == (next_node, flit) and can_use_T0:
                                # 使用T0专用entry
                                self._occupy_entry("TU", next_node, "T0", flit)
                                flit.is_delay = False
                                flit.is_arrive = True
                                link[flit.current_seat_index] = None
                                flit.current_seat_index = 0
                                self.T0_Etag_Order_FIFO.popleft()
                            elif can_use_T1:
                                # 使用T1 entry
                                self._occupy_entry("TU", next_node, "T1", flit)
                                self.T0_Etag_Order_FIFO.remove((next_node, flit))
                                flit.is_delay = False
                                flit.is_arrive = True
                                link[flit.current_seat_index] = None
                                flit.current_seat_index = 0
                            elif can_use_T2:
                                # 使用T2 entry
                                self._occupy_entry("TU", next_node, "T2", flit)
                                self.T0_Etag_Order_FIFO.remove((next_node, flit))
                                flit.is_delay = False
                                flit.is_arrive = True
                                link[flit.current_seat_index] = None
                                flit.current_seat_index = 0
                            else:
                                # 无法下环，继续绕环
                                link[flit.current_seat_index] = None
                                next_pos = next_node - self.config.NUM_COL * 2
                                flit.current_link = (next_node, next_pos)
                                flit.current_seat_index = 0
                        else:
                            # link_eject满，无法下环
                            link[flit.current_seat_index] = None
                            next_pos = next_node - self.config.NUM_COL * 2
                            flit.current_link = (next_node, next_pos)
                            flit.current_seat_index = 0
                else:
                    link[flit.current_seat_index] = None
                    next_pos = next_node - self.config.NUM_COL * 2
                    flit.current_link = (next_node, next_pos)
                    flit.current_seat_index = 0
        # B. 非边界横向环情况
        elif abs(current - next_node) == 1:
            if next_node == flit.current_position:
                flit.circuits_completed_h += 1
                if current - next_node == 1:
                    link_station = self.ring_bridge["TL"].get((next_node, target_eject_node_id))
                    can_use_T0 = self._entry_available("TL", (next_node, target_eject_node_id), "T0")
                    can_use_T1 = self._entry_available("TL", (next_node, target_eject_node_id), "T1")
                    can_use_T2 = self._entry_available("TL", (next_node, target_eject_node_id), "T2")

                    if flit.ETag_priority in ["T1", "T2"]:
                        if len(link_station) < self.config.RB_IN_FIFO_DEPTH and (
                            (flit.ETag_priority == "T1" and can_use_T1)
                            or (flit.ETag_priority == "T2" and can_use_T2)
                            or (flit.ETag_priority == "T1" and not can_use_T1 and can_use_T2)  # T1使用T2 entry
                        ):
                            flit.is_delay = False
                            flit.current_link = (next_node, target_eject_node_id)
                            link[flit.current_seat_index] = None
                            flit.current_seat_index = 0

                            if flit.ETag_priority == "T2":
                                self._occupy_entry("TL", (next_node, target_eject_node_id), "T2", flit)
                            elif flit.ETag_priority == "T1":
                                if can_use_T1:
                                    # T1使用T1 entry
                                    self._occupy_entry("TL", (next_node, target_eject_node_id), "T1", flit)
                                else:
                                    # T1使用T2 entry
                                    self._occupy_entry("TL", (next_node, target_eject_node_id), "T2", flit)

                        else:
                            if flit.ETag_priority == "T2":
                                flit.ETag_priority = "T1"
                            elif flit.ETag_priority == "T1":
                                self.T0_Etag_Order_FIFO.append((next_node, flit))
                                flit.ETag_priority = "T0"
                            link[flit.current_seat_index] = None
                            next_pos = max(next_node - 1, row_start)
                            flit.current_link = (next_node, next_pos)
                            flit.current_seat_index = 0

                    elif flit.ETag_priority == "T0":
                        if len(link_station) < self.config.RB_IN_FIFO_DEPTH:
                            # 按优先级尝试: T0专用 > T1 > T2
                            if self.T0_Etag_Order_FIFO[0] == (next_node, flit) and can_use_T0:
                                # 使用T0专用entry
                                self._occupy_entry("TL", (next_node, target_eject_node_id), "T0", flit)
                                flit.is_delay = False
                                flit.current_link = (next_node, target_eject_node_id)
                                link[flit.current_seat_index] = None
                                flit.current_seat_index = 0
                                self.T0_Etag_Order_FIFO.popleft()
                            elif can_use_T1:
                                # 使用T1 entry
                                self._occupy_entry("TL", (next_node, target_eject_node_id), "T1", flit)
                                self.T0_Etag_Order_FIFO.remove((next_node, flit))
                                flit.is_delay = False
                                flit.current_link = (next_node, target_eject_node_id)
                                link[flit.current_seat_index] = None
                                flit.current_seat_index = 0
                            elif can_use_T2:
                                # 使用T2 entry
                                self._occupy_entry("TL", (next_node, target_eject_node_id), "T2", flit)
                                self.T0_Etag_Order_FIFO.remove((next_node, flit))
                                flit.is_delay = False
                                flit.current_link = (next_node, target_eject_node_id)
                                link[flit.current_seat_index] = None
                                flit.current_seat_index = 0
                            else:
                                # 无法下环，继续绕环
                                link[flit.current_seat_index] = None
                                next_pos = max(next_node - 1, row_start)
                                flit.current_link = (next_node, next_pos)
                                flit.current_seat_index = 0
                        else:
                            # link_station满，无法下环
                            link[flit.current_seat_index] = None
                            next_pos = max(next_node - 1, row_start)
                            flit.current_link = (next_node, next_pos)
                            flit.current_seat_index = 0
                else:
                    # 横向环TR尝试下环
                    link_station = self.ring_bridge["TR"].get((next_node, target_eject_node_id))
                    can_use_T1 = self._entry_available("TR", (next_node, target_eject_node_id), "T1")
                    can_use_T2 = self._entry_available("TR", (next_node, target_eject_node_id), "T2")

                    if len(link_station) < self.config.RB_IN_FIFO_DEPTH and (
                        (flit.ETag_priority == "T1" and can_use_T1) or (flit.ETag_priority == "T2" and can_use_T2) or (flit.ETag_priority == "T0" and (can_use_T1 or can_use_T2))
                    ):
                        flit.is_delay = False
                        flit.current_link = (next_node, target_eject_node_id)
                        link[flit.current_seat_index] = None
                        flit.current_seat_index = 1

                        # 根据使用的entry类型更新计数器
                        if flit.ETag_priority == "T0":
                            self.T0_Etag_Order_FIFO.remove((next_node, flit))
                            if can_use_T1:
                                # T0使用T1 entry
                                self._occupy_entry("TR", (next_node, target_eject_node_id), "T1", flit)
                            else:
                                # T0使用T2 entry
                                self._occupy_entry("TR", (next_node, target_eject_node_id), "T2", flit)
                        elif flit.ETag_priority == "T2":
                            self._occupy_entry("TR", (next_node, target_eject_node_id), "T2", flit)
                        elif flit.ETag_priority == "T1":
                            if can_use_T1:
                                # T1使用T1 entry
                                self._occupy_entry("TR", (next_node, target_eject_node_id), "T1", flit)
                            else:
                                # T1使用T2 entry
                                self._occupy_entry("TR", (next_node, target_eject_node_id), "T2", flit)
                    else:
                        link[flit.current_seat_index] = None
                        next_pos = min(next_node + 1, row_end)
                        flit.current_link = (next_node, next_pos)
                        flit.current_seat_index = 0
                        if self.ETag_BOTHSIDE_UPGRADE and flit.ETag_priority == "T2":
                            flit.ETag_priority = "T1"
            else:
                link[flit.current_seat_index] = None
                if current - next_node == 1:
                    next_pos = max(next_node - 1, row_start)
                else:
                    next_pos = min(next_node + 1, row_end)
                flit.current_link = (next_node, next_pos)
                flit.current_seat_index = 0
        # C. 非边界纵向环情况
        else:
            if next_node == flit.current_position:
                flit.circuits_completed_v += 1
                if current - next_node == self.config.NUM_COL * 2:
                    link_eject = self.eject_queues["TU"][next_node]
                    can_use_T0 = self._entry_available("TU", next_node, "T0")
                    can_use_T1 = self._entry_available("TU", next_node, "T1")
                    can_use_T2 = self._entry_available("TU", next_node, "T2")
                    if flit.ETag_priority in ["T1", "T2"]:
                        # up move
                        if len(link_eject) < self.config.EQ_IN_FIFO_DEPTH and (
                            (flit.ETag_priority == "T1" and can_use_T1)
                            or (flit.ETag_priority == "T2" and can_use_T2)
                            or (flit.ETag_priority == "T1" and not can_use_T1 and can_use_T2)  # T1使用T2 entry
                        ):
                            flit.is_delay = False
                            flit.is_arrive = True
                            link[flit.current_seat_index] = None
                            flit.current_seat_index = 0
                            if flit.ETag_priority == "T2":
                                self._occupy_entry("TU", next_node, "T2", flit)
                            elif flit.ETag_priority == "T1":
                                if can_use_T1:
                                    # T1使用T1 entry
                                    self._occupy_entry("TU", next_node, "T1", flit)
                                else:
                                    # T1使用T2 entry
                                    self._occupy_entry("TU", next_node, "T2", flit)
                        else:
                            if flit.ETag_priority == "T2":
                                flit.ETag_priority = "T1"
                            elif flit.ETag_priority == "T1":
                                self.T0_Etag_Order_FIFO.append((next_node, flit))
                                flit.ETag_priority = "T0"
                            link[flit.current_seat_index] = None
                            next_pos = next_node - self.config.NUM_COL * 2 if next_node - self.config.NUM_COL * 2 >= col_start else col_start
                            flit.current_link = (next_node, next_pos)
                            flit.current_seat_index = 0
                    elif flit.ETag_priority == "T0":
                        if len(link_eject) < self.config.EQ_IN_FIFO_DEPTH:
                            # 按优先级尝试: T0专用 > T1 > T2
                            if self.T0_Etag_Order_FIFO[0] == (next_node, flit) and can_use_T0:
                                # 使用T0专用entry
                                self._occupy_entry("TU", next_node, "T0", flit)
                                flit.is_delay = False
                                flit.is_arrive = True
                                link[flit.current_seat_index] = None
                                flit.current_seat_index = 0
                                self.T0_Etag_Order_FIFO.popleft()
                            elif can_use_T1:
                                # 使用T1 entry
                                self._occupy_entry("TU", next_node, "T1", flit)
                                self.T0_Etag_Order_FIFO.remove((next_node, flit))
                                flit.is_delay = False
                                flit.is_arrive = True
                                link[flit.current_seat_index] = None
                                flit.current_seat_index = 0
                            elif can_use_T2:
                                # 使用T2 entry
                                self._occupy_entry("TU", next_node, "T2", flit)
                                self.T0_Etag_Order_FIFO.remove((next_node, flit))
                                flit.is_delay = False
                                flit.is_arrive = True
                                link[flit.current_seat_index] = None
                                flit.current_seat_index = 0
                            else:
                                # 无法下环，继续绕环
                                link[flit.current_seat_index] = None
                                next_pos = max(next_node - self.config.NUM_COL * 2, col_start)
                                flit.current_link = (next_node, next_pos)
                                flit.current_seat_index = 0
                        else:
                            # link_eject满，无法下环
                            link[flit.current_seat_index] = None
                            next_pos = max(next_node - self.config.NUM_COL * 2, col_start)
                            flit.current_link = (next_node, next_pos)
                            flit.current_seat_index = 0
                else:
                    # down move
                    link_eject = self.eject_queues["TD"][next_node]
                    can_use_T1 = self._entry_available("TD", next_node, "T1")
                    can_use_T2 = self._entry_available("TD", next_node, "T2")

                    if len(link_eject) < self.config.EQ_IN_FIFO_DEPTH and (
                        (flit.ETag_priority == "T1" and can_use_T1) or (flit.ETag_priority == "T2" and can_use_T2) or (flit.ETag_priority == "T0" and (can_use_T1 or can_use_T2))
                    ):
                        flit.is_delay = False
                        flit.is_arrive = True
                        link[flit.current_seat_index] = None
                        flit.current_seat_index = 0
                        # 根据使用的entry类型更新计数器
                        if flit.ETag_priority == "T0":
                            self.T0_Etag_Order_FIFO.remove((next_node, flit))
                            if can_use_T1:
                                # T0使用T1 entry
                                self._occupy_entry("TD", next_node, "T1", flit)
                            else:
                                # T0使用T2 entry
                                self._occupy_entry("TD", next_node, "T2", flit)
                        elif flit.ETag_priority == "T2":
                            self.EQ_UE_Counters["TD"][next_node]["T2"] += 1
                        elif flit.ETag_priority == "T1":
                            if can_use_T1:
                                # T1使用T1 entry
                                self._occupy_entry("TD", next_node, "T1", flit)
                            else:
                                # T1使用T2 entry
                                self._occupy_entry("TD", next_node, "T2", flit)
                    else:
                        link[flit.current_seat_index] = None
                        next_pos = min(next_node + self.config.NUM_COL * 2, col_end)
                        flit.current_link = (next_node, next_pos)
                        flit.current_seat_index = 0
            else:
                link[flit.current_seat_index] = None
                if current - next_node == self.config.NUM_COL * 2:
                    next_pos = max(next_node - self.config.NUM_COL * 2, col_start)
                else:
                    next_pos = min(next_node + self.config.NUM_COL * 2, col_end)
                flit.current_link = (next_node, next_pos)
                flit.current_seat_index = 0
        return

    def _handle_regular_flit(self, flit: Flit, link, current, next_node, row_start, row_end, col_start, col_end):
        # 1. 非链路末端: 在当前链路上前进一步
        if flit.current_seat_index < len(link) - 1:
            link[flit.current_seat_index] = None
            flit.current_seat_index += 1
            return
        # self.error_log(flit, 7, -1)

        # 2. 已经到达
        flit.current_position = next_node
        flit.path_index += 1
        # 检查是否还有后续路径
        if flit.path_index + 1 < len(flit.path):
            new_current, new_next_node = next_node, flit.path[flit.path_index + 1]

            # A. 处理横边界情况（非自环）
            if current == next_node and new_next_node != new_current:
                # 这里可以添加特殊处理逻辑
                pass

            # 2a. 正常绕环
            if new_current - new_next_node != self.config.NUM_COL:
                flit.current_link = (new_current, new_next_node)
                link[flit.current_seat_index] = None
                flit.current_seat_index = 0

            # 2b. 横向环向左进入Ring Bridge
            elif current - next_node == 1:
                station = self.ring_bridge["TL"].get((new_current, new_next_node))
                # TL有空位
                if self.config.RB_IN_FIFO_DEPTH > len(station) and self.RB_UE_Counters["TL"].get((new_current, new_next_node))["T2"] < self.config.TL_Etag_T2_UE_MAX:
                    flit.current_link = (new_current, new_next_node)
                    link[flit.current_seat_index] = None
                    flit.current_seat_index = 0
                    # 更新计数器
                    self._occupy_entry("TL", (new_current, new_next_node), "T2", flit)
                else:
                    # TL无空位: 预留到右侧等待队列，设置延迟标志，ETag升级
                    flit.ETag_priority = "T1"
                    next_pos = next_node - 1 if next_node - 1 >= row_start else row_start
                    flit.is_delay = True
                    link[flit.current_seat_index] = None
                    flit.current_link = (new_current, next_pos)
                    flit.current_seat_index = 0

            # 2c. 横向环向右进入Ring Bridge
            elif current - next_node == -1:
                station = self.ring_bridge["TR"].get((new_current, new_next_node))
                if self.config.RB_IN_FIFO_DEPTH > len(station) and self.RB_UE_Counters["TR"].get((new_current, new_next_node))["T2"] < self.config.TR_Etag_T2_UE_MAX:
                    flit.current_link = (new_current, new_next_node)
                    link[flit.current_seat_index] = None
                    flit.current_seat_index = 1
                    self._occupy_entry("TR", (new_current, new_next_node), "T2", flit)
                else:
                    # TR无空位: 设置延迟标志，如果双边ETag升级，则升级ETag。
                    if self.ETag_BOTHSIDE_UPGRADE:
                        flit.ETag_priority = "T1"
                    next_pos = next_node + 1 if next_node + 1 <= row_end else row_end
                    flit.is_delay = True
                    link[flit.current_seat_index] = None
                    flit.current_link = (new_current, next_pos)
                    flit.current_seat_index = 0
        else:
            # 3. 已经到达目的地，执行eject逻辑
            if current - next_node == self.config.NUM_COL * 2:  # 纵向环向上TU
                eject_queue = self.eject_queues["TU"][next_node]
                if self.config.EQ_IN_FIFO_DEPTH > len(eject_queue) and self.EQ_UE_Counters["TU"][next_node]["T2"] < self.config.TU_Etag_T2_UE_MAX:
                    link[flit.current_seat_index] = None
                    flit.current_seat_index = 0
                    flit.is_arrive = True
                    self._occupy_entry("TU", next_node, "T2", flit)
                else:
                    flit.ETag_priority = "T1"
                    next_pos = next_node - self.config.NUM_COL * 2 if next_node - self.config.NUM_COL * 2 >= col_start else col_start
                    flit.is_delay = True
                    link[flit.current_seat_index] = None
                    flit.current_link = (next_node, next_pos)
                    flit.current_seat_index = 0
            elif current - next_node == -self.config.NUM_COL * 2:  # 纵向环向下TD
                eject_queue = self.eject_queues["TD"][next_node]
                if self.config.EQ_IN_FIFO_DEPTH > len(eject_queue) and self.EQ_UE_Counters["TD"][next_node]["T2"] < self.config.TD_Etag_T2_UE_MAX:
                    link[flit.current_seat_index] = None
                    flit.current_seat_index = 0
                    flit.is_arrive = True
                    self._occupy_entry("TD", next_node, "T2", flit)
                else:
                    if self.ETag_BOTHSIDE_UPGRADE:
                        flit.ETag_priority = "T1"
                    # TODO: next_pos 名称不好
                    next_pos = next_node + self.config.NUM_COL * 2 if next_node + self.config.NUM_COL * 2 <= col_end else col_end
                    flit.is_delay = True
                    link[flit.current_seat_index] = None
                    flit.current_link = (next_node, next_pos)
                    flit.current_seat_index = 0

    def execute_moves(self, flit: Flit, cycle):
        if not flit.is_arrive:
            current, next_node = flit.current_link
            if current - next_node != self.config.NUM_COL:
                link = self.links.get(flit.current_link)
                self.set_link_slice(flit.current_link, flit.current_seat_index, flit, cycle)
                # link[flit.current_seat_index] = flit
                if (flit.current_seat_index == len(link) - 1 and len(link) > 2) or (flit.current_seat_index == 1 and len(link) == 2):
                    self.links_flow_stat[flit.req_type][flit.current_link] += 1
            else:
                # 将 flit 放入 ring_bridge 的相应方向
                if not flit.is_on_station:
                    # 使用字典映射 seat_index 到 ring_bridge 的方向和深度限制
                    direction, max_depth = self.ring_bridge_map.get(flit.current_seat_index, (None, None))
                    if direction is None:
                        return False
                    if direction in self.ring_bridge.keys() and len(self.ring_bridge[direction][flit.current_link]) < max_depth and self.ring_bridge_pre[direction][flit.current_link] is None:
                        # flit.flit_position = f"RB_{direction}"
                        # self.ring_bridge[direction][flit.current_link].append(flit)
                        self.ring_bridge_pre[direction][flit.current_link] = flit
                        flit.is_on_station = True
            return False
        else:
            if flit.current_link is not None:
                current, next_node = flit.current_link
            flit.arrival_network_cycle = cycle

            if flit.source - flit.destination == self.config.NUM_COL:
                flit.flit_position = f"IQ_EQ"
                flit.is_arrived = True

                return True
            elif current - next_node == self.config.NUM_COL * 2 or (current == next_node and current not in range(0, self.config.NUM_COL)):
                direction = "TU"
                queue = self.eject_queues["TU"]
                queue_pre = self.eject_queues_in_pre["TU"]
            else:
                direction = "TD"
                queue = self.eject_queues["TD"]
                queue_pre = self.eject_queues_in_pre["TD"]

            # flit.flit_position = f"EQ_{direction}"
            # queue[next_node].append(flit)
            if queue_pre[next_node]:
                return False
            else:
                queue_pre[next_node] = flit
                flit.itag_v = False
                return True

    @property
    def all_ip_positions(self):
        """Cached property for all IP positions"""
        if self._all_ip_positions is None:
            self._all_ip_positions = list(set(
                self.config.GDMA_SEND_POSITION_LIST + 
                self.config.SDMA_SEND_POSITION_LIST + 
                self.config.CDMA_SEND_POSITION_LIST + 
                self.config.DDR_SEND_POSITION_LIST + 
                self.config.L2M_SEND_POSITION_LIST
            ))
        return self._all_ip_positions

    @property
    def rn_positions(self):
        """Cached property for RN positions"""
        if self._rn_positions is None:
            self._rn_positions = list(set(
                self.config.GDMA_SEND_POSITION_LIST + 
                self.config.SDMA_SEND_POSITION_LIST + 
                self.config.CDMA_SEND_POSITION_LIST
            ))
        return self._rn_positions

    @property
    def sn_positions(self):
        """Cached property for SN positions"""
        if self._sn_positions is None:
            self._sn_positions = list(set(
                self.config.DDR_SEND_POSITION_LIST + 
                self.config.L2M_SEND_POSITION_LIST
            ))
        return self._sn_positions
    
    def clear_position_cache(self):
        """Clear position cache when network configuration changes"""
        self._all_ip_positions = None
        self._rn_positions = None
        self._sn_positions = None