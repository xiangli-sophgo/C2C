"""
Node class for NoC simulation.
Handles network node functionality including RN/SN trackers and databases.
"""
from __future__ import annotations
from collections import defaultdict
from config.config import CrossRingConfig


class Node:
    global_packet_id = -1

    def __init__(self, config: CrossRingConfig):
        self.config = config
        self.initialize_data_structures()
        self.initialize_rn()
        self.initialize_sn()

    @classmethod
    def get_next_packet_id(cls):
        cls.global_packet_id += 1
        return cls.global_packet_id

    @classmethod
    def clear_packet_id(cls):
        cls.global_packet_id = -1

    def initialize_data_structures(self):
        """Initialize the data structures for read and write databases."""

        self.rn_rdb = self.config._make_channels(("sdma", "gdma", "cdma"))
        self.rn_rdb_reserve = self.config._make_channels(("sdma", "gdma", "cdma"))
        self.rn_rdb_recv = self.config._make_channels(("sdma", "gdma", "cdma"))
        self.rn_rdb_count = self.config._make_channels(("sdma", "gdma", "cdma"))
        self.rn_wdb = self.config._make_channels(("sdma", "gdma", "cdma"))
        self.rn_wdb_reserve = self.config._make_channels(("sdma", "gdma", "cdma"))
        self.rn_wdb_count = self.config._make_channels(("sdma", "gdma", "cdma"))
        self.rn_wdb_send = self.config._make_channels(("sdma", "gdma", "cdma"))
        self.rn_tracker = {"read": self.config._make_channels(("sdma", "gdma", "cdma")), "write": self.config._make_channels(("sdma", "gdma", "cdma"))}
        self.rn_tracker_wait = {"read": self.config._make_channels(("sdma", "gdma", "cdma")), "write": self.config._make_channels(("sdma", "gdma", "cdma"))}
        self.rn_tracker_count = {"read": self.config._make_channels(("sdma", "gdma", "cdma")), "write": self.config._make_channels(("sdma", "gdma", "cdma"))}
        self.rn_tracker_pointer = {"read": self.config._make_channels(("sdma", "gdma", "cdma")), "write": self.config._make_channels(("sdma", "gdma", "cdma"))}
        self.sn_rdb = self.config._make_channels(("ddr", "l2m"))
        self.sn_rsp_queue = self.config._make_channels(("ddr", "l2m"))
        self.sn_req_wait = {"read": self.config._make_channels(("ddr", "l2m")), "write": self.config._make_channels(("ddr", "l2m"))}
        self.sn_tracker = self.config._make_channels(("ddr", "l2m"))
        self.sn_tracker_count = self.config._make_channels(("ddr", "l2m"), value_factory={"ro": {}, "share": {}})
        self.sn_wdb = self.config._make_channels(("ddr", "l2m"))
        self.sn_wdb_recv = self.config._make_channels(("ddr", "l2m"))
        self.sn_wdb_count = self.config._make_channels(("ddr", "l2m"))
        self.rn_wait_to_inject = []

    def initialize_rn(self):
        """Initialize RN structures."""
        # 针对 Ring 拓扑，所有节点都有 RN tracker
        is_ring_topology = getattr(self.config, "TOPO_TYPE", "").startswith("Ring")
        if is_ring_topology:
            for ip_type in self.rn_rdb.keys():
                for ip_pos in range(self.config.NUM_NODE):
                    self.rn_rdb[ip_type][ip_pos] = defaultdict(list)
                    self.rn_wdb[ip_type][ip_pos] = defaultdict(list)
                    self.setup_rn_trackers(ip_type, ip_pos)
        else:
            # 原始 CrossRing 拓扑：只在特定位置创建 RN tracker
            for ip_type in self.rn_rdb.keys():
                positions = getattr(self.config, f"{ip_type[:-2].upper()}_SEND_POSITION_LIST")
                for ip_pos in positions:
                    self.rn_rdb[ip_type][ip_pos] = defaultdict(list)
                    self.rn_wdb[ip_type][ip_pos] = defaultdict(list)
                    self.setup_rn_trackers(ip_type, ip_pos)

    def setup_rn_trackers(self, ip_type, ip_pos):
        """Setup read and write trackers for RN."""
        self.rn_rdb_recv[ip_type][ip_pos] = []
        self.rn_rdb_count[ip_type][ip_pos] = self.config.RN_RDB_SIZE
        self.rn_rdb_reserve[ip_type][ip_pos] = 0
        self.rn_wdb_count[ip_type][ip_pos] = self.config.RN_WDB_SIZE
        self.rn_wdb_send[ip_type][ip_pos] = []
        self.rn_wdb_reserve[ip_type][ip_pos] = 0

        for req_type in ["read", "write"]:
            self.rn_tracker[req_type][ip_type][ip_pos] = []
            self.rn_tracker_wait[req_type][ip_type][ip_pos] = []
            self.rn_tracker_count[req_type][ip_type][ip_pos] = self.config.RN_R_TRACKER_OSTD if req_type == "read" else self.config.RN_W_TRACKER_OSTD
            self.rn_tracker_pointer[req_type][ip_type][ip_pos] = -1

    def initialize_sn(self):
        """Initialize SN structures."""
        self.sn_tracker_release_time = defaultdict(list)

        # 检查是否为Ring拓扑 - 通过TOPO_TYPE字段判断
        is_ring_topology = getattr(self.config, "TOPO_TYPE", "").startswith("Ring")

        if is_ring_topology:
            # Ring拓扑：每个节点都有所有IP类型的SN tracker
            for ip_pos in range(getattr(self.config, "RING_NUM_NODE", self.config.NUM_NODE)):
                for key in self.sn_tracker:
                    self.sn_rdb[key][ip_pos] = []
                    self.sn_wdb[key][ip_pos] = defaultdict(list)
                    self.setup_sn_trackers(key, ip_pos)
        else:
            # 原始CrossRing拓扑：只在特定位置创建SN tracker
            for ip_pos in set(self.config.DDR_SEND_POSITION_LIST + self.config.L2M_SEND_POSITION_LIST):
                for key in self.sn_tracker:
                    self.sn_rdb[key][ip_pos] = []
                    self.sn_wdb[key][ip_pos] = defaultdict(list)
                    self.setup_sn_trackers(key, ip_pos)

    def setup_sn_trackers(self, key, ip_pos):
        """Setup trackers for SN."""
        self.sn_rsp_queue[key][ip_pos] = []
        for req_type in ["read", "write"]:
            self.sn_req_wait[req_type][key][ip_pos] = []
        self.sn_wdb_recv[key][ip_pos] = []
        self.sn_tracker[key][ip_pos] = []
        # if self.config.topo_type != "3x3":
        if key.startswith("ddr"):
            self.sn_wdb_count[key][ip_pos] = self.config.SN_DDR_WDB_SIZE
            self.sn_tracker_count[key]["ro"][ip_pos] = self.config.SN_DDR_R_TRACKER_OSTD
            self.sn_tracker_count[key]["share"][ip_pos] = self.config.SN_DDR_W_TRACKER_OSTD
        elif key.startswith("l2m"):
            self.sn_wdb_count[key][ip_pos] = self.config.SN_L2M_WDB_SIZE
            self.sn_tracker_count[key]["ro"][ip_pos] = self.config.SN_L2M_R_TRACKER_OSTD
            self.sn_tracker_count[key]["share"][ip_pos] = self.config.SN_L2M_W_TRACKER_OSTD