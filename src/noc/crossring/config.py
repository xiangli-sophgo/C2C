"""
CrossRing NoC专用配置类。

本模块提供专门针对CrossRing拓扑的配置实现，
替换现有的CrossRingConfig类，同时保持向后兼容性。
"""

from typing import Dict, Any, List, Optional, Tuple
import json
from dataclasses import dataclass, field

from src.noc.base.config import BaseNoCConfig
from src.noc.types import TopologyType, ValidationResult


@dataclass
class BasicConfiguration:
    """拓扑基础配置"""

    burst: int = 4
    network_frequency: int = 2.0
    slice_per_link: int = 8


@dataclass
class IPConfiguration:
    """IP配置数据类。"""

    gdma_count: int = 16
    sdma_count: int = 16
    cdma_count: int = 16
    ddr_count: int = 16
    l2m_count: int = 16

    # 带宽限制 (GB/s)
    gdma_bw_limit: float = 128.0
    sdma_bw_limit: float = 128.0
    cdma_bw_limit: float = 32.0
    ddr_bw_limit: float = 128.0
    l2m_bw_limit: float = 128.0

    # 读写间隔配置
    gdma_rw_gap: int = 1e9
    sdma_rw_gap: int = 1e9


@dataclass
class FIFOConfiguration:
    """FIFO缓冲区配置数据类。"""

    rb_in_depth: int = 16
    rb_out_depth: int = 8
    iq_out_depth: int = 8
    eq_in_depth: int = 16
    iq_ch_depth: int = 10
    eq_ch_depth: int = 10


@dataclass
class TagConfiguration:
    """Tag配置数据类。"""

    # ITag 配置
    itag_trigger_th_h: int = 80
    itag_trigger_th_v: int = 80
    itag_max_num_h: int = 1
    itag_max_num_v: int = 1

    # ETag 配置
    etag_bothside_upgrade: int = 0
    tl_etag_t1_ue_max: int = 15
    tl_etag_t2_ue_max: int = 8
    tr_etag_t2_ue_max: int = 12
    tu_etag_t1_ue_max: int = 15
    tu_etag_t2_ue_max: int = 8
    td_etag_t2_ue_max: int = 12


@dataclass
class TrackerConfiguration:
    """Tracker 配置数据类。"""

    rn_r_tracker_ostd: int = 64
    rn_w_tracker_ostd: int = 32
    sn_ddr_r_tracker_ostd: int = 96
    sn_ddr_w_tracker_ostd: int = 48
    sn_l2m_r_tracker_ostd: int = 96
    sn_l2m_w_tracker_ostd: int = 48
    sn_tracker_release_latency: int = 40


@dataclass
class LatencyConfiguration:
    """延迟配置数据类。"""

    ddr_r_latency: int = 100
    ddr_r_latency_var: int = 0
    ddr_w_latency: int = 0
    ddr_w_latency_var: int = 0
    l2m_r_latency: int = 12
    l2m_r_latency_var: int = 0
    l2m_w_latency: int = 16
    l2m_w_latency_var: int = 0


class CrossRingConfig(BaseNoCConfig):
    """
    CrossRing拓扑专用配置类。

    该类扩展BaseNoCConfig，提供专门针对CrossRing拓扑的配置管理，
    包括拓扑参数、IP配置、Tag、缓冲区等各种CrossRing特有的参数。
    """

    def __init__(self, num_col: int = 2, num_row: int = 4, config_name: str = "default"):
        """
        初始化CrossRing配置。

        Args:
            num_col: 列数
            num_row: 行数
            config_name: 配置名称
        """
        super().__init__(TopologyType.CROSSRING)

        # 基本拓扑参数
        self.config_name = config_name
        self.num_col = num_col
        self.num_row = num_row
        self.num_nodes = num_col * num_row
        self.num_ip = num_col * num_row
        self.num_rn = self.num_ip
        self.num_sn = self.num_ip

        # 使用组合配置
        self.basic_config = BasicConfiguration()
        self.ip_config = IPConfiguration()
        self.fifo_config = FIFOConfiguration()
        self.tag_config = TagConfiguration()
        self.tracker_config = TrackerConfiguration()
        self.latency_config = LatencyConfiguration()

        # 通道规格
        self.channel_spec = {
            "gdma": 2,
            "sdma": 2,
            "cdma": 2,
            "ddr": 2,
            "l2m": 2,
        }

        # 自动生成相关配置
        self._generate_derived_config()
        self._generate_ip_positions()

    def _generate_derived_config(self) -> None:
        """生成派生配置参数。"""
        if not hasattr(self.basic_config, "burst"):
            self.basic_config = BasicConfiguration()
        # 安全地获取tracker配置
        if hasattr(self.tracker_config, "rn_r_tracker_ostd"):
            # 计算缓冲区大小
            self.rn_rdb_size = self.tracker_config.rn_r_tracker_ostd * self.basic_config.burst
            self.rn_wdb_size = self.tracker_config.rn_w_tracker_ostd * self.basic_config.burst
            self.sn_ddr_rdb_size = self.tracker_config.sn_ddr_r_tracker_ostd * self.basic_config.burst
            self.sn_ddr_wdb_size = self.tracker_config.sn_ddr_w_tracker_ostd * self.basic_config.burst
            self.sn_l2m_rdb_size = self.tracker_config.sn_l2m_r_tracker_ostd * self.basic_config.burst
            self.sn_l2m_wdb_size = self.tracker_config.sn_l2m_w_tracker_ostd * self.basic_config.burst
        else:
            # 如果tracker_config是字典或未初始化，使用默认值
            tracker = TrackerConfiguration()
            self.rn_rdb_size = tracker.rn_r_tracker_ostd * self.basic_config.burst
            self.rn_wdb_size = tracker.rn_w_tracker_ostd * self.basic_config.burst
            self.sn_ddr_rdb_size = tracker.sn_ddr_r_tracker_ostd * self.basic_config.burst
            self.sn_ddr_wdb_size = tracker.sn_ddr_w_tracker_ostd * self.basic_config.burst
            self.sn_l2m_rdb_size = tracker.sn_l2m_r_tracker_ostd * self.basic_config.burst
            self.sn_l2m_wdb_size = tracker.sn_l2m_w_tracker_ostd * self.basic_config.burst

        # 生成通道名称列表
        self.ch_name_list = []
        for key in self.channel_spec:
            for idx in range(self.channel_spec[key]):
                self.ch_name_list.append(f"{key}_{idx}")

    def _generate_ip_positions(self) -> None:
        """生成IP位置列表。"""
        # 默认IP分布策略：每个节点都会挂所有的IP
        ip_positions = [i for i in range(self.num_nodes)]

        # 各类IP使用相同的位置列表
        self.ddr_send_position_list = ip_positions.copy()
        self.l2m_send_position_list = ip_positions.copy()
        self.gdma_send_position_list = ip_positions.copy()
        self.sdma_send_position_list = ip_positions.copy()
        self.cdma_send_position_list = ip_positions.copy()

    def validate_config(self) -> ValidationResult:
        """
        验证CrossRing配置参数。

        Returns:
            ValidationResult: (是否有效, 错误信息)
        """
        # 先验证基础参数
        basic_valid, basic_error = self.validate_basic_params()
        if not basic_valid:
            return basic_valid, basic_error

        errors = []

        # 拓扑参数验证
        if self.num_nodes != self.num_row * self.num_col:
            errors.append(f"节点数必须等于行数×列数 (num_nodes={self.num_nodes}, num_row={self.num_row}, num_col={self.num_col})")

        if self.num_col < 2 or self.num_row < 2:
            errors.append(f"CrossRing拓扑至少需要2×2节点 (num_row={self.num_row}, num_col={self.num_col})")

        # FIFO深度验证 - 安全地访问属性
        fifo_cfg = self.fifo_config
        if hasattr(fifo_cfg, "rb_in_depth"):
            if fifo_cfg.rb_in_depth <= 0:
                errors.append(f"RB输入FIFO深度必须为正数 (rb_in_depth={fifo_cfg.rb_in_depth})")
            if fifo_cfg.eq_in_depth <= 0:
                errors.append(f"EQ输入FIFO深度必须为正数 (eq_in_depth={fifo_cfg.eq_in_depth})")
        elif isinstance(fifo_cfg, dict):
            if fifo_cfg.get("rb_in_depth", 0) <= 0:
                errors.append(f"RB输入FIFO深度必须为正数 (rb_in_depth={fifo_cfg.get('rb_in_depth', 0)})")
            if fifo_cfg.get("eq_in_depth", 0) <= 0:
                errors.append(f"EQ输入FIFO深度必须为正数 (eq_in_depth={fifo_cfg.get('eq_in_depth', 0)})")

        # Tag参数验证 - 放宽约束
        tag_cfg = self.tag_config

        # 安全地获取Tag参数
        if hasattr(tag_cfg, "tl_etag_t2_ue_max"):
            tl_t2 = tag_cfg.tl_etag_t2_ue_max
            tl_t1 = tag_cfg.tl_etag_t1_ue_max
            tr_t2 = tag_cfg.tr_etag_t2_ue_max
            tu_t2 = tag_cfg.tu_etag_t2_ue_max
            tu_t1 = tag_cfg.tu_etag_t1_ue_max
            td_t2 = tag_cfg.td_etag_t2_ue_max
        elif isinstance(tag_cfg, dict):
            tr_t2 = tag_cfg.get("tr_etag_t2_ue_max", 0)
            tl_t2 = tag_cfg.get("tl_etag_t2_ue_max", 0)
            tl_t1 = tag_cfg.get("tl_etag_t1_ue_max", 0)
            td_t2 = tag_cfg.get("td_etag_t2_ue_max", 0)
            tu_t2 = tag_cfg.get("tu_etag_t2_ue_max", 0)
            tu_t1 = tag_cfg.get("tu_etag_t1_ue_max", 0)
        else:
            tl_t2 = tl_t1 = tr_t2 = td_t2 = tu_t2 = tu_t1 = 0

        # 安全地获取FIFO深度
        if hasattr(fifo_cfg, "rb_in_depth"):
            rb_depth = fifo_cfg.rb_in_depth
            eq_depth = fifo_cfg.eq_in_depth
        elif isinstance(fifo_cfg, dict):
            rb_depth = fifo_cfg.get("rb_in_depth", 16)
            eq_depth = fifo_cfg.get("eq_in_depth", 16)
        else:
            rb_depth = eq_depth = 8

        if tl_t2 <= 0:
            errors.append(f"TL ETag T2必须为正数 (tl_etag_t2_ue_max={tl_t2})")
        if tl_t1 <= tl_t2:
            errors.append(f"TL ETag T1必须大于T2 (tl_etag_t1_ue_max={tl_t1}, tl_etag_t2_ue_max={tl_t2})")
        if tl_t1 >= rb_depth:
            errors.append(f"TL ETag T1必须小于RB_IN_FIFO_DEPTH (tl_etag_t1_ue_max={tl_t1}, rb_in_depth={rb_depth})")
        if tr_t2 >= rb_depth:
            errors.append(f"TR ETag T2必须小于RB_IN_FIFO_DEPTH (tr_etag_t2_ue_max={tr_t2}, rb_in_depth={rb_depth})")

        if tu_t2 <= 0:
            errors.append(f"TU ETag T2必须为正数 (tu_etag_t2_ue_max={tu_t2})")
        if tu_t1 <= tu_t2:
            errors.append(f"TU ETag T1必须大于T2 (tu_etag_t1_ue_max={tu_t1}, tu_etag_t2_ue_max={tu_t2})")
        if tu_t1 >= eq_depth:
            errors.append(f"TU ETag T1必须小于EQ_IN_FIFO_DEPTH (tu_etag_t1_ue_max={tu_t1}, eq_in_depth={eq_depth})")
        if td_t2 >= eq_depth:
            errors.append(f"TD ETag T2必须小于EQ_IN_FIFO_DEPTH (td_etag_t2_ue_max={td_t2}, eq_in_depth={eq_depth})")

        #  Tracker 配置验证 - 安全地访问属性
        tracker = self.tracker_config
        if hasattr(tracker, "rn_r_tracker_ostd"):
            if tracker.rn_r_tracker_ostd <= 0 or tracker.rn_w_tracker_ostd <= 0:
                errors.append(f"RN Tracker OSTD必须为正数 (rn_r_tracker_ostd={tracker.rn_r_tracker_ostd}, rn_w_tracker_ostd={tracker.rn_w_tracker_ostd})")
            if tracker.sn_ddr_r_tracker_ostd <= 0 or tracker.sn_l2m_r_tracker_ostd <= 0:
                errors.append(f"SN Tracker OSTD必须为正数 (sn_ddr_r_tracker_ostd={tracker.sn_ddr_r_tracker_ostd}, sn_l2m_r_tracker_ostd={tracker.sn_l2m_r_tracker_ostd})")
            # 缓冲区大小一致性验证
            if hasattr(self, "rn_rdb_size") and self.rn_rdb_size != tracker.rn_r_tracker_ostd * self.basic_config.burst:
                errors.append(f"RN_RDB_SIZE必须等于RN_R_TRACKER_OSTD × BURST (rn_rdb_size={self.rn_rdb_size}, rn_r_tracker_ostd={tracker.rn_r_tracker_ostd}, burst={self.basic_config.burst})")
        elif isinstance(tracker, dict):
            rn_r_ostd = tracker.get("rn_r_tracker_ostd", 64)
            rn_w_ostd = tracker.get("rn_w_tracker_ostd", 32)
            if rn_r_ostd <= 0 or rn_w_ostd <= 0:
                errors.append(f"RN Tracker OSTD必须为正数 (rn_r_tracker_ostd={rn_r_ostd}, rn_w_tracker_ostd={rn_w_ostd})")
            # 缓冲区大小一致性验证
            if hasattr(self, "rn_rdb_size") and self.rn_rdb_size != rn_r_ostd * self.basic_config.burst:
                errors.append(f"RN_RDB_SIZE必须等于RN_R_TRACKER_OSTD × BURST (rn_rdb_size={self.rn_rdb_size}, rn_r_tracker_ostd={rn_r_ostd}, burst={self.basic_config.burst})")

        # IP配置验证 - 安全地访问属性
        ip_cfg = self.ip_config
        if hasattr(ip_cfg, "gdma_count"):
            if ip_cfg.gdma_count < 0 or ip_cfg.sdma_count < 0:
                errors.append(f"IP数量不能为负数 (gdma_count={ip_cfg.gdma_count}, sdma_count={ip_cfg.sdma_count})")
            if ip_cfg.gdma_bw_limit <= 0 or ip_cfg.ddr_bw_limit <= 0:
                errors.append(f"带宽限制必须为正数 (gdma_bw_limit={ip_cfg.gdma_bw_limit}, ddr_bw_limit={ip_cfg.ddr_bw_limit})")
        elif isinstance(ip_cfg, dict):
            gdma_count = ip_cfg.get("gdma_count", 0)
            sdma_count = ip_cfg.get("sdma_count", 0)
            if gdma_count < 0 or sdma_count < 0:
                errors.append(f"IP数量不能为负数 (gdma_count={gdma_count}, sdma_count={sdma_count})")
            gdma_bw = ip_cfg.get("gdma_bw_limit", 8.0)
            ddr_bw = ip_cfg.get("ddr_bw_limit", 80.0)
            if gdma_bw <= 0 or ddr_bw <= 0:
                errors.append(f"带宽限制必须为正数 (gdma_bw_limit={gdma_bw}, ddr_bw_limit={ddr_bw})")

        # 位置列表验证
        for pos in self.ddr_send_position_list:
            if pos >= self.num_nodes:
                errors.append(f"IP位置{pos}超出节点范围 (num_nodes={self.num_nodes})")

        if errors:
            return False, "; ".join(errors)

        return True, None

    def get_topology_params(self) -> Dict[str, Any]:
        """
        获取CrossRing拓扑参数。

        Returns:
            包含拓扑特定参数的字典
        """
        return {
            "topology_type": self.topology_type,
            "num_nodes": self.num_nodes,
            "num_col": self.num_col,
            "num_row": self.num_row,
            "num_ip": self.num_ip,
            "num_rn": self.num_rn,
            "num_sn": self.num_sn,
            "channel_spec": self.channel_spec,
            "ch_name_list": self.ch_name_list,
            "ip_positions": {
                "ddr": self.ddr_send_position_list,
                "l2m": self.l2m_send_position_list,
                "gdma": self.gdma_send_position_list,
                "sdma": self.sdma_send_position_list,
                "cdma": self.cdma_send_position_list,
            },
        }

    def update_topology_size(self, num_row: int, num_col: int) -> None:
        """
        更新拓扑大小。

        Args:
            num_col: 新的列数
            num_row: 新的行数
        """
        self.num_row = num_row
        self.num_col = num_col
        self.num_nodes = num_col * num_row

        # 重新生成相关配置
        self._generate_ip_positions()

    def update_basic_config(self, **kwargs) -> None:
        """
        更新网络基础配置。

        Args:
            **kwargs: IP配置参数
        """
        for key, value in kwargs.items():
            if hasattr(self.basic_config, key):
                setattr(self.basic_config, key, value)

    def update_ip_config(self, **kwargs) -> None:
        """
        更新IP配置。

        Args:
            **kwargs: IP配置参数
        """
        for key, value in kwargs.items():
            if hasattr(self.ip_config, key):
                setattr(self.ip_config, key, value)

    def update_fifo_config(self, **kwargs) -> None:
        """
        更新FIFO配置。

        Args:
            **kwargs: FIFO配置参数
        """
        for key, value in kwargs.items():
            if hasattr(self.fifo_config, key):
                setattr(self.fifo_config, key, value)

    def update_tag_config(self, **kwargs) -> None:
        """
        更新Tag配置。

        Args:
            **kwargs: Tag配置参数
        """
        for key, value in kwargs.items():
            if hasattr(self.tag_config, key):
                setattr(self.tag_config, key, value)

    def set_preset_configuration(self, preset: str) -> None:
        """
        设置预设配置。

        Args:
            preset: 预设配置名称 ('2260E', '2262')
        """
        if preset == "2260E":
            self.update_topology_size(num_row=3, num_col=3)
            self.num_ip = 8
            self.update_fifo_config(rb_in_depth=16, eq_in_depth=16)
            self.update_ip_config(gdma_count=4, sdma_count=4, ddr_count=16)
            self.update_tag_config(
                tl_etag_t2_ue_max=8,
                tl_etag_t1_ue_max=15,
                tr_etag_t2_ue_max=12,
                tu_etag_t2_ue_max=8,
                tu_etag_t1_ue_max=15,
                td_etag_t2_ue_max=12,
            )
        elif preset == "2262":
            self.update_topology_size(num_row=5, num_col=4)
            self.num_ip = 12
            self.update_fifo_config(rb_in_depth=16, eq_in_depth=16)
            self.update_ip_config(gdma_count=32, ddr_count=32)
            self.update_tag_config(
                tl_etag_t2_ue_max=8,
                tl_etag_t1_ue_max=15,
                tr_etag_t2_ue_max=12,
                tu_etag_t2_ue_max=8,
                tu_etag_t1_ue_max=15,
                td_etag_t2_ue_max=12,
            )
        else:
            raise ValueError(f"未知的预设配置: {preset}")

        # 重新生成派生配置
        self._generate_derived_config()
        self._generate_ip_positions()

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式。

        Returns:
            配置的字典表示
        """
        base_dict = super().to_dict()

        # 添加CrossRing特有参数
        crossring_dict = {
            "config_name": self.config_name,
            "num_col": self.num_col,
            "num_row": self.num_row,
            "num_ip": self.num_ip,
            "topo_type": self.topology_type,
            # "burst": self.basic_config.burst,
            # "network_frequency": self.network_frequency,
            # 组合配置
            "basic_config": self.basic_config.__dict__,
            "ip_config": self.ip_config.__dict__,
            "fifo_config": self.fifo_config.__dict__,
            "tag_config": self.tag_config.__dict__,
            "tracker_config": self.tracker_config.__dict__,
            "latency_config": self.latency_config.__dict__,
            # 其他参数
            "channel_spec": self.channel_spec,
            "ch_name_list": self.ch_name_list,
            "ip_positions": {
                "ddr": self.ddr_send_position_list,
                "l2m": self.l2m_send_position_list,
                "gdma": self.gdma_send_position_list,
                "sdma": self.sdma_send_position_list,
                "cdma": self.cdma_send_position_list,
            },
        }

        base_dict.update(crossring_dict)
        return base_dict

    def from_dict(self, config_dict: Dict[str, Any]) -> None:
        """
        从字典加载配置。

        Args:
            config_dict: 配置字典
        """
        # 先调用父类方法
        super().from_dict(config_dict)

        # 加载CrossRing特有参数
        if "config_name" in config_dict:
            self.config_name = config_dict["config_name"]
        if "num_col" in config_dict:
            self.num_col = config_dict["num_col"]
        if "num_row" in config_dict:
            self.num_row = config_dict["num_row"]
        if "num_ip" in config_dict:
            self.num_ip = config_dict["num_ip"]
        # if "topo_type" in config_dict:
        #     self.topo_type = config_dict["topo_type"]
        # if "burst" in config_dict:
        #     self.basic_config.burst = config_dict["burst"]
        # if "network_frequency" in config_dict:
        #     self.network_frequency = config_dict["network_frequency"]

        # 确保配置对象已初始化
        if not hasattr(self, "basic_config") or self.ip_config is None:
            self.basic_config = BasicConfiguration()
        if not hasattr(self, "ip_config") or self.ip_config is None:
            self.ip_config = IPConfiguration()
        if not hasattr(self, "fifo_config") or self.fifo_config is None:
            self.fifo_config = FIFOConfiguration()
        if not hasattr(self, "flow_control_config") or self.tag_config is None:
            self.tag_config = TagConfiguration()
        if not hasattr(self, "tracker_config") or self.tracker_config is None:
            self.tracker_config = TrackerConfiguration()
        if not hasattr(self, "latency_config") or self.latency_config is None:
            self.latency_config = LatencyConfiguration()

        # 加载组合配置 - 安全地处理可能是字典的情况
        if "basic_config" in config_dict:
            ip_cfg = config_dict["basic_config"]
            if isinstance(ip_cfg, dict):
                for key, value in ip_cfg.items():
                    if hasattr(self.basic_config, key):
                        setattr(self.basic_config, key, value)

        if "ip_config" in config_dict:
            ip_cfg = config_dict["ip_config"]
            if isinstance(ip_cfg, dict):
                for key, value in ip_cfg.items():
                    if hasattr(self.ip_config, key):
                        setattr(self.ip_config, key, value)

        if "fifo_config" in config_dict:
            fifo_cfg = config_dict["fifo_config"]
            if isinstance(fifo_cfg, dict):
                for key, value in fifo_cfg.items():
                    if hasattr(self.fifo_config, key):
                        setattr(self.fifo_config, key, value)

        if "tag_config" in config_dict:
            flow_cfg = config_dict["tag_config"]
            if isinstance(flow_cfg, dict):
                for key, value in flow_cfg.items():
                    if hasattr(self.tag_config, key):
                        setattr(self.tag_config, key, value)

        if "tracker_config" in config_dict:
            tracker_cfg = config_dict["tracker_config"]
            if isinstance(tracker_cfg, dict):
                for key, value in tracker_cfg.items():
                    if hasattr(self.tracker_config, key):
                        setattr(self.tracker_config, key, value)

        if "latency_config" in config_dict:
            latency_cfg = config_dict["latency_config"]
            if isinstance(latency_cfg, dict):
                for key, value in latency_cfg.items():
                    if hasattr(self.latency_config, key):
                        setattr(self.latency_config, key, value)

        # 加载其他参数
        if "channel_spec" in config_dict:
            self.channel_spec = config_dict["channel_spec"]
        if "ch_name_list" in config_dict:
            self.ch_name_list = config_dict["ch_name_list"]

        # 加载IP位置
        if "ip_positions" in config_dict:
            ip_pos = config_dict["ip_positions"]
            if isinstance(ip_pos, dict):
                if "ddr" in ip_pos:
                    self.ddr_send_position_list = ip_pos["ddr"]
                if "l2m" in ip_pos:
                    self.l2m_send_position_list = ip_pos["l2m"]
                if "gdma" in ip_pos:
                    self.gdma_send_position_list = ip_pos["gdma"]
                if "sdma" in ip_pos:
                    self.sdma_send_position_list = ip_pos["sdma"]
                if "cdma" in ip_pos:
                    self.cdma_send_position_list = ip_pos["cdma"]

        # 更新节点数
        self.num_nodes = self.num_col * self.num_row

        # 重新生成派生配置
        self._generate_derived_config()

    def __str__(self) -> str:
        """字符串表示。"""
        return f"CrossRingConfig({self.config_name}, {self.num_row}×{self.num_col})"

    def __repr__(self) -> str:
        """详细字符串表示。"""
        return f"CrossRingConfig(name='{self.config_name}', topology={self.num_row}×{self.num_col})"
