"""
CrossRing NoC专用配置类。

本模块提供专门针对CrossRing拓扑的配置实现，
替换现有的CrossRingConfig类，同时保持向后兼容性。
"""

from typing import Dict, Any, List, Optional, Tuple
import json
from dataclasses import dataclass, field
from enum import Enum

from ..base.config import BaseNoCConfig
from src.noc.utils.types import TopologyType, ValidationResult


class RoutingStrategy(Enum):
    """CrossRing路由策略枚举"""

    XY = "XY"  # 先水平环后垂直环
    YX = "YX"  # 先垂直环后水平环
    ADAPTIVE = "ADAPTIVE"  # 自适应路由（未来扩展）


@dataclass
class BasicConfiguration:
    """拓扑基础配置"""

    BURST: int = 4
    NETWORK_FREQUENCY: int = 2.0
    SLICE_PER_LINK: int = 8

    # Flit大小配置（单位：字节）
    FLIT_SIZE: int = 128  # 128字节 = 1024位

    # 路由策略配置
    ROUTING_STRATEGY: str = "XY"  # 默认使用XY路由

    # Link Slice配置 - CrossRing非环绕设计专用
    SELF_LINK_SLICES: int = 2  # 自连接（边界节点到自己）的slice数量
    
    # 性能优化配置
    FIFO_STATS_SAMPLE_INTERVAL: int = 100  # FIFO统计采样间隔（周期数）


@dataclass
class IPConfiguration:
    """IP配置数据类。"""

    GDMA_COUNT: int = 16
    SDMA_COUNT: int = 16
    CDMA_COUNT: int = 16
    DDR_COUNT: int = 16
    L2M_COUNT: int = 16

    # 带宽限制 (GB/s)
    GDMA_BW_LIMIT: float = 128.0
    SDMA_BW_LIMIT: float = 128.0
    CDMA_BW_LIMIT: float = 32.0
    DDR_BW_LIMIT: float = 128.0
    L2M_BW_LIMIT: float = 128.0

    # 读写间隔配置
    GDMA_RW_GAP: int = 1e9
    SDMA_RW_GAP: int = 1e9

    # IP接口FIFO深度配置
    IP_L2H_FIFO_DEPTH: int = 3  # L2H FIFO深度
    IP_H2L_H_FIFO_DEPTH: int = 2  # 网络域高级FIFO深度
    IP_H2L_L_FIFO_DEPTH: int = 2  # IP域低级FIFO深度


@dataclass
class FIFOConfiguration:
    """FIFO缓冲区配置数据类。"""

    RB_IN_FIFO_DEPTH: int = 16
    RB_OUT_FIFO_DEPTH: int = 8
    IQ_OUT_FIFO_DEPTH: int = 8
    EQ_IN_FIFO_DEPTH: int = 16
    IQ_CH_DEPTH: int = 10
    EQ_CH_DEPTH: int = 10


@dataclass
class TagConfiguration:
    """Tag配置数据类。"""

    # ITag 配置
    ITAG_TRIGGER_TH_H: int = 80
    ITAG_TRIGGER_TH_V: int = 80
    ITAG_MAX_NUM_H: int = 1
    ITAG_MAX_NUM_V: int = 1

    # ETag 配置
    ETAG_BOTHSIDE_UPGRADE: int = 0
    TL_ETAG_T1_UE_MAX: int = 15
    TL_ETAG_T2_UE_MAX: int = 8
    TR_ETAG_T2_UE_MAX: int = 12
    TU_ETAG_T1_UE_MAX: int = 15
    TU_ETAG_T2_UE_MAX: int = 8
    TD_ETAG_T2_UE_MAX: int = 12


@dataclass
class TrackerConfiguration:
    """Tracker 配置数据类。"""

    RN_R_TRACKER_OSTD: int = 128
    RN_W_TRACKER_OSTD: int = 32
    SN_DDR_R_TRACKER_OSTD: int = 32
    SN_DDR_W_TRACKER_OSTD: int = 16
    SN_L2M_R_TRACKER_OSTD: int = 96
    SN_L2M_W_TRACKER_OSTD: int = 48
    SN_TRACKER_RELEASE_LATENCY: int = 40


@dataclass
class LatencyConfiguration:
    """延迟配置数据类。"""

    DDR_R_LATENCY: int = 155
    DDR_R_LATENCY_VAR: int = 0
    DDR_W_LATENCY: int = 0
    DDR_W_LATENCY_VAR: int = 0
    L2M_R_LATENCY: int = 12
    L2M_R_LATENCY_VAR: int = 0
    L2M_W_LATENCY: int = 16
    L2M_W_LATENCY_VAR: int = 0


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
        self.CONFIG_NAME = config_name
        self.NUM_COL = num_col
        self.NUM_ROW = num_row
        self.NUM_NODE = num_col * num_row
        self.NUM_IP = num_col * num_row
        self.NUM_RN = self.NUM_IP
        self.NUM_SN = self.NUM_IP

        # 使用组合配置
        self.basic_config = BasicConfiguration()
        self.ip_config = IPConfiguration()
        self.fifo_config = FIFOConfiguration()
        self.tag_config = TagConfiguration()
        self.tracker_config = TrackerConfiguration()
        self.latency_config = LatencyConfiguration()

        # 路由策略属性
        self.ROUTING_STRATEGY = RoutingStrategy(self.basic_config.ROUTING_STRATEGY)

        # 通道规格
        self.CHANNEL_SPEC = {
            "gdma": 2,
            "sdma": 2,
            "cdma": 2,
            "ddr": 2,
            "l2m": 2,
        }
        self.CH_NAME_LIST = []
        for key in self.CHANNEL_SPEC:
            for idx in range(self.CHANNEL_SPEC[key]):
                self.CH_NAME_LIST.append(f"{key}_{idx}")

        # 自动生成相关配置
        self._generate_derived_config()
        self._generate_ip_positions()

    def _generate_derived_config(self) -> None:
        """生成派生配置参数。"""
        if not hasattr(self.basic_config, "BURST"):
            self.basic_config = BasicConfiguration()
        # 安全地获取tracker配置
        if hasattr(self.tracker_config, "RN_R_TRACKER_OSTD"):
            # 计算缓冲区大小
            self.RN_RDB_SIZE = self.tracker_config.RN_R_TRACKER_OSTD * self.basic_config.BURST
            self.RN_WDB_SIZE = self.tracker_config.RN_W_TRACKER_OSTD * self.basic_config.BURST
            self.SN_DDR_RDB_SIZE = self.tracker_config.SN_DDR_R_TRACKER_OSTD * self.basic_config.BURST
            self.SN_DDR_WDB_SIZE = self.tracker_config.SN_DDR_W_TRACKER_OSTD * self.basic_config.BURST
            self.SN_L2M_RDB_SIZE = self.tracker_config.SN_L2M_R_TRACKER_OSTD * self.basic_config.BURST
            self.SN_L2M_WDB_SIZE = self.tracker_config.SN_L2M_W_TRACKER_OSTD * self.basic_config.BURST

        # 生成通道名称列表
        self.ch_name_list = []
        for key in self.CHANNEL_SPEC:
            for idx in range(self.CHANNEL_SPEC[key]):
                self.ch_name_list.append(f"{key}_{idx}")

    def _generate_ip_positions(self) -> None:
        """生成IP位置列表。"""
        # 默认IP分布策略：每个节点都会挂所有的IP
        ip_positions = [i for i in range(self.NUM_NODE)]

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
        if self.NUM_NODE != self.NUM_ROW * self.NUM_COL:
            errors.append(f"节点数必须等于行数×列数 (NUM_NODE={self.NUM_NODE}, NUM_ROW={self.NUM_ROW}, NUM_COL={self.NUM_COL})")

        if self.NUM_COL < 2 or self.NUM_ROW < 2:
            errors.append(f"CrossRing拓扑至少需要2×2节点 (num_row={self.NUM_ROW}, num_col={self.NUM_COL})")

        # 路由策略验证
        try:
            RoutingStrategy(self.basic_config.ROUTING_STRATEGY)
        except ValueError:
            errors.append(f"无效的路由策略: {self.basic_config.ROUTING_STRATEGY}，支持的策略: {[s.value for s in RoutingStrategy]}")

        # FIFO深度验证 - 安全地访问属性
        fifo_cfg = self.fifo_config
        if hasattr(fifo_cfg, "RB_IN_DEPTH"):
            if fifo_cfg.RB_IN_FIFO_DEPTH <= 0:
                errors.append(f"RB输入FIFO深度必须为正数 (RB_IN_DEPTH={fifo_cfg.RB_IN_FIFO_DEPTH})")
            if fifo_cfg.EQ_IN_FIFO_DEPTH <= 0:
                errors.append(f"EQ输入FIFO深度必须为正数 (EQ_IN_DEPTH={fifo_cfg.EQ_IN_FIFO_DEPTH})")
        elif isinstance(fifo_cfg, dict):
            if fifo_cfg.get("RB_IN_FIFO_DEPTH", 0) <= 0:
                errors.append(f"RB输入FIFO深度必须为正数 (RB_IN_FIFO_DEPTH={fifo_cfg.get('RB_IN_FIFO_DEPTH', 0)})")
            if fifo_cfg.get("EQ_IN_FIFO_DEPTH", 0) <= 0:
                errors.append(f"EQ输入FIFO深度必须为正数 (EQ_IN_FIFO_DEPTH={fifo_cfg.get('EQ_IN_FIFO_DEPTH', 0)})")

        # Tag参数验证 - 放宽约束
        tag_cfg = self.tag_config

        # 安全地获取Tag参数
        if hasattr(tag_cfg, "TL_ETAG_T2_UE_MAX"):
            tl_t2 = tag_cfg.TL_ETAG_T2_UE_MAX
            tl_t1 = tag_cfg.TL_ETAG_T1_UE_MAX
            tr_t2 = tag_cfg.TR_ETAG_T2_UE_MAX
            tu_t2 = tag_cfg.TU_ETAG_T2_UE_MAX
            tu_t1 = tag_cfg.TU_ETAG_T1_UE_MAX
            td_t2 = tag_cfg.TD_ETAG_T2_UE_MAX
        else:
            raise ValueError

        # 安全地获取FIFO深度
        if hasattr(fifo_cfg, "RB_IN_FIFO_DEPTH"):
            rb_depth = fifo_cfg.RB_IN_FIFO_DEPTH
            eq_depth = fifo_cfg.EQ_IN_FIFO_DEPTH
        else:
            raise ValueError

        if tl_t2 <= 0:
            errors.append(f"TL ETag T2必须为正数 (TL_ETAG_T2_UE_MAX={tl_t2})")
        if tl_t1 <= tl_t2:
            errors.append(f"TL ETag T1必须大于T2 (TL_ETAG_T1_UE_MAX={tl_t1}, TL_ETAG_T2_UE_MAX={tl_t2})")
        if tl_t1 >= rb_depth:
            errors.append(f"TL ETag T1必须小于RB_IN_FIFO_DEPTH (TL_ETAG_T1_UE_MAX={tl_t1}, RB_IN_FIFO_DEPTH={rb_depth})")
        if tr_t2 >= rb_depth:
            errors.append(f"TR ETag T2必须小于RB_IN_FIFO_DEPTH (TR_ETAG_T2_UE_MAX={tr_t2}, RB_IN_FIFO_DEPTH={rb_depth})")

        if tu_t2 <= 0:
            errors.append(f"TU ETag T2必须为正数 (TU_ETAG_T2_UE_MAX={tu_t2})")
        if tu_t1 <= tu_t2:
            errors.append(f"TU ETag T1必须大于T2 (TU_ETAG_T1_UE_MAX={tu_t1}, TU_ETAG_T2_UE_MAX={tu_t2})")
        if tu_t1 >= eq_depth:
            errors.append(f"TU ETag T1必须小于EQ_IN_FIFO_DEPTH (TU_ETAG_T1_UE_MAX={tu_t1}, EQ_IN_FIFO_DEPTH={eq_depth})")
        if td_t2 >= eq_depth:
            errors.append(f"TD ETag T2必须小于EQ_IN_FIFO_DEPTH (TD_ETAG_T2_UE_MAX={td_t2}, EQ_IN_FIFO_DEPTH={eq_depth})")

        #  Tracker 配置验证 - 安全地访问属性
        tracker = self.tracker_config
        if hasattr(tracker, "RN_R_TRACKER_OSTD"):
            if tracker.RN_R_TRACKER_OSTD <= 0 or tracker.RN_W_TRACKER_OSTD <= 0:
                errors.append(f"RN Tracker OSTD必须为正数 (RN_R_TRACKER_OSTD={tracker.RN_R_TRACKER_OSTD}, RN_W_TRACKER_OSTD={tracker.RN_W_TRACKER_OSTD})")
            if tracker.SN_DDR_R_TRACKER_OSTD <= 0 or tracker.SN_L2M_R_TRACKER_OSTD <= 0:
                errors.append(f"SN Tracker OSTD必须为正数 (SN_DDR_R_TRACKER_OSTD={tracker.SN_DDR_R_TRACKER_OSTD}, SN_L2M_R_TRACKER_OSTD={tracker.SN_L2M_R_TRACKER_OSTD})")
            # 缓冲区大小一致性验证
            if hasattr(self, "RN_RDB_SIZE") and self.RN_RDB_SIZE != tracker.RN_R_TRACKER_OSTD * self.basic_config.BURST:
                errors.append(f"RN_RDB_SIZE必须等于RN_R_TRACKER_OSTD × BURST (RN_RDB_SIZE={self.RN_RDB_SIZE}, RN_R_TRACKER_OSTD={tracker.RN_R_TRACKER_OSTD}, BURST={self.basic_config.BURST})")
        elif isinstance(tracker, dict):
            rn_r_ostd = tracker.get("RN_R_TRACKER_OSTD", 64)
            rn_w_ostd = tracker.get("RN_W_TRACKER_OSTD", 32)
            if rn_r_ostd <= 0 or rn_w_ostd <= 0:
                errors.append(f"RN Tracker OSTD必须为正数 (RN_R_TRACKER_OSTD={rn_r_ostd}, RN_W_TRACKER_OSTD={rn_w_ostd})")
            # 缓冲区大小一致性验证
            if hasattr(self, "RN_RDB_SIZE") and self.RN_RDB_SIZE != rn_r_ostd * self.basic_config.BURST:
                errors.append(f"RN_RDB_SIZE必须等于RN_R_TRACKER_OSTD × BURST (RN_RDB_SIZE={self.RN_RDB_SIZE}, RN_R_TRACKER_OSTD={rn_r_ostd}, BURST={self.basic_config.BURST})")

        # IP配置验证 - 安全地访问属性
        ip_cfg = self.ip_config
        if hasattr(ip_cfg, "GDMA_COUNT"):
            if ip_cfg.GDMA_COUNT < 0 or ip_cfg.SDMA_COUNT < 0:
                errors.append(f"IP数量不能为负数 (GDMA_COUNT={ip_cfg.GDMA_COUNT}, SDMA_COUNT={ip_cfg.SDMA_COUNT})")
            if ip_cfg.GDMA_BW_LIMIT <= 0 or ip_cfg.DDR_BW_LIMIT <= 0:
                errors.append(f"带宽限制必须为正数 (GDMA_BW_LIMIT={ip_cfg.GDMA_BW_LIMIT}, DDR_BW_LIMIT={ip_cfg.DDR_BW_LIMIT})")
        elif isinstance(ip_cfg, dict):
            gdma_count = ip_cfg.get("GDMA_COUNT", 0)
            sdma_count = ip_cfg.get("SDMA_COUNT", 0)
            if gdma_count < 0 or sdma_count < 0:
                errors.append(f"IP数量不能为负数 (GDMA_COUNT={gdma_count}, SDMA_COUNT={sdma_count})")
            gdma_bw = ip_cfg.get("GDMA_BW_LIMIT", 8.0)
            ddr_bw = ip_cfg.get("DDR_BW_LIMIT", 80.0)
            if gdma_bw <= 0 or ddr_bw <= 0:
                errors.append(f"带宽限制必须为正数 (GDMA_BW_LIMIT={gdma_bw}, DDR_BW_LIMIT={ddr_bw})")

        # 位置列表验证
        for pos in self.ddr_send_position_list:
            if pos >= self.NUM_NODE:
                errors.append(f"IP位置{pos}超出节点范围 (num_nodes={self.NUM_NODE})")

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
            "num_nodes": self.NUM_NODE,
            "num_col": self.NUM_COL,
            "num_row": self.NUM_ROW,
            "num_ip": self.NUM_IP,
            "num_rn": self.NUM_RN,
            "num_sn": self.NUM_SN,
            "channel_spec": self.CHANNEL_SPEC,
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
        self.NUM_ROW = num_row
        self.NUM_COL = num_col
        self.NUM_NODE = num_col * num_row

        # 重新生成相关配置
        self._generate_ip_positions()

    def update_basic_config(self, **kwargs) -> None:
        """
        更新网络基础配置。

        Args:
            **kwargs: 基础配置参数
        """
        for key, value in kwargs.items():
            if hasattr(self.basic_config, key):
                setattr(self.basic_config, key, value)

        # 如果更新了路由策略，需要重新设置ROUTING_STRATEGY属性
        if "ROUTING_STRATEGY" in kwargs:
            self.ROUTING_STRATEGY = RoutingStrategy(self.basic_config.ROUTING_STRATEGY)

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

    def set_routing_strategy(self, strategy: str) -> None:
        """
        设置路由策略。

        Args:
            strategy: 路由策略 ("XY", "YX", "ADAPTIVE")
        """
        # 验证策略是否有效
        strategy_enum = RoutingStrategy(strategy)

        # 更新配置
        self.basic_config.ROUTING_STRATEGY = strategy
        self.ROUTING_STRATEGY = strategy_enum

    def get_routing_strategy(self) -> str:
        """
        获取当前路由策略。

        Returns:
            当前路由策略字符串
        """
        return self.ROUTING_STRATEGY.value

    def set_preset_configuration(self, preset: str) -> None:
        """
        设置预设配置。

        Args:
            preset: 预设配置名称 ('2260E', '2262')
        """
        if preset == "2260E":
            self.update_topology_size(num_row=3, num_col=3)
            self.NUM_IP = 8
            self.update_fifo_config(RB_IN_DEPTH=16, EQ_IN_DEPTH=16)
            self.update_ip_config(GDMA_COUNT=4, SDMA_COUNT=4, DDR_COUNT=16)
            self.update_tag_config(
                TL_ETAG_T2_UE_MAX=8,
                TL_ETAG_T1_UE_MAX=15,
                TR_ETAG_T2_UE_MAX=12,
                TU_ETAG_T2_UE_MAX=8,
                TU_ETAG_T1_UE_MAX=15,
                TD_ETAG_T2_UE_MAX=12,
            )
        elif preset == "2262":
            self.update_topology_size(num_row=5, num_col=4)
            self.NUM_IP = 12
            self.update_fifo_config(RB_IN_DEPTH=16, EQ_IN_DEPTH=16)
            self.update_ip_config(GDMA_COUNT=32, DDR_COUNT=32)
            self.update_tag_config(
                TL_ETAG_T2_UE_MAX=8,
                TL_ETAG_T1_UE_MAX=15,
                TR_ETAG_T2_UE_MAX=12,
                TU_ETAG_T2_UE_MAX=8,
                TU_ETAG_T1_UE_MAX=15,
                TD_ETAG_T2_UE_MAX=12,
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
            "config_name": self.CONFIG_NAME,
            "num_col": self.NUM_COL,
            "num_row": self.NUM_ROW,
            "num_ip": self.NUM_IP,
            "topo_type": self.topology_type,
            # 组合配置
            "basic_config": self.basic_config.__dict__,
            "ip_config": self.ip_config.__dict__,
            "fifo_config": self.fifo_config.__dict__,
            "tag_config": self.tag_config.__dict__,
            "tracker_config": self.tracker_config.__dict__,
            "latency_config": self.latency_config.__dict__,
            # 其他参数
            "channel_spec": self.CHANNEL_SPEC,
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
            self.CONFIG_NAME = config_dict["config_name"]
        if "num_col" in config_dict:
            self.NUM_COL = config_dict["num_col"]
        if "num_row" in config_dict:
            self.NUM_ROW = config_dict["num_row"]
        if "num_ip" in config_dict:
            self.NUM_IP = config_dict["num_ip"]

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
            self.CHANNEL_SPEC = config_dict["channel_spec"]
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
        self.NUM_NODE = self.NUM_COL * self.NUM_ROW

        # 重新生成派生配置
        self._generate_derived_config()

    @classmethod
    def create_preset_config(cls, preset_name: str, **kwargs) -> "CrossRingConfig":
        """
        创建预设配置。

        Args:
            preset_name: 预设配置名称 ('2260E', '2262', 'default')
            **kwargs: 额外配置参数

        Returns:
            CrossRing配置实例
        """
        if preset_name == "2260E":
            config = cls(num_row=3, num_col=3, config_name=preset_name)
            config.set_preset_configuration(preset_name)
        elif preset_name == "2262":
            config = cls(num_row=5, num_col=4, config_name=preset_name)
            config.set_preset_configuration(preset_name)
        elif preset_name == "default":
            config = cls(num_row=4, num_col=4, config_name=preset_name)
        else:
            raise ValueError(f"未知的预设配置: {preset_name}")

        # 应用额外参数
        for key, value in kwargs.items():
            config.set_parameter(key, value)

        return config

    def get_supported_presets(self) -> List[str]:
        """
        获取支持的预设配置列表。

        Returns:
            预设配置名称列表
        """
        return ["default", "2260E", "2262"]

    @classmethod
    def create_custom_config(cls, num_row: int = 4, num_col: int = 4, config_name: str = "custom", **kwargs) -> "CrossRingConfig":
        """
        创建自定义配置。

        Args:
            num_row: 行数
            num_col: 列数
            config_name: 配置名称
            **kwargs: 其他配置参数

        Returns:
            CrossRing配置实例
        """
        config = cls(num_row=num_row, num_col=num_col, config_name=config_name)

        # 应用自定义参数
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            elif hasattr(config.basic_config, key):
                setattr(config.basic_config, key, value)
            elif hasattr(config.ip_config, key):
                setattr(config.ip_config, key, value)
            elif hasattr(config.fifo_config, key):
                setattr(config.fifo_config, key, value)
            elif hasattr(config.tag_config, key):
                setattr(config.tag_config, key, value)
            elif hasattr(config.tracker_config, key):
                setattr(config.tracker_config, key, value)
            elif hasattr(config.latency_config, key):
                setattr(config.latency_config, key, value)

        # 重新生成派生配置
        config._generate_derived_config()
        return config

    @classmethod
    def load_from_file(cls, file_path: str) -> "CrossRingConfig":
        """
        从文件加载配置。

        Args:
            file_path: 配置文件路径

        Returns:
            CrossRing配置实例
        """
        import json

        with open(file_path, "r") as f:
            config_dict = json.load(f)

        # 创建基础配置
        config = cls(num_row=config_dict.get("num_row", 4), num_col=config_dict.get("num_col", 4), config_name=config_dict.get("config_name", "loaded"))

        # 从字典加载
        config.from_dict(config_dict)

        return config

    def get_resource_config(self) -> Dict[str, Any]:
        """获取资源配置信息"""
        return {
            "rn_resources": {
                "read_tracker_count": self.tracker_config.RN_R_TRACKER_OSTD,
                "write_tracker_count": self.tracker_config.RN_W_TRACKER_OSTD,
                "rdb_size": self.RN_RDB_SIZE,
                "wdb_size": self.RN_WDB_SIZE,
            },
            "sn_resources": {
                "ddr": {
                    "read_tracker_count": self.tracker_config.SN_DDR_R_TRACKER_OSTD,
                    "write_tracker_count": self.tracker_config.SN_DDR_W_TRACKER_OSTD,
                    "wdb_size": self.SN_DDR_WDB_SIZE,
                },
                "l2m": {
                    "read_tracker_count": self.tracker_config.SN_L2M_R_TRACKER_OSTD,
                    "write_tracker_count": self.tracker_config.SN_L2M_W_TRACKER_OSTD,
                    "wdb_size": self.SN_L2M_WDB_SIZE,
                },
                "tracker_release_latency": self.tracker_config.SN_TRACKER_RELEASE_LATENCY,
            },
            "fifo_depths": {
                "rb_in": self.fifo_config.RB_IN_FIFO_DEPTH,
                "rb_out": self.fifo_config.RB_OUT_FIFO_DEPTH,
                "iq_out": self.fifo_config.IQ_OUT_FIFO_DEPTH,
                "eq_in": self.fifo_config.EQ_IN_FIFO_DEPTH,
                "iq_ch": self.fifo_config.IQ_CH_DEPTH,
                "eq_ch": self.fifo_config.EQ_CH_DEPTH,
            },
        }

    def get_traffic_config(self) -> Dict[str, Any]:
        """获取流量配置信息"""
        return {
            "burst_length": self.basic_config.BURST,
            "ip_bandwidth_limits": {
                "gdma": self.ip_config.GDMA_BW_LIMIT,
                "sdma": self.ip_config.SDMA_BW_LIMIT,
                "cdma": self.ip_config.CDMA_BW_LIMIT,
                "ddr": self.ip_config.DDR_BW_LIMIT,
                "l2m": self.ip_config.L2M_BW_LIMIT,
            },
            "latencies": {
                "ddr_read": self.latency_config.DDR_R_LATENCY,
                "ddr_write": self.latency_config.DDR_W_LATENCY,
                "l2m_read": self.latency_config.L2M_R_LATENCY,
                "l2m_write": self.latency_config.L2M_W_LATENCY,
            },
        }

    def get_etag_config(self) -> Dict[str, Any]:
        """获取ETag配置信息"""
        return {
            "bothside_upgrade": self.tag_config.ETAG_BOTHSIDE_UPGRADE,
            "tl_settings": {
                "t1_ue_max": self.tag_config.TL_ETAG_T1_UE_MAX,
                "t2_ue_max": self.tag_config.TL_ETAG_T2_UE_MAX,
            },
            "tr_settings": {
                "t2_ue_max": self.tag_config.TR_ETAG_T2_UE_MAX,
            },
            "tu_settings": {
                "t1_ue_max": self.tag_config.TU_ETAG_T1_UE_MAX,
                "t2_ue_max": self.tag_config.TU_ETAG_T2_UE_MAX,
            },
            "td_settings": {
                "t2_ue_max": self.tag_config.TD_ETAG_T2_UE_MAX,
            },
        }

    def get_itag_config(self) -> Dict[str, Any]:
        """获取ITag配置信息"""
        return {
            "horizontal": {
                "trigger_threshold": self.tag_config.ITAG_TRIGGER_TH_H,
                "max_num": self.tag_config.ITAG_MAX_NUM_H,
            },
            "vertical": {
                "trigger_threshold": self.tag_config.ITAG_TRIGGER_TH_V,
                "max_num": self.tag_config.ITAG_MAX_NUM_V,
            },
        }

    def update_channel_names(self, ch_name_list: List[str]) -> None:
        """
        更新通道名称列表（CH_NAME_LIST）。

        这个方法允许在traffic生成时动态修改CH_NAME_LIST，
        使得节点挂载的IP能够使用自定义的通道名称。

        Args:
            ch_name_list: 新的通道名称列表，例如 ["gdma_0", "gdma_1", "ddr_0", "ddr_1"]

        Note:
            - 这是所有节点的通道名称集合，不需要分不同的节点
            - 更新后的CH_NAME_LIST会影响IP接口创建和可视化显示
            - 建议在模型初始化之前调用此方法
        """
        self.CH_NAME_LIST = ch_name_list.copy()
