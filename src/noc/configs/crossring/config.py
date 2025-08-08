"""
CrossRing NoC专用配置类。

本模块提供专门针对CrossRing拓扑的配置实现，
替换现有的CrossRingConfig类，同时保持向后兼容性。
"""

from typing import Dict, Any, List, Optional, Tuple
import json
import yaml
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from ..base_config import BaseNoCConfig
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

    @classmethod
    def from_yaml(cls, config_name: str) -> "CrossRingConfig":
        """
        从YAML文件加载配置，支持继承默认配置

        Args:
            config_name: 配置文件名（不包含.yaml扩展名），如 "3x3", "5x2", "5x4", "default"

        Returns:
            CrossRingConfig: 加载的配置实例

        Example:
            config = CrossRingConfig.from_yaml("3x3")
        """
        config_dir = Path(__file__).parent
        
        # 首先加载默认配置
        default_file = config_dir / "default.yaml"
        if default_file.exists():
            with open(default_file, 'r', encoding='utf-8') as f:
                default_data = yaml.safe_load(f)
        else:
            default_data = {}
            
        # 如果请求的是默认配置，直接使用
        if config_name == "default":
            yaml_data = default_data
        else:
            # 加载具体配置文件
            yaml_file = config_dir / f"{config_name}.yaml"
            if not yaml_file.exists():
                raise FileNotFoundError(f"配置文件不存在: {yaml_file}")
                
            with open(yaml_file, 'r', encoding='utf-8') as f:
                specific_data = yaml.safe_load(f)
                
            # 合并配置：具体配置覆盖默认配置
            yaml_data = cls._merge_configs(default_data, specific_data)
            
        # 创建基础配置实例
        topology = yaml_data.get('topology', {})
        config = cls(
            num_row=topology.get('NUM_ROW', 4),
            num_col=topology.get('NUM_COL', 4)
        )
        
        # 应用基础配置
        basic = yaml_data.get('basic', {})
        for key, value in basic.items():
            if hasattr(config.basic_config, key):
                setattr(config.basic_config, key, value)
                
        # 应用IP配置
        ip = yaml_data.get('ip', {})
        for key, value in ip.items():
            if hasattr(config.ip_config, key):
                setattr(config.ip_config, key, value)
            
        # 应用FIFO配置
        fifo = yaml_data.get('fifo', {})
        for key, value in fifo.items():
            if hasattr(config.fifo_config, key):
                setattr(config.fifo_config, key, value)
            
        # 应用Tag配置
        tag = yaml_data.get('tag', {})
        for key, value in tag.items():
            if hasattr(config.tag_config, key):
                setattr(config.tag_config, key, value)
                
        # 应用Tracker配置
        tracker = yaml_data.get('tracker', {})
        for key, value in tracker.items():
            if hasattr(config.tracker_config, key):
                setattr(config.tracker_config, key, value)
            
        # 应用延迟配置
        latency = yaml_data.get('latency', {})
        for key, value in latency.items():
            if hasattr(config.latency_config, key):
                setattr(config.latency_config, key, value)
            
        # 验证配置
        config.validate_config()
        
        return config

    @staticmethod
    def _merge_configs(default_config: dict, specific_config: dict) -> dict:
        """
        深度合并配置字典，specific_config覆盖default_config
        
        Args:
            default_config: 默认配置字典
            specific_config: 具体配置字典
            
        Returns:
            dict: 合并后的配置字典
        """
        merged = default_config.copy()
        
        for key, value in specific_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                # 递归合并嵌套字典
                merged[key] = CrossRingConfig._merge_configs(merged[key], value)
            else:
                # 直接覆盖
                merged[key] = value
                
        return merged

    @classmethod
    def create_preset_config(cls, preset_name: str, **kwargs) -> "CrossRingConfig":
        """创建预设配置 - 委托给from_yaml方法"""
        if preset_name in ["3x3", "5x2", "5x4"]:
            return cls.from_yaml(preset_name)
        else:
            raise ValueError(f"不支持的预设配置: {preset_name}")

    @classmethod 
    def create_custom_config(cls, **kwargs) -> "CrossRingConfig":
        """创建自定义配置 - 使用传统参数创建"""
        return cls(**kwargs)

    @classmethod
    def load_from_file(cls, file_path: str) -> "CrossRingConfig":
        """从文件加载配置 - 委托给from_yaml方法"""
        config_name = Path(file_path).stem
        return cls.from_yaml(config_name)

    def get_supported_presets(self) -> List[str]:
        """获取支持的预设配置列表"""
        return ["3x3", "5x2", "5x4"]


# 便捷的配置加载函数，保持向后兼容
def create_3x3_config() -> CrossRingConfig:
    """创建3x3拓扑配置"""
    return CrossRingConfig.from_yaml("3x3")


def create_5x2_config() -> CrossRingConfig:
    """创建5x2拓扑配置"""
    return CrossRingConfig.from_yaml("5x2")


def create_5x4_config() -> CrossRingConfig:
    """创建5x4拓扑配置"""
    return CrossRingConfig.from_yaml("5x4")
