"""
重构的CrossRing Flit类，基于通用NoC基类。

继承BaseFlit，添加CrossRing特有的字段和方法，
保持与原有实现的兼容性。
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any
import numpy as np
from dataclasses import dataclass, field

from ..base.flit import BaseFlit, FlitPool
from src.noc.utils.types import NodeId


@dataclass
class CrossRingFlit(BaseFlit):
    """
    CrossRing专用Flit类，继承BaseFlit。

    在通用基类基础上添加CrossRing特有的字段：
    - ETag/ITag优先级控制
    - 环路状态管理
    - CrossRing特有的时间戳
    """

    # ========== CrossRing特有字段 ==========

    # ETag优先级控制
    etag_priority: str = "T2"  # "T2" | "T1" | "T0"
    used_entry_level: Optional[str] = None  # 实际占用的entry级别

    # ITag状态
    itag_h: bool = False  # 水平ITag
    itag_v: bool = False  # 垂直ITag
    itag_reservation: int = 0  # I-Tag预留信息

    # 环路移动状态
    moving_direction: int = 1  # 1 (顺时针) | -1 (逆时针)
    moving_direction_v: int = 1  # 垂直方向
    circuits_completed_h: int = 0  # 水平环路完成数
    circuits_completed_v: int = 0  # 垂直环路完成数
    wait_cycle_h: int = 0  # 水平等待周期
    wait_cycle_v: int = 0  # 垂直等待周期

    # CrossRing坐标信息
    dest_xid: int = 0  # 目标X坐标
    dest_yid: int = 0  # 目标Y坐标
    dest_coordinates: tuple = field(default_factory=lambda: (0, 0))  # 目标坐标元组

    @property
    def dest_coords_tuple(self) -> tuple:
        """返回目标坐标元组，用于路由计算"""
        return (self.dest_xid, self.dest_yid)

    # 位置状态
    station_position: int = -1
    current_seat_index: int = -1
    current_link: Optional[tuple] = None
    is_on_station: bool = False

    # 详细位置追踪
    current_node_id: int = -1  # 当前所在节点ID
    current_link_id: str = ""  # 当前所在链路ID
    current_slice_index: int = -1  # 当前所在slice索引
    current_slot_index: int = -1  # 当前所在slot索引
    current_tag_info: str = ""  # 当前slot的tag信息
    crosspoint_direction: str = ""  # 在CrossPoint中的方向(arrival/departure)

    # 特有的tracker信息
    rn_tracker_type: Optional[str] = None  # RN端tracker类型
    sn_tracker_type: Optional[str] = None  # SN端tracker类型("ro", "share")

    # 四方向系统相关（新增）
    current_ring_direction: Optional[Any] = None  # 当前环形方向
    remaining_directions: List[Any] = field(default_factory=list)  # 剩余路由方向
    dimension_turn_cycle: int = -1  # 维度转换周期
    current_direction: str = ""  # 当前传输方向（"horizontal"或"vertical"）

    # 延迟发送支持
    departure_cycle: float = 0.0  # 允许发送的周期

    def __post_init__(self):
        """初始化后处理"""
        super().__post_init__()

        # 设置CrossRing特有的默认值
        if self.channel == "req":
            self.etag_priority = "T2"  # 默认优先级

        # 初始化坐标
        if hasattr(self, "_num_col") and self._num_col > 0:
            self.dest_xid = self.destination % self._num_col
            self.dest_yid = self.destination // self._num_col

    # ========== 实现抽象方法 ==========

    def get_routing_info(self) -> Dict[str, Any]:
        """获取CrossRing路由信息"""
        return {
            "etag_priority": self.etag_priority,
            "itag_h": self.itag_h,
            "itag_v": self.itag_v,
            "dest_xid": self.dest_xid,
            "dest_yid": self.dest_yid,
            "moving_direction": self.moving_direction,
            "moving_direction_v": self.moving_direction_v,
            "circuits_completed_h": self.circuits_completed_h,
            "circuits_completed_v": self.circuits_completed_v,
        }

    def calculate_expected_hops(self) -> int:
        """计算CrossRing预期跳数"""
        # CrossRing的跳数计算比较复杂，涉及环路传输
        # 这里提供简化版本
        if hasattr(self, "_num_col") and self._num_col > 0:
            src_x = self.source % self._num_col
            src_y = self.source // self._num_col
            dst_x = self.destination % self._num_col
            dst_y = self.destination // self._num_col

            # 简化的曼哈顿距离估算
            return abs(dst_x - src_x) + abs(dst_y - src_y)
        else:
            return len(self.path) - 1

    def is_valid_next_hop(self, next_node: NodeId) -> bool:
        """检查下一跳是否有效（CrossRing特定）"""
        # CrossRing的有效性检查需要考虑环路结构
        # 这里提供简化版本
        return next_node in self.path

    def is_next_hop_destination(self, next_node: NodeId) -> bool:
        """检查下一跳是否是最终目的地"""
        return next_node == self.destination

    def should_eject_at_node(self, current_node: NodeId) -> bool:
        """检查是否应该在当前节点下环（到达目标节点）"""
        return current_node == self.destination

    # ========== CrossRing特有方法 ==========

    def set_crossring_coordinates(self, num_col: int, num_row: int = 3) -> None:
        """设置CrossRing坐标信息。使用直角坐标系，原点在左下角。"""
        self._num_col = num_col
        self.dest_xid = self.destination % num_col  # x坐标：水平方向
        self.dest_yid = self.destination // num_col  # 这是原始row，需要转换

        # 计算实际的y坐标（直角坐标系）
        dest_y = num_row - 1 - self.dest_yid  # y坐标：垂直方向，从下到上

        # 设置dest_coordinates属性（用于路由计算）
        # 使用(x, y)格式，与topology保持一致
        self.dest_coordinates = (self.dest_xid, dest_y)

        # 设置源坐标到custom_fields
        src_x = self.source % num_col
        src_y = self.source // num_col
        self.custom_fields.update({"src_xid": src_x, "src_yid": src_y, "num_col": num_col})

    def upgrade_etag_priority(self) -> bool:
        """升级ETag优先级"""
        if self.etag_priority == "T2":
            self.etag_priority = "T1"
            return True
        elif self.etag_priority == "T1":
            self.etag_priority = "T0"
            return True
        return False

    def set_itag(self, direction: str) -> None:
        """设置ITag"""
        if direction == "horizontal":
            self.itag_h = True
        elif direction == "vertical":
            self.itag_v = True
        self.is_tagged = self.itag_h or self.itag_v

    def clear_itag(self, direction: str = "all") -> None:
        """清除ITag"""
        if direction == "horizontal" or direction == "all":
            self.itag_h = False
        if direction == "vertical" or direction == "all":
            self.itag_v = False
        self.is_tagged = self.itag_h or self.itag_v

    def increment_circuit(self, direction: str) -> None:
        """增加环路完成计数"""
        if direction == "horizontal":
            self.circuits_completed_h += 1
        elif direction == "vertical":
            self.circuits_completed_v += 1

    def add_wait_cycles(self, direction: str, cycles: int) -> None:
        """添加等待周期"""
        if direction == "horizontal":
            self.wait_cycle_h += cycles
        elif direction == "vertical":
            self.wait_cycle_v += cycles

    def reset_for_retry(self) -> None:
        """重试重置"""
        # 调用基类重试方法
        self.prepare_for_retry("crossring_retry")

        # 重置CrossRing特有状态
        self.circuits_completed_h = 0
        self.circuits_completed_v = 0
        self.wait_cycle_h = 0
        self.wait_cycle_v = 0
        self.clear_itag()
        self.station_position = -1
        self.current_seat_index = -1
        self.current_link = None
        self.is_on_station = False

    def get_crossring_status(self) -> Dict[str, Any]:
        """获取CrossRing状态摘要"""
        status = self.get_status_summary()

        # 添加CrossRing特有信息
        status.update(
            {
                "etag_priority": self.etag_priority,
                "itag_status": {"h": self.itag_h, "v": self.itag_v},
                "circuits": {"h": self.circuits_completed_h, "v": self.circuits_completed_v},
                "wait_cycles": {"h": self.wait_cycle_h, "v": self.wait_cycle_v},
                "coordinates": {"dest_x": self.dest_xid, "dest_y": self.dest_yid},
                "tracker_types": {"rn": self.rn_tracker_type, "sn": self.sn_tracker_type},
                "crossring_latencies": self.calculate_latencies(),
            }
        )

        return status

    def __repr__(self) -> str:
        """CrossRing特有的字符串表示"""
        # 基础信息
        req_attr = "O" if self.req_attr == "old" else "N"
        type_display = self.rsp_type[:3] if self.rsp_type else self.req_type[0]

        # 详细位置信息
        position_str = self._get_detailed_position_string()

        # 状态标识
        status = []
        if self.is_finish:
            status.append("F")
        if self.is_ejected:
            status.append("E")
        if self.itag_h:
            status.append("IH")
        if self.itag_v:
            status.append("IV")

        status_str = "".join(status) if status else ""

        # IP类型显示 - 简化格式：节点ID:IP类型首字母+索引
        src_type = self._get_simplified_ip_type(getattr(self, "source_type", None), self.source) if hasattr(self, "source_type") and self.source_type else "??"
        dst_type = self._get_simplified_ip_type(getattr(self, "destination_type", None), self.destination) if hasattr(self, "destination_type") and self.destination_type else "??"

        # Tag信息
        tag_info = ""
        if self.current_tag_info:
            tag_info = f"[{self.current_tag_info}]"

        return f"{self.flit_type.upper()},{self.packet_id}.{self.flit_id},{src_type}->{dst_type}:{position_str}{tag_info},{req_attr},{type_display},{status_str},{self.etag_priority}"

    def _get_simplified_ip_type(self, ip_type_str: str, node_id: int) -> str:
        """
        获取简化的IP类型显示格式

        Args:
            ip_type_str: IP类型字符串，如 "gdma_0_node_0"
            node_id: 节点ID

        Returns:
            简化格式字符串，如 "0:g0"
        """
        if not ip_type_str:
            return "??"

        # 解析IP类型字符串
        # 格式：gdma_0_node_0 -> 0:g0
        # 格式：ddr_4_node_4 -> 4:d4
        parts = ip_type_str.split("_")
        if len(parts) >= 2:
            ip_type = parts[0]  # gdma, ddr, sdma, cdma, l2m
            ip_index = parts[1]  # 0, 1, 2, ...

            # 获取IP类型简写
            if ip_type == "ddr":
                ip_char = "D"
            elif ip_type == "l2m":
                ip_char = "L"
            else:
                ip_char = ip_type[0].upper() if ip_type else "?"

            # 返回格式：节点ID:IP类型首字母+索引
            return f"{node_id}:{ip_char}{ip_index}"
        else:
            return "??"

    def _get_detailed_position_string(self) -> str:
        """获取简化的位置字符串"""
        if self.flit_position == "Ring_slice":
            # 在环路slice中：显示source->dest:slice格式
            slice_pos = getattr(self, "current_slice_index", -1)
            source_node = getattr(self, "link_source_node", -1)
            dest_node = getattr(self, "link_dest_node", -1)

            if source_node >= 0 and dest_node >= 0 and slice_pos >= 0:
                return f"{source_node}->{dest_node}:{slice_pos}"
            elif slice_pos >= 0:
                return f"S{slice_pos}"
            return "Link"
        elif self.flit_position in ["CP_arrival", "CP_departure"]:
            # 在CrossPoint中：显示节点ID.CP
            return f"N{self.current_node_id}.CP"
        elif self.flit_position in ["TR_FIFO", "TL_FIFO", "TU_FIFO", "TD_FIFO"]:
            # 在注入方向FIFO中，简化为IQ_方向
            direction = self.flit_position.replace("_FIFO", "")
            return f"N{self.current_node_id}.IQ_{direction}"
        elif "eject_" in self.flit_position and "_FIFO" in self.flit_position:
            # 在eject FIFO中，简化显示为EQ_方向
            direction = self.flit_position.replace("eject_", "").replace("_FIFO", "")
            return f"N{self.current_node_id}.EQ_{direction}"
        elif self.flit_position == "channel":
            # 在节点channel_buffer中
            return f"N{self.current_node_id}.channel"
        elif self.flit_position in ["l2h_fifo", "h2l_fifo"]:
            # 在IP接口FIFO中
            return f"N{self.current_node_id}.{self.flit_position}"
        elif self.flit_position == "pending":
            # 在IP接口pending队列中
            return f"N{self.current_node_id}.pending"
        elif self.flit_position == "RB":
            # 在Ring Buffer中 - 使用具体的FIFO名称
            rb_fifo = getattr(self, "rb_fifo_name", None)
            if rb_fifo:
                return f"N{self.current_node_id}.{rb_fifo}"
            else:
                return f"N{self.current_node_id}.RB"
        elif self.current_node_id >= 0:
            # 有节点信息但位置类型不在预期范围内
            return f"N{self.current_node_id}.{self.flit_position}"
        else:
            # 简化显示
            return self.flit_position

    def _reset_for_reuse(self):
        """重置CrossRing Flit以供重用"""
        # 调用基类重置方法
        super()._reset_for_reuse()

        # 重置CrossRing特有字段
        self.etag_priority = "T2"
        self.used_entry_level = None
        self.itag_h = False
        self.itag_v = False
        self.itag_reservation = 0
        self.moving_direction = 1
        self.moving_direction_v = 1
        self.circuits_completed_h = 0
        self.circuits_completed_v = 0
        self.wait_cycle_h = 0
        self.wait_cycle_v = 0
        self.dest_xid = 0
        self.dest_yid = 0
        self.station_position = -1
        self.current_seat_index = -1

        # 重置详细位置追踪字段
        self.current_node_id = -1
        self.current_link_id = ""
        self.current_slice_index = -1
        self.current_slot_index = -1
        self.current_tag_info = ""
        self.crosspoint_direction = ""

        # 重置延迟发送
        self.departure_cycle = 0.0

    def reset(self):
        """重置方法的简化接口"""
        self._reset_for_reuse()


# 工厂函数
def create_crossring_flit(source: NodeId, destination: NodeId, path: List[NodeId] = None, num_col: int = 4, **kwargs) -> CrossRingFlit:
    """
    创建CrossRing Flit的工厂函数

    Args:
        source: 源节点
        destination: 目标节点
        path: 路径（可选）
        num_col: 列数（用于坐标计算）
        **kwargs: 其他参数

    Returns:
        CrossRingFlit实例
    """
    if path is None:
        path = [source, destination]

    flit = CrossRingFlit(source=source, destination=destination, path=path, **kwargs)

    flit.set_crossring_coordinates(num_col, num_row=3)
    return flit


# ========== Flit对象池管理 ==========


# 全局CrossRing flit池实例
_global_crossring_flit_pool = FlitPool(CrossRingFlit, initial_size=1000)


def create_crossring_flit(source: NodeId, destination: NodeId, path: Optional[List[NodeId]] = None, **kwargs) -> CrossRingFlit:
    """
    创建CrossRing flit的便捷函数（使用对象池）。

    Args:
        source: 源节点
        destination: 目标节点
        path: 路径（可选）
        **kwargs: 其他参数

    Returns:
        CrossRingFlit实例
    """
    if path is None:
        path = [source, destination]

    # 提取num_col和num_row参数，避免传递给Flit构造函数
    num_col = kwargs.pop("num_col", 3)
    num_row = kwargs.pop("num_row", 3)

    flit = _global_crossring_flit_pool.get_flit(source=source, destination=destination, path=path, **kwargs)

    # 设置CrossRing坐标信息 - 使用正确的num_row参数
    flit.set_crossring_coordinates(num_col, num_row=num_row)

    return flit


def return_crossring_flit(flit: CrossRingFlit) -> None:
    """
    返回CrossRing flit到对象池。

    Args:
        flit: 要返回的flit对象
    """
    _global_crossring_flit_pool.return_flit(flit)


def get_crossring_flit_pool_stats() -> Dict[str, Any]:
    """
    获取CrossRing flit池统计信息。

    Returns:
        统计信息字典
    """
    return _global_crossring_flit_pool.get_stats()
