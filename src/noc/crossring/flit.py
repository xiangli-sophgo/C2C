"""
重构的CrossRing Flit类，基于通用NoC基类。

继承BaseFlit，添加CrossRing特有的字段和方法，
保持与原有实现的兼容性。
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any
import numpy as np
from dataclasses import dataclass, field

from ..base.flit import BaseFlit
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

    # ========== CrossRing特有方法 ==========

    def set_crossring_coordinates(self, num_col: int) -> None:
        """设置CrossRing坐标信息"""
        self._num_col = num_col
        self.dest_xid = self.destination % num_col
        self.dest_yid = self.destination // num_col
        
        # 设置dest_coordinates属性（用于路由计算）
        self.dest_coordinates = (self.dest_xid, self.dest_yid)

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
        src_type = self._get_simplified_ip_type(getattr(self, 'source_type', None), self.source) if hasattr(self, 'source_type') and self.source_type else "??"
        dst_type = self._get_simplified_ip_type(getattr(self, 'destination_type', None), self.destination) if hasattr(self, 'destination_type') and self.destination_type else "??"

        # Tag信息
        tag_info = ""
        if self.current_tag_info:
            tag_info = f"[{self.current_tag_info}]"

        return (
            f"{self.packet_id}.{self.flit_id} {src_type}->{dst_type}: "
            f"{position_str}{tag_info} "
            f"{req_attr},{self.flit_type},{type_display} "
            f"{status_str},{self.etag_priority}"
        )
    
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
        parts = ip_type_str.split('_')
        if len(parts) >= 2:
            ip_type = parts[0]  # gdma, ddr, sdma, cdma, l2m
            ip_index = parts[1]  # 0, 1, 2, ...
            
            # 获取IP类型首字母
            ip_char = ip_type[0] if ip_type else '?'
            
            # 返回格式：节点ID:IP类型首字母+索引
            return f"{node_id}:{ip_char}{ip_index}"
        else:
            return "??"

    def _get_detailed_position_string(self) -> str:
        """获取详细的位置字符串"""
        if self.flit_position == "Ring_slice":
            # 在环路slice中：简洁显示格式 source->dest:slice
            slice_pos = getattr(self, 'current_slice_index', -1)
            source_node = getattr(self, 'link_source_node', -1)
            dest_node = getattr(self, 'link_dest_node', -1)
            
            if source_node >= 0 and dest_node >= 0 and slice_pos >= 0:
                return f"{source_node}->{dest_node}:{slice_pos}"
            elif slice_pos >= 0:
                return f"Link:S{slice_pos}"
            return "Link"
        elif self.flit_position in ["CP_arrival", "CP_departure"]:
            # 在CrossPoint中：显示节点ID.CP.方向
            cp_dir = self.crosspoint_direction if self.crosspoint_direction else "unknown"
            return f"N{self.current_node_id}.CP.{cp_dir}"
        elif self.flit_position in ["TR_FIFO", "TL_FIFO", "TU_FIFO", "TD_FIFO"]:
            # 在注入方向FIFO中
            return f"N{self.current_node_id}.{self.flit_position}"
        elif self.flit_position == "channel":
            # 在节点channel_buffer中
            return f"N{self.current_node_id}.channel"
        elif self.flit_position in ["l2h_fifo", "h2l_fifo"]:
            # 在IP接口FIFO中
            return f"N{self.current_node_id}.IP.{self.flit_position}"
        elif self.flit_position == "pending":
            # 在IP接口pending队列中
            return f"N{self.current_node_id}.IP.pending"
        elif self.flit_position == "RB":
            # 在Ring Buffer中
            return f"N{self.current_node_id}.RB"
        elif self.current_node_id >= 0:
            # 有节点信息但位置类型不在预期范围内
            return f"N{self.current_node_id}.{self.flit_position}"
        else:
            # 传统显示方式（兼容旧代码）
            if self.flit_position == "Link" and self.current_link:
                return f"({self.current_position}: {self.current_link[0]}->{self.current_link[1]}).{self.current_seat_index}"
            else:
                return f"{self.current_position}:{self.flit_position}"


# 重写基类的重置方法以支持CrossRing特有字段
def _reset_crossring_for_reuse(self):
    """重置CrossRing Flit以供重用"""
    # 调用基类重置方法
    BaseFlit._reset_for_reuse(self)

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
    self.current_link = None
    self.is_on_station = False

    # 重置时间戳
    self.cmd_entry_cake0_cycle = np.inf
    self.cmd_entry_noc_from_cake0_cycle = np.inf
    self.cmd_entry_noc_from_cake1_cycle = np.inf
    self.cmd_received_by_cake0_cycle = np.inf
    self.cmd_received_by_cake1_cycle = np.inf
    self.data_entry_noc_from_cake0_cycle = np.inf
    self.data_entry_noc_from_cake1_cycle = np.inf
    self.data_received_complete_cycle = np.inf
    self.sn_rsp_generate_cycle = np.inf

    # 重置tracker信息
    self.rn_tracker_type = None
    self.sn_tracker_type = None
    
    # 重置详细位置追踪字段
    self.current_node_id = -1
    self.current_link_id = ""
    self.current_slice_index = -1
    self.current_slot_index = -1
    self.current_tag_info = ""
    self.crosspoint_direction = ""


CrossRingFlit._reset_for_reuse = _reset_crossring_for_reuse


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

    flit.set_crossring_coordinates(num_col)
    return flit


# ========== Flit对象池管理 ==========


class CrossRingFlitPool:
    """CrossRing Flit对象池，用于高效的内存管理。"""

    def __init__(self, initial_size: int = 1000):
        """
        初始化Flit对象池。

        Args:
            initial_size: 初始池大小
        """
        self.pool: List[CrossRingFlit] = []
        self.stats = {"created": 0, "reused": 0, "returned": 0, "peak_usage": 0, "current_usage": 0}

        # 预创建一些flit对象
        for _ in range(initial_size):
            flit = CrossRingFlit(source=0, destination=0)
            self.pool.append(flit)
            self.stats["created"] += 1

    def get_flit(self, source: NodeId, destination: NodeId, **kwargs) -> CrossRingFlit:
        """
        从池中获取flit对象。

        Args:
            source: 源节点
            destination: 目标节点
            **kwargs: 其他参数

        Returns:
            CrossRingFlit实例
        """
        if self.pool:
            flit = self.pool.pop()
            self.stats["reused"] += 1
        else:
            flit = CrossRingFlit(source=0, destination=0)
            self.stats["created"] += 1

        # 重置flit状态
        flit.reset()
        flit.source = source
        flit.destination = destination

        # 设置其他参数
        for key, value in kwargs.items():
            if hasattr(flit, key):
                setattr(flit, key, value)

        self.stats["current_usage"] += 1
        if self.stats["current_usage"] > self.stats["peak_usage"]:
            self.stats["peak_usage"] = self.stats["current_usage"]

        return flit

    def return_flit(self, flit: CrossRingFlit) -> None:
        """
        将flit对象返回到池中。

        Args:
            flit: 要返回的flit对象
        """
        if flit is not None:
            self.pool.append(flit)
            self.stats["returned"] += 1
            self.stats["current_usage"] -= 1

    def get_stats(self) -> Dict[str, Any]:
        """获取池统计信息。"""
        return self.stats.copy()

    def clear(self) -> None:
        """清空池。"""
        self.pool.clear()
        self.stats = {"created": 0, "reused": 0, "returned": 0, "peak_usage": 0, "current_usage": 0}


# 全局flit池实例
_global_flit_pool = CrossRingFlitPool()


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

    flit = _global_flit_pool.get_flit(source, destination, path=path, **kwargs)
    
    # 设置CrossRing坐标信息
    # 尝试从kwargs获取num_col，如果没有则使用默认值3
    num_col = kwargs.get('num_col', 3)
    flit.set_crossring_coordinates(num_col)
    
    return flit


def return_crossring_flit(flit: CrossRingFlit) -> None:
    """
    返回CrossRing flit到对象池。

    Args:
        flit: 要返回的flit对象
    """
    _global_flit_pool.return_flit(flit)


def get_crossring_flit_pool_stats() -> Dict[str, Any]:
    """
    获取CrossRing flit池统计信息。

    Returns:
        统计信息字典
    """
    return _global_flit_pool.get_stats()


def reset(self):
    self._reset_for_reuse()


CrossRingFlit.reset = reset
