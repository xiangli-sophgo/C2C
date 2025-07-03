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
from ..types import NodeId


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
    
    # 位置状态
    station_position: int = -1
    current_seat_index: int = -1
    current_link: Optional[tuple] = None
    is_on_station: bool = False
    
    # ========== CrossRing特有时间戳 ==========
    cmd_entry_cake0_cycle: float = np.inf      # RN端发出请求
    cmd_entry_noc_from_cake0_cycle: float = np.inf    # 进入网络
    cmd_entry_noc_from_cake1_cycle: float = np.inf    # SN端处理
    cmd_received_by_cake0_cycle: float = np.inf   # RN端收到响应  
    cmd_received_by_cake1_cycle: float = np.inf   # SN端收到请求
    data_entry_noc_from_cake0_cycle: float = np.inf   # 数据进网络(写)
    data_entry_noc_from_cake1_cycle: float = np.inf   # 数据进网络(读)
    data_received_complete_cycle: float = np.inf      # 数据传输完成
    sn_rsp_generate_cycle: float = np.inf       # SN响应生成时间
    
    # 特有的tracker信息
    rn_tracker_type: Optional[str] = None  # RN端tracker类型
    sn_tracker_type: Optional[str] = None  # SN端tracker类型("ro", "share")
    
    def __post_init__(self):
        """初始化后处理"""
        super().__post_init__()
        
        # 设置CrossRing特有的默认值
        if self.channel == "req":
            self.etag_priority = "T2"  # 默认优先级
        
        # 初始化坐标
        if hasattr(self, '_num_col') and self._num_col > 0:
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
        if hasattr(self, '_num_col') and self._num_col > 0:
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
        
        # 设置源坐标到custom_fields
        src_x = self.source % num_col
        src_y = self.source // num_col
        self.custom_fields.update({
            "src_xid": src_x,
            "src_yid": src_y,
            "num_col": num_col
        })
    
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
    
    def reset_for_crossring_retry(self) -> None:
        """CrossRing特有的重试重置"""
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
    
    def sync_crossring_latency_record(self, other_flit: 'CrossRingFlit') -> None:
        """同步CrossRing特有的延迟记录"""
        # 先调用基类同步方法
        self.sync_latency_record(other_flit)
        
        # 同步CrossRing特有时间戳
        if other_flit.req_type == "read":
            self.cmd_entry_cake0_cycle = min(other_flit.cmd_entry_cake0_cycle, self.cmd_entry_cake0_cycle)
            self.cmd_entry_noc_from_cake0_cycle = min(other_flit.cmd_entry_noc_from_cake0_cycle, self.cmd_entry_noc_from_cake0_cycle)
            self.cmd_received_by_cake1_cycle = min(other_flit.cmd_received_by_cake1_cycle, self.cmd_received_by_cake1_cycle)
            self.data_entry_noc_from_cake1_cycle = min(other_flit.data_entry_noc_from_cake1_cycle, self.data_entry_noc_from_cake1_cycle)
            self.data_received_complete_cycle = min(other_flit.data_received_complete_cycle, self.data_received_complete_cycle)
        elif other_flit.req_type == "write":
            self.cmd_entry_cake0_cycle = min(other_flit.cmd_entry_cake0_cycle, self.cmd_entry_cake0_cycle)
            self.cmd_entry_noc_from_cake0_cycle = min(other_flit.cmd_entry_noc_from_cake0_cycle, self.cmd_entry_noc_from_cake0_cycle)
            self.cmd_received_by_cake1_cycle = min(other_flit.cmd_received_by_cake1_cycle, self.cmd_received_by_cake1_cycle)
            self.cmd_entry_noc_from_cake1_cycle = min(other_flit.cmd_entry_noc_from_cake1_cycle, self.cmd_entry_noc_from_cake1_cycle)
            self.cmd_received_by_cake0_cycle = min(other_flit.cmd_received_by_cake0_cycle, self.cmd_received_by_cake0_cycle)
            self.data_entry_noc_from_cake0_cycle = min(other_flit.data_entry_noc_from_cake0_cycle, self.data_entry_noc_from_cake0_cycle)
            self.data_received_complete_cycle = min(other_flit.data_received_complete_cycle, self.data_received_complete_cycle)
    
    def calculate_crossring_latencies(self) -> Dict[str, float]:
        """计算CrossRing特有的延迟指标"""
        latencies = {}
        
        # 命令延迟
        if (self.cmd_entry_noc_from_cake0_cycle < np.inf and 
            self.cmd_received_by_cake1_cycle < np.inf):
            latencies["cmd_latency"] = self.cmd_received_by_cake1_cycle - self.cmd_entry_noc_from_cake0_cycle
        
        # 数据延迟
        if self.req_type == "read":
            if (self.data_entry_noc_from_cake1_cycle < np.inf and 
                self.data_received_complete_cycle < np.inf):
                latencies["data_latency"] = self.data_received_complete_cycle - self.data_entry_noc_from_cake1_cycle
        elif self.req_type == "write":
            if (self.data_entry_noc_from_cake0_cycle < np.inf and 
                self.data_received_complete_cycle < np.inf):
                latencies["data_latency"] = self.data_received_complete_cycle - self.data_entry_noc_from_cake0_cycle
        
        # 事务延迟
        if (self.cmd_entry_cake0_cycle < np.inf and 
            self.data_received_complete_cycle < np.inf):
            latencies["transaction_latency"] = self.data_received_complete_cycle - self.cmd_entry_cake0_cycle
        
        return latencies
    
    def get_crossring_status(self) -> Dict[str, Any]:
        """获取CrossRing状态摘要"""
        status = self.get_status_summary()
        
        # 添加CrossRing特有信息
        status.update({
            "etag_priority": self.etag_priority,
            "itag_status": {"h": self.itag_h, "v": self.itag_v},
            "circuits": {"h": self.circuits_completed_h, "v": self.circuits_completed_v},
            "wait_cycles": {"h": self.wait_cycle_h, "v": self.wait_cycle_v},
            "coordinates": {"dest_x": self.dest_xid, "dest_y": self.dest_yid},
            "tracker_types": {"rn": self.rn_tracker_type, "sn": self.sn_tracker_type},
            "crossring_latencies": self.calculate_crossring_latencies(),
        })
        
        return status
    
    def __repr__(self) -> str:
        """CrossRing特有的字符串表示"""
        # 基础信息
        req_attr = "O" if self.req_attr == "old" else "N"
        type_display = self.rsp_type[:3] if self.rsp_type else self.req_type[0]
        
        # 位置信息
        if self.flit_position == "Link" and self.current_link:
            position_str = f"({self.current_position}: {self.current_link[0]}->{self.current_link[1]}).{self.current_seat_index}"
        else:
            position_str = f"{self.current_position}:{self.flit_position}"
        
        # 状态标识
        status = []
        if self.is_finish: status.append("F")
        if self.is_ejected: status.append("E")
        if self.itag_h: status.append("H")
        if self.itag_v: status.append("V")
        
        status_str = "".join(status) if status else ""
        
        # IP类型显示
        src_type = self.source_ip_type[0] + self.source_ip_type[-1] if self.source_ip_type else "??"
        dst_type = self.dest_ip_type[0] + self.dest_ip_type[-1] if self.dest_ip_type else "??"
        
        return (
            f"{self.packet_id}.{self.flit_id} {self.source}.{src_type}->{self.destination}.{dst_type}: "
            f"{position_str}, "
            f"{req_attr}, {self.flit_type}, {type_display}, "
            f"{status_str}, "
            f"{self.etag_priority}"
        )


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

CrossRingFlit._reset_for_reuse = _reset_crossring_for_reuse


# 工厂函数
def create_crossring_flit(source: NodeId,
                         destination: NodeId,
                         path: List[NodeId] = None,
                         num_col: int = 4,
                         **kwargs) -> CrossRingFlit:
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
    
    flit = CrossRingFlit(
        source=source,
        destination=destination,
        path=path,
        **kwargs
    )
    
    flit.set_crossring_coordinates(num_col)
    return flit