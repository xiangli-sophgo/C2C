"""
CrossRing环形桥接组件实现。

本模块实现CrossRing拓扑中的环形桥接（Ring Bridge）功能，
负责处理水平环和垂直环之间的交叉点逻辑，包括：
- 维度转换（水平→垂直，垂直→水平）
- 环形仲裁（解决水平环和垂直环之间的冲突）
- 流量控制（背压和拥塞控制）
- 环形切片流水线处理
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
from collections import deque

from .flit import CrossRingFlit
from .config import CrossRingConfig, RoutingStrategy
from src.noc.utils.types import NodeId


class RingSlice:
    """
    环形切片类，实现环形桥接中的流水线处理阶段。
    
    每个环形切片负责处理一个流水线阶段的数据包处理：
    1. 输入阶段：从环形接收数据包
    2. 仲裁阶段：解决环形之间的冲突
    3. 转换阶段：执行维度转换（如需要）
    4. 输出阶段：将数据包转发到目标环形
    """
    
    def __init__(self, slice_id: int, config: CrossRingConfig):
        """
        初始化环形切片
        
        Args:
            slice_id: 切片标识符
            config: CrossRing配置
        """
        self.slice_id = slice_id
        self.config = config
        
        # 流水线阶段缓冲区
        self.input_buffer: List[CrossRingFlit] = []
        self.arbitration_buffer: List[CrossRingFlit] = []
        self.turning_buffer: List[CrossRingFlit] = []
        self.output_buffer: List[CrossRingFlit] = []
        
        # 仲裁状态
        self.arbitration_winner = None
        self.last_arbitration_cycle = 0
        
        # 日志记录器
        self.logger = logging.getLogger(f"RingSlice_{slice_id}")
    
    def process_pipeline_stage(self, stage: str, cycle: int) -> None:
        """
        处理指定的流水线阶段
        
        Args:
            stage: 流水线阶段名称
            cycle: 当前仿真周期
        """
        if stage == "input":
            self._process_input_stage(cycle)
        elif stage == "arbitration":
            self._process_arbitration_stage(cycle)
        elif stage == "turning":
            self._process_turning_stage(cycle)
        elif stage == "output":
            self._process_output_stage(cycle)
    
    def _process_input_stage(self, cycle: int) -> None:
        """处理输入阶段：从环形接收数据包"""
        # 输入阶段的具体实现将在后续添加
        pass
    
    def _process_arbitration_stage(self, cycle: int) -> None:
        """处理仲裁阶段：解决环形之间的冲突"""
        # 仲裁阶段的具体实现将在后续添加
        pass
    
    def _process_turning_stage(self, cycle: int) -> None:
        """处理转换阶段：执行维度转换"""
        # 转换阶段的具体实现将在后续添加
        pass
    
    def _process_output_stage(self, cycle: int) -> None:
        """处理输出阶段：将数据包转发到目标环形"""
        # 输出阶段的具体实现将在后续添加
        pass


class CrossPointModule:
    """
    交叉点模块类，管理环形交叉点的通信。
    
    交叉点模块位于环形交叉处，负责：
    - 维度转换功能（水平→垂直，垂直→水平）
    - 环形仲裁（管理水平环和垂直环之间的冲突）
    - 流量控制（实施背压和拥塞控制）
    """
    
    def __init__(self, node_id: NodeId, config: CrossRingConfig):
        """
        初始化交叉点模块
        
        Args:
            node_id: 节点标识符
            config: CrossRing配置
        """
        self.node_id = node_id
        self.config = config
        
        # 维度转换缓冲区
        self.h2v_buffer: deque[CrossRingFlit] = deque(maxlen=config.crosspoint_buffer_depth)
        self.v2h_buffer: deque[CrossRingFlit] = deque(maxlen=config.crosspoint_buffer_depth)
        
        # 仲裁状态
        self.horizontal_priority = True  # True表示水平环优先
        self.arbitration_counter = 0
        
        # 流量控制状态
        self.congestion_detected = False
        self.backpressure_active = False
        
        # 日志记录器
        self.logger = logging.getLogger(f"CrossPoint_{node_id}")
        self.logger.debug(f"交叉点模块初始化完成：节点{node_id}")
    
    def process_dimension_turning(self, flit: CrossRingFlit, from_direction: str, to_direction: str, cycle: int) -> bool:
        """
        处理维度转换
        
        Args:
            flit: 要转换的数据包
            from_direction: 源方向（"horizontal"或"vertical"）
            to_direction: 目标方向（"horizontal"或"vertical"）
            cycle: 当前仿真周期
            
        Returns:
            转换是否成功
        """
        if from_direction == "horizontal" and to_direction == "vertical":
            return self._process_h2v_turning(flit, cycle)
        elif from_direction == "vertical" and to_direction == "horizontal":
            return self._process_v2h_turning(flit, cycle)
        else:
            self.logger.warning(f"无效的维度转换请求：{from_direction} → {to_direction}")
            return False
    
    def _process_h2v_turning(self, flit: CrossRingFlit, cycle: int) -> bool:
        """
        处理水平到垂直的维度转换
        
        Args:
            flit: 要转换的数据包
            cycle: 当前仿真周期
            
        Returns:
            转换是否成功
        """
        if len(self.h2v_buffer) >= self.h2v_buffer.maxlen:
            self.logger.debug(f"周期{cycle}：H2V缓冲区已满，无法转换数据包{flit.packet_id}")
            return False
        
        # 执行维度转换
        flit.dimension_turn_cycle = cycle
        flit.current_direction = "vertical"
        self.h2v_buffer.append(flit)
        
        self.logger.debug(f"周期{cycle}：成功执行H2V维度转换，数据包{flit.packet_id}")
        return True
    
    def _process_v2h_turning(self, flit: CrossRingFlit, cycle: int) -> bool:
        """
        处理垂直到水平的维度转换
        
        Args:
            flit: 要转换的数据包
            cycle: 当前仿真周期
            
        Returns:
            转换是否成功
        """
        if len(self.v2h_buffer) >= self.v2h_buffer.maxlen:
            self.logger.debug(f"周期{cycle}：V2H缓冲区已满，无法转换数据包{flit.packet_id}")
            return False
        
        # 执行维度转换
        flit.dimension_turn_cycle = cycle
        flit.current_direction = "horizontal"
        self.v2h_buffer.append(flit)
        
        self.logger.debug(f"周期{cycle}：成功执行V2H维度转换，数据包{flit.packet_id}")
        return True
    
    def arbitrate_ring_access(self, horizontal_request: bool, vertical_request: bool, cycle: int, routing_strategy: RoutingStrategy = RoutingStrategy.XY) -> Tuple[bool, bool]:
        """
        仲裁环形访问权限 - 支持基于路由策略的优先级调整
        
        Args:
            horizontal_request: 水平环是否有访问请求
            vertical_request: 垂直环是否有访问请求
            cycle: 当前仿真周期
            routing_strategy: 路由策略，影响仲裁优先级
            
        Returns:
            (水平环是否获得访问权, 垂直环是否获得访问权)
        """
        # 如果只有一个方向有请求，直接授权
        if horizontal_request and not vertical_request:
            return True, False
        elif vertical_request and not horizontal_request:
            return False, True
        elif not horizontal_request and not vertical_request:
            return False, False
        
        # 两个方向都有请求时，根据路由策略调整仲裁优先级
        self.arbitration_counter += 1
        
        if routing_strategy == RoutingStrategy.XY:
            # XY路由：优先水平环
            if self.arbitration_counter % 3 == 0:
                # 偶尔给垂直环机会
                self.logger.debug(f"周期{cycle}：XY路由仲裁 - 垂直环获得访问权")
                return False, True
            else:
                self.logger.debug(f"周期{cycle}：XY路由仲裁 - 水平环获得访问权")
                return True, False
        elif routing_strategy == RoutingStrategy.YX:
            # YX路由：优先垂直环
            if self.arbitration_counter % 3 == 0:
                # 偶尔给水平环机会
                self.logger.debug(f"周期{cycle}：YX路由仲裁 - 水平环获得访问权")
                return True, False
            else:
                self.logger.debug(f"周期{cycle}：YX路由仲裁 - 垂直环获得访问权")
                return False, True
        else:
            # ADAPTIVE或其他：使用轮询仲裁
            if self.arbitration_counter % 2 == 0:
                self.logger.debug(f"周期{cycle}：自适应仲裁 - 水平环获得访问权")
                return True, False
            else:
                self.logger.debug(f"周期{cycle}：自适应仲裁 - 垂直环获得访问权")
                return False, True
    
    def detect_congestion(self, cycle: int) -> bool:
        """
        检测拥塞状态
        
        Args:
            cycle: 当前仿真周期
            
        Returns:
            是否检测到拥塞
        """
        # 基于缓冲区占用率检测拥塞
        h2v_occupancy = len(self.h2v_buffer) / self.h2v_buffer.maxlen
        v2h_occupancy = len(self.v2h_buffer) / self.v2h_buffer.maxlen
        
        congestion_threshold = 0.8  # 80%阈值
        
        self.congestion_detected = (h2v_occupancy >= congestion_threshold or 
                                   v2h_occupancy >= congestion_threshold)
        
        if self.congestion_detected:
            self.logger.debug(f"周期{cycle}：检测到拥塞 - H2V占用率:{h2v_occupancy:.2f}, V2H占用率:{v2h_occupancy:.2f}")
        
        return self.congestion_detected
    
    def apply_backpressure(self, direction: str, cycle: int) -> None:
        """
        应用背压控制
        
        Args:
            direction: 应用背压的方向
            cycle: 当前仿真周期
        """
        self.backpressure_active = True
        self.logger.debug(f"周期{cycle}：对{direction}方向应用背压控制")
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取交叉点模块状态
        
        Returns:
            状态信息字典
        """
        return {
            "node_id": self.node_id,
            "h2v_buffer_occupancy": len(self.h2v_buffer),
            "v2h_buffer_occupancy": len(self.v2h_buffer),
            "congestion_detected": self.congestion_detected,
            "backpressure_active": self.backpressure_active,
            "arbitration_counter": self.arbitration_counter,
        }


class RingBridge:
    """
    环形桥接主类，管理整个环形桥接系统。
    
    环形桥接负责：
    - 管理多个交叉点模块
    - 协调环形切片流水线
    - 实施全局流量控制策略
    """
    
    def __init__(self, config: CrossRingConfig):
        """
        初始化环形桥接
        
        Args:
            config: CrossRing配置
        """
        self.config = config
        
        # 创建所有节点的交叉点模块
        self.cross_points: Dict[NodeId, CrossPointModule] = {}
        for node_id in range(config.num_nodes):
            self.cross_points[node_id] = CrossPointModule(node_id, config)
        
        # 创建环形切片
        self.ring_slices: Dict[int, RingSlice] = {}
        for slice_id in range(config.slice_per_link):
            self.ring_slices[slice_id] = RingSlice(slice_id, config)
        
        # 全局统计信息
        self.stats = {
            "total_h2v_turns": 0,
            "total_v2h_turns": 0,
            "total_arbitrations": 0,
            "congestion_events": 0,
        }
        
        # 日志记录器
        self.logger = logging.getLogger("RingBridge")
        self.logger.info(f"环形桥接初始化完成：{len(self.cross_points)}个交叉点，{len(self.ring_slices)}个环形切片")
    
    def process_all_cross_points(self, cycle: int) -> None:
        """
        处理所有交叉点模块
        
        Args:
            cycle: 当前仿真周期
        """
        for cross_point in self.cross_points.values():
            # 检测拥塞
            if cross_point.detect_congestion(cycle):
                self.stats["congestion_events"] += 1
            
            # 处理维度转换缓冲区中的数据包
            self._process_cross_point_buffers(cross_point, cycle)
    
    def _process_cross_point_buffers(self, cross_point: CrossPointModule, cycle: int) -> None:
        """
        处理交叉点缓冲区中的数据包
        
        Args:
            cross_point: 交叉点模块
            cycle: 当前仿真周期
        """
        # 处理H2V缓冲区
        if cross_point.h2v_buffer:
            flit = cross_point.h2v_buffer.popleft()
            # 这里应该将flit转发到垂直环，具体实现将在模型中完成
            self.stats["total_h2v_turns"] += 1
        
        # 处理V2H缓冲区
        if cross_point.v2h_buffer:
            flit = cross_point.v2h_buffer.popleft()
            # 这里应该将flit转发到水平环，具体实现将在模型中完成
            self.stats["total_v2h_turns"] += 1
    
    def get_global_status(self) -> Dict[str, Any]:
        """
        获取全局状态信息
        
        Returns:
            全局状态信息字典
        """
        cross_point_status = {}
        for node_id, cross_point in self.cross_points.items():
            cross_point_status[node_id] = cross_point.get_status()
        
        return {
            "stats": self.stats,
            "cross_points": cross_point_status,
            "total_cross_points": len(self.cross_points),
            "total_ring_slices": len(self.ring_slices),
        }
