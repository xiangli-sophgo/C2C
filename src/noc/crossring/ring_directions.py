"""
CrossRing环形方向系统实现。

本模块实现CrossRing拓扑中的四方向通道系统：
- TL (Towards Left): 向左方向（水平逆时针）
- TR (Towards Right): 向右方向（水平顺时针）
- TU (Towards Up): 向上方向（垂直向上）
- TD (Towards Down): 向下方向（垂直向下）

提供方向映射、转换和选择功能。
"""

from typing import Dict, Tuple, Optional, List
from enum import Enum
import logging

from src.noc.utils.types import NodeId


class RingDirection(Enum):
    """环形方向枚举"""

    TL = "TL"  # Towards Left: 向左方向（水平逆时针）
    TR = "TR"  # Towards Right: 向右方向（水平顺时针）
    TU = "TU"  # Towards Up: 向上方向（垂直向上）
    TD = "TD"  # Towards Down: 向下方向（垂直向下）


class RingDirectionMapper:
    """
    环形方向映射器。

    负责在CrossRing的四方向系统和基本的顺时针/逆时针方向之间进行映射和转换。
    """

    def __init__(self, num_rows: int, num_cols: int):
        """
        初始化方向映射器

        Args:
            num_rows: 网格行数
            num_cols: 网格列数
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_nodes = num_rows * num_cols

        # 方向映射表
        self.direction_mapping = {
            # 水平方向映射
            ("horizontal", "clockwise"): RingDirection.TR,
            ("horizontal", "counter_clockwise"): RingDirection.TL,
            # 垂直方向映射
            ("vertical", "clockwise"): RingDirection.TD,
            ("vertical", "counter_clockwise"): RingDirection.TU,
        }

        # 反向映射表
        self.reverse_mapping = {
            RingDirection.TL: ("horizontal", "counter_clockwise"),
            RingDirection.TR: ("horizontal", "clockwise"),
            RingDirection.TU: ("vertical", "counter_clockwise"),
            RingDirection.TD: ("vertical", "clockwise"),
        }

        # 日志记录器
        self.logger = logging.getLogger("RingDirectionMapper")

    def get_node_coordinates(self, node_id: NodeId) -> Tuple[int, int]:
        """
        获取节点在网格中的坐标

        Args:
            node_id: 节点标识符

        Returns:
            (x, y)坐标元组
        """
        x = node_id % self.num_cols
        y = node_id // self.num_cols
        return x, y

    def get_node_id(self, x: int, y: int) -> NodeId:
        """
        根据坐标获取节点标识符

        Args:
            x: X坐标
            y: Y坐标

        Returns:
            节点标识符
        """
        return y * self.num_cols + x

    def determine_ring_direction(self, source: NodeId, destination: NodeId) -> Tuple[Optional[RingDirection], Optional[RingDirection]]:
        """
        确定从源节点到目标节点的环形方向

        Args:
            source: 源节点标识符
            destination: 目标节点标识符

        Returns:
            (水平方向, 垂直方向)元组，如果不需要移动则为None
        """
        if source == destination:
            return None, None

        src_x, src_y = self.get_node_coordinates(source)
        dst_x, dst_y = self.get_node_coordinates(destination)

        horizontal_direction = None
        vertical_direction = None

        # 确定水平方向
        if src_x != dst_x:
            horizontal_direction = self._determine_horizontal_direction(src_x, dst_x)

        # 确定垂直方向
        if src_y != dst_y:
            vertical_direction = self._determine_vertical_direction(src_y, dst_y)

        return horizontal_direction, vertical_direction

    def _determine_horizontal_direction(self, src_x: int, dst_x: int) -> RingDirection:
        """
        确定水平环形方向

        Args:
            src_x: 源X坐标
            dst_x: 目标X坐标

        Returns:
            水平环形方向
        """
        # 计算顺时针和逆时针距离
        if dst_x > src_x:
            # 目标在右侧
            clockwise_distance = dst_x - src_x
            counter_clockwise_distance = src_x + (self.num_cols - dst_x)
        else:
            # 目标在左侧
            clockwise_distance = dst_x + (self.num_cols - src_x)
            counter_clockwise_distance = src_x - dst_x

        # 选择最短路径
        if clockwise_distance <= counter_clockwise_distance:
            return RingDirection.TR  # 顺时针
        else:
            return RingDirection.TL  # 逆时针

    def _determine_vertical_direction(self, src_y: int, dst_y: int) -> RingDirection:
        """
        确定垂直环形方向

        Args:
            src_y: 源Y坐标
            dst_y: 目标Y坐标

        Returns:
            垂直环形方向
        """
        # 计算向下和向上距离
        if dst_y > src_y:
            # 目标在下方
            down_distance = dst_y - src_y
            up_distance = src_y + (self.num_rows - dst_y)
        else:
            # 目标在上方
            down_distance = dst_y + (self.num_rows - src_y)
            up_distance = src_y - dst_y

        # 选择最短路径
        if down_distance <= up_distance:
            return RingDirection.TD  # 向下
        else:
            return RingDirection.TU  # 向上

    def convert_to_basic_direction(self, ring_direction: RingDirection) -> Tuple[str, str]:
        """
        将环形方向转换为基本方向

        Args:
            ring_direction: 环形方向

        Returns:
            (维度, 基本方向)元组
        """
        return self.reverse_mapping[ring_direction]

    def convert_from_basic_direction(self, dimension: str, basic_direction: str) -> RingDirection:
        """
        从基本方向转换为环形方向

        Args:
            dimension: 维度（"horizontal"或"vertical"）
            basic_direction: 基本方向（"clockwise"或"counter_clockwise"）

        Returns:
            环形方向
        """
        return self.direction_mapping[(dimension, basic_direction)]

    def get_next_node_in_direction(self, current_node: NodeId, ring_direction: RingDirection) -> NodeId:
        """
        获取指定方向上的下一个节点

        Args:
            current_node: 当前节点标识符
            ring_direction: 环形方向

        Returns:
            下一个节点标识符
        """
        x, y = self.get_node_coordinates(current_node)

        if ring_direction == RingDirection.TR:
            # 水平顺时针：向右移动，带环绕
            next_x = (x + 1) % self.num_cols
            return self.get_node_id(next_x, y)

        elif ring_direction == RingDirection.TL:
            # 水平逆时针：向左移动，带环绕
            next_x = (x - 1) % self.num_cols
            return self.get_node_id(next_x, y)

        elif ring_direction == RingDirection.TD:
            # 垂直向下：向下移动，带环绕
            next_y = (y + 1) % self.num_rows
            return self.get_node_id(x, next_y)

        elif ring_direction == RingDirection.TU:
            # 垂直向上：向上移动，带环绕
            next_y = (y - 1) % self.num_rows
            return self.get_node_id(x, next_y)

        else:
            raise ValueError(f"未知的环形方向：{ring_direction}")

    def get_ring_path(self, source: NodeId, destination: NodeId) -> List[Tuple[NodeId, RingDirection]]:
        """
        获取从源节点到目标节点的完整环形路径

        Args:
            source: 源节点标识符
            destination: 目标节点标识符

        Returns:
            路径列表，每个元素为(节点ID, 使用的环形方向)
        """
        if source == destination:
            return [(source, None)]

        path = []
        current_node = source

        # 获取需要的方向
        horizontal_direction, vertical_direction = self.determine_ring_direction(source, destination)

        # 先水平移动（XY路由）
        if horizontal_direction:
            while True:
                next_node = self.get_next_node_in_direction(current_node, horizontal_direction)
                path.append((current_node, horizontal_direction))
                current_node = next_node

                # 检查是否到达目标列
                current_x, _ = self.get_node_coordinates(current_node)
                target_x, _ = self.get_node_coordinates(destination)
                if current_x == target_x:
                    break

        # 再垂直移动
        if vertical_direction:
            while current_node != destination:
                next_node = self.get_next_node_in_direction(current_node, vertical_direction)
                path.append((current_node, vertical_direction))
                current_node = next_node

        # 添加目标节点
        if not path or path[-1][0] != destination:
            path.append((destination, None))

        return path

    def validate_ring_connectivity(self) -> bool:
        """
        验证环形连接的正确性

        Returns:
            验证是否通过
        """
        try:
            # 验证水平环的连接
            for row in range(self.num_rows):
                for col in range(self.num_cols):
                    node_id = self.get_node_id(col, row)

                    # 测试顺时针连接
                    next_tr = self.get_next_node_in_direction(node_id, RingDirection.TR)
                    expected_tr = self.get_node_id((col + 1) % self.num_cols, row)
                    if next_tr != expected_tr:
                        self.logger.error(f"水平顺时针连接错误：节点{node_id} -> {next_tr}，期望{expected_tr}")
                        return False

                    # 测试逆时针连接
                    next_tl = self.get_next_node_in_direction(node_id, RingDirection.TL)
                    expected_tl = self.get_node_id((col - 1) % self.num_cols, row)
                    if next_tl != expected_tl:
                        self.logger.error(f"水平逆时针连接错误：节点{node_id} -> {next_tl}，期望{expected_tl}")
                        return False

            # 验证垂直环的连接
            for col in range(self.num_cols):
                for row in range(self.num_rows):
                    node_id = self.get_node_id(col, row)

                    # 测试向下连接
                    next_td = self.get_next_node_in_direction(node_id, RingDirection.TD)
                    expected_td = self.get_node_id(col, (row + 1) % self.num_rows)
                    if next_td != expected_td:
                        self.logger.error(f"垂直向下连接错误：节点{node_id} -> {next_td}，期望{expected_td}")
                        return False

                    # 测试向上连接
                    next_tu = self.get_next_node_in_direction(node_id, RingDirection.TU)
                    expected_tu = self.get_node_id(col, (row - 1) % self.num_rows)
                    if next_tu != expected_tu:
                        self.logger.error(f"垂直向上连接错误：节点{node_id} -> {next_tu}，期望{expected_tu}")
                        return False

            self.logger.info("环形连接验证通过")
            return True

        except Exception as e:
            self.logger.error(f"环形连接验证失败：{e}")
            return False

    def get_direction_statistics(self) -> Dict[str, int]:
        """
        获取方向使用统计信息

        Returns:
            方向统计字典
        """
        return {
            "total_directions": len(RingDirection),
            "horizontal_directions": 2,  # TL, TR
            "vertical_directions": 2,  # TU, TD
            "grid_size": f"{self.num_rows}x{self.num_cols}",
            "total_nodes": self.num_nodes,
        }
