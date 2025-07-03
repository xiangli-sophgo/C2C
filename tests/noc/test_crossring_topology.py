"""
CrossRing拓扑测试模块。

本模块包含CrossRing拓扑实现的完整测试用例，包括：
- 拓扑构建测试
- 路径计算测试
- 拓扑分析测试
- 性能指标测试
- 边界情况测试
"""

import unittest
import numpy as np
import logging
from typing import List, Dict, Tuple

# 导入待测试模块
# import sys
# import os
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.noc.crossring.topology import CrossRingTopology
from src.noc.crossring.config import CrossRingConfig
from src.noc.types import RoutingStrategy, TopologyType
from src.noc.utils.adjacency import create_crossring_adjacency_matrix, validate_adjacency_matrix


class TestCrossRingTopology(unittest.TestCase):
    """CrossRing拓扑测试类。"""

    def setUp(self):
        """测试前置设置。"""
        # 禁用日志输出以避免测试时的干扰
        logging.disable(logging.CRITICAL)

        # 创建测试配置
        self.small_config = CrossRingConfig(num_row=3, num_col=3)

        self.medium_config = CrossRingConfig(num_row=4, num_col=4)

        self.large_config = CrossRingConfig(num_row=8, num_col=8)

    def tearDown(self):
        """测试后清理。"""
        # 重新启用日志
        logging.disable(logging.NOTSET)

    def test_crossring_initialization(self):
        """测试CrossRing拓扑初始化。"""
        # 测试小规模拓扑
        topology = CrossRingTopology(self.small_config)
        self.assertEqual(topology.num_nodes, 9)
        self.assertEqual(topology.num_rows, 3)
        self.assertEqual(topology.num_cols, 3)
        self.assertEqual(topology.topology_type, TopologyType.CROSSRING)

        # 测试中等规模拓扑
        topology_medium = CrossRingTopology(self.medium_config)
        self.assertEqual(topology_medium.num_nodes, 16)
        self.assertEqual(topology_medium.num_rows, 4)
        self.assertEqual(topology_medium.num_cols, 4)

    def test_invalid_configuration(self):
        """测试无效配置处理。"""
        # 测试无效的行列数
        with self.assertRaises(ValueError):
            invalid_config = CrossRingConfig(num_row=1, num_col=3)
            CrossRingTopology(invalid_config)

        with self.assertRaises(ValueError):
            invalid_config = CrossRingConfig(num_row=3, num_col=1)
            CrossRingTopology(invalid_config)

    def test_topology_building(self):
        """测试拓扑构建。"""
        topology = CrossRingTopology(self.small_config)

        # 验证邻接矩阵存在且正确
        adj_matrix = topology.get_adjacency_matrix()
        self.assertIsNotNone(adj_matrix)
        self.assertEqual(len(adj_matrix), 9)

        # 验证邻接矩阵有效性
        is_valid, error_msg = validate_adjacency_matrix(adj_matrix)
        self.assertTrue(is_valid, f"邻接矩阵无效: {error_msg}")

        # 验证节点位置映射
        for node_id in range(topology.num_nodes):
            position = topology.get_node_position(node_id)
            self.assertIsInstance(position, tuple)
            self.assertEqual(len(position), 2)
            row, col = position
            self.assertTrue(0 <= row < topology.num_rows)
            self.assertTrue(0 <= col < topology.num_cols)

    def test_neighbor_calculation(self):
        """测试邻居节点计算。"""
        topology = CrossRingTopology(self.small_config)

        # 在3x3网格中，不同位置的节点有不同数量的邻居
        # 角落节点：2个邻居
        # 边缘节点：3个邻居
        # 中心节点：4个邻居
        expected_neighbors = {
            0: 2,  # 左上角
            1: 3,  # 上边
            2: 2,  # 右上角
            3: 3,  # 左边
            4: 4,  # 中心
            5: 3,  # 右边
            6: 2,  # 左下角
            7: 3,  # 下边
            8: 2,  # 右下角
        }

        # 测试所有节点的邻居数量
        for node_id in range(topology.num_nodes):
            neighbors = topology.get_neighbors(node_id)
            expected_count = expected_neighbors[node_id]
            self.assertEqual(len(neighbors), expected_count, f"节点{node_id}应该有{expected_count}个邻居")

            # 验证邻居节点有效性
            for neighbor in neighbors:
                self.assertTrue(0 <= neighbor < topology.num_nodes)
                self.assertNotEqual(neighbor, node_id)

        # 测试具体节点的邻居关系（以3x3网格为例）
        # 节点0 (0,0) 的邻居应该是: 1(右), 3(下)
        neighbors_0 = topology.get_neighbors(0)
        self.assertIn(1, neighbors_0)  # 右邻居
        self.assertIn(3, neighbors_0)  # 下邻居

        # 测试中心节点4 (1,1) 的邻居
        neighbors_4 = topology.get_neighbors(4)
        self.assertIn(3, neighbors_4)  # 左邻居
        self.assertIn(5, neighbors_4)  # 右邻居
        self.assertIn(1, neighbors_4)  # 上邻居
        self.assertIn(7, neighbors_4)  # 下邻居

    def test_shortest_path_calculation(self):
        """测试最短路径计算。"""
        topology = CrossRingTopology(self.small_config)

        # 测试同一节点的路径
        path = topology.calculate_shortest_path(0, 0)
        self.assertEqual(path, [0])

        # 测试相邻节点的路径
        path = topology.calculate_shortest_path(0, 1)
        self.assertEqual(len(path), 2)
        self.assertEqual(path[0], 0)
        self.assertEqual(path[-1], 1)

        # 测试对角线路径（节点0到节点8）
        path = topology.calculate_shortest_path(0, 8)
        self.assertGreater(len(path), 1)
        self.assertEqual(path[0], 0)
        self.assertEqual(path[-1], 8)

        # 验证路径连续性
        for i in range(len(path) - 1):
            neighbors = topology.get_neighbors(path[i])
            self.assertIn(path[i + 1], neighbors)

    def test_hv_path_calculation(self):
        """测试水平优先(HV)路径计算。"""
        topology = CrossRingTopology(self.small_config)

        # 测试从左上角到右下角的HV路径
        path = topology.calculate_hv_path(0, 8)
        self.assertEqual(path[0], 0)
        self.assertEqual(path[-1], 8)

        # 验证HV路径特性：先水平移动
        start_row, start_col = topology.get_node_position(path[0])

        # 找到第一个垂直移动的位置
        horizontal_phase = True
        for i in range(1, len(path)):
            curr_row, curr_col = topology.get_node_position(path[i])
            if horizontal_phase and curr_row != start_row:
                horizontal_phase = False
            elif not horizontal_phase and curr_col != topology.get_node_position(path[i - 1])[1]:
                self.fail("HV路径在垂直移动后不应再水平移动")

    def test_vh_path_calculation(self):
        """测试垂直优先(VH)路径计算。"""
        topology = CrossRingTopology(self.small_config)

        # 测试从左上角到右下角的VH路径
        path = topology.calculate_vh_path(0, 8)
        self.assertEqual(path[0], 0)
        self.assertEqual(path[-1], 8)

        # 验证VH路径特性：先垂直移动
        start_row, start_col = topology.get_node_position(path[0])

        # 找到第一个水平移动的位置
        vertical_phase = True
        for i in range(1, len(path)):
            curr_row, curr_col = topology.get_node_position(path[i])
            if vertical_phase and curr_col != start_col:
                vertical_phase = False
            elif not vertical_phase and curr_row != topology.get_node_position(path[i - 1])[0]:
                self.fail("VH路径在水平移动后不应再垂直移动")

    def test_ring_distance_calculation(self):
        """测试环内距离计算。"""
        topology = CrossRingTopology(self.small_config)

        # 测试水平环距离
        # 节点0和节点2在同一行，距离为2（0->1->2）
        h_distance = topology.get_ring_distance(0, 2, "horizontal")
        self.assertEqual(h_distance, 2)

        # 测试垂直环距离
        # 节点0和节点6在同一列，距离为2（0->3->6）
        v_distance = topology.get_ring_distance(0, 6, "vertical")
        self.assertEqual(v_distance, 2)

        # 测试相邻节点的距离
        # 节点0和节点1在同一行，距离为1
        h_distance_adjacent = topology.get_ring_distance(0, 1, "horizontal")
        self.assertEqual(h_distance_adjacent, 1)

        # 测试不在同一环的情况
        # 节点0和节点4不在同一行
        h_distance_diff_row = topology.get_ring_distance(0, 4, "horizontal")
        self.assertEqual(h_distance_diff_row, float("inf"))

        # 测试无效方向
        with self.assertRaises(ValueError):
            topology.get_ring_distance(0, 1, "invalid")

    def test_ring_structure_analysis(self):
        """测试环结构分析。"""
        topology = CrossRingTopology(self.small_config)

        # 测试水平环
        h_rings = topology.get_horizontal_rings()
        self.assertEqual(len(h_rings), 3)  # 3行

        # 验证每个水平环的内容
        self.assertEqual(h_rings[0], [0, 1, 2])  # 第一行
        self.assertEqual(h_rings[1], [3, 4, 5])  # 第二行
        self.assertEqual(h_rings[2], [6, 7, 8])  # 第三行

        # 测试垂直环
        v_rings = topology.get_vertical_rings()
        self.assertEqual(len(v_rings), 3)  # 3列

        # 验证每个垂直环的内容
        self.assertEqual(v_rings[0], [0, 3, 6])  # 第一列
        self.assertEqual(v_rings[1], [1, 4, 7])  # 第二列
        self.assertEqual(v_rings[2], [2, 5, 8])  # 第三列

    def test_hop_count_calculation(self):
        """测试跳数计算。"""
        topology = CrossRingTopology(self.small_config)

        # 测试同一节点的跳数
        hop_count = topology.get_hop_count(0, 0)
        self.assertEqual(hop_count, 0)

        # 测试相邻节点的跳数
        hop_count = topology.get_hop_count(0, 1)
        self.assertEqual(hop_count, 1)

        # 测试对角线节点的跳数
        hop_count = topology.get_hop_count(0, 8)
        self.assertGreater(hop_count, 0)
        self.assertLessEqual(hop_count, 4)  # 在3x3网格中最大跳数不应超过4

    def test_diameter_calculation(self):
        """测试网络直径计算。"""
        topology = CrossRingTopology(self.small_config)

        diameter = topology.get_diameter()
        self.assertGreater(diameter, 0)
        self.assertLessEqual(diameter, 4)  # 3x3 CrossRing的理论最大直径

        # 测试更大规模拓扑的直径
        topology_large = CrossRingTopology(self.large_config)
        diameter_large = topology_large.get_diameter()
        self.assertGreater(diameter_large, diameter)

    def test_average_hop_count(self):
        """测试平均跳数计算。"""
        topology = CrossRingTopology(self.small_config)

        avg_hop_count = topology.get_average_hop_count()
        self.assertGreater(avg_hop_count, 0)
        self.assertLess(avg_hop_count, topology.get_diameter())

    def test_topology_efficiency(self):
        """测试拓扑效率计算。"""
        topology = CrossRingTopology(self.small_config)

        efficiency = topology.get_topology_efficiency()
        self.assertGreater(efficiency, 0.0)
        self.assertLessEqual(efficiency, 1.0)

    def test_load_distribution_analysis(self):
        """测试负载分布分析。"""
        topology = CrossRingTopology(self.small_config)

        # 初始状态下的负载分布
        load_dist = topology.calculate_load_distribution()
        self.assertIn("total_links", load_dist)
        self.assertIn("active_links", load_dist)
        self.assertIn("load_variance", load_dist)
        self.assertIn("load_distribution", load_dist)
        self.assertIn("utilization_stats", load_dist)

        # 验证初始状态
        self.assertEqual(load_dist["active_links"], 0)
        self.assertEqual(load_dist["load_variance"], 0.0)

    def test_topology_validation(self):
        """测试拓扑验证。"""
        topology = CrossRingTopology(self.small_config)

        # 验证拓扑有效性
        is_valid, error_msg = topology.validate_topology()
        self.assertTrue(is_valid, f"拓扑验证失败: {error_msg}")

        # 验证连通性
        self.assertTrue(topology.is_connected())

    def test_adaptive_routing(self):
        """测试自适应路由。"""
        config = CrossRingConfig(num_row=3, num_col=3)
        topology = CrossRingTopology(config)

        # 测试自适应路由路径计算
        path = topology.calculate_route(0, 8, RoutingStrategy.ADAPTIVE)
        self.assertGreater(len(path), 1)
        self.assertEqual(path[0], 0)
        self.assertEqual(path[-1], 8)

    def test_load_balanced_routing(self):
        """测试负载均衡路由。"""
        config = CrossRingConfig(num_row=3, num_col=3)
        topology = CrossRingTopology(config)

        # 测试负载均衡路由路径计算
        path = topology.calculate_route(0, 8, RoutingStrategy.LOAD_BALANCED)
        self.assertGreater(len(path), 1)
        self.assertEqual(path[0], 0)
        self.assertEqual(path[-1], 8)

    def test_crossring_info(self):
        """测试CrossRing信息获取。"""
        topology = CrossRingTopology(self.small_config)

        info = topology.get_crossring_info()
        self.assertIn("num_rows", info)
        self.assertIn("num_cols", info)
        self.assertIn("horizontal_rings", info)
        self.assertIn("vertical_rings", info)
        self.assertIn("topology_efficiency", info)
        self.assertIn("supported_routing", info)
        self.assertIn("load_distribution", info)

        # 验证信息正确性
        self.assertEqual(info["num_rows"], 3)
        self.assertEqual(info["num_cols"], 3)
        self.assertEqual(info["horizontal_rings"], 3)
        self.assertEqual(info["vertical_rings"], 3)
        self.assertIn("HV", info["supported_routing"])
        self.assertIn("VH", info["supported_routing"])

    def test_boundary_cases(self):
        """测试边界情况。"""
        topology = CrossRingTopology(self.small_config)

        # 测试无效节点ID
        with self.assertRaises(ValueError):
            topology.get_neighbors(-1)

        with self.assertRaises(ValueError):
            topology.get_neighbors(topology.num_nodes)

        with self.assertRaises(ValueError):
            topology.get_node_position(-1)

        with self.assertRaises(ValueError):
            topology.get_node_position(topology.num_nodes)

    def test_performance_large_topology(self):
        """测试大规模拓扑性能。"""
        import time

        # 测试16x16拓扑构建时间
        start_time = time.time()
        large_config = CrossRingConfig(num_row=16, num_col=16)
        topology = CrossRingTopology(large_config)
        build_time = time.time() - start_time

        # 验证构建时间小于1秒
        self.assertLess(build_time, 1.0, "16x16拓扑构建时间应小于1秒")

        # 验证拓扑正确性
        self.assertEqual(topology.num_nodes, 256)
        is_valid, error_msg = topology.validate_topology()
        self.assertTrue(is_valid, f"大规模拓扑验证失败: {error_msg}")

    def test_string_representations(self):
        """测试字符串表示。"""
        topology = CrossRingTopology(self.small_config)

        # 测试__str__方法
        str_repr = str(topology)
        self.assertIn("CrossRingTopology", str_repr)
        self.assertIn("3×3", str_repr)
        self.assertIn("nodes=9", str_repr)

        # 测试__repr__方法
        repr_str = repr(topology)
        self.assertIn("CrossRingTopology", repr_str)
        self.assertIn("rows=3", repr_str)
        self.assertIn("cols=3", repr_str)


class TestCrossRingTopologyIntegration(unittest.TestCase):
    """CrossRing拓扑集成测试类。"""

    def setUp(self):
        """集成测试前置设置。"""
        logging.disable(logging.CRITICAL)
        self.config = CrossRingConfig(num_row=4, num_col=4)
        self.topology = CrossRingTopology(self.config)

    def tearDown(self):
        """集成测试后清理。"""
        logging.disable(logging.NOTSET)

    def test_routing_consistency(self):
        """测试路由一致性。"""
        # 对比不同路由策略的结果
        src, dst = 0, 15

        shortest_path = self.topology.calculate_route(src, dst, RoutingStrategy.SHORTEST)
        hv_path = self.topology.calculate_route(src, dst, RoutingStrategy.DETERMINISTIC)
        vh_path = self.topology.calculate_route(src, dst, RoutingStrategy.MINIMAL)

        # 所有路径都应该有效
        self.assertGreater(len(shortest_path), 1)
        self.assertGreater(len(hv_path), 1)
        self.assertGreater(len(vh_path), 1)

        # 所有路径都应该到达目标
        self.assertEqual(shortest_path[-1], dst)
        self.assertEqual(hv_path[-1], dst)
        self.assertEqual(vh_path[-1], dst)

    def test_topology_metrics_correlation(self):
        """测试拓扑指标相关性。"""
        diameter = self.topology.get_diameter()
        avg_hop_count = self.topology.get_average_hop_count()
        efficiency = self.topology.get_topology_efficiency()

        # 平均跳数应该小于直径
        self.assertLess(avg_hop_count, diameter)

        # 效率应该与平均跳数成反比
        self.assertGreater(efficiency, 0)
        self.assertLess(efficiency * avg_hop_count, 2.0)


if __name__ == "__main__":
    # 设置测试运行器
    unittest.main(verbosity=2, buffer=True)
