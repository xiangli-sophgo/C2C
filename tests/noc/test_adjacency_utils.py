"""
邻接矩阵工具测试模块。

本模块包含邻接矩阵生成和验证工具的完整测试用例，包括：
- CrossRing邻接矩阵生成测试
- 邻接矩阵验证测试
- 图连通性测试
- 节点度数分析测试
- 图算法测试
"""

import unittest
import numpy as np
import tempfile
import os
import json
import csv

# 导入待测试模块
# import sys
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.noc.utils.adjacency import (
    create_crossring_adjacency_matrix,
    validate_adjacency_matrix,
    check_connectivity,
    analyze_node_degrees,
    get_node_neighbors,
    calculate_graph_diameter,
    calculate_clustering_coefficient,
    export_adjacency_matrix,
)


class TestCrossRingAdjacencyMatrix(unittest.TestCase):
    """CrossRing邻接矩阵生成测试类。"""

    def test_small_crossring_matrix(self):
        """测试小规模CrossRing邻接矩阵生成。"""
        # 测试2x2矩阵
        adj_matrix = create_crossring_adjacency_matrix(2, 2)
        self.assertEqual(len(adj_matrix), 4)
        self.assertEqual(len(adj_matrix[0]), 4)

        # 验证2x2网格的连接关系 (4-cycle: 0-1-3-2-0)
        # 节点0应该连接到节点1和2
        self.assertEqual(adj_matrix[0][1], 1)  # 连接到节点1
        self.assertEqual(adj_matrix[0][2], 1)  # 连接到节点2
        self.assertEqual(adj_matrix[0][3], 0)  # 不连接到节点3
        self.assertEqual(adj_matrix[0][0], 0)  # 不自连

        # 验证其他节点的连接
        self.assertEqual(sum(adj_matrix[0]), 2)  # 节点0的度为2
        self.assertEqual(sum(adj_matrix[1]), 2)  # 节点1的度为2
        self.assertEqual(sum(adj_matrix[2]), 2)  # 节点2的度为2
        self.assertEqual(sum(adj_matrix[3]), 2)  # 节点3的度为2

    def test_medium_crossring_matrix(self):
        """测试中等规模CrossRing邻接矩阵生成。"""
        # 测试3x3矩阵
        adj_matrix = create_crossring_adjacency_matrix(3, 3)
        adj_matrix = np.array(adj_matrix)
        node_num = 9
        # 1. 矩阵尺寸
        self.assertEqual(len(adj_matrix), node_num)
        # 2. 无自环，主对角线全 0
        self.assertTrue(np.all(np.diag(adj_matrix) == 0))
        # 3. 对称（无向图）
        self.assertTrue(np.array_equal(adj_matrix, adj_matrix.T))
        # 4. 每个点的邻居集合与预期一致
        expected = {
            0: {1, 3},
            1: {0, 2, 4},
            2: {1, 5},
            3: {0, 4, 6},
            4: {1, 3, 5, 7},
            5: {2, 4, 8},
            6: {3, 7},
            7: {4, 6, 8},
            8: {5, 7},
        }
        for i in range(node_num):
            actual_neighbors = {j for j in range(node_num) if adj_matrix[i, j] == 1}
            self.assertEqual(actual_neighbors, expected[i], f"节点 {i} 的邻居应是 {expected[i]}，但得到 {actual_neighbors}")
        # 5. 校验度分布
        degrees = {i: len(expected[i]) for i in range(node_num)}
        for i in range(node_num):
            self.assertEqual(adj_matrix[i].sum(), degrees[i], f"节点 {i} 的度应是 {degrees[i]}，但矩阵中是 {adj_matrix[i].sum()}")

    def test_rectangular_crossring_matrix(self):
        """测试矩形CrossRing邻接矩阵生成。"""
        # 测试3x4矩阵
        adj_matrix = create_crossring_adjacency_matrix(3, 4)
        self.assertEqual(len(adj_matrix), 12)

        # 验证特定节点的连接
        # 节点0 (0,0) 应该连接到: 1(右), 4(下)
        self.assertEqual(adj_matrix[0][1], 1)  # 右邻居
        self.assertEqual(adj_matrix[0][4], 1)  # 下邻居

    def test_large_crossring_matrix(self):
        """测试大规模CrossRing邻接矩阵生成。"""
        # 测试8x8矩阵
        adj_matrix = create_crossring_adjacency_matrix(8, 8)
        self.assertEqual(len(adj_matrix), 64)

        # 验证矩阵对称性
        for i in range(64):
            for j in range(64):
                self.assertEqual(adj_matrix[i][j], adj_matrix[j][i])

    def test_invalid_crossring_parameters(self):
        """测试无效参数处理。"""
        # 测试行数或列数小于2的情况
        with self.assertRaises(ValueError):
            create_crossring_adjacency_matrix(0, 3)

        with self.assertRaises(ValueError):
            create_crossring_adjacency_matrix(3, 0)

        with self.assertRaises(ValueError):
            create_crossring_adjacency_matrix(0, 5)

        with self.assertRaises(ValueError):
            create_crossring_adjacency_matrix(5, 0)

    def test_crossring_is_mesh(self):
        """验证CrossRing是Mesh拓扑（无环形连接）。"""
        adj_matrix = create_crossring_adjacency_matrix(4, 4)

        # 验证水平方向无环（第0行：节点0,1,2,3）
        self.assertEqual(adj_matrix[0][1], 1)  # 0->1
        self.assertEqual(adj_matrix[1][2], 1)  # 1->2
        self.assertEqual(adj_matrix[2][3], 1)  # 2->3
        self.assertEqual(adj_matrix[3][0], 0)  # 3->0 (无环)

        # 验证垂直方向无环（第0列：节点0,4,8,12）
        self.assertEqual(adj_matrix[0][4], 1)  # 0->4
        self.assertEqual(adj_matrix[4][8], 1)  # 4->8
        self.assertEqual(adj_matrix[8][12], 1)  # 8->12
        self.assertEqual(adj_matrix[12][0], 0)  # 12->0 (无环)


class TestAdjacencyMatrixValidation(unittest.TestCase):
    """邻接矩阵验证测试类。"""

    def test_valid_matrix_validation(self):
        """测试有效矩阵验证。"""
        # 创建有效的CrossRing邻接矩阵
        adj_matrix = create_crossring_adjacency_matrix(3, 3)
        is_valid, error_msg = validate_adjacency_matrix(adj_matrix)
        self.assertTrue(is_valid)
        self.assertIsNone(error_msg)

    def test_empty_matrix_validation(self):
        """测试空矩阵验证。"""
        is_valid, error_msg = validate_adjacency_matrix([])
        self.assertFalse(is_valid)
        self.assertIn("邻接矩阵不能为空", error_msg)

    def test_non_square_matrix_validation(self):
        """测试非方阵验证。"""
        # 创建非方阵
        invalid_matrix = [[1, 0], [1, 0, 1]]
        is_valid, error_msg = validate_adjacency_matrix(invalid_matrix)
        self.assertFalse(is_valid)
        self.assertIn("长度不匹配", error_msg)

    def test_invalid_elements_validation(self):
        """测试无效元素验证。"""
        # 包含非0/1元素的矩阵
        invalid_matrix = [[0, 1, 2], [1, 0, 1], [2, 1, 0]]
        is_valid, error_msg = validate_adjacency_matrix(invalid_matrix)
        self.assertFalse(is_valid)
        self.assertIn("必须为0或1", error_msg)

    def test_self_loop_validation(self):
        """测试自环验证。"""
        # 包含自环的矩阵
        invalid_matrix = [[1, 1, 0], [1, 0, 1], [0, 1, 0]]
        is_valid, error_msg = validate_adjacency_matrix(invalid_matrix)
        self.assertFalse(is_valid)
        self.assertIn("不允许自环", error_msg)

    def test_asymmetric_matrix_validation(self):
        """测试非对称矩阵验证。"""
        # 非对称矩阵
        invalid_matrix = [[0, 1, 0], [0, 0, 1], [0, 1, 0]]
        is_valid, error_msg = validate_adjacency_matrix(invalid_matrix)
        self.assertFalse(is_valid)
        self.assertIn("不对称", error_msg)


class TestGraphConnectivity(unittest.TestCase):
    """图连通性测试类。"""

    def test_connected_graph(self):
        """测试连通图。"""
        # CrossRing拓扑应该是连通的
        adj_matrix = create_crossring_adjacency_matrix(3, 3)
        self.assertTrue(check_connectivity(adj_matrix))

    def test_disconnected_graph(self):
        """测试非连通图。"""
        # 创建两个分离的子图
        disconnected_matrix = [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
        self.assertFalse(check_connectivity(disconnected_matrix))

    def test_single_node_connectivity(self):
        """测试单节点连通性。"""
        single_node_matrix = [[0]]
        self.assertTrue(check_connectivity(single_node_matrix))

    def test_empty_graph_connectivity(self):
        """测试空图连通性。"""
        self.assertFalse(check_connectivity([]))  # 空图根据实现定义为不连通


class TestNodeDegreeAnalysis(unittest.TestCase):
    """节点度数分析测试类。"""

    def test_crossring_degree_analysis(self):
        """测试CrossRing度数分析。"""
        adj_matrix = create_crossring_adjacency_matrix(3, 3)
        degrees, min_deg, max_deg, avg_deg = analyze_node_degrees(adj_matrix)

        # CrossRing中每个节点度数都是4
        self.assertEqual(len(degrees), 9)
        self.assertEqual(min_deg, 2)
        self.assertEqual(max_deg, 4)
        # self.assertEqual(avg_deg, 4.0)

    def test_empty_matrix_degree_analysis(self):
        """测试空矩阵度数分析。"""
        degrees, min_deg, max_deg, avg_deg = analyze_node_degrees([])
        self.assertEqual(degrees, [])
        self.assertEqual(min_deg, 0)
        self.assertEqual(max_deg, 0)
        self.assertEqual(avg_deg, 0.0)

    def test_variable_degree_graph(self):
        """测试不同度数的图。"""
        # 星形图：中心节点度数为3，其他节点度数为1
        star_matrix = [[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]
        degrees, min_deg, max_deg, avg_deg = analyze_node_degrees(star_matrix)

        self.assertEqual(min_deg, 1)
        self.assertEqual(max_deg, 3)
        self.assertEqual(avg_deg, 1.5)


class TestNodeNeighbors(unittest.TestCase):
    """节点邻居测试类。"""

    def test_get_crossring_neighbors(self):
        """测试获取CrossRing节点邻居。"""
        adj_matrix = create_crossring_adjacency_matrix(3, 3)

        # 测试节点0的邻居
        neighbors_0 = get_node_neighbors(adj_matrix, 0)
        self.assertEqual(len(neighbors_0), 2)
        expected_neighbors = [1, 3]  # 右、下
        self.assertEqual(set(neighbors_0), set(expected_neighbors))

        # 测试中心节点4的邻居
        neighbors_4 = get_node_neighbors(adj_matrix, 4)
        self.assertEqual(len(neighbors_4), 4)
        expected_neighbors = [1, 3, 5, 7]  # 上、左、右、下
        self.assertEqual(set(neighbors_4), set(expected_neighbors))

    def test_invalid_node_id(self):
        """测试无效节点ID。"""
        adj_matrix = create_crossring_adjacency_matrix(3, 3)

        with self.assertRaises(ValueError):
            get_node_neighbors(adj_matrix, -1)

        with self.assertRaises(ValueError):
            get_node_neighbors(adj_matrix, 9)

    def test_empty_matrix_neighbors(self):
        """测试空矩阵邻居查询。"""
        with self.assertRaises(ValueError):
            get_node_neighbors([], 0)


class TestGraphDiameter(unittest.TestCase):
    """图直径测试类。"""

    def test_crossring_diameter(self):
        """测试CrossRing直径。"""
        # 3x3 CrossRing的直径
        adj_matrix = create_crossring_adjacency_matrix(3, 3)
        diameter = calculate_graph_diameter(adj_matrix)
        self.assertGreater(diameter, 0)
        self.assertLessEqual(diameter, 4)

        # 4x4 CrossRing的直径应该更大
        adj_matrix_4x4 = create_crossring_adjacency_matrix(4, 4)
        diameter_4x4 = calculate_graph_diameter(adj_matrix_4x4)
        self.assertGreaterEqual(diameter_4x4, diameter)

    def test_small_graph_diameter(self):
        """测试小图直径。"""
        # 2x2图的直径
        adj_matrix = create_crossring_adjacency_matrix(2, 2)
        diameter = calculate_graph_diameter(adj_matrix)
        self.assertEqual(diameter, 2)  # 2x2环的直径为2

    def test_empty_graph_diameter(self):
        """测试空图直径。"""
        diameter = calculate_graph_diameter([])
        self.assertEqual(diameter, 0)

    def test_single_node_diameter(self):
        """测试单节点图直径。"""
        single_node_matrix = [[0]]
        diameter = calculate_graph_diameter(single_node_matrix)
        self.assertEqual(diameter, 0)


class TestClusteringCoefficient(unittest.TestCase):
    """聚类系数测试类。"""

    def test_crossring_clustering(self):
        """测试CrossRing聚类系数。"""
        adj_matrix = create_crossring_adjacency_matrix(3, 3)
        clustering = calculate_clustering_coefficient(adj_matrix)
        self.assertGreaterEqual(clustering, 0.0)
        self.assertLessEqual(clustering, 1.0)

    def test_complete_graph_clustering(self):
        """测试完全图聚类系数。"""
        # 4节点完全图（除了自环）
        complete_matrix = [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]]
        clustering = calculate_clustering_coefficient(complete_matrix)
        self.assertAlmostEqual(clustering, 1.0, places=5)

    def test_star_graph_clustering(self):
        """测试星形图聚类系数。"""
        # 星形图的聚类系数应该很低
        star_matrix = [[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]
        clustering = calculate_clustering_coefficient(star_matrix)
        self.assertAlmostEqual(clustering, 0.0, places=5)

    def test_small_graph_clustering(self):
        """测试小图聚类系数。"""
        # 单节点和双节点图
        single_node = [[0]]
        clustering = calculate_clustering_coefficient(single_node)
        self.assertEqual(clustering, 0.0)

        two_nodes = [[0, 1], [1, 0]]
        clustering = calculate_clustering_coefficient(two_nodes)
        self.assertEqual(clustering, 0.0)


class TestMatrixExport(unittest.TestCase):
    """矩阵导出测试类。"""

    def setUp(self):
        """测试前置设置。"""
        self.test_matrix = create_crossring_adjacency_matrix(3, 3)
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """测试后清理。"""
        # 清理临时文件
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_txt_export(self):
        """测试TXT格式导出。"""
        filename = os.path.join(self.temp_dir, "test_matrix.txt")
        export_adjacency_matrix(self.test_matrix, filename, "txt")

        # 验证文件存在
        self.assertTrue(os.path.exists(filename))

        # 验证文件内容
        with open(filename, "r", encoding="utf-8") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 9)  # 9行

            # 验证第一行
            first_row = lines[0].strip().split()
            self.assertEqual(len(first_row), 9)  # 9列

    def test_csv_export(self):
        """测试CSV格式导出。"""
        filename = os.path.join(self.temp_dir, "test_matrix.csv")
        export_adjacency_matrix(self.test_matrix, filename, "csv")

        # 验证文件存在
        self.assertTrue(os.path.exists(filename))

        # 验证文件内容
        with open(filename, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
            self.assertEqual(len(rows), 9)
            self.assertEqual(len(rows[0]), 9)

    def test_json_export(self):
        """测试JSON格式导出。"""
        filename = os.path.join(self.temp_dir, "test_matrix.json")
        export_adjacency_matrix(self.test_matrix, filename, "json")

        # 验证文件存在
        self.assertTrue(os.path.exists(filename))

        # 验证文件内容
        with open(filename, "r", encoding="utf-8") as f:
            loaded_matrix = json.load(f)
            self.assertEqual(loaded_matrix, self.test_matrix)

    def test_invalid_format_export(self):
        """测试无效格式导出。"""
        filename = os.path.join(self.temp_dir, "test_matrix.xyz")

        with self.assertRaises(ValueError):
            export_adjacency_matrix(self.test_matrix, filename, "xyz")

    def test_empty_matrix_export(self):
        """测试空矩阵导出。"""
        filename = os.path.join(self.temp_dir, "empty_matrix.txt")

        with self.assertRaises(ValueError):
            export_adjacency_matrix([], filename, "txt")


class TestAdjacencyUtilsIntegration(unittest.TestCase):
    """邻接矩阵工具集成测试类。"""

    def test_complete_workflow(self):
        """测试完整工作流程。"""
        # 1. 生成CrossRing邻接矩阵
        adj_matrix = create_crossring_adjacency_matrix(4, 4)

        # 2. 验证矩阵有效性
        is_valid, error_msg = validate_adjacency_matrix(adj_matrix)
        self.assertTrue(is_valid, f"邻接矩阵无效: {error_msg}")

        # 3. 检查连通性
        self.assertTrue(check_connectivity(adj_matrix))

        # 4. 分析节点度数
        degrees, min_deg, max_deg, avg_deg = analyze_node_degrees(adj_matrix)
        self.assertEqual(len(degrees), 16)
        self.assertEqual(min_deg, 2)
        self.assertEqual(max_deg, 4)

        # 5. 计算图直径
        diameter = calculate_graph_diameter(adj_matrix)
        self.assertGreater(diameter, 0)

        # 6. 计算聚类系数
        clustering = calculate_clustering_coefficient(adj_matrix)
        self.assertGreaterEqual(clustering, 0.0)
        self.assertLessEqual(clustering, 1.0)

    def test_different_topology_sizes(self):
        """测试不同拓扑规模。"""
        sizes = [(2, 2), (3, 3), (4, 4), (5, 5)]

        for rows, cols in sizes:
            with self.subTest(rows=rows, cols=cols):
                # 生成矩阵
                adj_matrix = create_crossring_adjacency_matrix(rows, cols)

                # 基本验证
                expected_nodes = rows * cols
                self.assertEqual(len(adj_matrix), expected_nodes)

                # 验证有效性
                is_valid, _ = validate_adjacency_matrix(adj_matrix)
                self.assertTrue(is_valid)

                # 验证连通性
                self.assertTrue(check_connectivity(adj_matrix))

                # 验证每个节点都有4个邻居
                degrees, min_deg, max_deg, avg_deg = analyze_node_degrees(adj_matrix)
                if rows == 2 and cols == 2:
                    self.assertEqual(min_deg, 2)
                    self.assertEqual(max_deg, 2)
                    # self.assertEqual(avg_deg, 2.0)
                elif rows >= 2 and cols >= 2:  # 正常CrossRing
                    self.assertEqual(min_deg, 2)
                    self.assertEqual(max_deg, 4)
                    # self.assertEqual(avg_deg, 4.0)


if __name__ == "__main__":
    # 设置测试运行器
    unittest.main(verbosity=2, buffer=True)
