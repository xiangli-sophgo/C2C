"""
测试CrossRing网络传输实现。

本模块测试新实现的CrossRing网络传输逻辑，包括：
- inject_queue → ring网络传输
- ring网络内部传输
- ring网络 → eject_queue传输
- ETag/ITag拥塞控制机制
"""

import unittest
from unittest.mock import Mock, patch
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

try:
    from src.noc.crossring.model import CrossRingModel
    from src.noc.crossring.config import CrossRingConfig
    from src.noc.crossring.flit import CrossRingFlit
except ImportError as e:
    print(f"Import error: {e}")
    # 如果导入失败，跳过测试
    import pytest

    pytest.skip("CrossRing modules not available", allow_module_level=True)


class TestCrossRingNetworkTransmission(unittest.TestCase):
    """测试CrossRing网络传输功能。"""

    def setUp(self):
        """设置测试环境。"""
        self.config = CrossRingConfig(num_col=2, num_row=2)  # 2x2网格
        self.model = CrossRingModel(self.config)

    def test_crossring_pieces_initialization(self):
        """测试CrossRing pieces的初始化。"""
        # 验证所有节点都已创建
        self.assertEqual(len(self.model.crossring_pieces), 4)  # 2x2 = 4个节点

        # 验证每个piece的结构
        for node_id, piece in self.model.crossring_pieces.items():
            self.assertIn("node_id", piece)
            self.assertIn("coordinates", piece)
            self.assertIn("inject_queues", piece)
            self.assertIn("eject_queues", piece)
            self.assertIn("ring_buffers", piece)
            self.assertIn("etag_status", piece)
            self.assertIn("itag_status", piece)
            self.assertIn("arbitration_state", piece)

            # 验证队列结构
            for channel in ["req", "rsp", "data"]:
                self.assertIn(channel, piece["inject_queues"])
                self.assertIn(channel, piece["eject_queues"])

            # 验证ring缓冲区结构
            for direction in ["horizontal", "vertical"]:
                self.assertIn(direction, piece["ring_buffers"])
                for channel in ["req", "rsp", "data"]:
                    self.assertIn(channel, piece["ring_buffers"][direction])
                    self.assertIn("clockwise", piece["ring_buffers"][direction][channel])
                    self.assertIn("counter_clockwise", piece["ring_buffers"][direction][channel])

    def test_node_coordinates(self):
        """测试节点坐标计算。"""
        # 测试2x2网格的坐标
        expected_coords = {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)}  # 左上  # 右上  # 左下  # 右下

        for node_id, expected_coord in expected_coords.items():
            actual_coord = self.model._get_node_coordinates(node_id)
            self.assertEqual(actual_coord, expected_coord)

    def test_route_direction_determination(self):
        """测试路由方向确定。"""
        # 测试水平路由（同行不同列）
        direction = self.model._determine_route_direction(0, 1)  # (0,0) -> (1,0)
        self.assertEqual(direction, "horizontal")

        # 测试垂直路由（同列不同行）
        direction = self.model._determine_route_direction(0, 2)  # (0,0) -> (0,1)
        self.assertEqual(direction, "vertical")

        # 测试相同节点
        direction = self.model._determine_route_direction(0, 0)
        self.assertIsNone(direction)

    def test_ring_direction_determination(self):
        """测试ring传输方向确定。"""
        # 测试水平方向
        # 从(0,0)到(1,0)，应该选择顺时针（向右）
        ring_dir = self.model._determine_ring_direction(0, 1, "horizontal")
        self.assertEqual(ring_dir, "clockwise")

        # 从(1,0)到(0,0)，应该选择逆时针（向左）
        ring_dir = self.model._determine_ring_direction(1, 0, "horizontal")
        self.assertEqual(ring_dir, "counter_clockwise")

        # 测试垂直方向
        # 从(0,0)到(0,1)，应该选择顺时针（向下）
        ring_dir = self.model._determine_ring_direction(0, 2, "vertical")
        self.assertEqual(ring_dir, "clockwise")

    def test_inject_to_ring_basic(self):
        """测试基本的inject到ring传输。"""
        # 创建测试flit
        test_flit = CrossRingFlit(source=0, destination=1, channel="req", packet_id="test_packet_1")

        # 将flit添加到inject队列
        piece = self.model.crossring_pieces[0]
        piece["inject_queues"]["req"].append(test_flit)

        # 执行一个仿真步骤
        self.model._process_inject_to_ring()

        # 验证flit已从inject队列移动到ring缓冲区
        self.assertEqual(len(piece["inject_queues"]["req"]), 0)

        # 验证flit在正确的ring缓冲区中
        horizontal_cw_buffer = piece["ring_buffers"]["horizontal"]["req"]["clockwise"]
        self.assertEqual(len(horizontal_cw_buffer), 1)
        self.assertEqual(horizontal_cw_buffer[0], test_flit)

    def test_local_delivery(self):
        """测试本地传输（源和目标相同）。"""
        # 创建本地传输的flit
        test_flit = CrossRingFlit(source=0, destination=0, channel="req", packet_id="local_packet")

        # 将flit添加到inject队列
        piece = self.model.crossring_pieces[0]
        piece["inject_queues"]["req"].append(test_flit)

        # 执行inject处理
        self.model._process_inject_to_ring()

        # 验证flit直接移动到eject队列
        self.assertEqual(len(piece["inject_queues"]["req"]), 0)
        self.assertEqual(len(piece["eject_queues"]["req"]), 1)
        self.assertTrue(test_flit.is_arrive)

    def test_etag_congestion_control(self):
        """测试ETag拥塞控制。"""
        piece = self.model.crossring_pieces[0]

        # 填满eject队列以触发拥塞
        for i in range(self.config.eject_buffer_depth):
            dummy_flit = CrossRingFlit(source=1, destination=0, channel="req")
            piece["eject_queues"]["req"].append(dummy_flit)

        # 更新ETag状态
        self.model._update_etag_status(piece)

        # 验证ETag被设置
        self.assertTrue(piece["etag_status"]["horizontal"]["req"])
        self.assertTrue(piece["etag_status"]["vertical"]["req"])

    def test_arbitration_mechanism(self):
        """测试仲裁机制。"""
        piece = self.model.crossring_pieces[0]

        # 测试inject优先级
        self.assertTrue(self.model._arbitrate_ring_access(piece, "horizontal", "inject"))

        # 验证优先级已更新
        self.assertEqual(piece["arbitration_state"]["horizontal_priority"], "ring_cw")

        # 测试下一个优先级
        self.assertTrue(self.model._arbitrate_ring_access(piece, "horizontal", "ring_cw"))
        self.assertEqual(piece["arbitration_state"]["horizontal_priority"], "ring_ccw")

    def test_complete_transmission_flow(self):
        """测试完整的传输流程。"""
        # 创建从节点0到节点3的flit（对角传输）
        test_flit = CrossRingFlit(source=0, destination=3, channel="req", packet_id="diagonal_packet")

        # 将flit添加到源节点的inject队列
        source_piece = self.model.crossring_pieces[0]
        source_piece["inject_queues"]["req"].append(test_flit)

        # 执行多个仿真周期
        max_cycles = 20
        for cycle in range(max_cycles):
            self.model.cycle = cycle + 1
            self.model._step_crossring_networks()

            # 检查是否到达目标
            dest_piece = self.model.crossring_pieces[3]
            if dest_piece["eject_queues"]["req"]:
                arrived_flit = dest_piece["eject_queues"]["req"][0]
                self.assertEqual(arrived_flit.packet_id, "diagonal_packet")
                self.assertTrue(arrived_flit.is_arrive)
                break
        else:
            self.fail("Flit did not arrive at destination within expected cycles")


if __name__ == "__main__":
    unittest.main()
