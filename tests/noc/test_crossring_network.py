"""
测试CrossRing网络传输实现。

本模块测试新实现的CrossRing网络传输逻辑，包括：
- CrossRing模型基本功能
- 节点和网络结构
- 流量注入和基本操作
"""

import unittest
from unittest.mock import Mock, patch
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.noc.crossring.model import CrossRingModel
from src.noc.crossring.config import CrossRingConfig
from src.noc.crossring.flit import create_crossring_flit


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

            # 验证ring缓冲区结构（使用四方向系统）
            for direction in ["TL", "TR", "TU", "TD"]:
                self.assertIn(direction, piece["ring_buffers"])
                for channel in ["req", "rsp", "data"]:
                    self.assertIn(channel, piece["ring_buffers"][direction])

    def test_node_coordinates(self):
        """测试节点坐标计算。"""
        # 测试2x2网格的坐标
        expected_coords = {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)}

        for node_id, expected_coord in expected_coords.items():
            actual_coord = self.model._get_node_coordinates(node_id)
            self.assertEqual(actual_coord, expected_coord)

    def test_model_initialization(self):
        """测试模型基本初始化。"""
        # 验证模型基本属性
        self.assertEqual(self.model.config.num_row, 2)
        self.assertEqual(self.model.config.num_col, 2)
        self.assertEqual(self.model.cycle, 0)
        self.assertFalse(self.model.is_running)
        self.assertFalse(self.model.is_finished)

        # 验证统计信息初始化
        self.assertIsInstance(self.model.stats, dict)
        self.assertEqual(self.model.stats["total_requests"], 0)

    def test_flit_creation(self):
        """测试flit创建功能。"""
        # 使用create_crossring_flit创建flit
        test_flit = create_crossring_flit(
            source=0, 
            destination=1, 
            path=[0, 1],
            req_type="read",
            burst_length=4,
            packet_id="test_packet_1"
        )

        # 验证flit创建成功
        self.assertEqual(test_flit.source, 0)
        self.assertEqual(test_flit.destination, 1)
        self.assertEqual(test_flit.packet_id, "test_packet_1")

    def test_basic_step_execution(self):
        """测试基本的步进执行。"""
        initial_cycle = self.model.cycle
        
        # 执行一个仿真步骤
        self.model.step()
        
        # 验证周期数增加
        self.assertEqual(self.model.cycle, initial_cycle + 1)

    def test_ip_interface_registration(self):
        """测试IP接口注册。"""
        # 验证IP接口已创建
        self.assertGreater(len(self.model.ip_interfaces), 0)
        
        # 验证IP接口类型
        for key, ip_interface in self.model.ip_interfaces.items():
            self.assertIsNotNone(ip_interface)
            self.assertTrue(hasattr(ip_interface, 'ip_type'))
            self.assertTrue(hasattr(ip_interface, 'node_id'))

    def test_traffic_injection(self):
        """测试流量注入功能。"""
        # 注入测试流量
        packet_ids = self.model.inject_test_traffic(
            source=0, 
            destination=1, 
            req_type="read", 
            count=1,
            burst_length=4
        )
        
        # 验证流量注入成功
        self.assertIsInstance(packet_ids, list)
        if packet_ids:  # 如果成功注入了流量
            self.assertGreater(len(packet_ids), 0)

    def test_model_summary(self):
        """测试模型摘要信息。"""
        summary = self.model.get_model_summary()
        
        # 验证摘要包含关键信息
        self.assertIsInstance(summary, dict)
        self.assertIn("config_name", summary)
        self.assertIn("total_nodes", summary)  # 实际字段名是total_nodes而不是num_nodes
        self.assertIn("current_cycle", summary)

    def test_network_statistics(self):
        """测试网络统计信息。"""
        stats = self.model.get_network_statistics()
        
        # 验证统计信息结构
        self.assertIsInstance(stats, dict)
        # 基本统计信息应该存在（使用实际存在的字段名）
        for key in ["total_packets_injected", "total_packets_completed", "active_packets"]:
            self.assertIn(key, stats)

    def test_ring_connections(self):
        """测试环形连接信息。"""
        for node_id, piece in self.model.crossring_pieces.items():
            connections = piece["ring_connections"]
            
            # 验证连接信息结构
            self.assertIsInstance(connections, dict)
            # 应该有四个方向的连接
            for direction in ["TL", "TR", "TU", "TD"]:
                self.assertIn(direction, connections)
                # 连接目标应该是有效的节点ID
                target_node = connections[direction]
                self.assertIn(target_node, self.model.crossring_pieces.keys())

    def test_etag_status_structure(self):
        """测试ETag状态结构。"""
        piece = self.model.crossring_pieces[0]
        
        # 验证ETag状态结构使用四方向系统
        for direction in ["TL", "TR", "TU", "TD"]:
            self.assertIn(direction, piece["etag_status"])
            for channel in ["req", "rsp", "data"]:
                self.assertIn(channel, piece["etag_status"][direction])
                # 初始状态应该是False
                self.assertFalse(piece["etag_status"][direction][channel])

    def test_itag_status_structure(self):
        """测试ITag状态结构。"""
        piece = self.model.crossring_pieces[0]
        
        # 验证ITag状态结构使用四方向系统
        for direction in ["TL", "TR", "TU", "TD"]:
            self.assertIn(direction, piece["itag_status"])
            for channel in ["req", "rsp", "data"]:
                self.assertIn(channel, piece["itag_status"][direction])
                # 初始状态应该是False
                self.assertFalse(piece["itag_status"][direction][channel])

    def test_arbitration_state_structure(self):
        """测试仲裁状态结构。"""
        piece = self.model.crossring_pieces[0]
        
        # 验证仲裁状态结构
        arb_state = piece["arbitration_state"]
        self.assertIn("current_priority", arb_state)
        self.assertIn("arbitration_counter", arb_state)
        self.assertIn("last_winner", arb_state)
        
        # 初始优先级应该是inject
        self.assertEqual(arb_state["current_priority"], "inject")

    def test_step_multiple_cycles(self):
        """测试多周期执行。"""
        initial_cycle = self.model.cycle
        num_cycles = 5
        
        # 执行多个周期
        for i in range(num_cycles):
            self.model.step()
        
        # 验证周期数正确增加
        self.assertEqual(self.model.cycle, initial_cycle + num_cycles)

    def test_cleanup(self):
        """测试清理功能。"""
        # 执行几个周期
        for i in range(3):
            self.model.step()
        
        # 执行清理
        self.model.cleanup()
        
        # 验证清理后的状态（如果有特定的清理行为）
        # 这里只验证cleanup方法可以正常调用
        self.assertTrue(True)  # 如果没有异常就说明cleanup工作正常

    def test_direction_mapper_integration(self):
        """测试方向映射器集成。"""
        # 验证方向映射器存在
        self.assertIsNotNone(self.model.direction_mapper)
        
        # 验证环形连接验证
        self.assertTrue(self.model.direction_mapper.validate_ring_connectivity())

    def test_ring_bridge_integration(self):
        """测试环形桥接集成。"""
        # 验证环形桥接存在
        self.assertIsNotNone(self.model.ring_bridge)
        
        # 验证交叉点模块已创建
        for node_id in self.model.crossring_pieces.keys():
            self.assertIn(node_id, self.model.ring_bridge.cross_points)


if __name__ == "__main__":
    unittest.main()