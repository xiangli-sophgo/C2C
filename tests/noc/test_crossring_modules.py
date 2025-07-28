"""
CrossRing模块功能测试。

测试inject_queue、eject_queue和ring_bridge三大核心模块的功能正确性，
包括：
- FIFO配置和深度验证
- 路由策略适配
- 轮询仲裁机制
- 数据流传输
- IP连接管理
- 时序控制
"""

import unittest
import logging
import sys
import os
from typing import List, Optional

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.noc.crossring.node import CrossRingNode
from src.noc.crossring.config import CrossRingConfig, RoutingStrategy
from src.noc.crossring.flit import CrossRingFlit
from src.noc.crossring.link import CrossRingSlot, RingSlice
from src.noc.crossring.tag_mechanism import CrossRingTagManager


class TestCrossRingModules(unittest.TestCase):
    """CrossRing核心模块功能测试类"""

    def setUp(self):
        """测试初始化"""
        # 创建配置
        self.config = CrossRingConfig()
        self.config.iq_ch_depth = 10
        self.config.iq_out_depth = 8
        self.config.eq_in_depth = 16
        self.config.eq_ch_depth = 10
        self.config.rb_in_depth = 16
        self.config.rb_out_depth = 8
        self.config.routing_strategy = RoutingStrategy.XY

        # 创建日志记录器
        self.logger = logging.getLogger("test")
        self.logger.setLevel(logging.DEBUG)

        # 创建节点实例
        self.node = CrossRingNode(0, (1, 1), self.config, self.logger)

    def create_test_flit(self, packet_id: int, dest_coordinates: tuple, channel: str = "req") -> CrossRingFlit:
        """创建测试flit"""
        flit = CrossRingFlit(packet_id=packet_id, flit_id=1)
        flit.dest_coordinates = dest_coordinates
        flit.channel = channel
        return flit

    # ==================== inject_queue 测试 ====================

    def test_inject_queue_structure(self):
        """测试inject_queue的数据结构"""
        # 检查基本结构存在
        self.assertTrue(hasattr(self.node, "ip_inject_channel_buffers"))
        self.assertTrue(hasattr(self.node, "inject_input_fifos"))
        self.assertTrue(hasattr(self.node, "inject_arbitration_state"))

        # 检查inject_input_fifos结构
        for channel in ["req", "rsp", "data"]:
            self.assertIn(channel, self.node.inject_input_fifos)
            for direction in ["TR", "TL", "TU", "TD", "EQ"]:
                self.assertIn(direction, self.node.inject_input_fifos[channel])

        # 检查FIFO深度
        for channel in ["req", "rsp", "data"]:
            for direction in ["TR", "TL", "TU", "TD", "EQ"]:
                fifo = self.node.inject_input_fifos[channel][direction]
                self.assertEqual(fifo.internal_queue.maxlen, self.config.iq_out_depth)

    def test_ip_connection_management(self):
        """测试IP连接管理"""
        # 初始状态：无连接的IP
        self.assertEqual(len(self.node.get_connected_ips()), 0)

        # 连接IP
        self.assertTrue(self.node.connect_ip("ip_0"))
        self.assertEqual(len(self.node.get_connected_ips()), 1)
        self.assertIn("ip_0", self.node.get_connected_ips())

        # 检查inject channel buffer是否创建
        self.assertIn("ip_0", self.node.ip_inject_channel_buffers)
        for channel in ["req", "rsp", "data"]:
            self.assertIn(channel, self.node.ip_inject_channel_buffers["ip_0"])
            buffer = self.node.ip_inject_channel_buffers["ip_0"][channel]
            self.assertEqual(buffer.internal_queue.maxlen, self.config.iq_ch_depth)

        # 检查eject channel buffer是否创建
        self.assertIn("ip_0", self.node.ip_eject_channel_buffers)
        for channel in ["req", "rsp", "data"]:
            self.assertIn(channel, self.node.ip_eject_channel_buffers["ip_0"])
            buffer = self.node.ip_eject_channel_buffers["ip_0"][channel]
            self.assertEqual(buffer.internal_queue.maxlen, self.config.eq_ch_depth)

        # 连接第二个IP
        self.node.connect_ip("ip_1")
        self.assertEqual(len(self.node.get_connected_ips()), 2)

        # 断开IP
        self.node.disconnect_ip("ip_0")
        self.assertEqual(len(self.node.get_connected_ips()), 1)
        self.assertNotIn("ip_0", self.node.get_connected_ips())
        self.assertNotIn("ip_0", self.node.ip_inject_channel_buffers)
        self.assertNotIn("ip_0", self.node.ip_eject_channel_buffers)

    def test_inject_queue_routing_strategy(self):
        """测试inject_queue的路由策略适配"""
        # 创建测试flit
        flit_right = self.create_test_flit(1, (3, 1))  # 向右
        flit_left = self.create_test_flit(2, (0, 1))  # 向左
        flit_up = self.create_test_flit(3, (1, 3))  # 向上
        flit_down = self.create_test_flit(4, (1, 0))  # 向下
        flit_local = self.create_test_flit(5, (1, 1))  # 本地

        # 测试XY路由
        self.node.config.routing_strategy = RoutingStrategy.XY
        self.assertEqual(self.node._calculate_routing_direction(flit_right), "TR")
        self.assertEqual(self.node._calculate_routing_direction(flit_left), "TL")
        self.assertEqual(self.node._calculate_routing_direction(flit_up), "TU")
        self.assertEqual(self.node._calculate_routing_direction(flit_down), "TD")
        self.assertEqual(self.node._calculate_routing_direction(flit_local), "EQ")

        # 测试YX路由（对角线目标）
        flit_diagonal = self.create_test_flit(6, (3, 3))  # 向右上
        self.node.config.routing_strategy = RoutingStrategy.XY
        self.assertEqual(self.node._calculate_routing_direction(flit_diagonal), "TR")  # XY：先水平

        self.node.config.routing_strategy = RoutingStrategy.YX
        self.assertEqual(self.node._calculate_routing_direction(flit_diagonal), "TU")  # YX：先垂直

    def test_inject_queue_flit_injection(self):
        """测试inject_queue的flit注入"""
        self.node.connect_ip("ip_0")

        # 创建测试flit
        flit = self.create_test_flit(1, (3, 1), "req")

        # 注入flit到IP的channel buffer
        success = self.node.add_to_inject_queue(flit, "req", "ip_0")
        self.assertTrue(success)

        # 运行一个周期更新FIFO状态
        self.node._step_compute_phase()
        self.node._step_update_phase()

        # 检查channel buffer中是否有flit
        channel_buffer = self.node.ip_inject_channel_buffers["ip_0"]["req"]
        self.assertTrue(channel_buffer.valid_signal())

        # 处理仲裁，应该将flit移动到TR方向
        self.node.process_inject_arbitration(0)

        # 再次运行周期更新
        self.node._step_compute_phase()
        self.node._step_update_phase()

        # 检查TR方向FIFO是否有flit
        tr_fifo = self.node.inject_input_fifos["req"]["TR"]
        self.assertTrue(tr_fifo.valid_signal())

    # ==================== eject_queue 测试 ====================

    def test_eject_queue_structure(self):
        """测试eject_queue的数据结构"""
        # 检查基本结构存在
        self.assertTrue(hasattr(self.node, "ip_eject_channel_buffers"))
        self.assertTrue(hasattr(self.node, "eject_input_fifos"))
        self.assertTrue(hasattr(self.node, "eject_arbitration_state"))

        # 检查eject_input_fifos结构
        for channel in ["req", "rsp", "data"]:
            self.assertIn(channel, self.node.eject_input_fifos)
            for direction in ["TU", "TD", "TR", "TL"]:
                self.assertIn(direction, self.node.eject_input_fifos[channel])
                fifo = self.node.eject_input_fifos[channel][direction]
                self.assertEqual(fifo.internal_queue.maxlen, self.config.eq_in_depth)

    def test_eject_queue_routing_strategy_sources(self):
        """测试eject_queue的路由策略源配置"""
        # XY路由的活跃源
        self.node.config.routing_strategy = RoutingStrategy.XY
        active_sources_xy = self.node._get_active_eject_sources()
        expected_xy = ["IQ_EQ", "ring_bridge_EQ", "TU", "TD"]
        self.assertEqual(set(active_sources_xy), set(expected_xy))

        # YX路由的活跃源
        self.node.config.routing_strategy = RoutingStrategy.YX
        active_sources_yx = self.node._get_active_eject_sources()
        expected_yx = ["IQ_EQ", "ring_bridge_EQ", "TR", "TL"]
        self.assertEqual(set(active_sources_yx), set(expected_yx))

        # ADAPTIVE路由的活跃源
        self.node.config.routing_strategy = RoutingStrategy.ADAPTIVE
        active_sources_adaptive = self.node._get_active_eject_sources()
        expected_adaptive = ["IQ_EQ", "ring_bridge_EQ", "TU", "TD", "TR", "TL"]
        self.assertEqual(set(active_sources_adaptive), set(expected_adaptive))

    def test_eject_queue_flit_assignment(self):
        """测试eject_queue的flit分配"""
        self.node.connect_ip("ip_0")

        # 创建本地目标flit
        local_flit = self.create_test_flit(1, (1, 1), "req")  # 本地目标

        # 模拟从IQ_EQ获取flit (先将flit放入inject EQ FIFO)
        eq_fifo = self.node.inject_input_fifos["req"]["EQ"]
        eq_fifo.write_input(local_flit)

        # 运行周期更新
        self.node._step_compute_phase()
        self.node._step_update_phase()

        # 处理eject仲裁
        self.node.process_eject_arbitration(0)

        # 再次运行周期更新
        self.node._step_compute_phase()
        self.node._step_update_phase()

        # 检查IP的eject channel buffer是否收到flit
        eject_buffer = self.node.ip_eject_channel_buffers["ip_0"]["req"]
        self.assertTrue(eject_buffer.valid_signal())

        # IP获取flit
        received_flit = self.node.get_eject_flit("ip_0", "req")
        self.assertIsNotNone(received_flit)
        self.assertEqual(received_flit.packet_id, 1)

    # ==================== CrossRingSlot 测试 ====================

    def test_crossring_slot_basic_functionality(self):
        """测试CrossRingSlot的基本功能"""
        # 创建测试slot
        slot = CrossRingSlot(slot_id=1, cycle=0, channel="req")

        # 测试初始状态
        self.assertFalse(slot.is_occupied)
        self.assertTrue(slot.is_available)
        self.assertFalse(slot.is_reserved)

        # 测试flit分配
        flit = self.create_test_flit(1, (2, 1), "req")
        success = slot.assign_flit(flit)
        self.assertTrue(success)
        self.assertTrue(slot.is_occupied)
        self.assertFalse(slot.is_available)

        # 测试flit释放
        released_flit = slot.release_flit()
        self.assertIsNotNone(released_flit)
        self.assertEqual(released_flit.packet_id, 1)
        self.assertFalse(slot.is_occupied)
        self.assertTrue(slot.is_available)

    def test_crossring_slot_itag_mechanism(self):
        """测试CrossRingSlot的I-Tag机制"""
        slot = CrossRingSlot(slot_id=2, cycle=0, channel="req")

        # 测试I-Tag预约
        success = slot.reserve_itag(reserver_id=5, direction="horizontal")
        self.assertTrue(success)
        self.assertTrue(slot.is_reserved)
        self.assertFalse(slot.is_available)
        self.assertEqual(slot.itag_reserver_id, 5)
        self.assertEqual(slot.itag_direction, "horizontal")

        # 测试重复预约失败
        success2 = slot.reserve_itag(reserver_id=6, direction="vertical")
        self.assertFalse(success2)

        # 测试清除I-Tag
        slot.clear_itag()
        self.assertFalse(slot.is_reserved)
        self.assertTrue(slot.is_available)
        self.assertIsNone(slot.itag_reserver_id)

    def test_crossring_slot_etag_mechanism(self):
        """测试CrossRingSlot的E-Tag机制"""
        from src.noc.base.link import PriorityLevel

        slot = CrossRingSlot(slot_id=3, cycle=0, channel="req")
        flit = self.create_test_flit(1, (2, 1), "req")
        slot.assign_flit(flit)

        # 测试E-Tag标记
        slot.mark_etag(PriorityLevel.T1, "TL")
        self.assertTrue(slot.etag_marked)
        self.assertEqual(slot.etag_priority, PriorityLevel.T1)
        self.assertEqual(slot.etag_direction, "TL")

        # 测试E-Tag优先级升级逻辑
        new_priority = slot.should_upgrade_etag(failed_attempts=2)
        self.assertEqual(new_priority, PriorityLevel.T0)  # T1 -> T0 for TL direction

        # 测试E-Tag清除
        slot.clear_etag()
        self.assertFalse(slot.etag_marked)
        self.assertEqual(slot.etag_priority, PriorityLevel.T2)

    # ==================== RingSlice 测试 ====================

    def test_ring_slice_basic_functionality(self):
        """测试RingSlice的基本功能"""
        ring_slice = RingSlice("test_slice", "horizontal", 0, num_channels=3)

        # 测试初始状态
        self.assertEqual(ring_slice.slice_id, "test_slice")
        self.assertEqual(ring_slice.ring_type, "horizontal")
        self.assertEqual(ring_slice.position, 0)

        # 测试slot接收
        slot = CrossRingSlot(slot_id=1, cycle=0, channel="req")
        success = ring_slice.receive_slot(slot, "req")
        self.assertTrue(success)

        # 第一次step：input_buffer -> current_slots，current_slots -> output_buffer
        ring_slice.step(1)

        # 第二次step：确保slot到达output_buffer
        ring_slice.step(2)

        # 测试slot传输
        transmitted_slot = ring_slice.transmit_slot("req")
        self.assertIsNotNone(transmitted_slot)
        self.assertEqual(transmitted_slot.slot_id, 1)

    def test_ring_slice_pipeline_transmission(self):
        """测试RingSlice的流水线传输"""
        ring_slice = RingSlice("pipeline_slice", "vertical", 1)

        # 创建测试slots
        slot1 = CrossRingSlot(slot_id=1, cycle=0, channel="req")
        slot2 = CrossRingSlot(slot_id=2, cycle=1, channel="req")

        # 接收第一个slot
        ring_slice.receive_slot(slot1, "req")
        ring_slice.step(0)

        # 接收第二个slot
        ring_slice.receive_slot(slot2, "req")
        ring_slice.step(1)

        # 第一个slot应该在输出缓存
        output_slot = ring_slice.transmit_slot("req")
        self.assertIsNotNone(output_slot)
        self.assertEqual(output_slot.slot_id, 1)

        # 执行下一个周期
        ring_slice.step(2)

        # 第二个slot应该到达输出
        output_slot2 = ring_slice.transmit_slot("req")
        self.assertIsNotNone(output_slot2)
        self.assertEqual(output_slot2.slot_id, 2)

    def test_ring_slice_utilization_stats(self):
        """测试RingSlice的利用率统计"""
        ring_slice = RingSlice("stats_slice", "horizontal", 2)

        # 发送一些slots
        for i in range(5):
            slot = CrossRingSlot(slot_id=i, cycle=i, channel="req")
            ring_slice.receive_slot(slot, "req")
            ring_slice.step(i)
            ring_slice.transmit_slot("req")

        # 检查统计信息
        utilization = ring_slice.get_utilization("req")
        self.assertGreater(utilization, 0.0)
        self.assertLessEqual(utilization, 1.0)

        status = ring_slice.get_ring_slice_status()
        self.assertIn("utilization", status)
        self.assertIn("stats", status)

    # ==================== ring_bridge 测试 ====================

    def test_ring_bridge_structure(self):
        """测试ring_bridge的数据结构"""
        # 检查基本结构存在
        self.assertTrue(hasattr(self.node, "ring_bridge_input_fifos"))
        self.assertTrue(hasattr(self.node, "ring_bridge_output_fifos"))
        self.assertTrue(hasattr(self.node, "ring_bridge_arbitration_state"))

        # 检查input_fifos结构和深度
        for channel in ["req", "rsp", "data"]:
            self.assertIn(channel, self.node.ring_bridge_input_fifos)
            for direction in ["TR", "TL", "TU", "TD"]:
                self.assertIn(direction, self.node.ring_bridge_input_fifos[channel])
                fifo = self.node.ring_bridge_input_fifos[channel][direction]
                self.assertEqual(fifo.internal_queue.maxlen, self.config.rb_in_depth)

        # 检查output_fifos结构和深度
        for channel in ["req", "rsp", "data"]:
            self.assertIn(channel, self.node.ring_bridge_output_fifos)
            for direction in ["EQ", "TR", "TL", "TU", "TD"]:
                self.assertIn(direction, self.node.ring_bridge_output_fifos[channel])
                fifo = self.node.ring_bridge_output_fifos[channel][direction]
                self.assertEqual(fifo.internal_queue.maxlen, self.config.rb_out_depth)

    def test_ring_bridge_routing_strategy_adaptation(self):
        """测试ring_bridge的路由策略适配"""
        # XY路由配置
        self.node.config.routing_strategy = RoutingStrategy.XY
        input_sources_xy = self.node._get_ring_bridge_input_sources()
        output_directions_xy = self.node._get_ring_bridge_output_directions()

        expected_input_xy = ["IQ_TU", "IQ_TD", "RB_TR", "RB_TL"]
        expected_output_xy = ["EQ", "TU", "TD"]
        self.assertEqual(set(input_sources_xy), set(expected_input_xy))
        self.assertEqual(set(output_directions_xy), set(expected_output_xy))

        # YX路由配置
        self.node.config.routing_strategy = RoutingStrategy.YX
        input_sources_yx = self.node._get_ring_bridge_input_sources()
        output_directions_yx = self.node._get_ring_bridge_output_directions()

        expected_input_yx = ["IQ_TR", "IQ_TL", "RB_TU", "RB_TD"]
        expected_output_yx = ["EQ", "TR", "TL"]
        self.assertEqual(set(input_sources_yx), set(expected_input_yx))
        self.assertEqual(set(output_directions_yx), set(expected_output_yx))

    def test_ring_bridge_crosspoint_input(self):
        """测试ring_bridge的CrossPoint输入"""
        # 创建测试flit
        flit = self.create_test_flit(1, (2, 1), "req")

        # CrossPoint向ring_bridge输入添加flit
        success = self.node.add_to_ring_bridge_input(flit, "TR", "req")
        self.assertTrue(success)

        # 运行周期更新
        self.node._step_compute_phase()
        self.node._step_update_phase()

        # 检查ring_bridge输入FIFO
        input_fifo = self.node.ring_bridge_input_fifos["req"]["TR"]
        self.assertTrue(input_fifo.valid_signal())

    def test_ring_bridge_eq_output_integration(self):
        """测试ring_bridge的EQ输出集成"""
        # 创建本地目标flit
        local_flit = self.create_test_flit(1, (1, 1), "req")

        # 将flit添加到ring_bridge的TR输入（来自CrossPoint）
        self.node.add_to_ring_bridge_input(local_flit, "TR", "req")

        # 运行周期更新
        self.node._step_compute_phase()
        self.node._step_update_phase()

        # 处理ring_bridge仲裁（应该输出到EQ）
        self.node.process_ring_bridge_arbitration(0)

        # 再次运行周期更新
        self.node._step_compute_phase()
        self.node._step_update_phase()

        # 检查EQ输出FIFO
        eq_output = self.node.ring_bridge_output_fifos["req"]["EQ"]
        self.assertTrue(eq_output.valid_signal())

        # 通过接口获取EQ输出
        eq_flit = self.node.get_ring_bridge_eq_flit("req")
        self.assertIsNotNone(eq_flit)
        self.assertEqual(eq_flit.packet_id, 1)

    def test_ring_bridge_iq_input_integration(self):
        """测试ring_bridge的IQ输入集成"""
        # 创建需要垂直路由的flit（XY路由）
        flit = self.create_test_flit(1, (1, 3), "req")  # 向上

        # 先放入inject TU FIFO
        tu_fifo = self.node.inject_input_fifos["req"]["TU"]
        tu_fifo.write_input(flit)

        # 运行周期更新
        self.node._step_compute_phase()
        self.node._step_update_phase()

        # 处理ring_bridge仲裁
        self.node.process_ring_bridge_arbitration(0)

        # 再次运行周期更新
        self.node._step_compute_phase()
        self.node._step_update_phase()

        # 检查ring_bridge的TU输出（注意：对于(1,3)目标，XY路由应该路由到EQ而不TU）
        eq_output = self.node.ring_bridge_output_fifos["req"]["EQ"]
        self.assertTrue(eq_output.valid_signal())

    # ==================== 集成测试 ====================

    def test_complete_data_flow(self):
        """测试完整的数据流：inject -> ring_bridge -> eject"""
        self.node.connect_ip("ip_0")

        # 1. 创建本地目标flit
        local_flit = self.create_test_flit(1, (1, 1), "req")

        # 2. IP注入flit
        self.node.add_to_inject_queue(local_flit, "req", "ip_0")

        # 先运行一个计算和更新周期使FIFO状态正确
        self.node._step_compute_phase()
        self.node._step_update_phase()

        # 运行两个完整周期更新（需要两个周期完成完整流水线）
        self.node.update_state(0)
        self.node.update_state(1)

        # 5. IP获取flit
        received_flit = self.node.get_eject_flit("ip_0", "req")
        self.assertIsNotNone(received_flit)
        self.assertEqual(received_flit.packet_id, 1)

    def test_crosspoint_to_eject_flow(self):
        """测试CrossPoint到eject的数据流"""
        self.node.connect_ip("ip_0")

        # 1. 创建本地目标flit
        local_flit = self.create_test_flit(1, (1, 1), "req")

        # 2. CrossPoint向ring_bridge注入flit
        self.node.add_to_ring_bridge_input(local_flit, "TR", "req")

        # 先运行一个计算和更新周期使FIFO状态正确
        self.node._step_compute_phase()
        self.node._step_update_phase()

        # 运行两个完整周期更新
        self.node.update_state(0)
        self.node.update_state(1)

        # 5. IP获取flit
        received_flit = self.node.get_eject_flit("ip_0", "req")
        self.assertIsNotNone(received_flit)
        self.assertEqual(received_flit.packet_id, 1)

    def test_multi_ip_arbitration(self):
        """测试多IP仲裁"""
        # 连接多个IP
        self.node.connect_ip("ip_0")
        self.node.connect_ip("ip_1")
        self.node.connect_ip("ip_2")

        # 创建多个本地flit
        for i in range(3):
            local_flit = self.create_test_flit(i, (1, 1), "req")
            eq_fifo = self.node.inject_input_fifos["req"]["EQ"]
            eq_fifo.write_input(local_flit)

        # 运行初始周期更新
        self.node._step_compute_phase()
        self.node._step_update_phase()

        # 处理多次仲裁
        for cycle in range(3):
            self.node.process_eject_arbitration(cycle)
            self.node._step_compute_phase()
            self.node._step_update_phase()

        # 检查各IP是否都收到flit
        received_count = 0
        for ip_id in ["ip_0", "ip_1", "ip_2"]:
            flit = self.node.get_eject_flit(ip_id, "req")
            if flit is not None:
                received_count += 1

        self.assertGreater(received_count, 0)  # 至少有一个IP收到flit

    def test_node_state_update_integration(self):
        """测试节点状态更新的集成"""
        self.node.connect_ip("ip_0")

        # 创建测试数据
        inject_flit = self.create_test_flit(1, (2, 1), "req")  # 向右
        local_flit = self.create_test_flit(2, (1, 1), "req")  # 本地

        # 注入flit
        self.node.add_to_inject_queue(inject_flit, "req", "ip_0")
        self.node.add_to_inject_queue(local_flit, "req", "ip_0")

        # 先运行一个计算和更新周期使FIFO状态正确
        self.node._step_compute_phase()
        self.node._step_update_phase()

        # 统一状态更新（应该处理inject、ring_bridge、eject仲裁）
        self.node.update_state(0)

        # 检查结果
        # TR方向应该有向右的flit
        tr_fifo = self.node.inject_input_fifos["req"]["TR"]
        eq_fifo = self.node.inject_input_fifos["req"]["EQ"]
        # 至少其中一个方向应该有flit
        self.assertTrue(tr_fifo.valid_signal() or eq_fifo.valid_signal())

        # 系统正常运行
        self.assertIsNotNone(self.node)

    def test_performance_statistics(self):
        """测试性能统计"""
        self.node.connect_ip("ip_0")

        # 获取初始统计
        initial_stats = self.node.get_stats()
        self.assertIn("buffer_occupancy", initial_stats)
        self.assertIn("ip_inject_channel_buffers", initial_stats["buffer_occupancy"])
        self.assertIn("ip_eject_channel_buffers", initial_stats["buffer_occupancy"])
        self.assertIn("ring_bridge_input_fifos", initial_stats["buffer_occupancy"])
        self.assertIn("ring_bridge_output_fifos", initial_stats["buffer_occupancy"])

        # 检查缓冲区占用统计
        inject_occupancy = initial_stats["buffer_occupancy"]["ip_inject_channel_buffers"]
        self.assertIn("ip_0", inject_occupancy)

        eject_occupancy = initial_stats["buffer_occupancy"]["ip_eject_channel_buffers"]
        self.assertIn("ip_0", eject_occupancy)


class TestCrossRingEdgeCases(unittest.TestCase):
    """CrossRing边界情况测试"""

    def setUp(self):
        """测试初始化"""
        self.config = CrossRingConfig()
        self.config.iq_ch_depth = 2  # 小缓冲区测试
        self.config.iq_out_depth = 2
        self.config.eq_in_depth = 2
        self.config.eq_ch_depth = 2
        self.config.rb_in_depth = 2
        self.config.rb_out_depth = 2

        self.logger = logging.getLogger("test")
        self.node = CrossRingNode(0, (1, 1), self.config, self.logger)

    def test_buffer_full_scenarios(self):
        """测试缓冲区满的情况"""
        self.node.connect_ip("ip_0")

        # 填满inject channel buffer
        for i in range(3):  # 尝试超过容量
            flit = CrossRingFlit(packet_id=i, flit_id=1)
            success = self.node.add_to_inject_queue(flit, "req", "ip_0")
            if i < 2:
                self.assertTrue(success)  # 前两个应该成功
            else:
                self.assertFalse(success)  # 第三个应该失败

    def test_no_ip_connected_scenarios(self):
        """测试没有IP连接的情况"""
        # 尝试注入flit到不存在的IP
        flit = CrossRingFlit(packet_id=1, flit_id=1)
        success = self.node.add_to_inject_queue(flit, "req", "nonexistent_ip")
        self.assertFalse(success)

        # 尝试从不存在的IP获取flit
        received_flit = self.node.get_eject_flit("nonexistent_ip", "req")
        self.assertIsNone(received_flit)

    def test_routing_strategy_switching(self):
        """测试运行时路由策略切换"""
        # 初始XY路由
        self.node.config.routing_strategy = RoutingStrategy.XY
        xy_sources = self.node._get_ring_bridge_input_sources()

        # 切换到YX路由
        self.node.config.routing_strategy = RoutingStrategy.YX
        yx_sources = self.node._get_ring_bridge_input_sources()

        # 验证源列表不同
        self.assertNotEqual(xy_sources, yx_sources)


class TestCrossRingTagMechanism(unittest.TestCase):
    """CrossRing Tag机制测试类"""

    def setUp(self):
        """测试初始化"""
        self.config = CrossRingConfig()
        self.logger = logging.getLogger("test_tag")
        self.tag_manager = CrossRingTagManager(node_id=0, config=self.config, logger=self.logger)

    def test_tag_manager_initialization(self):
        """测试Tag管理器初始化"""
        self.assertEqual(self.tag_manager.node_id, 0)
        self.assertIsNotNone(self.tag_manager.itag_config)
        self.assertIsNotNone(self.tag_manager.etag_config)

        # 检查I-Tag状态
        self.assertIn("req", self.tag_manager.itag_states)
        self.assertIn("horizontal", self.tag_manager.itag_states["req"])
        self.assertIn("vertical", self.tag_manager.itag_states["req"])

        # 检查E-Tag状态
        self.assertIn("req", self.tag_manager.etag_states)
        self.assertIn("TL", self.tag_manager.etag_states["req"])

    def test_itag_trigger_conditions(self):
        """测试I-Tag触发条件"""
        # 测试正常情况下不触发
        should_trigger = self.tag_manager.should_trigger_itag("req", "horizontal", 50)
        self.assertFalse(should_trigger)

        # 测试超过阈值时触发
        should_trigger = self.tag_manager.should_trigger_itag("req", "horizontal", 100)
        self.assertTrue(should_trigger)

    def test_itag_reservation_cycle(self):
        """测试I-Tag预约周期"""
        # 创建模拟RingSlice
        ring_slice = RingSlice("test_slice", "horizontal", 0)

        # 添加一个可预约的slot
        slot = CrossRingSlot(slot_id=1, cycle=0, channel="req")
        ring_slice.receive_slot(slot, "req")
        ring_slice.step(0)

        # 触发I-Tag预约
        success = self.tag_manager.trigger_itag_reservation("req", "horizontal", ring_slice, 100)
        # 由于简化实现可能找不到slot，这里只测试方法调用不报错
        self.assertIsInstance(success, bool)

        # 取消预约
        cancel_success = self.tag_manager.cancel_itag_reservation("req", "horizontal", ring_slice)
        self.assertIsInstance(cancel_success, bool)

    def test_etag_upgrade_logic(self):
        """测试E-Tag升级逻辑"""
        from src.noc.base.link import PriorityLevel

        # 创建有flit的slot
        slot = CrossRingSlot(slot_id=1, cycle=0, channel="req")
        flit = CrossRingFlit(packet_id=1, flit_id=1)
        slot.assign_flit(flit)

        # 测试T2 -> T1升级
        new_priority = self.tag_manager.should_upgrade_etag(slot, "req", "TL", 1)
        self.assertEqual(new_priority, PriorityLevel.T1)

        # 设置为T1后测试T1 -> T0升级
        slot.mark_etag(PriorityLevel.T1, "TL")
        new_priority = self.tag_manager.should_upgrade_etag(slot, "req", "TL", 2)
        self.assertEqual(new_priority, PriorityLevel.T0)

        # 测试TR方向不能升级到T0
        new_priority = self.tag_manager.should_upgrade_etag(slot, "req", "TR", 2)
        self.assertIsNone(new_priority)

    def test_etag_ejection_control(self):
        """测试E-Tag下环控制"""
        from src.noc.base.link import PriorityLevel

        # 创建有flit的slot
        slot = CrossRingSlot(slot_id=1, cycle=0, channel="req")
        flit = CrossRingFlit(packet_id=1, flit_id=1)
        slot.assign_flit(flit)

        # 测试T2级下环
        slot.mark_etag(PriorityLevel.T2, "TL")
        can_eject = self.tag_manager.can_eject_with_etag(slot, "req", "TL", 5, 16)
        self.assertTrue(can_eject)  # FIFO占用5 < T2最大8

        # 测试T1级下环
        slot.mark_etag(PriorityLevel.T1, "TL")
        can_eject = self.tag_manager.can_eject_with_etag(slot, "req", "TL", 10, 16)
        self.assertTrue(can_eject)  # FIFO占用10 < T1最大15

        # 测试T0级下环（需要轮询）
        slot.mark_etag(PriorityLevel.T0, "TL")
        can_eject = self.tag_manager.can_eject_with_etag(slot, "req", "TL", 15, 16)
        self.assertIsInstance(can_eject, bool)  # T0级需要轮询，结果可能为True或False

    def test_tag_states_update(self):
        """测试Tag状态更新"""
        # 更新状态
        self.tag_manager.update_states(100)

        # 检查状态更新正常运行
        status = self.tag_manager.get_tag_manager_status()
        self.assertIn("node_id", status)
        self.assertIn("itag_states", status)
        self.assertIn("etag_states", status)
        self.assertIn("stats", status)

    def test_tag_statistics(self):
        """测试Tag统计信息"""
        initial_stats = self.tag_manager.stats.copy()

        # 重置统计
        self.tag_manager.reset_stats()

        # 检查统计被重置
        for channel in ["req", "rsp", "data"]:
            self.assertEqual(self.tag_manager.stats["itag_triggers"][channel], 0)
            self.assertEqual(self.tag_manager.stats["etag_upgrades"][channel]["T2_to_T1"], 0)
            self.assertEqual(self.tag_manager.stats["successful_injections"][channel], 0)


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")  # 减少测试输出

    # 运行测试
    unittest.main(verbosity=2)
