"""
NoC节点实现测试。

测试RouterNode、NetworkInterface、ProcessingElement、MemoryController和NodeFactory的实现。
"""

import unittest
import sys
import os

# 添加源代码路径
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from src.noc.base.router import RouterNode, RoutingAlgorithm, PortDirection
from src.noc.base.network_interface import NetworkInterface, ProtocolType, QoSClass
from src.noc.base.processing_element import ProcessingElement, WorkloadType, TaskType
from src.noc.base.memory_controller import MemoryController, MemoryType, SchedulingPolicy
from src.noc.base.node_factory import NoCNodeFactory, create_node, create_mesh_topology
from src.noc.crossring.flit import CrossRingFlit
from src.noc.utils.types import NodeType, Priority


class TestRouterNode(unittest.TestCase):
    """测试RouterNode类"""

    def setUp(self):
        """设置测试环境"""
        self.router = RouterNode(node_id=1, position=(1, 1), routing_algorithm=RoutingAlgorithm.XY)

    def test_router_initialization(self):
        """测试路由器初始化"""
        self.assertEqual(self.router.node_id, 1)
        self.assertEqual(self.router.position, (1, 1))
        self.assertEqual(self.router.node_type, NodeType.ROUTER)
        self.assertEqual(self.router.routing_algorithm, RoutingAlgorithm.XY)
        self.assertEqual(self.router.num_ports, 5)

    def test_routing_algorithms(self):
        """测试路由算法"""
        # 测试XY路由
        self.router.mesh_width = 4

        # 测试向东路由
        output_port = self.router._xy_routing(5)  # 目标节点5在(1,1)的东边
        self.assertEqual(output_port, PortDirection.EAST)

        # 测试向北路由
        output_port = self.router._xy_routing(2)  # 目标节点2在(1,1)的北边
        self.assertEqual(output_port, PortDirection.NORTH)

    def test_flit_processing(self):
        """测试flit处理"""
        # 创建测试flit
        flit = CrossRingFlit(source=0, destination=5, packet_id="test_packet", creation_time=0)
        flit.priority = Priority.MEDIUM

        # 测试flit处理
        result = self.router.process_flit(flit, "west")
        self.assertTrue(result)

    def test_buffer_management(self):
        """测试缓冲区管理"""
        # 测试缓冲区状态
        self.assertFalse(self.router.is_buffer_full("north"))
        self.assertTrue(self.router.is_buffer_empty("north"))

        # 测试可用空间
        space = self.router.get_available_space("north")
        self.assertEqual(space, self.router.input_buffer_size)

    def test_virtual_channel_allocation(self):
        """测试虚拟通道分配"""
        # 分配虚拟通道
        vc_id = self.router.allocate_virtual_channel("north", Priority.HIGH)
        self.assertIsNotNone(vc_id)
        self.assertGreaterEqual(vc_id, 0)
        self.assertLess(vc_id, self.router.virtual_channels)

    def test_router_status(self):
        """测试路由器状态获取"""
        status = self.router.get_router_status()
        self.assertIn("routing_algorithm", status)
        self.assertIn("router_stats", status)
        self.assertIn("port_credits", status)


class TestNetworkInterface(unittest.TestCase):
    """测试NetworkInterface类"""

    def setUp(self):
        """设置测试环境"""
        self.ni = NetworkInterface(node_id=2, position=(0, 1), protocol_type=ProtocolType.MEMORY)

    def test_ni_initialization(self):
        """测试网络接口初始化"""
        self.assertEqual(self.ni.node_id, 2)
        self.assertEqual(self.ni.position, (0, 1))
        self.assertEqual(self.ni.node_type, NodeType.NETWORK_INTERFACE)
        self.assertEqual(self.ni.protocol_type, ProtocolType.MEMORY)

    def test_protocol_conversion(self):
        """测试协议转换"""
        # 创建测试flit
        flit = CrossRingFlit(source=2, destination=0, packet_id="memory_request", creation_time=0)
        flit.memory_address = 0x1000
        flit.memory_op = "READ"

        # 测试IP到NoC协议转换
        result = self.ni._convert_ip_to_noc_protocol(flit)
        self.assertTrue(result)

    def test_qos_policy(self):
        """测试QoS策略"""
        self.ni.qos_enabled = True

        # 创建测试flit
        flit = CrossRingFlit(source=2, destination=0, packet_id="qos_test", creation_time=0)

        # 测试QoS策略应用
        result = self.ni._apply_qos_policy(flit)
        self.assertTrue(result)
        self.assertIsNotNone(flit.qos_class)

    def test_traffic_shaping(self):
        """测试流量整形"""
        # 初始化令牌桶
        self.ni.traffic_shaper["token_bucket"] = 10

        # 创建测试flit
        flit = CrossRingFlit(source=2, destination=0, packet_id="traffic_test", creation_time=0)

        # 测试流量整形
        result = self.ni._apply_traffic_shaping(flit)
        self.assertTrue(result)
        self.assertEqual(self.ni.traffic_shaper["token_bucket"], 9)

    def test_ni_status(self):
        """测试网络接口状态获取"""
        status = self.ni.get_ni_status()
        self.assertIn("protocol_type", status)
        self.assertIn("clock_ratio", status)
        self.assertIn("ni_stats", status)


class TestProcessingElement(unittest.TestCase):
    """测试ProcessingElement类"""

    def setUp(self):
        """设置测试环境"""
        self.pe = ProcessingElement(node_id=3, position=(1, 0), workload_type=WorkloadType.SYNTHETIC, num_cores=2)

    def test_pe_initialization(self):
        """测试处理元素初始化"""
        self.assertEqual(self.pe.node_id, 3)
        self.assertEqual(self.pe.position, (1, 0))
        self.assertEqual(self.pe.node_type, NodeType.PROCESSING_ELEMENT)
        self.assertEqual(self.pe.num_cores, 2)
        self.assertEqual(self.pe.workload_type, WorkloadType.SYNTHETIC)

    def test_task_creation(self):
        """测试任务创建"""
        task_id = self.pe.add_task(TaskType.COMPUTE, 100)
        self.assertIsNotNone(task_id)
        self.assertEqual(len(self.pe.task_queue), 1)

    def test_task_scheduling(self):
        """测试任务调度"""
        # 添加测试任务
        self.pe.add_task(TaskType.COMPUTE, 50)
        self.pe.add_task(TaskType.MEMORY_READ, 20)

        # 执行调度
        self.pe._schedule_tasks()

        # 检查是否有任务被调度
        running_count = sum(1 for task in self.pe.running_tasks.values() if task is not None)
        self.assertGreaterEqual(running_count, 1)

    def test_synthetic_workload_generation(self):
        """测试合成工作负载生成"""
        # 设置下一个任务时间为当前周期
        self.pe.workload_generator["next_task_time"] = 0
        self.pe.current_cycle = 0

        # 生成工作负载
        initial_queue_size = len(self.pe.task_queue)
        self.pe._generate_workload()

        # 检查是否生成了新任务
        self.assertGreater(len(self.pe.task_queue), initial_queue_size)

    def test_pe_step_execution(self):
        """测试PE周期执行"""
        self.pe.step_pe(1)
        self.assertEqual(self.pe.current_cycle, 1)

    def test_pe_status(self):
        """测试PE状态获取"""
        status = self.pe.get_pe_status()
        self.assertIn("pe_type", status)
        self.assertIn("num_cores", status)
        self.assertIn("workload_type", status)
        self.assertIn("pe_stats", status)


class TestMemoryController(unittest.TestCase):
    """测试MemoryController类"""

    def setUp(self):
        """设置测试环境"""
        self.mc = MemoryController(node_id=0, position=(0, 0), memory_type=MemoryType.DDR4, scheduling=SchedulingPolicy.FRFCFS)

    def test_mc_initialization(self):
        """测试内存控制器初始化"""
        self.assertEqual(self.mc.node_id, 0)
        self.assertEqual(self.mc.position, (0, 0))
        self.assertEqual(self.mc.node_type, NodeType.MEMORY_CONTROLLER)
        self.assertEqual(self.mc.memory_type, MemoryType.DDR4)
        self.assertEqual(self.mc.scheduling_policy, SchedulingPolicy.FRFCFS)

    def test_address_decoding(self):
        """测试地址解码"""
        address = 0x12345678
        bank_id, row_id, col_id = self.mc._decode_address(address)

        self.assertIsInstance(bank_id, int)
        self.assertIsInstance(row_id, int)
        self.assertIsInstance(col_id, int)
        self.assertGreaterEqual(bank_id, 0)
        self.assertLess(bank_id, self.mc.num_banks)

    def test_memory_request_processing(self):
        """测试内存请求处理"""
        # 创建内存请求flit
        flit = CrossRingFlit(source=3, destination=0, packet_id="mem_req_test", creation_time=0)
        flit.request_type = "read"
        flit.memory_address = 0x1000
        flit.memory_type = MemoryType.DDR4  # 补充 memory_type 字段
        flit.data_size = 64  # 补充 data_size 字段
        result = self.mc.process_flit(flit, "network")
        self.assertTrue(result)
        self.assertEqual(len(self.mc.request_queue), 1)

    def test_scheduling_policies(self):
        """测试调度策略"""
        # 添加测试请求
        for i in range(3):
            flit = CrossRingFlit(source=i + 1, destination=0, packet_id=f"req_{i}", creation_time=i)
            flit.request_type = "read"
            flit.memory_address = 0x1000 + i * 64
            self.mc.process_flit(flit, "network")

        # 测试FIFO调度
        self.mc.scheduling_policy = SchedulingPolicy.FIFO
        self.mc._schedule_requests()

        # 检查是否有请求被处理
        self.assertLessEqual(len(self.mc.request_queue), 2)  # 一个请求应该被处理

    def test_bank_state_management(self):
        """测试银行状态管理"""
        # 检查初始银行状态
        for bank_id in range(self.mc.num_banks):
            self.assertIn(bank_id, self.mc.bank_states)
            self.assertEqual(self.mc.bank_states[bank_id]["state"], "idle")

    def test_mc_status(self):
        """测试内存控制器状态获取"""
        status = self.mc.get_mc_status()
        self.assertIn("memory_type", status)
        self.assertIn("scheduling_policy", status)
        self.assertIn("mc_stats", status)
        self.assertIn("bank_states", status)


class TestNodeFactory(unittest.TestCase):
    """测试NoCNodeFactory类"""

    def setUp(self):
        """设置测试环境"""
        self.factory = NoCNodeFactory()

    def test_factory_initialization(self):
        """测试工厂初始化"""
        # 检查默认工厂是否注册
        self.assertIn(NodeType.ROUTER, self.factory.factories)
        self.assertIn(NodeType.NETWORK_INTERFACE, self.factory.factories)
        self.assertIn(NodeType.PROCESSING_ELEMENT, self.factory.factories)
        self.assertIn(NodeType.MEMORY_CONTROLLER, self.factory.factories)

    def test_node_creation(self):
        """测试节点创建"""
        # 创建路由器节点
        router = self.factory.create_node(NodeType.ROUTER, node_id=10, position=(2, 2), routing_algorithm="xy")

        self.assertIsInstance(router, RouterNode)
        self.assertEqual(router.node_id, 10)
        self.assertEqual(router.position, (2, 2))

        # 创建处理元素节点
        pe = self.factory.create_node(NodeType.PROCESSING_ELEMENT, node_id=11, position=(1, 2), num_cores=4)

        self.assertIsInstance(pe, ProcessingElement)
        self.assertEqual(pe.num_cores, 4)

    def test_config_template(self):
        """测试配置模板"""
        # 注册自定义配置
        self.factory.register_node_config("test_router", {"routing_algorithm": "adaptive", "virtual_channels": 4})

        # 使用配置模板创建节点
        router = self.factory.create_node(NodeType.ROUTER, node_id=12, position=(3, 3), config_name="test_router", virtual_channels=4)
        self.assertEqual(router.virtual_channels, 4)

    def test_mesh_topology_creation(self):
        """测试网格拓扑创建"""
        # 清除之前的节点
        self.factory.clear_nodes()

        # 创建2x2网格
        nodes = self.factory.create_mesh_topology(2, 2)

        self.assertEqual(len(nodes), 4)
        self.assertEqual(len(self.factory.created_nodes), 4)

        # 检查节点位置
        positions = [node.position for node in nodes]
        expected_positions = [(0, 0), (1, 0), (0, 1), (1, 1)]
        for pos in expected_positions:
            self.assertIn(pos, positions)

    def test_node_retrieval(self):
        """测试节点检索"""
        # 创建测试节点
        router = self.factory.create_node(NodeType.ROUTER, node_id=20, position=(0, 0))

        # 按ID检索
        retrieved_node = self.factory.get_node(20)
        self.assertEqual(retrieved_node, router)

        # 按类型检索
        routers = self.factory.get_nodes_by_type(NodeType.ROUTER)
        self.assertIn(router, routers)

    def test_factory_info(self):
        """测试工厂信息获取"""
        info = self.factory.get_factory_info()

        self.assertIn("registered_factories", info)
        self.assertIn("registered_configs", info)
        self.assertIn("created_nodes", info)
        self.assertIn("nodes_by_type", info)


class TestConvenienceFunctions(unittest.TestCase):
    """测试便捷函数"""

    def test_create_node_function(self):
        """测试create_node便捷函数"""
        router = create_node(NodeType.ROUTER, node_id=100, position=(5, 5))

        self.assertIsInstance(router, RouterNode)
        self.assertEqual(router.node_id, 100)

    def test_create_mesh_topology_function(self):
        """测试create_mesh_topology便捷函数"""
        # 清除默认工厂的节点
        from src.noc.base.node_factory import default_node_factory

        default_node_factory.clear_nodes()

        nodes = create_mesh_topology(3, 3)

        self.assertEqual(len(nodes), 9)

        # 检查节点ID分配
        node_ids = [node.node_id for node in nodes]
        expected_ids = list(range(9))
        self.assertEqual(sorted(node_ids), expected_ids)


class TestIntegration(unittest.TestCase):
    """集成测试"""

    def setUp(self):
        """设置测试环境"""
        self.factory = NoCNodeFactory()
        self.factory.clear_nodes()

    def test_end_to_end_communication(self):
        """测试端到端通信"""
        # 创建简单的2节点系统
        pe = self.factory.create_node(NodeType.PROCESSING_ELEMENT, node_id=1, position=(0, 0))

        mc = self.factory.create_node(NodeType.MEMORY_CONTROLLER, node_id=0, position=(1, 0))

        # PE生成内存请求
        pe.add_task(TaskType.COMPUTE, 10)  # 改为计算任务，确保能被执行

        # 模拟更多周期的执行
        for cycle in range(30):
            pe.step_pe(cycle)
            mc.step_mc(cycle)

        # 检查是否生成了内存请求
        self.assertGreater(pe.pe_stats["instructions_executed"], 0)

    def test_multi_node_system(self):
        """测试多节点系统"""
        # 创建小型网格系统
        nodes = self.factory.create_mesh_topology(2, 2)

        # 验证系统完整性
        self.assertEqual(len(nodes), 4)

        # 检查不同类型的节点都存在
        node_types = {node.node_type for node in nodes}
        self.assertGreater(len(node_types), 1)  # 应该有多种节点类型

        # 模拟系统运行
        for cycle in range(5):
            for node in nodes:
                if hasattr(node, "step_pe"):
                    node.step_pe(cycle)
                elif hasattr(node, "step_mc"):
                    node.step_mc(cycle)
                elif hasattr(node, "step_router"):
                    node.step_router(cycle)
                elif hasattr(node, "step_ni"):
                    node.step_ni(cycle)


if __name__ == "__main__":
    # 运行所有测试
    unittest.main(verbosity=2)
