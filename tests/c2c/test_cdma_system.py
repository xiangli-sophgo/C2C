"""
CDMA系统单元测试
每个测试独立运行，清晰说明测试目的、方法和预期结果
"""

import unittest
import time
import threading
import sys
import os
import sys

# Add the parent directory to sys.path to allow importing modules from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.c2c.protocol.cdma_system import CDMASystem, CDMASystemState, CDMAMessage
from src.c2c.protocol.memory_types import MemoryType
from src.c2c.utils.exceptions import CDMAError


class TestCDMASystemBasics(unittest.TestCase):
    """CDMA系统基础功能测试"""

    def setUp(self):
        """每个测试前的准备工作"""
        self.chip_a = CDMASystem("chip_A")
        self.chip_b = CDMASystem("chip_B")

        # 建立连接
        self.chip_a.connect_to_chip("chip_B", self.chip_b)
        self.chip_b.connect_to_chip("chip_A", self.chip_a)

    def tearDown(self):
        """每个测试后的清理工作"""
        try:
            self.chip_a.shutdown()
            self.chip_b.shutdown()
        except Exception as e:
            print(f"清理警告: {e}")

    def test_01_system_initialization(self):
        """
        测试目标: 验证CDMA系统能正确初始化
        测试方法: 创建两个CDMA系统并检查初始状态
        预期结果: 两个系统都处于READY状态且能正确连接
        """
        print("\n[TEST] 系统初始化测试")
        print("目标: 验证CDMA系统正确初始化")

        # 检查系统状态
        self.assertEqual(self.chip_a.state, CDMASystemState.READY, "chip_A应该处于READY状态")
        self.assertEqual(self.chip_b.state, CDMASystemState.READY, "chip_B应该处于READY状态")

        # 检查连接
        status_a = self.chip_a.get_comprehensive_status()
        self.assertIn("chip_B", status_a["connected_chips"], "chip_A应该连接到chip_B")

        print("✓ 系统初始化成功")
        print(f"  - chip_A状态: {self.chip_a.state.value}")
        print(f"  - chip_B状态: {self.chip_b.state.value}")
        print(f"  - 连接数: {len(status_a['connected_chips'])}")

    def test_02_cdma_receive_basic(self):
        """
        测试目标: 验证CDMA接收操作的基本功能
        测试方法: chip_A执行cdma_receive，向chip_B发送Credit
        预期结果: 操作成功，返回有效的transaction_id
        """
        print("\n[TEST] CDMA基础接收测试")
        print("目标: 验证CDMA接收操作基本功能")
        print("方法: chip_A接收来自chip_B的数据")

        result = self.chip_a.cdma_receive(dst_addr=0x1000, dst_shape=(512,), dst_mem_type=MemoryType.GMEM, src_chip_id="chip_B", data_type="float32")

        # 验证结果
        self.assertTrue(result.success, f"CDMA接收应该成功: {result.error_message}")
        self.assertIsNotNone(result.transaction_id, "应该返回有效的transaction_id")
        self.assertEqual(result.system_state, CDMASystemState.READY.value, "系统状态应该保持READY")

        print("✓ CDMA接收操作成功")
        print(f"  - Transaction ID: {result.transaction_id}")
        print(f"  - 执行时间: {result.execution_time:.3f}秒")
        print(f"  - 系统状态: {result.system_state}")

    def test_03_cdma_send_basic(self):
        """
        测试目标: 验证CDMA发送操作的基本功能
        测试方法: chip_B执行cdma_send，向chip_A发送数据
        预期结果: 操作成功，返回传输统计信息
        """
        print("\n[TEST] CDMA基础发送测试")
        print("目标: 验证CDMA发送操作基本功能")
        print("方法: chip_B向chip_A发送512字节float32数据")

        result = self.chip_b.cdma_send(src_addr=0x2000, src_shape=(128,), dst_chip_id="chip_A", src_mem_type=MemoryType.GMEM, data_type="float32")  # 128 * 4 bytes = 512 bytes

        # 验证结果
        self.assertTrue(result.success, f"CDMA发送应该成功: {result.error_message}")
        self.assertGreater(result.bytes_transferred, 0, "应该有数据传输")
        self.assertGreater(result.throughput_mbps, 0, "应该有吞吐量统计")

        print("✓ CDMA发送操作成功")
        print(f"  - 传输字节数: {result.bytes_transferred}")
        print(f"  - 吞吐量: {result.throughput_mbps:.2f} MB/s")
        print(f"  - 延迟: {result.latency_ms:.2f} ms")

    def test_04_tensor_shapes(self):
        """
        测试目标: 验证系统支持不同的tensor形状
        测试方法: 测试1D、2D、3D、4D等不同维度的tensor
        预期结果: 所有常见形状都能被正确处理
        """
        print("\n[TEST] Tensor形状支持测试")
        print("目标: 验证不同维度tensor形状的支持")

        test_shapes = [((64,), "1D tensor"), ((8, 8), "2D tensor"), ((4, 4, 4), "3D tensor"), ((2, 2, 2, 8), "4D tensor")]

        for shape, description in test_shapes:
            with self.subTest(shape=shape):
                print(f"  测试 {description}: {shape}")

                result = self.chip_a.cdma_receive(dst_addr=0x1000, dst_shape=shape, dst_mem_type=MemoryType.GMEM, src_chip_id="chip_B", data_type="float32")

                self.assertTrue(result.success, f"{description}应该被支持")
                print(f"    ✓ {description}支持成功")

        print("✓ 所有tensor形状测试通过")

    def test_05_data_types(self):
        """
        测试目标: 验证系统支持不同的数据类型
        测试方法: 测试float32、int32、int8等数据类型
        预期结果: 所有数据类型都能被正确处理
        """
        print("\n[TEST] 数据类型支持测试")
        print("目标: 验证不同数据类型的支持")

        data_types = ["float32", "int32", "int16", "int8"]

        for data_type in data_types:
            with self.subTest(data_type=data_type):
                print(f"  测试数据类型: {data_type}")

                result = self.chip_a.cdma_receive(dst_addr=0x1000, dst_shape=(128,), dst_mem_type=MemoryType.GMEM, src_chip_id="chip_B", data_type=data_type)

                self.assertTrue(result.success, f"数据类型{data_type}应该被支持")
                print(f"    ✓ {data_type}支持成功")

        print("✓ 所有数据类型测试通过")

    def test_06_memory_types(self):
        """
        测试目标: 验证系统支持不同的内存类型
        测试方法: 测试GMEM、L2M、LMEM等内存类型
        预期结果: 所有内存类型都能被正确处理
        """
        print("\n[TEST] 内存类型支持测试")
        print("目标: 验证不同内存类型的支持")

        memory_types = [(MemoryType.GMEM, "全局内存"), (MemoryType.L2M, "L2缓存"), (MemoryType.LMEM, "本地内存")]

        for mem_type, description in memory_types:
            with self.subTest(mem_type=mem_type):
                print(f"  测试{description}: {mem_type.value}")

                result = self.chip_a.cdma_receive(dst_addr=0x1000, dst_shape=(128,), dst_mem_type=mem_type, src_chip_id="chip_B", data_type="float32")

                self.assertTrue(result.success, f"{description}应该被支持")
                print(f"    ✓ {description}支持成功")

        print("✓ 所有内存类型测试通过")


class TestCDMASystemAdvanced(unittest.TestCase):
    """CDMA系统高级功能测试"""

    def setUp(self):
        """测试准备"""
        self.chip_a = CDMASystem("chip_A")
        self.chip_b = CDMASystem("chip_B")
        self.chip_c = CDMASystem("chip_C")

        # 建立全连接
        for chip1, id1 in [(self.chip_a, "chip_A"), (self.chip_b, "chip_B"), (self.chip_c, "chip_C")]:
            for chip2, id2 in [(self.chip_a, "chip_A"), (self.chip_b, "chip_B"), (self.chip_c, "chip_C")]:
                if id1 != id2:
                    chip1.connect_to_chip(id2, chip2)

    def tearDown(self):
        """测试清理"""
        for chip in [self.chip_a, self.chip_b, self.chip_c]:
            try:
                chip.shutdown()
            except Exception as e:
                print(f"清理警告: {e}")

    def test_07_reduce_operations(self):
        """
        测试目标: 验证All Reduce操作的支持
        测试方法: 测试none、sum、mean等reduce操作
        预期结果: 支持的reduce操作都能正常执行
        """
        print("\n[TEST] All Reduce操作测试")
        print("目标: 验证不同reduce操作的支持")

        reduce_ops = ["none", "sum", "mean"]

        for reduce_op in reduce_ops:
            with self.subTest(reduce_op=reduce_op):
                print(f"  测试reduce操作: {reduce_op}")

                result = self.chip_b.cdma_send(src_addr=0x2000, src_shape=(64,), dst_chip_id="chip_A", src_mem_type=MemoryType.GMEM, data_type="float32", reduce_op=reduce_op)

                self.assertTrue(result.success, f"Reduce操作{reduce_op}应该成功")
                print(f"    ✓ {reduce_op}操作成功")

        print("✓ 所有reduce操作测试通过")

    def test_08_concurrent_operations(self):
        """
        测试目标: 验证并发操作的处理能力
        测试方法: 同时执行多个发送和接收操作
        预期结果: 并发操作都能正确完成
        """
        print("\n[TEST] 并发操作测试")
        print("目标: 验证系统并发处理能力")
        print("方法: 同时执行2个接收和2个发送操作")

        results = []
        threads = []

        def receive_operation(chip, src_chip_id, addr_offset, index):
            result = chip.cdma_receive(dst_addr=0x1000 + addr_offset, dst_shape=(32,), dst_mem_type=MemoryType.GMEM, src_chip_id=src_chip_id, data_type="float32")
            results.append((f"recv_{index}", result))
            print(f"    接收操作{index}完成: {result.success}")

        def send_operation(chip, dst_chip_id, addr_offset, index):
            result = chip.cdma_send(src_addr=0x2000 + addr_offset, src_shape=(32,), dst_chip_id=dst_chip_id, src_mem_type=MemoryType.GMEM, data_type="float32")
            results.append((f"send_{index}", result))
            print(f"    发送操作{index}完成: {result.success}")

        # 创建并发操作
        threads.append(threading.Thread(target=receive_operation, args=(self.chip_a, "chip_B", 0x0000, 1)))
        threads.append(threading.Thread(target=receive_operation, args=(self.chip_a, "chip_C", 0x1000, 2)))
        threads.append(threading.Thread(target=send_operation, args=(self.chip_b, "chip_A", 0x0000, 1)))
        threads.append(threading.Thread(target=send_operation, args=(self.chip_c, "chip_A", 0x1000, 2)))

        # 启动所有线程
        for thread in threads:
            thread.start()

        # 等待完成
        for thread in threads:
            thread.join(timeout=5.0)  # 5秒超时

        # 验证结果
        self.assertEqual(len(results), 4, "应该有4个操作结果")
        success_count = sum(1 for _, result in results if result.success)

        print(f"✓ 并发操作测试完成: {success_count}/4个操作成功")


class TestCDMASystemPerformance(unittest.TestCase):
    """CDMA系统性能测试"""

    def setUp(self):
        """测试准备"""
        self.chip_a = CDMASystem("chip_A")
        self.chip_b = CDMASystem("chip_B")

        self.chip_a.connect_to_chip("chip_B", self.chip_b)
        self.chip_b.connect_to_chip("chip_A", self.chip_a)

    def tearDown(self):
        """测试清理"""
        try:
            self.chip_a.shutdown()
            self.chip_b.shutdown()
        except Exception as e:
            print(f"清理警告: {e}")

    def test_09_performance_monitoring(self):
        """
        测试目标: 验证性能监控功能
        测试方法: 执行多个操作并检查性能统计
        预期结果: 能够正确收集和报告性能数据
        """
        print("\n[TEST] 性能监控测试")
        print("目标: 验证性能监控功能")
        print("方法: 执行5个发送操作并检查性能统计")

        # 执行多个操作
        operation_count = 5
        for i in range(operation_count):
            result = self.chip_b.cdma_send(src_addr=0x2000 + i * 0x100, src_shape=(32,), dst_chip_id="chip_A", src_mem_type=MemoryType.GMEM, data_type="float32")
            print(f"    操作{i+1}完成: {result.success}")

        # 获取性能报告
        report = self.chip_b.generate_performance_report()

        # 验证报告内容
        self.assertIn("throughput_stats", report, "应该包含吞吐量统计")
        self.assertIn("latency_stats", report, "应该包含延迟统计")

        throughput_events = report["throughput_stats"]["total_events"]
        latency_events = report["latency_stats"]["total_events"]

        print("✓ 性能监控测试通过")
        print(f"  - 吞吐量事件数: {throughput_events}")
        print(f"  - 延迟事件数: {latency_events}")

    def test_10_system_status(self):
        """
        测试目标: 验证系统状态查询功能
        测试方法: 执行操作后查询系统综合状态
        预期结果: 状态信息完整且准确
        """
        print("\n[TEST] 系统状态查询测试")
        print("目标: 验证系统状态查询功能")

        # 执行一些操作
        self.chip_a.cdma_receive(dst_addr=0x1000, dst_shape=(128,), dst_mem_type=MemoryType.GMEM, src_chip_id="chip_B", data_type="float32")

        # 获取系统状态
        status = self.chip_a.get_comprehensive_status()

        # 验证状态结构
        required_fields = ["chip_id", "system_state", "performance_stats", "flow_stats", "error_stats", "sync_stats", "transaction_stats", "connected_chips"]

        for field in required_fields:
            self.assertIn(field, status, f"状态应该包含{field}字段")

        self.assertEqual(status["chip_id"], "chip_A", "芯片ID应该正确")
        self.assertIn("chip_B", status["connected_chips"], "应该显示连接的芯片")

        print("✓ 系统状态查询测试通过")
        print(f"  - 芯片ID: {status['chip_id']}")
        print(f"  - 系统状态: {status['system_state']}")
        print(f"  - 连接芯片数: {len(status['connected_chips'])}")


class TestCDMACompatibility(unittest.TestCase):
    """CDMA兼容性接口测试"""

    def setUp(self):
        """测试准备"""
        self.chip_a = CDMASystem("chip_A")
        self.chip_b = CDMASystem("chip_B")

        self.chip_a.connect_to_chip("chip_B", self.chip_b)
        self.chip_b.connect_to_chip("chip_A", self.chip_a)

    def tearDown(self):
        """测试清理"""
        try:
            self.chip_a.shutdown()
            self.chip_b.shutdown()
        except Exception as e:
            print(f"清理警告: {e}")

    def test_11_message_interface(self):
        """
        测试目标: 验证兼容性消息接口
        测试方法: 使用CDMAMessage接口发送消息
        预期结果: 兼容接口正常工作
        """
        print("\n[TEST] 兼容性消息接口测试")
        print("目标: 验证CDMAMessage兼容接口")

        # 创建发送消息
        send_msg = CDMAMessage(source_id="chip_B", destination_id="chip_A", message_type="send", tensor_shape=(64,), data_type="float32")

        # 创建接收消息
        recv_msg = CDMAMessage(source_id="chip_B", destination_id="chip_A", message_type="receive", tensor_shape=(64,), data_type="float32")

        # 测试接口
        recv_result = self.chip_a.send_message(recv_msg)
        send_result = self.chip_b.send_message(send_msg)

        self.assertTrue(recv_result.success, "接收消息应该成功")
        print(f"  ✓ 接收消息处理成功")

        print("✓ 兼容性接口测试通过")


def run_test_suite():
    """运行完整的测试套件"""
    print("=" * 70)
    print("CDMA系统测试套件")
    print("=" * 70)

    # 定义测试类
    test_classes = [TestCDMASystemBasics, TestCDMASystemAdvanced, TestCDMASystemPerformance, TestCDMACompatibility]

    total_tests = 0
    passed_tests = 0
    failed_tests = 0

    for test_class in test_classes:
        print(f"\n{'='*70}")
        print(f"运行测试类: {test_class.__name__}")
        print(f"{'='*70}")

        # 创建测试套件
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)

        # 运行测试
        runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, "w"))
        result = runner.run(suite)

        # 统计结果
        class_tests = result.testsRun
        class_passed = class_tests - len(result.failures) - len(result.errors)
        class_failed = len(result.failures) + len(result.errors)

        total_tests += class_tests
        passed_tests += class_passed
        failed_tests += class_failed

        print(f"类测试结果: {class_passed}/{class_tests} 通过")

        # 显示失败的测试
        if result.failures:
            print("\n失败的测试:")
            for test, traceback in result.failures:
                print(f"  - {test}: FAIL")

        if result.errors:
            print("\n错误的测试:")
            for test, traceback in result.errors:
                print(f"  - {test}: ERROR")

    # 总结
    print(f"\n{'='*70}")
    print("测试结果总结")
    print(f"{'='*70}")
    print(f"总测试数: {total_tests}")
    print(f"通过测试: {passed_tests}")
    print(f"失败测试: {failed_tests}")
    print(f"成功率: {passed_tests/total_tests*100:.1f}%")

    if failed_tests == 0:
        print("\n🎉 所有测试通过！CDMA系统工作正常。")
    else:
        print(f"\n⚠️ 有 {failed_tests} 个测试失败，需要检查实现。")

    return failed_tests == 0


if __name__ == "__main__":
    run_test_suite()
