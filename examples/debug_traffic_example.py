#!/usr/bin/env python3
"""
带调试功能的流量验证示例

这个示例展示了如何使用调试功能来验证每个请求的响应和数据的正确性
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.noc.crossring.model import CrossRingModel
from src.noc.crossring.config import CrossRingConfig
from src.noc.utils.traffic_scheduler import TrafficScheduler


class DebugTrafficValidator:
    """带调试功能的流量验证器"""

    def __init__(self, config: CrossRingConfig):
        self.config = config
        self.model = None
        self.traffic_scheduler = None
        self.injected_packets = {}  # 记录注入的包

    def setup_model(self, topology_size: tuple = (4, 4)):
        """设置模型"""
        self.config.num_row, self.config.num_col = topology_size
        self.config.basic_config.network_frequency = 1.0

        # 创建模型
        self.model = CrossRingModel(config=self.config)
        self.model.initialize_network()

        print(f"CrossRing模型初始化完成: {topology_size}")
        print(f"总节点数: {self.model.get_node_count()}")
        print(f"配置验证: num_row={self.config.num_row}, num_col={self.config.num_col}, num_nodes={self.config.num_nodes}")

    def setup_traffic_scheduler(self, traffic_file_path: str, traffic_files: list):
        """设置流量调度器"""
        self.traffic_scheduler = TrafficScheduler(config=self.config, traffic_file_path=traffic_file_path)
        self.traffic_scheduler.setup_single_chain(traffic_files)
        self.traffic_scheduler.set_verbose(True)

    def enable_debug_mode(self, trace_packets: list = None):
        """启用调试模式"""
        self.model.enable_debug(level=2, trace_packets=trace_packets)
        print(f"调试模式已启用")
        if trace_packets:
            print(f"追踪包: {trace_packets}")

    def run_simulation_with_validation(self, max_cycles: int = 2000):
        """运行仿真并验证结果"""
        print(f"\n开始运行仿真，最大周期: {max_cycles}")
        print("=" * 50)

        # 启动初始流量
        self.traffic_scheduler.start_initial_traffics()

        cycle = 0
        requests_injected = 0
        packet_counter = 0

        while cycle < max_cycles:
            # 获取准备好的请求
            ready_requests = self.traffic_scheduler.get_ready_requests(cycle)

            # 注入请求到网络
            for request in ready_requests:
                req_time, src, src_type, dst, dst_type, op, burst, traffic_id = request

                # 生成唯一的包ID
                packet_id = f"req_{packet_counter}_{src}_{dst}_{op}"
                packet_counter += 1

                # 注入包并记录
                success = self.model.inject_packet(src_node=src, dst_node=dst, op_type=op, burst_size=burst, cycle=cycle, packet_id=packet_id)

                if success:
                    requests_injected += 1
                    self.injected_packets[packet_id] = {"source": src, "destination": dst, "op_type": op, "burst_size": burst, "injected_cycle": cycle, "traffic_id": traffic_id}

                    print(f"周期 {cycle}: 注入包 {packet_id} - {src}({src_type}) -> {dst}({dst_type}), {op}, burst={burst}")

                    # 更新流量统计
                    self.traffic_scheduler.update_traffic_stats(traffic_id, "injected_req")

            # 推进网络一个周期
            self.model.advance_cycle()

            # 检查完成的包
            completed_packets = self.model.get_completed_packets()
            for packet in completed_packets:
                self._validate_completed_packet(packet, cycle)

            # 检查并推进流量链
            completed_traffics = self.traffic_scheduler.check_and_advance_chains(cycle)

            # 进度报告
            if cycle % 200 == 0 and cycle > 0:
                active_count = self.traffic_scheduler.get_active_traffic_count()
                debug_stats = self.model.get_debug_statistics()
                print(f"周期 {cycle}: 注入={requests_injected}, 活跃流量={active_count}, 完成请求={debug_stats.get('completed_requests', 0)}")

            # 检查是否所有流量都完成
            if self.traffic_scheduler.is_all_completed():
                print(f"所有流量在周期 {cycle} 完成")
                break

            cycle += 1

        print(f"\n仿真在周期 {cycle} 结束")
        return self._generate_validation_report(cycle, requests_injected)

    def _validate_completed_packet(self, packet_info: dict, current_cycle: int):
        """验证已完成的包"""
        packet_id = packet_info.get("packet_id", "unknown")

        if packet_id in self.injected_packets:
            original = self.injected_packets[packet_id]

            # 验证基本信息
            if packet_info.get("source") != original["source"]:
                print(f"错误: 包 {packet_id} 源节点不匹配")

            if packet_info.get("destination") != original["destination"]:
                print(f"错误: 包 {packet_id} 目标节点不匹配")

            # 验证延迟
            latency = current_cycle - original["injected_cycle"]
            if latency > 100:  # 假设合理延迟上限
                print(f"警告: 包 {packet_id} 延迟过高: {latency} 周期")

            # 验证数据完整性
            if original["op_type"] == "R":
                expected_data_flits = original["burst_size"]
                actual_data_flits = packet_info.get("data_flit_count", 0)
                if expected_data_flits != actual_data_flits:
                    print(f"错误: 包 {packet_id} 数据flit数量不匹配: 期望 {expected_data_flits}, 实际 {actual_data_flits}")

    def _generate_validation_report(self, final_cycle: int, total_injected: int) -> dict:
        """生成验证报告"""
        print("\n" + "=" * 60)
        print("流量验证报告")
        print("=" * 60)

        # 获取调试统计
        debug_stats = self.model.get_debug_statistics()
        validation_result = self.model.validate_traffic_correctness()

        print(f"仿真周期: {final_cycle}")
        print(f"注入请求总数: {total_injected}")
        print(f"完成请求数: {debug_stats.get('completed_requests', 0)}")
        print(f"失败请求数: {debug_stats.get('failed_requests', 0)}")
        print(f"完成率: {validation_result.get('completion_rate', 0):.1f}%")
        print(f"平均延迟: {debug_stats.get('avg_latency', 0):.1f} 周期")
        print(f"最大延迟: {debug_stats.get('max_latency', 0)} 周期")
        print(f"响应错误: {debug_stats.get('response_errors', 0)}")
        print(f"数据错误: {debug_stats.get('data_errors', 0)}")

        # 验证结果
        is_correct = validation_result.get("is_correct", False)
        print(f"\n总体验证结果: {'通过' if is_correct else '失败'}")

        if not is_correct:
            print("发现的问题:")
            if debug_stats.get("response_errors", 0) > 0:
                print(f"  - {debug_stats['response_errors']} 个响应错误")
            if debug_stats.get("data_errors", 0) > 0:
                print(f"  - {debug_stats['data_errors']} 个数据错误")

        # 流量调度器统计
        finish_stats = self.traffic_scheduler.get_finish_time_stats()
        print(f"\n流量调度器统计:")
        print(f"  读完成时间: {finish_stats.get('R_finish_time', 0)} ns")
        print(f"  写完成时间: {finish_stats.get('W_finish_time', 0)} ns")
        print(f"  总完成时间: {finish_stats.get('Total_finish_time', 0)} ns")

        # 链状态
        chain_status = self.traffic_scheduler.get_chain_status()
        print(f"\n链状态:")
        for chain_id, status in chain_status.items():
            print(f"  {chain_id}: {status['current_index']}/{status['total_traffics']} 完成")
            print(f"    时间偏移: {status['time_offset']} ns")
            print(f"    已完成: {status['is_completed']}")

        print("=" * 60)

        return {
            "is_correct": is_correct,
            "completion_rate": validation_result.get("completion_rate", 0),
            "total_injected": total_injected,
            "completed_requests": debug_stats.get("completed_requests", 0),
            "response_errors": debug_stats.get("response_errors", 0),
            "data_errors": debug_stats.get("data_errors", 0),
            "avg_latency": debug_stats.get("avg_latency", 0),
        }


def main():
    """主函数"""
    print("CrossRing NoC 调试流量验证示例")
    print("=" * 50)

    # 创建配置
    config = CrossRingConfig()
    config.basic_config.network_frequency = 1.0  # 简化为1GHz

    # 创建验证器
    validator = DebugTrafficValidator(config)

    # 设置模型
    validator.setup_model(topology_size=(4, 4))

    # 设置流量调度器
    traffic_file_path = r"traffic_data"
    traffic_files = ["samle_traffic.txt"]

    # 检查流量文件是否存在
    test_file = os.path.join(traffic_file_path, traffic_files[0])
    if not os.path.exists(test_file):
        print(f"错误: 流量文件不存在: {test_file}")
        return

    validator.setup_traffic_scheduler(traffic_file_path, traffic_files)

    # 启用调试模式，追踪前几个包
    trace_packets = ["req_0_0_1_R", "req_1_0_1_R", "req_2_0_1_R"]
    validator.enable_debug_mode(trace_packets)

    # 运行仿真并验证
    result = validator.run_simulation_with_validation(max_cycles=3000)

    # 打印最终调试报告
    validator.model.print_debug_report()

    # 判断测试结果
    if result["is_correct"] and result["completion_rate"] > 95.0:
        print("\n✅ 流量验证测试通过!")
    else:
        print("\n❌ 流量验证测试失败!")
        if result["completion_rate"] <= 95.0:
            print(f"   完成率过低: {result['completion_rate']:.1f}%")
        if not result["is_correct"]:
            print(f"   发现错误: 响应错误={result['response_errors']}, 数据错误={result['data_errors']}")


if __name__ == "__main__":
    main()
