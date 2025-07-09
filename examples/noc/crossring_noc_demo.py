#!/usr/bin/env python3
"""
CrossRing NoC仿真演示
====================

本演示展示如何使用CrossRing NoC仿真框架进行完整的仿真流程：
1. 加载traffic文件
2. 配置CrossRing模型
3. 运行仿真
4. 进行结果分析

Usage:
    python crossring_noc_demo.py

Features:
    - 支持多种traffic文件格式
    - 可配置的网络拓扑
    - 实时仿真监控
    - 详细的性能分析
    - 可视化结果展示
"""

import sys
import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import json

from src.noc.crossring.model import CrossRingModel, create_crossring_model
from src.noc.crossring.config import CrossRingConfig
from src.noc.crossring.config_factory import CrossRingConfigFactory
from src.noc.utils.traffic_scheduler import TrafficFileReader, TrafficState
from src.noc.analysis.result_processor import BandwidthAnalyzer
from src.noc.analysis.performance_metrics import PerformanceMetrics, RequestMetrics


class CrossRingNoCDemo:
    """CrossRing NoC仿真演示主类"""

    def __init__(self):
        """初始化演示"""
        self.logger = self._setup_logging()
        self.model: Optional[CrossRingModel] = None
        self.traffic_readers: List[TrafficFileReader] = []
        self.simulation_results: Dict[str, Any] = {}

        # 默认路径
        self.base_path = Path(__file__).parent.parent.parent
        self.traffic_data_path = self.base_path / "traffic_data"
        self.output_path = self.base_path / "output" / "crossring_noc_demo"

        # 创建输出目录
        self.output_path.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self) -> logging.Logger:
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler("crossring_noc_demo.log")]
        )
        return logging.getLogger(self.__class__.__name__)

    # ========== 配置设置模块 ==========

    def create_default_config(self, topology_size: Tuple[int, int] = (3, 4)) -> CrossRingConfig:
        """创建默认CrossRing配置"""
        num_row, num_col = topology_size

        self.logger.info(f"创建CrossRing配置: {num_row}x{num_col} 拓扑")

        config = CrossRingConfig(num_row=num_row, num_col=num_col, config_name=f"demo_{num_row}x{num_col}")

        # 配置IP位置 - 分布式部署
        total_nodes = num_row * num_col

        # GDMA: 分布在前几个节点
        config.gdma_send_position_list = list(range(min(4, total_nodes)))

        # DDR: 分布在后几个节点
        config.ddr_send_position_list = list(range(max(0, total_nodes - 4), total_nodes))

        # 其他IP类型
        config.sdma_send_position_list = [0, 1] if total_nodes >= 2 else [0]
        config.cdma_send_position_list = [2, 3] if total_nodes >= 4 else [0]
        config.l2m_send_position_list = list(range(0, min(2, total_nodes)))

        self.logger.info(f"配置完成 - GDMA节点: {config.gdma_send_position_list}")
        self.logger.info(f"配置完成 - DDR节点: {config.ddr_send_position_list}")

        return config

    def create_custom_config(self, config_dict: Dict[str, Any]) -> CrossRingConfig:
        """创建自定义配置"""
        self.logger.info("创建自定义CrossRing配置")

        # 使用工厂方法创建配置
        factory = CrossRingConfigFactory()
        config = factory.create_config(**config_dict)

        return config

    # ========== Traffic加载模块 ==========

    def load_traffic_file(self, filename: str, time_offset: int = 0, traffic_id: str = None) -> TrafficFileReader:
        """加载单个traffic文件"""
        if traffic_id is None:
            traffic_id = f"traffic_{len(self.traffic_readers)}"

        self.logger.info(f"加载traffic文件: {filename}")

        # 检查文件是否存在
        traffic_file_path = self.traffic_data_path / filename
        if not traffic_file_path.exists():
            raise FileNotFoundError(f"Traffic文件不存在: {traffic_file_path}")

        # 创建traffic读取器
        reader = TrafficFileReader(
            filename=filename, traffic_file_path=str(self.traffic_data_path), config=self.model.config if self.model else None, time_offset=time_offset, traffic_id=traffic_id
        )

        self.traffic_readers.append(reader)

        # 打印文件统计信息
        self.logger.info(f"Traffic统计 - 总请求: {reader.total_req}")
        self.logger.info(f"Traffic统计 - 读请求: {reader.read_req}, 写请求: {reader.write_req}")
        self.logger.info(f"Traffic统计 - 总flit: {reader.total_flit}")

        return reader

    def load_multiple_traffic_files(self, file_configs: List[Dict[str, Any]]) -> List[TrafficFileReader]:
        """加载多个traffic文件"""
        self.logger.info(f"加载 {len(file_configs)} 个traffic文件")

        readers = []
        for config in file_configs:
            reader = self.load_traffic_file(**config)
            readers.append(reader)

        return readers

    # ========== 仿真运行模块 ==========

    def create_model(self, config: CrossRingConfig) -> CrossRingModel:
        """创建CrossRing模型"""
        self.logger.info("创建CrossRing仿真模型")

        self.model = CrossRingModel(config)

        self.logger.info(f"模型创建完成: {self.model}")
        self.logger.info(f"节点数量: {self.model.get_node_count()}")
        self.logger.info(f"IP接口数量: {len(self.model.ip_interfaces)}")

        return self.model

    def inject_traffic_from_files(self) -> None:
        """从traffic文件注入流量"""
        if not self.traffic_readers:
            self.logger.warning("没有加载的traffic文件")
            return

        self.logger.info("开始从traffic文件注入流量")

        # 获取所有traffic的下一个请求时间
        total_injected = 0

        while True:
            # 找到最早的请求
            earliest_time = float("inf")
            earliest_reader = None

            for reader in self.traffic_readers:
                next_time = reader.peek_next_cycle()
                if next_time is not None and next_time < earliest_time:
                    earliest_time = next_time
                    earliest_reader = reader

            if earliest_reader is None:
                break  # 没有更多请求

            # 获取当前周期的所有请求
            requests = earliest_reader.get_requests_until_cycle(earliest_time)

            for req in requests:
                t, src, src_t, dst, dst_t, op, burst, traffic_id = req
                # 使用src_t和dst_t来指定IP类型
                packet_ids = self.model.inject_test_traffic(
                    source=src,
                    destination=dst,
                    req_type=op.lower(),  # 将操作类型转换为小写
                    count=1,
                    burst_length=burst,
                    ip_type=src_t  # 使用源IP类型
                )

                if packet_ids:
                    total_injected += len(packet_ids)
                    if total_injected % 100 == 0:
                        self.logger.debug(f"已注入 {total_injected} 个请求")

        self.logger.info(f"Traffic注入完成，总计: {total_injected} 个请求")

    def run_simulation(self, max_cycles: int = 100000, warmup_cycles: int = 0, stats_start_cycle: int = 0, progress_interval: int = 5000) -> Dict[str, Any]:
        """运行完整仿真"""
        if not self.model:
            raise RuntimeError("模型未初始化，请先调用create_model()")

        self.logger.info(f"开始CrossRing NoC仿真")
        self.logger.info(f"参数: max_cycles={max_cycles}, warmup={warmup_cycles}")

        # 启用调试模式（可选）
        self.model.enable_debug(level=0)

        start_time = time.time()

        # 从traffic文件注入流量（不注入额外的测试流量）
        self.inject_traffic_from_files()

        # 运行仿真
        results = self.model.run_simulation(max_cycles=max_cycles, warmup_cycles=warmup_cycles, stats_start_cycle=stats_start_cycle)

        end_time = time.time()
        simulation_time = end_time - start_time

        self.logger.info(f"仿真完成！")
        self.logger.info(f"仿真时间: {simulation_time:.2f} 秒")
        self.logger.info(f"最终周期: {self.model.cycle}")

        # 保存结果
        self.simulation_results = results

        return results

    def _inject_test_traffic(self) -> None:
        """注入测试流量到模型"""
        if not self.traffic_readers:
            self.logger.info("没有traffic文件，生成测试流量")
            # 生成一些基础的测试流量
            test_packets = [
                (0, 1, "R", 4),  # 从节点0到节点1的读操作
                (1, 2, "W", 4),  # 从节点1到节点2的写操作
                (2, 0, "R", 4),  # 从节点2到节点0的读操作
                (0, 3, "W", 4),  # 从节点0到节点3的写操作
            ]

            injected = 0
            for src, dst, op, burst in test_packets:
                if src < self.model.get_node_count() and dst < self.model.get_node_count():
                    success = self.model.inject_packet(src_node=src, dst_node=dst, op_type=op, burst_size=burst, cycle=0)
                    if success:
                        injected += 1

            self.logger.info(f"注入测试流量: {injected} 个包")
            return

        # 从traffic文件注入流量
        self.logger.info("从traffic文件注入流量")
        total_injected = 0

        for reader in self.traffic_readers:
            # 获取前100个请求进行测试
            requests = reader.get_requests_until_cycle(10000)  # 获取前10000周期的请求

            for req in requests[:50]:  # 限制为前50个请求避免过载
                t, src, src_t, dst, dst_t, op, burst, traffic_id = req

                # 确保节点ID在有效范围内
                if src < self.model.get_node_count() and dst < self.model.get_node_count():
                    packet_ids = self.model.inject_test_traffic(
                        source=src,
                        destination=dst,
                        req_type=op.lower(),
                        count=1,
                        burst_length=burst,
                        ip_type=src_t  # 使用源IP类型
                    )

                    if packet_ids:
                        total_injected += len(packet_ids)

        self.logger.info(f"从traffic文件注入流量: {total_injected} 个包")

    # ========== 结果分析模块 ==========

    def analyze_performance(self) -> Dict[str, Any]:
        """分析仿真性能"""
        if not self.simulation_results:
            self.logger.warning("没有仿真结果可供分析")
            return {}

        self.logger.info("开始性能分析")

        # 使用模型内置的分析功能
        analysis = self.model.analyze_simulation_results(self.simulation_results)

        self.logger.info("性能分析完成")
        return analysis

    def _analyze_ip_performance(self, ip_stats: Dict[str, Any]) -> Dict[str, Any]:
        """分析IP接口性能"""
        ip_analysis = {
            "total_ips": len(ip_stats),
            "ip_details": {},
            "summary": {"total_read_transactions": 0, "total_write_transactions": 0, "total_retries": 0, "avg_utilization": 0.0},
        }

        total_read = 0
        total_write = 0
        total_retries = 0

        for ip_key, stats in ip_stats.items():
            ip_type = ip_key.split("_")[0]

            read_count = stats.get("rn_read_active", 0)
            write_count = stats.get("rn_write_active", 0)
            retries = stats.get("read_retries", 0) + stats.get("write_retries", 0)

            ip_analysis["ip_details"][ip_key] = {
                "ip_type": ip_type,
                "read_transactions": read_count,
                "write_transactions": write_count,
                "total_retries": retries,
                "utilization": (read_count + write_count) / max(1, read_count + write_count + 100),  # 简化计算
            }

            total_read += read_count
            total_write += write_count
            total_retries += retries

        ip_analysis["summary"]["total_read_transactions"] = total_read
        ip_analysis["summary"]["total_write_transactions"] = total_write
        ip_analysis["summary"]["total_retries"] = total_retries

        if ip_stats:
            avg_util = sum(ip["utilization"] for ip in ip_analysis["ip_details"].values()) / len(ip_stats)
            ip_analysis["summary"]["avg_utilization"] = avg_util

        return ip_analysis

    def _analyze_congestion(self) -> Dict[str, Any]:
        """分析网络拥塞情况"""
        congestion_analysis = {"congestion_detected": False, "bottleneck_nodes": [], "congestion_summary": {}}

        if hasattr(self.model, "get_congestion_statistics"):
            congestion_stats = self.model.get_congestion_statistics()

            total_congestion = congestion_stats.get("total_congestion_events", 0)
            total_injections = congestion_stats.get("total_injections", 1)

            congestion_analysis["congestion_detected"] = total_congestion > 0
            congestion_analysis["congestion_summary"] = {
                "total_congestion_events": total_congestion,
                "total_injections": total_injections,
                "congestion_rate": total_congestion / total_injections if total_injections > 0 else 0.0,
            }

        return congestion_analysis

    # ========== 可视化和报告 ==========

    def generate_report(self) -> str:
        """生成仿真报告"""
        if not self.simulation_results:
            return "无仿真结果可生成报告"

        analysis = self.analyze_performance()

        # 使用模型内置的报告生成功能
        report_text = self.model.generate_simulation_report(self.simulation_results, analysis)

        # 保存报告
        report_file = self.output_path / "simulation_report.txt"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report_text)

        self.logger.info(f"报告已保存: {report_file}")

        return report_text

    def save_results(self, filename: str = None) -> str:
        """保存仿真结果到JSON文件"""
        if filename is None:
            filename = f"simulation_results_{int(time.time())}.json"

        results_file = self.output_path / filename

        # 准备要保存的数据
        save_data = {
            "simulation_results": self.simulation_results,
            "performance_analysis": self.analyze_performance(),
            "timestamp": time.time(),
            "config_summary": {
                "topology_size": (self.model.config.num_row, self.model.config.num_col) if self.model else None,
                "total_nodes": self.model.config.num_nodes if self.model else None,
            },
        }

        # 保存为JSON
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)

        self.logger.info(f"结果已保存: {results_file}")
        return str(results_file)

    # ========== 清理资源 ==========

    def cleanup(self):
        """清理资源"""
        self.logger.info("清理仿真资源")

        # 关闭traffic文件读取器
        for reader in self.traffic_readers:
            reader.close()

        # 清理模型
        if self.model:
            self.model.cleanup()

        self.logger.info("资源清理完成")


# ========== 示例场景演示 ==========


def demo_basic_simulation():
    """基础仿真演示"""
    print("\n" + "=" * 60)
    print("演示 1: 基础CrossRing NoC仿真")
    print("=" * 60)

    demo = CrossRingNoCDemo()

    try:
        # 1. 创建配置
        config = demo.create_default_config(topology_size=(3, 4))

        # 2. 创建模型
        model = demo.create_model(config)

        # 3. 加载traffic文件
        demo.load_traffic_file("crossring_traffic.txt", traffic_id="main_traffic")

        # 4. 运行仿真
        results = demo.run_simulation(max_cycles=500)

        # 5. 分析结果
        analysis = demo.analyze_performance()

        # 6. 生成报告
        report = demo.generate_report()
        print(report)

        # 7. 保存结果
        results_file = demo.save_results()
        print(f"\n结果已保存到: {results_file}")

        return True

    except Exception as e:
        demo.logger.error(f"仿真过程中发生错误: {e}")
        return False
    finally:
        demo.cleanup()


def demo_performance_comparison():
    """性能对比演示"""
    print("\n" + "=" * 60)
    print("演示 2: 不同配置性能对比")
    print("=" * 60)

    # 对比不同拓扑大小的性能
    topologies = [(2, 3), (3, 4), (4, 4)]
    results = {}

    for topology in topologies:
        print(f"\n测试拓扑: {topology[0]}x{topology[1]}")
        demo = CrossRingNoCDemo()

        try:
            config = demo.create_default_config(topology_size=topology)
            model = demo.create_model(config)
            demo.load_traffic_file("crossring_traffic.txt")

            # 运行短时间仿真进行对比
            sim_results = demo.run_simulation(max_cycles=20000)
            analysis = demo.analyze_performance()

            topology_key = f"{topology[0]}x{topology[1]}"
            results[topology_key] = {
                "throughput": analysis.get("basic_metrics", {}).get("throughput", 0),
                "peak_active": analysis.get("basic_metrics", {}).get("peak_active_requests", 0),
                "total_transactions": analysis.get("basic_metrics", {}).get("total_transactions", 0),
            }

            print(f"  吞吐量: {results[topology_key]['throughput']:.4f}")
            print(f"  峰值活跃请求: {results[topology_key]['peak_active']}")

        except Exception as e:
            print(f"  拓扑 {topology} 测试失败: {e}")
        finally:
            demo.cleanup()

    # 显示对比结果
    print(f"\n性能对比总结:")
    print(f"{'拓扑':<10} {'吞吐量':<15} {'峰值活跃':<15} {'总事务':<15}")
    print("-" * 60)
    for topology, metrics in results.items():
        print(f"{topology:<10} {metrics['throughput']:<15.4f} {metrics['peak_active']:<15} {metrics['total_transactions']:<15}")


def main():
    """主函数"""
    print("CrossRing NoC仿真演示")
    print("=" * 80)
    print("本演示展示CrossRing NoC完整的仿真流程")
    print("=" * 80)

    # 检查traffic文件是否存在
    base_path = Path(__file__).parent.parent.parent
    traffic_file = base_path / "traffic_data" / "crossring_traffic.txt"

    if not traffic_file.exists():
        print(f"警告: Traffic文件不存在: {traffic_file}")
        print("请确保traffic文件存在后再运行演示")
        return 1

    demos = [
        ("基础仿真", demo_basic_simulation),
        # ("性能对比", demo_performance_comparison),
    ]

    passed = 0
    total = len(demos)

    for demo_name, demo_func in demos:
        try:
            print(f"\n{'='*80}")
            print(f"开始演示: {demo_name}")
            print("=" * 80)

            if demo_func():
                passed += 1
                print(f"\n✓ {demo_name} 演示完成")
            else:
                print(f"\n✗ {demo_name} 演示失败")

        except Exception as e:
            print(f"\n✗ {demo_name} 演示异常: {e}")
            import traceback

            traceback.print_exc()

    print(f"\n{'='*80}")
    print(f"演示结果: {passed}/{total} 成功")

    if passed == total:
        print("🎉 所有演示完成！")
        print("\nCrossRing NoC仿真特性总结:")
        print("- ✓ Traffic文件加载和解析")
        print("- ✓ 可配置的CrossRing拓扑")
        print("- ✓ 完整的仿真执行流程")
        print("- ✓ 详细的性能分析")
        print("- ✓ 结果保存和报告生成")
    else:
        print(f"❌ {total - passed} 个演示未完成")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
