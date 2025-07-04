#!/usr/bin/env python3
"""
C2C流量模式演示
展示不同应用场景下的C2C通信流量特征和模式
"""

import sys
import os
import time
import threading
import random
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.c2c.protocol.cdma_system import CDMASystem, CDMASystemState
from src.c2c.protocol.memory_types import MemoryType
from src.c2c.utils.exceptions import CDMAError


class TrafficPattern(Enum):
    """流量模式类型"""

    BROADCAST = "broadcast"  # 广播模式
    GATHER = "gather"  # 聚合模式
    SCATTER = "scatter"  # 分散模式
    ALL_TO_ALL = "all_to_all"  # 全互连模式
    PIPELINE = "pipeline"  # 流水线模式
    RING = "ring"  # 环形模式


@dataclass
class TrafficMetrics:
    """流量指标"""

    total_bytes: int = 0
    total_operations: int = 0
    avg_throughput: float = 0.0
    avg_latency: float = 0.0
    peak_throughput: float = 0.0
    min_latency: float = float("inf")
    max_latency: float = 0.0
    success_rate: float = 0.0

    def update(self, bytes_transferred: int, throughput: float, latency: float, success: bool):
        """更新指标"""
        self.total_bytes += bytes_transferred
        self.total_operations += 1

        if success:
            # 更新吞吐量统计
            self.avg_throughput = (self.avg_throughput * (self.total_operations - 1) + throughput) / self.total_operations
            if throughput > self.peak_throughput:
                self.peak_throughput = throughput

            # 更新延迟统计
            self.avg_latency = (self.avg_latency * (self.total_operations - 1) + latency) / self.total_operations
            if latency < self.min_latency:
                self.min_latency = latency
            if latency > self.max_latency:
                self.max_latency = latency

        # 更新成功率
        successful_ops = int(self.success_rate * (self.total_operations - 1))
        if success:
            successful_ops += 1
        self.success_rate = successful_ops / self.total_operations


class TrafficGenerator:
    """流量生成器"""

    def __init__(self, chips: Dict[str, CDMASystem]):
        self.chips = chips
        self.metrics = {pattern: TrafficMetrics() for pattern in TrafficPattern}
        self.is_running = False

    def setup_chip_topology(self):
        """设置芯片拓扑"""
        print("设置8芯片C2C通信拓扑")
        print("=" * 50)

        # 8芯片系统：4个计算芯片 + 2个存储芯片 + 1个主控芯片 + 1个IO芯片
        chip_types = {
            "master": "主控芯片",
            "compute_0": "计算芯片0",
            "compute_1": "计算芯片1",
            "compute_2": "计算芯片2",
            "compute_3": "计算芯片3",
            "storage_0": "存储芯片0",
            "storage_1": "存储芯片1",
            "io_chip": "IO芯片",
        }

        # 建立全连接拓扑
        chip_ids = list(self.chips.keys())
        for i, chip_id_a in enumerate(chip_ids):
            for j, chip_id_b in enumerate(chip_ids):
                if i != j:
                    self.chips[chip_id_a].connect_to_chip(chip_id_b, self.chips[chip_id_b])

        print(f"✓ 建立了{len(chip_ids)}芯片的全连接拓扑")
        for chip_id, chip_type in chip_types.items():
            print(f"  - {chip_id}: {chip_type}")

        return chip_types

    def generate_broadcast_traffic(self, duration: float = 5.0):
        """生成广播流量模式"""
        print(f"\n生成广播流量模式 (持续 {duration}s)")
        print("场景: 主控芯片向所有计算芯片广播模型参数")
        print("-" * 40)

        pattern = TrafficPattern.BROADCAST
        start_time = time.time()

        while time.time() - start_time < duration:
            # 主控芯片向所有计算芯片广播
            broadcast_targets = ["compute_0", "compute_1", "compute_2", "compute_3"]

            for target in broadcast_targets:
                try:
                    # 目标芯片准备接收
                    recv_result = self.chips[target].cdma_receive(
                        dst_addr=0x10000000 + random.randint(0, 0x1000000), dst_shape=(512, 256), dst_mem_type=MemoryType.GMEM, src_chip_id="master", data_type="float32"  # 模型参数
                    )

                    # 主控芯片发送数据
                    send_result = self.chips["master"].cdma_send(src_addr=0x20000000, src_shape=(512, 256), dst_chip_id=target, src_mem_type=MemoryType.GMEM, data_type="float32")

                    success = recv_result.success and send_result.success
                    if success:
                        self.metrics[pattern].update(send_result.bytes_transferred, send_result.throughput_mbps, send_result.latency_ms, True)
                        print(f"  ✓ master -> {target}: {send_result.throughput_mbps:.1f} MB/s")
                    else:
                        self.metrics[pattern].update(0, 0, 0, False)
                        print(f"  ✗ master -> {target}: 失败")

                except Exception as e:
                    self.metrics[pattern].update(0, 0, 0, False)
                    print(f"  ✗ master -> {target}: 异常 - {e}")

            time.sleep(0.1)  # 短暂间隔

    def generate_gather_traffic(self, duration: float = 5.0):
        """生成聚合流量模式"""
        print(f"\n生成聚合流量模式 (持续 {duration}s)")
        print("场景: 所有计算芯片向主控芯片聚合梯度")
        print("-" * 40)

        pattern = TrafficPattern.GATHER
        start_time = time.time()

        while time.time() - start_time < duration:
            # 主控芯片准备接收聚合数据
            try:
                recv_result = self.chips["master"].cdma_receive(dst_addr=0x30000000, dst_shape=(256, 128), dst_mem_type=MemoryType.GMEM, src_chip_id="compute_0", data_type="float32")  # 主要接收者

                # 所有计算芯片发送梯度到主控芯片
                gather_sources = ["compute_0", "compute_1", "compute_2", "compute_3"]

                for i, source in enumerate(gather_sources):
                    send_result = self.chips[source].cdma_send(
                        src_addr=0x40000000 + i * 0x100000,
                        src_shape=(256, 128),
                        dst_chip_id="master",
                        src_mem_type=MemoryType.GMEM,
                        data_type="float32",
                        reduce_op="sum" if i > 0 else "none",  # 第一个不需要reduce
                    )

                    success = send_result.success
                    if success:
                        self.metrics[pattern].update(send_result.bytes_transferred, send_result.throughput_mbps, send_result.latency_ms, True)
                        print(f"  ✓ {source} -> master: {send_result.throughput_mbps:.1f} MB/s")
                    else:
                        self.metrics[pattern].update(0, 0, 0, False)
                        print(f"  ✗ {source} -> master: 失败")

            except Exception as e:
                self.metrics[pattern].update(0, 0, 0, False)
                print(f"  ✗ 聚合操作异常: {e}")

            time.sleep(0.2)

    def generate_scatter_traffic(self, duration: float = 5.0):
        """生成分散流量模式"""
        print(f"\n生成分散流量模式 (持续 {duration}s)")
        print("场景: 存储芯片向计算芯片分散数据批次")
        print("-" * 40)

        pattern = TrafficPattern.SCATTER
        start_time = time.time()

        while time.time() - start_time < duration:
            # 存储芯片向计算芯片分散不同的数据批次
            compute_chips = ["compute_0", "compute_1", "compute_2", "compute_3"]
            storage_chips = ["storage_0", "storage_1"]

            for storage in storage_chips:
                for i, compute in enumerate(compute_chips):
                    try:
                        # 计算芯片准备接收数据批次
                        recv_result = self.chips[compute].cdma_receive(
                            dst_addr=0x50000000 + i * 0x200000, dst_shape=(128, 64, 4), dst_mem_type=MemoryType.L2M, src_chip_id=storage, data_type="float32"  # 3D数据批次  # 使用L2缓存
                        )

                        # 存储芯片发送不同的数据批次
                        send_result = self.chips[storage].cdma_send(src_addr=0x60000000 + i * 0x100000, src_shape=(128, 64, 4), dst_chip_id=compute, src_mem_type=MemoryType.GMEM, data_type="float32")

                        success = recv_result.success and send_result.success
                        if success:
                            self.metrics[pattern].update(send_result.bytes_transferred, send_result.throughput_mbps, send_result.latency_ms, True)
                            print(f"  ✓ {storage} -> {compute}: {send_result.throughput_mbps:.1f} MB/s")
                        else:
                            self.metrics[pattern].update(0, 0, 0, False)

                    except Exception as e:
                        self.metrics[pattern].update(0, 0, 0, False)

            time.sleep(0.15)

    def generate_all_to_all_traffic(self, duration: float = 3.0):
        """生成全互连流量模式"""
        print(f"\n生成全互连流量模式 (持续 {duration}s)")
        print("场景: 计算芯片之间交换中间结果")
        print("-" * 40)

        pattern = TrafficPattern.ALL_TO_ALL
        start_time = time.time()

        while time.time() - start_time < duration:
            compute_chips = ["compute_0", "compute_1", "compute_2", "compute_3"]

            # 每个计算芯片向其他所有计算芯片发送数据
            for src in compute_chips:
                for dst in compute_chips:
                    if src != dst:
                        try:
                            # 准备接收
                            recv_result = self.chips[dst].cdma_receive(
                                dst_addr=0x70000000 + hash(src + dst) % 0x1000000, dst_shape=(64, 32), dst_mem_type=MemoryType.LMEM, src_chip_id=src, data_type="int32"  # 本地内存
                            )

                            # 发送数据
                            send_result = self.chips[src].cdma_send(
                                src_addr=0x80000000 + hash(dst + src) % 0x1000000, src_shape=(64, 32), dst_chip_id=dst, src_mem_type=MemoryType.LMEM, data_type="int32"
                            )

                            success = recv_result.success and send_result.success
                            if success:
                                self.metrics[pattern].update(send_result.bytes_transferred, send_result.throughput_mbps, send_result.latency_ms, True)

                        except Exception as e:
                            self.metrics[pattern].update(0, 0, 0, False)

            time.sleep(0.1)
            print(f"  ✓ 完成一轮全互连通信")

    def generate_pipeline_traffic(self, duration: float = 4.0):
        """生成流水线流量模式"""
        print(f"\n生成流水线流量模式 (持续 {duration}s)")
        print("场景: 数据处理流水线 storage -> compute -> io")
        print("-" * 40)

        pattern = TrafficPattern.PIPELINE
        start_time = time.time()

        pipeline_stages = [("storage_0", "compute_0", "数据加载"), ("compute_0", "compute_1", "特征提取"), ("compute_1", "compute_2", "模型推理"), ("compute_2", "io_chip", "结果输出")]

        stage_index = 0

        while time.time() - start_time < duration:
            src, dst, stage_name = pipeline_stages[stage_index % len(pipeline_stages)]

            try:
                # 准备接收
                recv_result = self.chips[dst].cdma_receive(dst_addr=0x90000000 + stage_index * 0x100000, dst_shape=(256, 128), dst_mem_type=MemoryType.GMEM, src_chip_id=src, data_type="float32")

                # 发送数据
                send_result = self.chips[src].cdma_send(src_addr=0xA0000000 + stage_index * 0x100000, src_shape=(256, 128), dst_chip_id=dst, src_mem_type=MemoryType.GMEM, data_type="float32")

                success = recv_result.success and send_result.success
                if success:
                    self.metrics[pattern].update(send_result.bytes_transferred, send_result.throughput_mbps, send_result.latency_ms, True)
                    print(f"  ✓ {stage_name}: {src} -> {dst} ({send_result.throughput_mbps:.1f} MB/s)")
                else:
                    self.metrics[pattern].update(0, 0, 0, False)
                    print(f"  ✗ {stage_name}: {src} -> {dst} 失败")

            except Exception as e:
                self.metrics[pattern].update(0, 0, 0, False)
                print(f"  ✗ {stage_name}异常: {e}")

            stage_index += 1
            time.sleep(0.2)

    def generate_ring_traffic(self, duration: float = 3.0):
        """生成环形流量模式"""
        print(f"\n生成环形流量模式 (持续 {duration}s)")
        print("场景: 计算芯片环形AllReduce通信")
        print("-" * 40)

        pattern = TrafficPattern.RING
        start_time = time.time()

        # 定义环形拓扑
        ring_order = ["compute_0", "compute_1", "compute_2", "compute_3"]

        while time.time() - start_time < duration:
            # 环形传递数据
            for i in range(len(ring_order)):
                src = ring_order[i]
                dst = ring_order[(i + 1) % len(ring_order)]

                try:
                    # 准备接收
                    recv_result = self.chips[dst].cdma_receive(dst_addr=0xB0000000 + i * 0x50000, dst_shape=(128, 64), dst_mem_type=MemoryType.GMEM, src_chip_id=src, data_type="float32")

                    # 发送数据（环形传递）
                    send_result = self.chips[src].cdma_send(
                        src_addr=0xC0000000 + i * 0x50000, src_shape=(128, 64), dst_chip_id=dst, src_mem_type=MemoryType.GMEM, data_type="float32", reduce_op="sum"  # 累积求和
                    )

                    success = recv_result.success and send_result.success
                    if success:
                        self.metrics[pattern].update(send_result.bytes_transferred, send_result.throughput_mbps, send_result.latency_ms, True)

                except Exception as e:
                    self.metrics[pattern].update(0, 0, 0, False)

            print(f"  ✓ 完成一轮环形通信")
            time.sleep(0.3)

    def run_traffic_patterns(self):
        """运行所有流量模式"""
        print("\n" + "=" * 60)
        print("C2C流量模式演示")
        print("=" * 60)

        # 按顺序运行各种流量模式
        patterns = [
            ("广播模式", self.generate_broadcast_traffic, 3.0),
            ("聚合模式", self.generate_gather_traffic, 3.0),
            ("分散模式", self.generate_scatter_traffic, 3.0),
            ("全互连模式", self.generate_all_to_all_traffic, 2.0),
            ("流水线模式", self.generate_pipeline_traffic, 3.0),
            ("环形模式", self.generate_ring_traffic, 2.0),
        ]

        for name, func, duration in patterns:
            print(f"\n{'='*20} {name} {'='*20}")
            try:
                func(duration)
                print(f"✓ {name}完成")
            except Exception as e:
                print(f"✗ {name}失败: {e}")

            time.sleep(0.5)  # 模式间间隔

    def generate_performance_report(self):
        """生成性能报告"""
        print("\n" + "=" * 60)
        print("C2C流量模式性能报告")
        print("=" * 60)

        # 创建性能对比表
        print(f"{'流量模式':<12} {'总操作数':<8} {'总字节数':<12} {'平均吞吐量':<12} {'平均延迟':<10} {'成功率':<8}")
        print("-" * 70)

        for pattern in TrafficPattern:
            metrics = self.metrics[pattern]
            if metrics.total_operations > 0:
                print(
                    f"{pattern.value:<12} {metrics.total_operations:<8} "
                    f"{metrics.total_bytes:>10,}B {metrics.avg_throughput:>10.1f}MB/s "
                    f"{metrics.avg_latency:>8.2f}ms {metrics.success_rate:>7.1%}"
                )
            else:
                print(f"{pattern.value:<12} {'无数据':<8}")

        # 详细分析
        print(f"\n详细性能分析:")
        print("-" * 40)

        # 找出最佳和最差性能
        active_patterns = [(p, m) for p, m in self.metrics.items() if m.total_operations > 0]

        if active_patterns:
            best_throughput = max(active_patterns, key=lambda x: x[1].avg_throughput)
            best_latency = min(active_patterns, key=lambda x: x[1].avg_latency)
            highest_traffic = max(active_patterns, key=lambda x: x[1].total_bytes)

            print(f"🏆 最高吞吐量: {best_throughput[0].value} ({best_throughput[1].avg_throughput:.1f} MB/s)")
            print(f"⚡ 最低延迟: {best_latency[0].value} ({best_latency[1].avg_latency:.2f} ms)")
            print(f"📊 最高流量: {highest_traffic[0].value} ({highest_traffic[1].total_bytes:,} bytes)")

            # 计算总体统计
            total_ops = sum(m.total_operations for _, m in active_patterns)
            total_bytes = sum(m.total_bytes for _, m in active_patterns)
            avg_success_rate = sum(m.success_rate for _, m in active_patterns) / len(active_patterns)

            print(f"\n📈 总体统计:")
            print(f"   总操作数: {total_ops:,}")
            print(f"   总流量: {total_bytes:,} bytes ({total_bytes/(1024*1024):.1f} MB)")
            print(f"   平均成功率: {avg_success_rate:.1%}")

        # 流量模式特征分析
        print(f"\n🔍 流量模式特征分析:")
        print("-" * 40)

        pattern_analysis = {
            TrafficPattern.BROADCAST: "一对多通信，适合参数分发",
            TrafficPattern.GATHER: "多对一通信，适合梯度聚合",
            TrafficPattern.SCATTER: "一对多数据分发，适合批量处理",
            TrafficPattern.ALL_TO_ALL: "全互连通信，网络负载最高",
            TrafficPattern.PIPELINE: "流水线处理，延迟和吞吐量平衡",
            TrafficPattern.RING: "环形通信，适合大规模AllReduce",
        }

        for pattern, description in pattern_analysis.items():
            metrics = self.metrics[pattern]
            if metrics.total_operations > 0:
                print(f"   {pattern.value}: {description}")
                print(f"     - 吞吐量: {metrics.avg_throughput:.1f} MB/s")
                print(f"     - 延迟: {metrics.avg_latency:.2f} ms")
                print(f"     - 成功率: {metrics.success_rate:.1%}")


def main():
    """主函数"""
    print("C2C芯片间通信流量模式演示")
    print("展示不同应用场景下的通信模式和性能特征")

    # 创建8芯片系统
    chip_ids = ["master", "compute_0", "compute_1", "compute_2", "compute_3", "storage_0", "storage_1", "io_chip"]

    chips = {}

    try:
        # 初始化芯片
        print("\n初始化芯片系统...")
        for chip_id in chip_ids:
            chips[chip_id] = CDMASystem(chip_id)

        # 创建流量生成器
        traffic_gen = TrafficGenerator(chips)

        # 设置拓扑
        chip_types = traffic_gen.setup_chip_topology()

        # 运行流量模式演示
        traffic_gen.run_traffic_patterns()

        # 生成性能报告
        traffic_gen.generate_performance_report()

        print(f"\n🎉 流量模式演示完成!")
        print(f"演示了6种不同的C2C通信流量模式，展示了各种应用场景下的通信特征。")

    except KeyboardInterrupt:
        print(f"\n⏹️  用户中断演示")

    except Exception as e:
        print(f"\n❌ 演示过程中发生错误: {e}")

    finally:
        # 清理资源
        print(f"\n清理系统资源...")
        for chip_id, chip in chips.items():
            try:
                chip.shutdown()
                print(f"✓ {chip_id} 已关闭")
            except Exception as e:
                print(f"✗ {chip_id} 关闭失败: {e}")

        print("资源清理完成")


if __name__ == "__main__":
    main()
