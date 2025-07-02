#!/usr/bin/env python3
"""
C2C CDMA协议使用示例
演示如何使用CDMA协议进行芯片间通信的基本操作
"""

import sys
import os
import time
import threading
from typing import List, Dict, Any

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.protocol.cdma_system import CDMASystem, CDMASystemState
from src.protocol.memory_types import MemoryType
from src.utils.exceptions import CDMAError


class CDMAUsageExample:
    """CDMA协议使用示例类"""
    
    def __init__(self):
        self.chips = {}
        self.setup_multi_chip_system()
    
    def setup_multi_chip_system(self):
        """设置多芯片系统"""
        print("=" * 60)
        print("设置多芯片C2C通信系统")
        print("=" * 60)
        
        # 创建4个芯片
        chip_names = ["主控芯片", "计算芯片A", "计算芯片B", "存储芯片"]
        chip_ids = ["master", "compute_a", "compute_b", "storage"]
        
        for name, chip_id in zip(chip_names, chip_ids):
            self.chips[chip_id] = CDMASystem(chip_id)
            print(f"✓ 创建{name} (ID: {chip_id})")
        
        # 建立连接拓扑 - 星型结构，主控芯片连接所有其他芯片
        master = self.chips["master"]
        
        for chip_id, chip in self.chips.items():
            if chip_id != "master":
                master.connect_to_chip(chip_id, chip)
                chip.connect_to_chip("master", master)
                print(f"✓ {chip_id} 与 master 建立连接")
        
        # 计算芯片之间也建立连接
        self.chips["compute_a"].connect_to_chip("compute_b", self.chips["compute_b"])
        self.chips["compute_b"].connect_to_chip("compute_a", self.chips["compute_a"])
        print("✓ compute_a 与 compute_b 建立连接")
        
        print(f"\n多芯片系统设置完成，共{len(self.chips)}个芯片")
        
    def example_1_basic_data_transfer(self):
        """示例1: 基础数据传输"""
        print("\n" + "=" * 60)
        print("示例1: 基础数据传输")
        print("=" * 60)
        print("场景: 主控芯片向计算芯片A发送模型参数")
        
        master = self.chips["master"]
        compute_a = self.chips["compute_a"]
        
        # 1. 计算芯片A准备接收数据
        print("\n步骤1: 计算芯片A准备接收模型参数")
        recv_result = compute_a.cdma_receive(
            dst_addr=0x10000000,  # 目标地址
            dst_shape=(1024, 512),  # 模型参数形状
            dst_mem_type=MemoryType.GMEM,  # 全局内存
            src_chip_id="master",
            data_type="float32"
        )
        
        if recv_result.success:
            print(f"✓ 接收准备成功，事务ID: {recv_result.transaction_id}")
            print(f"  目标地址: 0x{recv_result.transaction_id.split('_')[-1]}")
            print(f"  数据形状: (1024, 512)")
            print(f"  数据类型: float32")
        
        # 2. 主控芯片发送数据
        print("\n步骤2: 主控芯片发送模型参数")
        send_result = master.cdma_send(
            src_addr=0x20000000,  # 源地址
            src_shape=(1024, 512),  # 数据形状
            dst_chip_id="compute_a",
            src_mem_type=MemoryType.GMEM,
            data_type="float32"
        )
        
        if send_result.success:
            print(f"✓ 数据发送成功")
            print(f"  传输字节数: {send_result.bytes_transferred:,}")
            print(f"  传输速度: {send_result.throughput_mbps:.2f} MB/s")
            print(f"  传输延迟: {send_result.latency_ms:.2f} ms")
        
        return recv_result.success and send_result.success
    
    def example_2_tensor_operations(self):
        """示例2: 张量操作和All-Reduce"""
        print("\n" + "=" * 60)
        print("示例2: 张量操作和All-Reduce通信")
        print("=" * 60)
        print("场景: 分布式训练中的梯度聚合")
        
        master = self.chips["master"]
        compute_a = self.chips["compute_a"]
        compute_b = self.chips["compute_b"]
        
        # 1. 主控芯片准备接收聚合后的梯度
        print("\n步骤1: 主控芯片准备接收聚合梯度")
        recv_result = master.cdma_receive(
            dst_addr=0x30000000,
            dst_shape=(512, 256),  # 梯度张量形状
            dst_mem_type=MemoryType.GMEM,
            src_chip_id="compute_a",
            data_type="float32"
        )
        
        # 2. 计算芯片A发送梯度（使用sum reduce）
        print("\n步骤2: 计算芯片A发送梯度（求和聚合）")
        send_result_a = compute_a.cdma_send(
            src_addr=0x40000000,
            src_shape=(512, 256),
            dst_chip_id="master",
            src_mem_type=MemoryType.GMEM,
            data_type="float32",
            reduce_op="sum"  # 求和聚合
        )
        
        # 3. 计算芯片B也发送梯度
        print("\n步骤3: 计算芯片B发送梯度（求和聚合）")
        send_result_b = compute_b.cdma_send(
            src_addr=0x50000000,
            src_shape=(512, 256),
            dst_chip_id="master",
            src_mem_type=MemoryType.GMEM,
            data_type="float32",
            reduce_op="sum"
        )
        
        if all([recv_result.success, send_result_a.success, send_result_b.success]):
            print(f"✓ All-Reduce操作成功完成")
            print(f"  梯度聚合字节数: {send_result_a.bytes_transferred + send_result_b.bytes_transferred:,}")
            print(f"  平均传输速度: {(send_result_a.throughput_mbps + send_result_b.throughput_mbps)/2:.2f} MB/s")
        
        return all([recv_result.success, send_result_a.success, send_result_b.success])
    
    def example_3_memory_hierarchy(self):
        """示例3: 内存层次结构使用"""
        print("\n" + "=" * 60)
        print("示例3: 内存层次结构使用")
        print("=" * 60)
        print("场景: 多级内存数据流动")
        
        compute_a = self.chips["compute_a"]
        storage = self.chips["storage"]
        
        # 1. 从存储芯片的全局内存读取数据到计算芯片的L2缓存
        print("\n步骤1: 存储芯片GMEM -> 计算芯片L2缓存")
        
        # 计算芯片准备接收到L2缓存
        recv_l2_result = compute_a.cdma_receive(
            dst_addr=0x60000000,
            dst_shape=(256, 128),
            dst_mem_type=MemoryType.L2M,  # L2缓存
            src_chip_id="storage",
            data_type="int32"
        )
        
        # 存储芯片从全局内存发送
        send_gmem_result = storage.cdma_send(
            src_addr=0x70000000,
            src_shape=(256, 128),
            dst_chip_id="compute_a",
            src_mem_type=MemoryType.GMEM,  # 全局内存
            data_type="int32"
        )
        
        print(f"  ✓ GMEM -> L2M: {send_gmem_result.throughput_mbps:.2f} MB/s")
        
        # 2. 从L2缓存移动数据到本地内存进行计算
        print("\n步骤2: L2缓存 -> 本地内存")
        
        recv_lmem_result = compute_a.cdma_receive(
            dst_addr=0x80000000,
            dst_shape=(128, 64),  # 处理后的数据更小
            dst_mem_type=MemoryType.LMEM,  # 本地内存
            src_chip_id="compute_a",  # 芯片内部传输
            data_type="int32"
        )
        
        send_l2_result = compute_a.cdma_send(
            src_addr=0x60000000,
            src_shape=(128, 64),
            dst_chip_id="compute_a",
            src_mem_type=MemoryType.L2M,
            data_type="int32"
        )
        
        print(f"  ✓ L2M -> LMEM: {send_l2_result.throughput_mbps:.2f} MB/s")
        
        # 显示内存层次结构性能
        print(f"\n内存层次结构性能对比:")
        print(f"  全局内存传输: {send_gmem_result.throughput_mbps:.2f} MB/s")
        print(f"  L2缓存传输: {send_l2_result.throughput_mbps:.2f} MB/s")
        print(f"  延迟对比: GMEM={send_gmem_result.latency_ms:.2f}ms, L2M={send_l2_result.latency_ms:.2f}ms")
        
        return all([recv_l2_result.success, send_gmem_result.success, 
                   recv_lmem_result.success, send_l2_result.success])
    
    def example_4_concurrent_operations(self):
        """示例4: 并发操作"""
        print("\n" + "=" * 60)
        print("示例4: 并发数据传输")
        print("=" * 60)
        print("场景: 多个芯片同时进行数据交换")
        
        results = []
        threads = []
        
        def transfer_task(src_chip, dst_chip, src_addr, dst_addr, shape, task_name):
            """数据传输任务"""
            try:
                # 目标芯片准备接收
                recv_result = self.chips[dst_chip].cdma_receive(
                    dst_addr=dst_addr,
                    dst_shape=shape,
                    dst_mem_type=MemoryType.GMEM,
                    src_chip_id=src_chip,
                    data_type="float32"
                )
                
                # 源芯片发送数据
                send_result = self.chips[src_chip].cdma_send(
                    src_addr=src_addr,
                    src_shape=shape,
                    dst_chip_id=dst_chip,
                    src_mem_type=MemoryType.GMEM,
                    data_type="float32"
                )
                
                success = recv_result.success and send_result.success
                results.append({
                    'task': task_name,
                    'success': success,
                    'bytes': send_result.bytes_transferred if success else 0,
                    'throughput': send_result.throughput_mbps if success else 0,
                    'latency': send_result.latency_ms if success else 0
                })
                
                print(f"  ✓ {task_name}完成: {send_result.throughput_mbps:.2f} MB/s")
                
            except Exception as e:
                print(f"  ✗ {task_name}失败: {e}")
                results.append({'task': task_name, 'success': False})
        
        # 定义并发传输任务
        tasks = [
            ("master", "compute_a", 0x90000000, 0x91000000, (128, 128), "主控->计算A"),
            ("master", "compute_b", 0x92000000, 0x93000000, (128, 128), "主控->计算B"),
            ("storage", "compute_a", 0x94000000, 0x95000000, (64, 256), "存储->计算A"),
            ("compute_a", "compute_b", 0x96000000, 0x97000000, (256, 64), "计算A->计算B")
        ]
        
        print(f"\n启动{len(tasks)}个并发传输任务:")
        
        # 创建并启动线程
        for task in tasks:
            thread = threading.Thread(target=transfer_task, args=task)
            threads.append(thread)
            thread.start()
        
        # 等待所有任务完成
        for thread in threads:
            thread.join(timeout=10.0)
        
        # 统计结果
        successful_tasks = [r for r in results if r['success']]
        total_bytes = sum(r['bytes'] for r in successful_tasks)
        avg_throughput = sum(r['throughput'] for r in successful_tasks) / len(successful_tasks) if successful_tasks else 0
        avg_latency = sum(r['latency'] for r in successful_tasks) / len(successful_tasks) if successful_tasks else 0
        
        print(f"\n并发传输统计:")
        print(f"  成功任务: {len(successful_tasks)}/{len(tasks)}")
        print(f"  总传输字节: {total_bytes:,}")
        print(f"  平均吞吐量: {avg_throughput:.2f} MB/s")
        print(f"  平均延迟: {avg_latency:.2f} ms")
        
        return len(successful_tasks) == len(tasks)
    
    def show_system_status(self):
        """显示系统状态"""
        print("\n" + "=" * 60)
        print("系统状态报告")
        print("=" * 60)
        
        for chip_id, chip in self.chips.items():
            status = chip.get_comprehensive_status()
            perf_report = chip.generate_performance_report()
            
            print(f"\n芯片: {chip_id}")
            print(f"  状态: {status['system_state']}")
            print(f"  连接数: {len(status['connected_chips'])}")
            print(f"  连接到: {', '.join(status['connected_chips'])}")
            
            # 性能统计
            throughput_stats = perf_report.get('throughput_stats', {})
            latency_stats = perf_report.get('latency_stats', {})
            
            if throughput_stats.get('total_events', 0) > 0:
                print(f"  吞吐量: 平均{throughput_stats.get('average', 0):.2f} MB/s")
                print(f"  延迟: 平均{latency_stats.get('average', 0):.2f} ms")
            
            # 错误统计
            error_stats = status.get('error_stats', {})
            if error_stats:
                print(f"  错误统计: {error_stats}")
    
    def cleanup(self):
        """清理资源"""
        print("\n" + "=" * 60)
        print("清理系统资源")
        print("=" * 60)
        
        for chip_id, chip in self.chips.items():
            try:
                chip.shutdown()
                print(f"✓ {chip_id} 已关闭")
            except Exception as e:
                print(f"✗ {chip_id} 关闭失败: {e}")
        
        print("系统清理完成")


def main():
    """主函数"""
    print("C2C CDMA协议使用示例")
    print("展示芯片间通信的各种使用场景")
    
    example = CDMAUsageExample()
    
    try:
        # 运行各种示例
        examples = [
            ("基础数据传输", example.example_1_basic_data_transfer),
            ("张量操作和All-Reduce", example.example_2_tensor_operations),
            ("内存层次结构", example.example_3_memory_hierarchy),
            ("并发操作", example.example_4_concurrent_operations)
        ]
        
        results = []
        for name, func in examples:
            print(f"\n{'='*20} 开始执行: {name} {'='*20}")
            try:
                success = func()
                results.append((name, success))
                status = "成功" if success else "失败"
                print(f"{'='*20} {name}: {status} {'='*20}")
            except Exception as e:
                print(f"{'='*20} {name}: 异常 - {e} {'='*20}")
                results.append((name, False))
        
        # 显示系统状态
        example.show_system_status()
        
        # 总结
        print("\n" + "=" * 60)
        print("示例执行总结")
        print("=" * 60)
        
        successful = sum(1 for _, success in results if success)
        total = len(results)
        
        for name, success in results:
            status = "✓" if success else "✗"
            print(f"  {status} {name}")
        
        print(f"\n总体结果: {successful}/{total} 个示例成功")
        
        if successful == total:
            print("🎉 所有示例都成功执行！C2C CDMA协议工作正常。")
        else:
            print(f"⚠️  有 {total - successful} 个示例失败，请检查系统配置。")
    
    finally:
        # 清理资源
        example.cleanup()


if __name__ == "__main__":
    main()