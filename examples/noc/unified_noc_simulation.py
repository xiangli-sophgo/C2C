#!/usr/bin/env python3
"""
Unified NoC Simulation Example

This example demonstrates how to run NoC simulations with either Mesh or CrossRing topology,
allowing users to choose the topology type and compare performance.
"""

import sys
import os
import time
import argparse
from typing import Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.noc.mesh.config import MeshConfig
from src.noc.mesh.model import MeshModel
from src.noc.crossring.config import CrossRingConfig
from src.noc.crossring.model import CrossRingModel
from src.noc.utils.traffic_scheduler import TrafficScheduler


class UnifiedNoCSimulation:
    """统一的NoC仿真类，支持Mesh和CrossRing拓扑"""
    
    def __init__(self, topology_type: str = "mesh", topology_size: tuple = (4, 4)):
        """
        初始化仿真。
        
        Args:
            topology_type: 拓扑类型 ("mesh" 或 "crossring")
            topology_size: 拓扑大小 (rows, cols)
        """
        self.topology_type = topology_type.lower()
        self.topology_size = topology_size
        self.config = None
        self.model = None
        self.traffic_scheduler = None
        
        self.stats = {
            'total_packets': 0,
            'total_flits': 0,
            'simulation_time': 0,
            'start_time': 0,
            'end_time': 0
        }
        
        self._setup_model()
    
    def _setup_model(self):
        """根据拓扑类型设置模型"""
        rows, cols = self.topology_size
        
        if self.topology_type == "mesh":
            self.config = MeshConfig(rows=rows, cols=cols, config_name=f"{rows}x{cols}_mesh")
            self.model = MeshModel(self.config)
            print(f"已创建 {rows}x{cols} Mesh拓扑")
        
        elif self.topology_type == "crossring":
            self.config = CrossRingConfig(num_row=rows, num_col=cols, config_name=f"{rows}x{cols}_crossring")
            self.model = CrossRingModel(self.config)
            print(f"已创建 {rows}x{cols} CrossRing拓扑")
        
        else:
            raise ValueError(f"不支持的拓扑类型: {self.topology_type}")
        
        # 初始化网络
        self.model.initialize_network()
        
        print(f"拓扑类型: {self.topology_type}")
        print(f"节点数量: {self.model.get_node_count()}")
        
        if hasattr(self.config, 'basic_config'):
            print(f"网络频率: {self.config.basic_config.network_frequency} GHz")
        elif hasattr(self.config, 'mesh_config'):
            print(f"路由器延迟: {self.config.mesh_config.router_latency} cycles")
            print(f"链路延迟: {self.config.mesh_config.link_latency} cycles")
    
    def setup_traffic(self, traffic_file_path: str, traffic_files: list):
        """设置traffic调度器"""
        self.traffic_scheduler = TrafficScheduler(self.config, traffic_file_path)
        self.traffic_scheduler.set_verbose(True)
        
        # 设置traffic链
        if isinstance(traffic_files[0], list):
            self.traffic_scheduler.setup_parallel_chains(traffic_files)
        else:
            self.traffic_scheduler.setup_single_chain(traffic_files)
        
        print(f"Traffic调度器设置完成")
        print(f"Traffic文件: {traffic_files}")
    
    def run_simulation(self, max_cycles: int = 10000):
        """运行仿真"""
        if not self.model or not self.traffic_scheduler:
            raise ValueError("模型和traffic调度器必须在运行仿真前设置")
        
        print(f"\n开始仿真 - 最大周期数: {max_cycles}")
        self.stats['start_time'] = time.time()
        
        # 启动初始traffic
        self.traffic_scheduler.start_initial_traffics()
        
        cycle = 0
        packets_injected = 0
        
        while cycle < max_cycles:
            # 获取当前周期的就绪请求
            ready_requests = self.traffic_scheduler.get_ready_requests(cycle)
            
            # 注入包到网络
            for request in ready_requests:
                req_time, src, src_type, dst, dst_type, op, burst, traffic_id = request
                
                # 注入包
                packet_injected = self.model.inject_packet(
                    src_node=src,
                    dst_node=dst,
                    op_type=op,
                    burst_size=burst,
                    cycle=cycle
                )
                
                if packet_injected:
                    packets_injected += 1
                    self.stats['total_packets'] += 1
                    self.stats['total_flits'] += burst
                    
                    # 更新traffic统计
                    self.traffic_scheduler.update_traffic_stats(traffic_id, "injected_req")
            
            # 推进网络仿真一个周期
            self.model.advance_cycle()
            
            # 检查已完成的包并更新统计
            completed_packets = self.model.get_completed_packets()
            for packet in completed_packets:
                traffic_id = packet.get('traffic_id', 'unknown')
                # 更新接收到的flit数量
                for _ in range(packet.get('flit_count', 1)):
                    self.traffic_scheduler.update_traffic_stats(traffic_id, "received_flit")
            
            # 检查并推进traffic链
            completed_traffics = self.traffic_scheduler.check_and_advance_chains(cycle)
            
            # 进度报告
            if cycle % 1000 == 0:
                active_count = self.traffic_scheduler.get_active_traffic_count()
                print(f"周期 {cycle}: 注入包数={packets_injected}, 活跃traffic={active_count}")
            
            # 检查是否所有traffic都已完成
            if self.traffic_scheduler.is_all_completed():
                print(f"所有traffic在周期 {cycle} 完成")
                break
            
            cycle += 1
        
        self.stats['end_time'] = time.time()
        self.stats['simulation_time'] = self.stats['end_time'] - self.stats['start_time']
        
        print(f"\n仿真在周期 {cycle} 完成")
        self._print_results()
    
    def _print_results(self):
        """打印仿真结果"""
        print(f"\n=== 仿真结果 ===")
        print(f"拓扑类型: {self.topology_type.upper()}")
        print(f"拓扑大小: {self.topology_size[0]}x{self.topology_size[1]}")
        print(f"总注入包数: {self.stats['total_packets']}")
        print(f"总flit数: {self.stats['total_flits']}")
        print(f"仿真时间: {self.stats['simulation_time']:.3f} 秒")
        
        # 获取网络统计
        network_stats = self.model.get_network_statistics()
        print(f"\n=== 网络性能 ===")
        print(f"网络利用率: {network_stats.get('utilization', 0):.2f}%")
        print(f"平均延迟: {network_stats.get('avg_latency', 0):.2f} 周期")
        print(f"平均跳数: {network_stats.get('avg_hops', 0):.2f}")
        print(f"吞吐量: {network_stats.get('throughput', 0):.2f} 包/周期")
        
        if self.topology_type == "mesh":
            print(f"网络直径: {network_stats.get('network_diameter', 0)}")
            print(f"平均度数: {network_stats.get('average_degree', 0):.2f}")
        
        # 获取traffic完成统计
        finish_stats = self.traffic_scheduler.get_finish_time_stats()
        print(f"\n=== Traffic完成统计 ===")
        print(f"读操作完成时间: {finish_stats.get('R_finish_time', 0)} ns")
        print(f"写操作完成时间: {finish_stats.get('W_finish_time', 0)} ns")
        print(f"总完成时间: {finish_stats.get('Total_finish_time', 0)} ns")
        
        # 链状态
        print(f"\n=== 链状态 ===")
        chain_status = self.traffic_scheduler.get_chain_status()
        for chain_id, status in chain_status.items():
            print(f"{chain_id}: {status['current_index']}/{status['total_traffics']} 完成")


def create_sample_traffic_file(filename: str, requests: int = 50):
    """创建样本traffic文件"""
    with open(filename, 'w') as f:
        for i in range(requests):
            time_ns = i * 16  # 16ns间隔
            src = i % 4
            dst = (i + 1) % 4
            op = 'R' if i % 2 == 0 else 'W'
            burst = 4
            
            f.write(f"{time_ns},{src},gdma_{src},{dst},ddr_{dst},{op},{burst}\n")
    
    print(f"创建样本traffic文件: {filename}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='统一NoC仿真示例')
    parser.add_argument('--topology', choices=['mesh', 'crossring'], default='mesh',
                       help='拓扑类型 (默认: mesh)')
    parser.add_argument('--rows', type=int, default=4, help='行数 (默认: 4)')
    parser.add_argument('--cols', type=int, default=4, help='列数 (默认: 4)')
    parser.add_argument('--cycles', type=int, default=5000, help='最大仿真周期 (默认: 5000)')
    parser.add_argument('--traffic', type=str, default='auto', 
                       help='traffic文件路径或"auto"使用默认文件')
    
    args = parser.parse_args()
    
    print("统一NoC仿真示例")
    print("=" * 50)
    
    # 创建仿真实例
    sim = UnifiedNoCSimulation(
        topology_type=args.topology,
        topology_size=(args.rows, args.cols)
    )
    
    # 设置traffic
    if args.traffic == 'auto':
        # 使用默认traffic文件或创建样本文件
        traffic_dir = "traffic_data"
        if not os.path.exists(traffic_dir):
            os.makedirs(traffic_dir)
        
        sample_file = os.path.join(traffic_dir, f"{args.topology}_sample.txt")
        if not os.path.exists(sample_file):
            create_sample_traffic_file(sample_file, 30)
        
        traffic_files = [f"{args.topology}_sample.txt"]
        traffic_path = traffic_dir
    else:
        # 使用用户指定的traffic文件
        traffic_path = os.path.dirname(args.traffic)
        traffic_files = [os.path.basename(args.traffic)]
    
    sim.setup_traffic(traffic_path, traffic_files)
    
    # 运行仿真
    sim.run_simulation(max_cycles=args.cycles)
    
    print("\n仿真完成!")


if __name__ == "__main__":
    main()