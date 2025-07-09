#!/usr/bin/env python3
"""
CrossRing Debug Demo
===================

专门用于详细的请求追踪和调试的演示程序。
可以追踪特定请求的完整生命周期，包括：
- Flit在网络中的位置
- 每个周期的状态变化
- 路由决策过程
- Tag机制的工作过程

Usage:
    python crossring_debug_demo.py [packet_id]
"""

import sys
import logging
from pathlib import Path
from typing import Optional

# 添加src路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.noc.crossring.model import CrossRingModel
from src.noc.crossring.config import CrossRingConfig
from src.noc.debug import RequestTracker, RequestState, FlitType


def setup_debug_logging():
    """设置详细的调试日志"""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("crossring_debug.log", mode="w")
        ]
    )
    return logging.getLogger(__name__)


def create_debug_config(rows=3, cols=3):
    """创建调试用的3x3 CrossRing配置"""
    config = CrossRingConfig(num_row=rows, num_col=cols, config_name="debug_3x3")
    
    # 配置IP接口：确保节点0有GDMA，节点4有DDR
    config.gdma_send_position_list = [0, 1, 2]  # 前三个节点有GDMA
    config.ddr_send_position_list = [3, 4, 5]   # 后三个节点有DDR
    config.l2m_send_position_list = [6, 7, 8]   # 最后三个节点有L2M
    
    # 调试配置
    config.debug_enabled = True
    config.verbose_mode = True
    
    return config


def get_debug_traffic_file():
    """获取专门的调试traffic文件"""
    traffic_file = Path(__file__).parent.parent.parent / "traffic_data" / "debug_3x3_traffic.txt"
    
    if not traffic_file.exists():
        # 如果文件不存在，创建一个临时文件
        traffic_content = """# Debug traffic: Node 0 (GDMA) -> Node 4 (DDR)
# Format: cycle,src_node,src_ip,dst_node,dst_ip,request_type,request_size
0,0,gdma_0,4,ddr_4,R,4
20,0,gdma_0,4,ddr_4,W,4
40,0,gdma_0,4,ddr_4,R,8
"""
        
        temp_file = Path("temp_debug_traffic.txt")
        with open(temp_file, "w") as f:
            f.write(traffic_content)
        return temp_file
    
    return traffic_file


def print_network_topology(rows, cols):
    """打印网络拓扑结构"""
    print("\n📊 网络拓扑结构:")
    print("=" * 40)
    
    for row in range(rows):
        row_str = ""
        for col in range(cols):
            node_id = row * cols + col
            row_str += f"[{node_id:2d}]"
            if col < cols - 1:
                row_str += " -- "
        print(row_str)
        
        # 打印垂直连接
        if row < rows - 1:
            col_str = ""
            for col in range(cols):
                col_str += " |  "
                if col < cols - 1:
                    col_str += "    "
            print(col_str)
    
    print("=" * 40)
    print("✅ 节点0 (GDMA) -> 节点4 (DDR) 的路径:")
    print("   HV路径: 0 -> 1 -> 4")
    print("   VH路径: 0 -> 3 -> 4")
    print()


def run_debug_simulation(target_packet_id: Optional[str] = None):
    """运行调试仿真"""
    logger = setup_debug_logging()
    
    print("🔍 CrossRing Debug Demo")
    print("=" * 50)
    print("追踪请求的完整生命周期...")
    print()
    
    # 创建配置
    config = create_debug_config()
    
    # 显示拓扑结构
    print_network_topology(3, 3)
    
    # 创建调试traffic文件
    traffic_file = get_debug_traffic_file()
    
    try:
        # 创建模型
        model = CrossRingModel(config, traffic_file_path=str(traffic_file))
        
        # 启用详细调试
        model.debug_enabled = True
        model.request_tracker.enable_debug(level=2)
        
        # 如果指定了packet_id，只追踪特定请求
        if target_packet_id:
            model.request_tracker.track_packet(target_packet_id)
            print(f"🎯 追踪目标: {target_packet_id}")
        else:
            # 否则追踪所有请求
            print("🎯 追踪目标: 所有请求")
            
        print()
        
        # 注入流量
        injected = model.inject_from_traffic_file(
            traffic_file_path=str(traffic_file),
            cycle_accurate=True,
            immediate_inject=False
        )
        
        print(f"✅ 注入了 {injected} 个请求")
        
        # 详细检查IP接口状态
        print(f"\n🔍 IP接口详细状态:")
        for ip_key, ip_interface in model.ip_interfaces.items():
            print(f"  {ip_key}: 节点{ip_interface.node_id}")
            if hasattr(ip_interface, 'active_requests'):
                print(f"    活跃请求: {len(ip_interface.active_requests)}")
            if hasattr(ip_interface, 'pending_requests'):
                print(f"    等待请求: {len(ip_interface.pending_requests)}")
                if ip_interface.pending_requests:
                    for req in ip_interface.pending_requests:
                        print(f"      - {req}")
        
        # 检查请求追踪器状态  
        print(f"\n🔍 请求追踪器状态:")
        print(f"  活跃请求: {len(model.request_tracker.active_requests)}")
        print(f"  完成请求: {len(model.request_tracker.completed_requests)}")
        print(f"  失败请求: {len(getattr(model.request_tracker, 'failed_requests', []))}")
        
        if model.request_tracker.active_requests:
            print("  活跃请求详情:")
            for packet_id, lifecycle in model.request_tracker.active_requests.items():
                print(f"    - {packet_id}: {lifecycle.current_state.value}")
        
        print()
        
        # 运行仿真，逐周期显示详细信息
        print("\n🔄 开始仿真...")
        print("=" * 50)
        
        max_cycles = 200
        for cycle in range(max_cycles):
            # 执行一个周期
            model.step()
            
            # 每个周期都检查详细状态
            active_count = len(model.request_tracker.active_requests)
            completed_count = len(model.request_tracker.completed_requests)
            
            # 检查IP接口状态
            ip_status = check_ip_interface_status(model)
            node_status = check_node_queue_status(model)
            
            # 如果有活动或者前几个周期，打印详细信息
            if active_count > 0 or completed_count > 0 or cycle < 20:
                print(f"\n周期 {cycle:3d}: 活跃请求={active_count}, 完成请求={completed_count}")
                
                # 显示IP接口队列状态
                if any(ip_status.values()):
                    print(f"         IP接口状态: {ip_status}")
                
                # 显示节点队列状态
                if any(any(queues.values()) for queues in node_status.values()):
                    print(f"         节点队列状态:")
                    for node_id, queues in node_status.items():
                        if any(queues.values()):
                            print(f"           节点{node_id}: {queues}")
                
                # 如果有追踪的特定请求，显示其状态
                if target_packet_id:
                    lifecycle = model.request_tracker.get_request_status(target_packet_id)
                    if lifecycle:
                        print(f"           目标请求 {target_packet_id}: {lifecycle.current_state.value}")
                        if lifecycle.request_path:
                            latest_pos = lifecycle.request_path[-1]
                            print(f"           最新位置: 节点{latest_pos[0]} (周期{latest_pos[1]})")
            
            # 打印网络状态
            if cycle < 20 or active_count > 0:
                model.request_tracker.print_network_state(cycle)
            
            # 检查是否所有请求都完成
            if len(model.request_tracker.active_requests) == 0 and cycle > 10:
                print(f"\n✅ 所有请求在周期 {cycle} 完成")
                break
                
        print("\n" + "=" * 50)
        print("🎯 仿真完成，生成详细报告...")
        
        # 生成详细的追踪报告
        print_detailed_trace_report(model.request_tracker, target_packet_id)
        
        return True
        
    except Exception as e:
        logger.error(f"Debug仿真失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if 'model' in locals():
            model.cleanup()
        # 清理临时文件
        if traffic_file.exists():
            traffic_file.unlink()


def print_detailed_trace_report(tracker: RequestTracker, target_packet_id: Optional[str] = None):
    """打印详细的追踪报告"""
    print("\n📋 详细追踪报告")
    print("=" * 60)
    
    # 统计信息
    stats = tracker.get_statistics()
    print(f"总请求数: {stats['total_requests']}")
    print(f"完成请求: {stats['completed_requests']}")
    print(f"失败请求: {stats['failed_requests']}")
    print(f"平均延迟: {stats['avg_latency']:.2f} 周期")
    print(f"最大延迟: {stats['max_latency']} 周期")
    print(f"最小延迟: {stats['min_latency']} 周期")
    
    # 如果指定了目标请求，显示详细信息
    if target_packet_id:
        print(f"\n🎯 目标请求 {target_packet_id} 详细信息:")
        print("-" * 40)
        
        lifecycle = tracker.get_request_status(target_packet_id)
        if lifecycle:
            print_request_lifecycle(lifecycle)
        else:
            print("未找到目标请求")
    else:
        # 显示所有完成请求的摘要
        print("\n📊 所有完成请求摘要:")
        print("-" * 40)
        
        for packet_id, lifecycle in tracker.completed_requests.items():
            print(f"请求 {packet_id}: {lifecycle.source} -> {lifecycle.destination}")
            print(f"  延迟: {lifecycle.get_total_latency()} 周期")
            print(f"  状态: {lifecycle.current_state.value}")
    
    print("\n" + "=" * 60)


def check_ip_interface_status(model):
    """检查IP接口状态"""
    status = {}
    for ip_key, ip_interface in model.ip_interfaces.items():
        ip_status = {}
        
        # 检查活跃请求
        if hasattr(ip_interface, 'active_requests'):
            active = len(ip_interface.active_requests)
            if active > 0:
                ip_status["活跃"] = active
                # 检查每个请求的阶段
                stages = {}
                for req_id, req_info in ip_interface.active_requests.items():
                    stage = req_info.get("stage", "unknown")
                    stages[stage] = stages.get(stage, 0) + 1
                if stages:
                    ip_status["阶段"] = stages
        
        # 检查L2H FIFO状态
        if hasattr(ip_interface, 'l2h_fifos'):
            l2h_status = {}
            for channel, fifo in ip_interface.l2h_fifos.items():
                if len(fifo) > 0:
                    l2h_status[f"L2H_{channel}"] = len(fifo)
            if l2h_status:
                ip_status["L2H"] = l2h_status
        
        # 检查inject_fifos状态  
        if hasattr(ip_interface, 'inject_fifos'):
            inject_status = {}
            for channel, fifo in ip_interface.inject_fifos.items():
                if len(fifo) > 0:
                    inject_status[f"inject_{channel}"] = len(fifo)
            if inject_status:
                ip_status["inject"] = inject_status
                
        # 检查其他队列
        if hasattr(ip_interface, 'completed_requests'):
            completed = len(ip_interface.completed_requests)
            if completed > 0:
                ip_status["完成"] = completed
        
        if ip_status:
            status[ip_key] = ip_status
    
    return status

def check_node_queue_status(model):
    """检查节点队列状态"""
    status = {}
    for node_id, node in model.crossring_nodes.items():
        node_queues = {}
        
        # 检查inject队列
        if hasattr(node, 'inject_queues'):
            for channel, queue in node.inject_queues.items():
                if len(queue) > 0:
                    node_queues[f"inject_{channel}"] = len(queue)
        
        # 检查eject队列  
        if hasattr(node, 'eject_queues'):
            for channel, queue in node.eject_queues.items():
                if len(queue) > 0:
                    node_queues[f"eject_{channel}"] = len(queue)
                    
        # 检查ring队列
        if hasattr(node, 'ring_queues'):
            for direction, queue in node.ring_queues.items():
                if len(queue) > 0:
                    node_queues[f"ring_{direction}"] = len(queue)
        
        if node_queues:
            status[node_id] = node_queues
    
    return status

def print_request_lifecycle(lifecycle):
    """打印请求生命周期详情"""
    print(f"请求ID: {lifecycle.packet_id}")
    print(f"源节点: {lifecycle.source} -> 目标节点: {lifecycle.destination}")
    print(f"操作类型: {lifecycle.op_type}")
    print(f"突发长度: {lifecycle.burst_size}")
    print()
    
    print("⏰ 时间线:")
    print(f"  创建: 周期 {lifecycle.created_cycle}")
    print(f"  注入: 周期 {lifecycle.injected_cycle}")
    print(f"  到达: 周期 {lifecycle.arrived_cycle}")
    print(f"  响应: 周期 {lifecycle.response_sent_cycle}")
    print(f"  完成: 周期 {lifecycle.completed_cycle}")
    print()
    
    print("📏 延迟分析:")
    print(f"  总延迟: {lifecycle.get_total_latency()} 周期")
    print(f"  请求延迟: {lifecycle.get_request_latency()} 周期")
    print(f"  数据延迟: {lifecycle.get_data_latency()} 周期")
    print()
    
    print("🛤️  请求路径:")
    if lifecycle.request_path:
        for i, (node_id, cycle) in enumerate(lifecycle.request_path):
            print(f"  步骤 {i+1}: 节点{node_id} (周期{cycle})")
    else:
        print("  无路径记录")
    print()
    
    print("✅ 验证结果:")
    print(f"  响应有效: {lifecycle.response_valid}")
    print(f"  数据有效: {lifecycle.data_valid}")
    print(f"  数据完整性: {lifecycle.data_integrity_ok}")


def main():
    """主函数"""
    # 解析命令行参数
    target_packet_id = sys.argv[1] if len(sys.argv) > 1 else None
    
    print("🔍 CrossRing Debug Demo")
    print("=" * 50)
    print("专门用于详细请求追踪和调试")
    print(f"📝 用法: python {Path(__file__).name} [packet_id]")
    
    if target_packet_id:
        print(f"🎯 将追踪特定请求: {target_packet_id}")
    else:
        print("🎯 将追踪所有请求")
    
    print("=" * 50)
    
    success = run_debug_simulation(target_packet_id)
    
    if success:
        print("\n✅ Debug演示完成！")
        print("\n📋 功能特点:")
        print("- ✅ 3x3 CrossRing拓扑")
        print("- ✅ 节点0 (GDMA) -> 节点4 (DDR)")
        print("- ✅ 完整的请求生命周期追踪")
        print("- ✅ 详细的路径分析")
        print("- ✅ 周期级别的状态监控")
        print("- ✅ 自动验证和报告")
        return 0
    else:
        print("\n❌ Debug演示失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())