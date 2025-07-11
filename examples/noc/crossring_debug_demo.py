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
import time
from pathlib import Path
from typing import Optional

# 立即禁用所有日志输出
logging.getLogger().setLevel(logging.CRITICAL)
logging.getLogger("src").setLevel(logging.CRITICAL)
logging.getLogger("src.noc").setLevel(logging.CRITICAL)

# 添加src路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.noc.crossring.model import CrossRingModel
from src.noc.crossring.config import CrossRingConfig
from src.noc.debug import RequestTracker, RequestState, FlitType


def setup_debug_logging():
    """设置简洁的调试日志"""
    # 获取项目根目录
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)

    log_file = output_dir / "crossring_debug.log"

    # 完全禁用日志输出到控制台
    logging.getLogger().handlers.clear()

    # 创建文件处理器但禁用控制台输出
    file_handler = logging.FileHandler(str(log_file), mode="w")
    file_handler.setLevel(logging.DEBUG)

    # 设置所有日志器到CRITICAL级别
    logging.getLogger().setLevel(logging.CRITICAL)
    logging.getLogger().addHandler(file_handler)

    # 特别禁用特定的模块日志
    logging.getLogger("src.noc").setLevel(logging.CRITICAL)
    logging.getLogger("src").setLevel(logging.CRITICAL)

    return logging.getLogger(__name__)


def create_debug_config(rows=3, cols=3):
    """创建调试用的3x3 CrossRing配置"""
    config = CrossRingConfig(num_row=rows, num_col=cols, config_name="debug_3x3")

    # 配置IP接口：确保节点0有GDMA，节点4有DDR
    config.gdma_send_position_list = [1]  # 节点1有GDMA
    config.ddr_send_position_list = [0]  # 节点0有DDR
    config.l2m_send_position_list = [6, 7, 8]  # 最后三个节点有L2M

    # 调试配置
    config.debug_enabled = True
    config.verbose_mode = True

    return config


def get_debug_traffic_file():
    """获取专门的调试traffic文件"""
    traffic_file = Path(__file__).parent.parent.parent / "traffic_data" / "temp_debug_traffic.txt"

    if not traffic_file.exists():
        # 如果文件不存在，创建一个临时文件
        traffic_content = """# Debug traffic: Node 1 (GDMA) -> Node 0 (DDR)
# Format: cycle,src_node,src_ip,dst_node,dst_ip,request_type,request_size
0,1,gdma_0,0,ddr_0,R,4
"""

        temp_file = Path(__file__).parent.parent.parent / "traffic_data" / "temp_debug_traffic.txt"
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


def run_debug_simulation(target_packet_id: Optional[str] = None, debug_sleep_time: float = 0.0):
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
    print("DEMO: 开始创建traffic文件...")
    traffic_file = get_debug_traffic_file()
    print(f"DEMO: traffic文件创建完成: {traffic_file}")

    try:
        # 创建模型
        print("DEMO: 开始创建CrossRingModel...")
        model = CrossRingModel(config, traffic_file_path=str(traffic_file))
        print("DEMO: CrossRingModel创建完成")

        # 启用详细调试
        model.debug_enabled = True
        model.request_tracker.enable_debug(level=2)
        print("DEMO: debug模式启用完成")

        # 设置debug休眠时间
        if debug_sleep_time > 0:
            model.set_debug_sleep_time(debug_sleep_time)
            print(f"🐌 Debug休眠模式: {debug_sleep_time}秒/周期")
            print("   在仿真过程中可以按 Ctrl+C 来暂停并退出")
        else:
            # 如果没有指定sleep时间，默认使用0.5秒
            debug_sleep_time = 0.3
            model.set_debug_sleep_time(debug_sleep_time)
            print(f"🐌 默认Debug休眠模式: {debug_sleep_time}秒/周期（方便观察flit状态）")
            print("   如需更快速度，请使用: python crossring_debug_demo.py [packet_id] 0")

        # 如果指定了packet_id，只追踪特定请求
        if target_packet_id:
            model.request_tracker.track_packet(target_packet_id)
            print(f"🎯 追踪目标: {target_packet_id}")
        else:
            # 否则追踪所有请求
            print("🎯 追踪目标: 所有请求")

        print()

        # 注入流量
        injected = model.inject_from_traffic_file(traffic_file_path=str(traffic_file), cycle_accurate=True, immediate_inject=False)

        # 显示RequestTracker追踪的请求状态
        tracked_requests = model.request_tracker.get_active_tracked_requests()

        # 运行仿真，逐周期显示详细信息
        print("\n🔄 开始仿真...")
        print("=" * 50)

        max_cycles = 200
        print(f"DEMO: 准备开始仿真循环，最大周期={max_cycles}")

        try:
            for cycle in range(max_cycles):
                # 执行一个周期
                model.step()

                # 每个周期都检查详细状态
                tracked_requests = model.request_tracker.get_active_tracked_requests()
                completed_count = len(model.request_tracker.completed_requests)

                # 检查是否有被追踪的包在活跃传输
                has_active_movement = model.request_tracker.has_actively_moving_tracked_packets()

                # 决定是否显示debug信息
                should_show_debug = has_active_movement or completed_count > 0 or cycle < 10  # 有活跃传输的追踪包  # 有完成的请求  # 前几个周期总是显示

                # 只有在有实际flit移动时才显示信息
                if should_show_debug and has_active_movement:
                    for packet_id, lifecycle in tracked_requests.items():
                        flit_info = []

                        # 显示请求flit状态
                        if lifecycle.request_flits:
                            req_flit = lifecycle.request_flits[-1]  # 最新的请求flit
                            flit_info.append(f"REQ: {req_flit}")

                        # 显示响应flit状态
                        if lifecycle.response_flits:
                            rsp_flit = lifecycle.response_flits[-1]  # 最新的响应flit
                            flit_info.append(f"RSP: {rsp_flit}")

                        # 显示数据flit状态
                        if lifecycle.data_flits:
                            data_positions = [f"{flit}" for flit in lifecycle.data_flits]
                            flit_info.append(f"DAT({len(lifecycle.data_flits)}): {', '.join(data_positions)}")

                        if flit_info:
                            print(f"周期{model.cycle}: {' | '.join(flit_info)}")

                    # 移除目标请求的详细状态显示

                    # 添加sleep以便观察
                    if debug_sleep_time > 0:
                        # print(f"\n⏱️  休眠 {debug_sleep_time} 秒...")
                        time.sleep(debug_sleep_time)

                # 检查是否所有追踪的请求都完成
                if len(tracked_requests) == 0 and cycle > 10:
                    if target_packet_id:
                        print(f"\n✅ 追踪的请求 {target_packet_id} 在周期 {cycle} 完成")
                    else:
                        print(f"\n✅ 所有追踪的请求在周期 {cycle} 完成")
                    break

        except KeyboardInterrupt:
            print(f"\n⚠️  用户中断仿真（周期 {cycle}）")
            print(f"📊 仿真统计:")
            current_tracked = model.request_tracker.get_active_tracked_requests()
            print(f"   - 追踪的活跃请求: {len(current_tracked)}")
            print(f"   - 完成请求: {len(model.request_tracker.completed_requests)}")
            print(f"   - 已仿真周期: {cycle}")
            return False

        print("\n" + "=" * 50)
        print("🎯 仿真完成，生成详细报告...")

        # 使用RequestTracker自带的最终报告
        model.request_tracker.print_final_report()

        # 如果有特定目标，显示详细信息
        if target_packet_id:
            print_target_request_details(model.request_tracker, target_packet_id)

        return True

    except Exception as e:
        logger.error(f"Debug仿真失败: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        if "model" in locals():
            model.cleanup()
        # 清理临时文件
        if traffic_file.exists():
            traffic_file.unlink()


def print_target_request_details(tracker: RequestTracker, target_packet_id: str):
    """打印目标请求的详细信息"""
    print(f"\n🎯 目标请求 {target_packet_id} 详细追踪信息")
    print("=" * 60)

    lifecycle = tracker.get_request_status(target_packet_id)
    if not lifecycle:
        print("❌ 未找到目标请求")
        return

    # 基础信息
    print(f"源节点: {lifecycle.source} -> 目标节点: {lifecycle.destination}")
    print(f"操作类型: {lifecycle.op_type}, 突发长度: {lifecycle.burst_size}")
    print(f"当前状态: {lifecycle.current_state.value}")

    # 时间戳信息
    print(f"\n⏱️  时间线:")
    print(f"  创建周期: {lifecycle.created_cycle}")
    print(f"  注入周期: {lifecycle.injected_cycle}")
    print(f"  到达周期: {lifecycle.arrived_cycle}")
    if lifecycle.response_sent_cycle > 0:
        print(f"  响应发送周期: {lifecycle.response_sent_cycle}")
    if lifecycle.data_start_cycle > 0:
        print(f"  数据开始周期: {lifecycle.data_start_cycle}")
    if lifecycle.completed_cycle > 0:
        print(f"  完成周期: {lifecycle.completed_cycle}")

    # 延迟信息
    print(f"\n📊 延迟统计:")
    print(f"  总延迟: {lifecycle.get_total_latency()} 周期")
    print(f"  请求延迟: {lifecycle.get_request_latency()} 周期")
    print(f"  数据延迟: {lifecycle.get_data_latency()} 周期")

    # Flit追踪信息
    print(f"\n🔍 Flit追踪:")
    if lifecycle.request_flits:
        print(f"  请求Flit ({len(lifecycle.request_flits)}):")
        for i, flit in enumerate(lifecycle.request_flits):
            print(f"    [{i}] {flit}")

    if lifecycle.response_flits:
        print(f"  响应Flit ({len(lifecycle.response_flits)}):")
        for i, flit in enumerate(lifecycle.response_flits):
            print(f"    [{i}] {flit}")

    if lifecycle.data_flits:
        print(f"  数据Flit ({len(lifecycle.data_flits)}):")
        for i, flit in enumerate(lifecycle.data_flits[:5]):  # 只显示前5个
            print(f"    [{i}] {flit}")
        if len(lifecycle.data_flits) > 5:
            print(f"    ... (共{len(lifecycle.data_flits)}个数据Flit)")

    # 验证信息
    print(f"\n✅ 验证状态:")
    print(f"  响应有效: {lifecycle.response_valid}")
    print(f"  数据有效: {lifecycle.data_valid}")
    print(f"  数据完整性: {lifecycle.data_integrity_ok}")

    print("=" * 60)


# 移除了冗余函数，现在完全依赖RequestTracker的内置功能进行调试


def main():
    """主函数"""
    # 解析命令行参数
    # target_packet_id = sys.argv[1] if len(sys.argv) > 1 else None
    # debug_sleep_time = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0

    target_packet_id = 1  # 修复：使用数字格式的packet_id
    debug_sleep_time = 0.3

    print("🔍 CrossRing Debug Demo")
    print("=" * 50)
    print("专门用于详细请求追踪和调试")
    print(f"📝 用法: python {Path(__file__).name} [packet_id] [sleep_time]")
    print("    packet_id: 要追踪的特定请求ID，使用简单数字如 1, 2, 3 (可选)")
    print("    sleep_time: debug模式下每周期休眠时间，单位秒 (可选)")
    print("    示例: python crossring_debug_demo.py 1 0.1")

    if target_packet_id:
        print(f"🎯 将追踪特定请求: {target_packet_id}")
    else:
        print("🎯 将追踪所有请求")

    if debug_sleep_time > 0:
        print(f"🐌 Debug休眠: {debug_sleep_time}秒/周期")

    print("=" * 50)

    success = run_debug_simulation(target_packet_id, debug_sleep_time)

    if success:
        print("\n✅ Debug演示完成！")
        print("\n📋 功能特点:")
        print("- ✅ 3x3 CrossRing拓扑")
        print("- ✅ 节点0 (GDMA) -> 节点4 (DDR)")
        print("- ✅ 特定packet_id追踪（请求→响应→数据）")
        print("- ✅ 智能延迟检测（跳过非活跃传输期间）")
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
