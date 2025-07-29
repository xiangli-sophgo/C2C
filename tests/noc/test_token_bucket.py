#!/usr/bin/env python3
"""
令牌桶(Token Bucket)测试脚本
测试不同带宽限制下的实际传输速率
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 添加src路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.noc.utils.token_bucket import TokenBucket
import logging
import matplotlib

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

if sys.platform == "darwin":  # macOS 的系统标识是 'darwin'
    matplotlib.use("macosx")  # 仅在 macOS 上使用该后端

# 导入跨平台字体配置
from src.utils.font_config import configure_matplotlib_fonts

# 配置matplotlib字体
configure_matplotlib_fonts()


def test_token_bucket_basic():
    """测试令牌桶基本功能"""
    print("=== 测试令牌桶基本功能 ===")

    # 创建令牌桶：每周期生成2个令牌，桶容量10
    tb = TokenBucket(rate=2.0, bucket_size=10.0)

    print(f"初始状态: {tb}")

    # 测试消耗令牌
    for i in range(5):
        success = tb.consume(3)
        print(f"周期{i}: 尝试消耗3个令牌, 成功={success}, 剩余={tb.get_tokens():.2f}")

    # 测试refill
    print("\n--- 等待5个周期后 ---")
    tb.refill(10)  # 跳到第10周期
    print(f"周期10: refill后, 剩余={tb.get_tokens():.2f}")

    # 再次尝试消耗
    for i in range(10, 15):
        success = tb.consume(1)
        print(f"周期{i}: 尝试消耗1个令牌, 成功={success}, 剩余={tb.get_tokens():.2f}")


def simulate_bandwidth_limit(bw_limit_gbps, network_freq_ghz, flit_size, simulation_cycles):
    """
    模拟带宽限制效果

    Args:
        bw_limit_gbps: 带宽限制 (GB/s)
        network_freq_ghz: 网络频率 (GHz)
        flit_size: flit大小 (字节)
        simulation_cycles: 仿真周期数

    Returns:
        tuple: (发送成功的周期列表, 累积传输字节数列表, 实际带宽列表)
    """
    # 计算令牌生成速率
    bytes_per_cycle = bw_limit_gbps / network_freq_ghz  # 字节/周期
    rate = bytes_per_cycle / flit_size  # flits/周期

    # 创建令牌桶
    tb = TokenBucket(rate=rate, bucket_size=bw_limit_gbps)

    # 记录数据
    send_cycles = []
    cumulative_bytes = []
    instant_bandwidth = []

    total_bytes = 0
    window_size = 100  # 用于计算瞬时带宽的窗口大小
    window_bytes = []

    # 模拟传输
    for cycle in range(simulation_cycles):
        # 每个周期refill
        tb.refill(cycle)

        # 尝试发送flit
        if tb.consume(1):
            send_cycles.append(cycle)
            total_bytes += flit_size
            cumulative_bytes.append(total_bytes)
            window_bytes.append((cycle, flit_size))

        # 移除窗口外的数据
        window_bytes = [(c, b) for c, b in window_bytes if c > cycle - window_size]

        # 计算瞬时带宽 (GB/s)
        if cycle > 0 and cycle % 10 == 0:  # 每10个周期计算一次
            window_total = sum(b for _, b in window_bytes)
            window_time = min(window_size, cycle) / (network_freq_ghz * 1e9)  # 秒
            instant_bw = window_total / window_time / 1e9  # GB/s
            instant_bandwidth.append((cycle, instant_bw))

    return send_cycles, cumulative_bytes, instant_bandwidth


def plot_token_bucket_behavior():
    """绘制令牌桶行为曲线"""
    print("\n=== 绘制令牌桶行为曲线 ===")

    # 测试参数
    network_freq_ghz = 2.0  # 2GHz
    flit_size = 128  # 128字节
    simulation_cycles = 2000

    # 不同的带宽限制
    bw_limits = [10, 20, 40, 80]  # GB/s

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Token Bucket 带宽限制效果测试", fontsize=16)

    # 1. 累积传输量对比
    ax1 = axes[0, 0]
    for bw_limit in bw_limits:
        send_cycles, cumulative_bytes, _ = simulate_bandwidth_limit(bw_limit, network_freq_ghz, flit_size, simulation_cycles)
        if send_cycles:
            # 转换为时间(ns)和数据量(GB)
            times_ns = [c / network_freq_ghz for c in send_cycles]
            cumulative_gb = [b / 1e9 for b in cumulative_bytes]
            ax1.plot(times_ns, cumulative_gb, label=f"{bw_limit} GB/s")

    ax1.set_xlabel("时间 (ns)")
    ax1.set_ylabel("累积传输量 (GB)")
    ax1.set_title("累积传输量随时间变化")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 瞬时带宽
    ax2 = axes[0, 1]
    for bw_limit in bw_limits:
        _, _, instant_bandwidth = simulate_bandwidth_limit(bw_limit, network_freq_ghz, flit_size, simulation_cycles)
        if instant_bandwidth:
            cycles, bws = zip(*instant_bandwidth)
            times_ns = [c / network_freq_ghz for c in cycles]
            ax2.plot(times_ns, bws, label=f"{bw_limit} GB/s", alpha=0.7)
            # 画出理论限制线
            ax2.axhline(y=bw_limit, color="gray", linestyle="--", alpha=0.5)

    ax2.set_xlabel("时间 (ns)")
    ax2.set_ylabel("瞬时带宽 (GB/s)")
    ax2.set_title("瞬时带宽测量")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 令牌桶状态变化（单个示例）
    ax3 = axes[1, 0]
    bw_limit = 20  # GB/s
    bytes_per_cycle = bw_limit / network_freq_ghz
    rate = bytes_per_cycle / flit_size
    tb = TokenBucket(rate=rate, bucket_size=bw_limit)

    token_levels = []
    send_success = []

    for cycle in range(200):
        tb.refill(cycle)
        token_levels.append(tb.get_tokens())
        success = tb.consume(1)
        send_success.append(1 if success else 0)

    cycles = list(range(200))
    times_ns = [c / network_freq_ghz for c in cycles]

    ax3.plot(times_ns, token_levels, "b-", label="令牌数量")
    ax3.fill_between(times_ns, 0, send_success, alpha=0.3, color="green", label="发送成功")
    ax3.set_xlabel("时间 (ns)")
    ax3.set_ylabel("令牌数量")
    ax3.set_title(f"令牌桶状态变化 (限制={bw_limit} GB/s)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 实际带宽 vs 理论带宽
    ax4 = axes[1, 1]
    actual_bandwidths = []

    for bw_limit in bw_limits:
        send_cycles, cumulative_bytes, _ = simulate_bandwidth_limit(bw_limit, network_freq_ghz, flit_size, simulation_cycles)
        if cumulative_bytes:
            # 计算平均带宽
            total_time_s = simulation_cycles / (network_freq_ghz * 1e9)
            avg_bandwidth = cumulative_bytes[-1] / total_time_s / 1e9  # GB/s
            actual_bandwidths.append(avg_bandwidth)
        else:
            actual_bandwidths.append(0)

    x = np.arange(len(bw_limits))
    width = 0.35

    bars1 = ax4.bar(x - width / 2, bw_limits, width, label="理论带宽", alpha=0.7)
    bars2 = ax4.bar(x + width / 2, actual_bandwidths, width, label="实际带宽", alpha=0.7)

    ax4.set_xlabel("带宽限制配置")
    ax4.set_ylabel("带宽 (GB/s)")
    ax4.set_title("理论带宽 vs 实际带宽")
    ax4.set_xticks(x)
    ax4.set_xticklabels([f"{bw} GB/s" for bw in bw_limits])
    ax4.legend()
    ax4.grid(True, axis="y", alpha=0.3)

    # 在柱状图上添加数值
    for bar, value in zip(bars1, bw_limits):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2.0, height, f"{value:.1f}", ha="center", va="bottom")

    for bar, value in zip(bars2, actual_bandwidths):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2.0, height, f"{value:.1f}", ha="center", va="bottom")

    plt.tight_layout()

    # 保存图表
    output_file = "token_bucket_test_results.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\n图表已保存到: {output_file}")

    plt.show()


def test_burst_behavior():
    """测试突发流量行为"""
    print("\n=== 测试突发流量行为 ===")

    # 参数设置
    bw_limit = 20  # GB/s
    network_freq_ghz = 2.0
    flit_size = 128

    bytes_per_cycle = bw_limit / network_freq_ghz
    rate = bytes_per_cycle / flit_size
    tb = TokenBucket(rate=rate, bucket_size=bw_limit)

    # 模拟突发流量
    results = []

    # 阶段1: 空闲100个周期（积累令牌）
    for cycle in range(100):
        tb.refill(cycle)
        results.append({"cycle": cycle, "tokens": tb.get_tokens(), "sent": 0, "phase": "idle"})

    # 阶段2: 突发传输50个周期
    for cycle in range(100, 150):
        tb.refill(cycle)
        sent = 0
        # 尝试连续发送多个flit
        while tb.consume(1):
            sent += 1
            if sent >= 5:  # 每周期最多尝试5个
                break

        results.append({"cycle": cycle, "tokens": tb.get_tokens(), "sent": sent, "phase": "burst"})

    # 阶段3: 正常传输100个周期
    for cycle in range(150, 250):
        tb.refill(cycle)
        sent = 1 if tb.consume(1) else 0

        results.append({"cycle": cycle, "tokens": tb.get_tokens(), "sent": sent, "phase": "normal"})

    # 绘制结果
    plt.figure(figsize=(12, 6))

    cycles = [r["cycle"] for r in results]
    tokens = [r["tokens"] for r in results]
    sent = [r["sent"] for r in results]

    plt.subplot(2, 1, 1)
    plt.plot(cycles, tokens, "b-", linewidth=2)
    plt.axhline(y=rate, color="r", linestyle="--", label=f"生成速率={rate:.2f}")
    plt.fill_between([0, 100], 0, max(tokens) * 1.1, alpha=0.2, color="gray", label="空闲期")
    plt.fill_between([100, 150], 0, max(tokens) * 1.1, alpha=0.2, color="orange", label="突发期")
    plt.fill_between([150, 250], 0, max(tokens) * 1.1, alpha=0.2, color="green", label="正常期")
    plt.ylabel("令牌数量")
    plt.title("突发流量场景下的令牌桶行为")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.bar(cycles, sent, width=1.0, alpha=0.7)
    plt.ylabel("发送的flit数")
    plt.xlabel("周期")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("token_bucket_burst_test.png", dpi=150, bbox_inches="tight")
    print(f"\n突发测试图表已保存到: token_bucket_burst_test.png")
    plt.show()


if __name__ == "__main__":
    # 运行测试
    test_token_bucket_basic()
    plot_token_bucket_behavior()
    test_burst_behavior()

    print("\n测试完成！")
