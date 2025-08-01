#!/usr/bin/env python3
"""
使用cProfile和memory_profiler分析noc_main.py的性能
运行前请安装：pip install memory_profiler psutil
"""
import cProfile
import pstats
import io
from memory_profiler import profile
import psutil
import os
import time
from pathlib import Path
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from noc_main import main


def analyze_with_cprofile():
    """使用cProfile分析时间性能"""
    print("=" * 50)
    print("🕐 使用cProfile分析时间性能")
    print("=" * 50)
    
    # 创建性能分析器
    profiler = cProfile.Profile()
    
    # 开始分析
    profiler.enable()
    main()  # 运行原始main函数
    profiler.disable()
    
    # 分析结果
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.sort_stats('cumulative')
    ps.print_stats(20)  # 显示前20个最耗时的函数
    
    print(s.getvalue())
    
    # 保存详细报告
    ps.dump_stats('noc_main_profile.prof')
    print("详细分析结果已保存到: noc_main_profile.prof")
    print("可使用 snakeviz noc_main_profile.prof 查看可视化结果")


def analyze_memory_usage():
    """分析内存使用情况"""
    print("=" * 50)
    print("💾 分析内存使用情况")
    print("=" * 50)
    
    process = psutil.Process(os.getpid())
    
    # 记录开始内存
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"程序启动内存: {start_memory:.2f} MB")
    
    # 运行主程序
    start_time = time.time()
    main()
    end_time = time.time()
    
    # 记录结束内存
    end_memory = process.memory_info().rss / 1024 / 1024  # MB
    peak_memory = process.memory_info().vms / 1024 / 1024  # MB
    
    print(f"程序结束内存: {end_memory:.2f} MB")
    print(f"峰值虚拟内存: {peak_memory:.2f} MB")
    print(f"内存增长: {end_memory - start_memory:.2f} MB")
    print(f"总运行时间: {end_time - start_time:.2f} 秒")


@profile
def main_with_memory_profile():
    """使用memory_profiler装饰器的main函数"""
    main()


def run_line_profiler():
    """使用line_profiler进行逐行分析"""
    print("=" * 50)
    print("📊 要使用line_profiler，请按以下步骤:")
    print("=" * 50)
    print("1. 安装: pip install line_profiler")
    print("2. 在要分析的函数前添加 @profile 装饰器")
    print("3. 运行: kernprof -l -v profile_noc_main.py")


if __name__ == "__main__":
    choice = input("选择分析类型:\n1. cProfile时间分析\n2. 内存使用分析\n3. memory_profiler逐行内存分析\n4. 显示line_profiler使用说明\n请输入数字(1-4): ")
    
    if choice == "1":
        analyze_with_cprofile()
    elif choice == "2":
        analyze_memory_usage()
    elif choice == "3":
        print("运行memory_profiler逐行分析...")
        main_with_memory_profile()
    elif choice == "4":
        run_line_profiler()
    else:
        print("无效选择，运行默认的cProfile分析")
        analyze_with_cprofile()