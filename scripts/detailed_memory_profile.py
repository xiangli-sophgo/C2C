#!/usr/bin/env python3
"""
详细的内存分析工具，可以追踪每个类的实例数量和内存占用
"""
import gc
import sys
import psutil
import os
from collections import defaultdict
from pathlib import Path
import tracemalloc
import time

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

def get_size(obj, seen=None):
    """递归计算对象的内存大小"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    
    seen.add(obj_id)
    
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    
    return size

def analyze_memory_by_class():
    """分析每个类的实例数量和内存占用"""
    gc.collect()
    
    # 统计每个类的实例
    class_stats = defaultdict(lambda: {"count": 0, "size": 0})
    
    for obj in gc.get_objects():
        try:
            obj_class = obj.__class__.__name__
            module = obj.__class__.__module__
            
            # 只统计项目相关的类
            if module and ('noc' in module or 'topology' in module or 'protocol' in module):
                full_name = f"{module}.{obj_class}"
                class_stats[full_name]["count"] += 1
                class_stats[full_name]["size"] += get_size(obj)
        except:
            pass
    
    # 按内存占用排序
    sorted_stats = sorted(class_stats.items(), key=lambda x: x[1]["size"], reverse=True)
    
    print("\n" + "="*80)
    print("按类统计的内存占用（只显示项目相关类）")
    print("="*80)
    print(f"{'类名':<50} {'实例数':>10} {'总内存(MB)':>15}")
    print("-"*80)
    
    total_size = 0
    for class_name, stats in sorted_stats[:30]:  # 显示前30个
        size_mb = stats["size"] / (1024 * 1024)
        if size_mb > 0.1:  # 只显示超过0.1MB的
            print(f"{class_name:<50} {stats['count']:>10} {size_mb:>15.2f}")
            total_size += stats["size"]
    
    print("-"*80)
    print(f"{'项目类总计':<50} {'':<10} {total_size/(1024*1024):>15.2f}")
    print("="*80)

def profile_with_tracemalloc():
    """使用tracemalloc进行内存分析"""
    print("\n开始tracemalloc内存分析...")
    
    # 启动内存追踪
    tracemalloc.start()
    
    # 记录开始状态
    process = psutil.Process(os.getpid())
    start_memory = process.memory_info().rss / 1024 / 1024
    print(f"初始内存: {start_memory:.2f} MB")
    
    # 导入并运行main
    from noc_main import main
    
    # 在运行前快照
    snapshot1 = tracemalloc.take_snapshot()
    
    # 运行main
    start_time = time.time()
    main()
    end_time = time.time()
    
    # 运行后快照
    snapshot2 = tracemalloc.take_snapshot()
    
    # 内存统计
    end_memory = process.memory_info().rss / 1024 / 1024
    peak_memory = process.memory_info().vms / 1024 / 1024
    
    print(f"\n运行后内存: {end_memory:.2f} MB")
    print(f"峰值虚拟内存: {peak_memory:.2f} MB")
    print(f"内存增长: {end_memory - start_memory:.2f} MB")
    print(f"运行时间: {end_time - start_time:.2f} 秒")
    
    # 分析内存差异
    print("\n" + "="*80)
    print("内存分配TOP 20（按增量排序）")
    print("="*80)
    
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    for stat in top_stats[:20]:
        print(f"{stat.traceback.format()[0]:<70} {stat.size_diff/(1024*1024):>10.2f} MB")
    
    # 分析当前内存状态
    print("\n" + "="*80)
    print("当前内存分配TOP 20")
    print("="*80)
    
    current_stats = snapshot2.statistics('lineno')
    for stat in current_stats[:20]:
        print(f"{stat.traceback.format()[0]:<70} {stat.size/(1024*1024):>10.2f} MB")
    
    # 按类分析
    analyze_memory_by_class()
    
    # 查找大对象
    find_large_objects()
    
    tracemalloc.stop()

def find_large_objects(threshold_mb=10):
    """查找内存中的大对象"""
    print("\n" + "="*80)
    print(f"大对象分析（超过{threshold_mb}MB）")
    print("="*80)
    
    large_objects = []
    threshold_bytes = threshold_mb * 1024 * 1024
    
    for obj in gc.get_objects():
        try:
            size = get_size(obj)
            if size > threshold_bytes:
                obj_type = type(obj).__name__
                obj_module = type(obj).__module__ if hasattr(type(obj), '__module__') else 'unknown'
                
                # 获取更多信息
                info = {
                    'type': obj_type,
                    'module': obj_module,
                    'size': size,
                    'size_mb': size / (1024 * 1024)
                }
                
                # 对于某些类型，尝试获取更多细节
                if isinstance(obj, list):
                    info['length'] = len(obj)
                elif isinstance(obj, dict):
                    info['keys'] = len(obj)
                elif hasattr(obj, '__len__'):
                    try:
                        info['length'] = len(obj)
                    except:
                        pass
                
                large_objects.append(info)
        except:
            pass
    
    # 按大小排序
    large_objects.sort(key=lambda x: x['size'], reverse=True)
    
    for obj in large_objects[:20]:
        details = f"{obj['module']}.{obj['type']}"
        if 'length' in obj:
            details += f" (长度: {obj['length']})"
        elif 'keys' in obj:
            details += f" (键数: {obj['keys']})"
        
        print(f"{details:<60} {obj['size_mb']:>10.2f} MB")
    
    if not large_objects:
        print(f"未发现超过{threshold_mb}MB的大对象")

def analyze_virtual_memory():
    """分析虚拟内存使用情况"""
    print("\n" + "="*80)
    print("虚拟内存分析")
    print("="*80)
    
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    print(f"物理内存 (RSS): {mem_info.rss / (1024*1024):.2f} MB")
    print(f"虚拟内存 (VMS): {mem_info.vms / (1024*1024):.2f} MB")
    print(f"VMS/RSS 比率: {mem_info.vms / mem_info.rss:.2f}")
    
    # 在macOS上，虚拟内存可能会非常大，这通常是正常的
    if sys.platform == 'darwin':
        print("\n注意：在macOS上，虚拟内存值可能会非常大，这是正常现象。")
        print("macOS的内存管理机制会预留大量虚拟地址空间，但不一定实际使用。")
        print("应该主要关注RSS（实际物理内存）的使用情况。")
    
    # 尝试获取更详细的内存映射信息
    try:
        memory_maps = process.memory_maps()
        print(f"\n内存映射区域数: {len(memory_maps)}")
        
        # 按类型统计内存映射
        map_stats = defaultdict(int)
        for mmap in memory_maps:
            map_stats[mmap.path or '[anonymous]'] += mmap.rss
        
        print("\n主要内存映射区域:")
        for path, size in sorted(map_stats.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {path:<50} {size/(1024*1024):>10.2f} MB")
    except:
        print("无法获取详细的内存映射信息")

if __name__ == "__main__":
    print("详细内存分析工具")
    print("1. 运行完整分析（包括tracemalloc）")
    print("2. 只分析当前内存状态")
    print("3. 分析虚拟内存")
    
    choice = input("请选择 (1-3): ")
    
    if choice == "1":
        profile_with_tracemalloc()
        analyze_virtual_memory()
    elif choice == "2":
        analyze_memory_by_class()
        find_large_objects()
        analyze_virtual_memory()
    elif choice == "3":
        analyze_virtual_memory()
    else:
        print("无效选择")