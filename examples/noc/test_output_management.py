#!/usr/bin/env python3
"""
测试输出管理功能的简化演示
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.noc.analysis.output_manager import OutputManager, SimulationContext
import json
import time


def test_basic_output_manager():
    """测试基本的输出管理功能"""
    print("=" * 50)
    print("测试基本输出管理功能")
    print("=" * 50)
    
    # 创建输出管理器
    output_manager = OutputManager()
    
    # 创建会话
    config = {
        'topology_type': 'mesh_4x4',
        'node_count': 16,
        'test_param': 'test_value'
    }
    
    session_id = output_manager.create_session(
        model_name="test_model",
        topology_type="4x4_mesh",
        config=config,
        session_name="basic_test"
    )
    
    print(f"创建的会话ID: {session_id}")
    print(f"会话目录: {output_manager.get_session_dir()}")
    
    # 保存一些测试数据
    test_data = {
        'test_key': 'test_value',
        'timestamp': time.time(),
        'data_array': [1, 2, 3, 4, 5]
    }
    
    # 保存JSON数据
    json_path = output_manager.save_data(test_data, "test_data", "json")
    print(f"保存JSON数据: {json_path}")
    
    # 保存配置
    extra_config = {'extra_param': 'extra_value'}
    config_path = output_manager.save_config(extra_config, "extra_config")
    print(f"保存配置文件: {config_path}")
    
    # 保存日志
    log_content = f"测试日志内容\n时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    log_path = output_manager.save_log(log_content, "test_log")
    print(f"保存日志文件: {log_path}")
    
    # 保存报告
    report_content = """# 测试报告

## 基本信息
- 测试时间: 2023-12-01
- 测试类型: 基本功能测试

## 结果
测试通过。
"""
    report_path = output_manager.save_report(report_content, "test_report", "md")
    print(f"保存报告文件: {report_path}")
    
    # 生成会话摘要
    summary = output_manager.generate_session_summary()
    print("\n会话摘要:")
    print(summary)
    
    return output_manager


def test_context_manager():
    """测试上下文管理器"""
    print("\n" + "=" * 50)
    print("测试上下文管理器")
    print("=" * 50)
    
    config = {
        'model': 'crossring',
        'topology': '8_ring',
        'frequency': 2.0
    }
    
    try:
        with SimulationContext(
            model_name="crossring",
            topology_type="8_ring",
            config=config,
            session_name="context_test"
        ) as output_manager:
            
            print(f"在上下文中的会话目录: {output_manager.get_session_dir()}")
            
            # 保存一些数据
            data = {'context_test': True, 'value': 42}
            output_manager.save_data(data, "context_data", "json")
            
            # 模拟一些处理
            time.sleep(1)
            
            print("上下文处理完成")
            
    except Exception as e:
        print(f"上下文管理器测试中出现错误: {e}")
    
    print("退出上下文管理器")


def test_session_list():
    """测试会话列表功能"""
    print("\n" + "=" * 50)
    print("测试会话列表功能")
    print("=" * 50)
    
    output_manager = OutputManager()
    sessions = output_manager.list_sessions()
    
    print(f"找到 {len(sessions)} 个历史会话:")
    for i, session in enumerate(sessions[:3], 1):  # 只显示前3个
        print(f"{i}. {session.get('session_id', 'Unknown')}")
        print(f"   模型: {session.get('model_name', 'Unknown')}")
        print(f"   拓扑: {session.get('topology_type', 'Unknown')}")
        print(f"   创建时间: {session.get('created_time', 'Unknown')}")
        print()


def main():
    """主函数"""
    print("NoC 输出管理系统测试")
    
    try:
        # 测试基本功能
        output_manager = test_basic_output_manager()
        
        # 测试上下文管理器
        test_context_manager()
        
        # 测试会话列表
        test_session_list()
        
        print("\n" + "=" * 50)
        print("所有测试完成!")
        print("=" * 50)
        print("检查 output/ 目录查看生成的文件")
        
        # 显示output目录结构
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
        if os.path.exists(output_dir):
            print(f"\noutput目录内容:")
            for item in os.listdir(output_dir):
                item_path = os.path.join(output_dir, item)
                if os.path.isdir(item_path):
                    print(f"  📁 {item}/")
                    # 显示子目录
                    for subitem in os.listdir(item_path):
                        subitem_path = os.path.join(item_path, subitem)
                        if os.path.isdir(subitem_path):
                            print(f"    📁 {subitem}/")
                        else:
                            print(f"    📄 {subitem}")
                else:
                    print(f"  📄 {item}")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()