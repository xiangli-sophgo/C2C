#!/usr/bin/env python3
"""
CrossRing NoC演示脚本。

展示在C2C仓库中重新实现的CrossRing NoC部分功能，
包括基础数据结构、IP接口和资源管理的验证。
"""

import sys
import os
import logging
from pathlib import Path

# 添加项目路径
# project_root = Path(__file__).parent.parent
# sys.path.insert(0, str(project_root / "src"))


def setup_logging():
    """配置日志"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler(f"../output/crossring_demo.log")])


def test_crossring_flit():
    """测试CrossRing Flit类"""
    print("\n=== 测试CrossRing Flit类 ===")

    try:
        from src.noc.crossring.flit import create_crossring_flit, return_crossring_flit, get_crossring_flit_pool_stats

        # 创建测试Flit
        flit = create_crossring_flit(source=0, destination=8, path=[0, 1, 5, 8], req_type="read", burst_length=4, packet_id="test_packet_001")

        print(f"创建Flit: {flit}")
        print(f"Flit坐标: {flit.get_coordinates(4)}")
        print(f"Flit字典: {flit.to_dict()}")

        # 测试路径前进
        print(f"初始位置: {flit.current_position}")
        flit.advance_path()
        print(f"前进后位置: {flit.current_position}")

        # 测试同步延迟记录
        flit2 = create_crossring_flit(0, 8, [0, 8], req_type="write")
        flit2.cmd_entry_cake0_cycle = 100
        flit.sync_latency_record(flit2)
        print(f"同步延迟记录后: {flit.cmd_entry_cake0_cycle}")

        # 返回到对象池
        return_crossring_flit(flit)
        return_crossring_flit(flit2)

        # 检查对象池统计
        pool_stats = get_crossring_flit_pool_stats()
        print(f"对象池统计: {pool_stats}")

        print("✓ CrossRing Flit测试通过")
        return True

    except Exception as e:
        print(f"✗ CrossRing Flit测试失败: {e}")
        return False


def test_crossring_config():
    """测试CrossRing配置类"""
    print("\n=== 测试CrossRing配置类 ===")

    try:
        from src.noc.crossring.config import create_crossring_config_2262, create_crossring_config_custom, CrossRingConfig

        # 测试预设配置
        config_2262 = create_crossring_config_2262()
        print(f"2262配置: {config_2262}")
        print(f"节点数: {config_2262.num_nodes}")
        print(f"拓扑参数: {config_2262.get_topology_params()}")

        # 测试自定义配置
        custom_config = create_crossring_config_custom(3, 3, "test_3x3", burst=8, gdma_bw_limit=256.0)
        print(f"自定义配置: {custom_config}")

        # 测试配置验证
        valid, error = custom_config.validate_config()
        print(f"配置验证: valid={valid}, error={error}")

        # 测试配置优化
        custom_config.optimize_for_workload("compute_intensive")
        print(f"计算密集型优化后: RN读tracker={custom_config.tracker_config.rn_r_tracker_ostd}")

        # 测试仿真配置生成
        sim_config = custom_config.create_simulation_config(max_cycles=5000)
        print(f"仿真配置: {sim_config['simulation']}")

        # 测试推荐周期数
        recommended = custom_config.get_recommended_simulation_cycles()
        print(f"推荐周期数: {recommended}")

        print("✓ CrossRing配置测试通过")
        return True

    except Exception as e:
        print(f"✗ CrossRing配置测试失败: {e}")
        return False


def test_crossring_ip_interface():
    """测试CrossRing IP接口"""
    print("\n=== 测试CrossRing IP接口 ===")

    try:
        from src.noc.crossring.config import create_crossring_config_custom
        from src.noc.crossring.ip_interface import CrossRingIPInterface

        # 创建配置和模拟模型
        config = create_crossring_config_custom(3, 3, "test_ip")

        class MockModel:
            def register_ip_interface(self, ip_interface):
                pass

        model = MockModel()

        # 创建IP接口
        ip_interface = CrossRingIPInterface(config=config, ip_type="gdma", node_id=0, model=model)

        print(f"IP接口创建成功: {ip_interface.ip_type}_{ip_interface.node_id}")

        # 测试状态获取
        status = ip_interface.get_status()
        print(f"IP接口状态: {status['rn_resources']}")

        # 测试请求入队
        success = ip_interface.enqueue_request(source=0, destination=8, req_type="read", burst_length=4, packet_id="test_ip_req_001")
        print(f"请求入队: {success}")
        print(f"inject_fifo状态: req={len(ip_interface.inject_fifos['req'])}")

        # 测试步进
        ip_interface.step(100)
        print(f"执行周期100后状态: {ip_interface.get_status()['current_cycle']}")

        print("✓ CrossRing IP接口测试通过")
        return True

    except Exception as e:
        print(f"✗ CrossRing IP接口测试失败: {e}")
        return False


def test_crossring_model():
    """测试CrossRing主模型"""
    print("\n=== 测试CrossRing主模型 ===")

    try:
        from src.noc.crossring.model import create_crossring_model

        # 创建模型
        model = create_crossring_model("test_model", 3, 3)
        print(f"模型创建成功: {model}")

        # 获取模型摘要
        summary = model.get_model_summary()
        print(f"模型摘要: {summary}")

        # 注入测试流量
        packet_ids = model.inject_test_traffic(source=0, destination=8, req_type="read", count=5, burst_length=4)
        print(f"注入测试流量: {len(packet_ids)}个包")

        # 执行几个周期
        for i in range(10):
            model.step()

        print(f"执行10个周期后: cycle={model.cycle}")

        # 检查活跃请求
        active_requests = model.get_active_request_count()
        print(f"活跃请求数: {active_requests}")

        # 获取全局状态
        global_status = model.get_global_tracker_status()
        print(f"全局tracker状态: {len(global_status)}个IP接口")

        # 打印调试状态
        model.print_debug_status()

        # 清理
        model.cleanup()

        print("✓ CrossRing模型测试通过")
        return True

    except Exception as e:
        print(f"✗ CrossRing模型测试失败: {e}")
        return False


def test_integration():
    """集成测试"""
    print("\n=== CrossRing集成测试 ===")

    try:
        from src.noc.crossring import quick_start_simulation, get_module_info, validate_installation

        # 验证安装
        installation_ok = validate_installation()
        print(f"安装验证: {'通过' if installation_ok else '失败'}")

        # 获取模块信息
        module_info = get_module_info()
        print(f"模块信息: {module_info['name']} v{module_info['version']}")
        print(f"功能特性: {len(module_info['features'])}项")

        # 快速启动仿真（小规模测试）
        print("启动快速仿真测试...")
        results = quick_start_simulation(config_name="custom", max_cycles=1000, num_test_requests=10)

        print(f"仿真完成: {results['simulation_info']['total_cycles']}个周期")
        print(f"全局统计: {results['global_stats']}")

        print("✓ CrossRing集成测试通过")
        return True

    except Exception as e:
        print(f"✗ CrossRing集成测试失败: {e}")
        return False


def run_performance_benchmark():
    """运行性能基准测试"""
    print("\n=== CrossRing性能基准测试 ===")

    try:
        import time
        from src.noc.crossring.model import create_crossring_model

        # 创建不同规模的模型进行测试
        configs = [
            ("3x3", 3, 3),
            ("4x4", 4, 4),
            ("5x4", 5, 4),
        ]

        results = {}

        for name, rows, cols in configs:
            print(f"\n测试配置: {name} ({rows}x{cols})")

            start_time = time.time()

            # 创建模型
            model = create_crossring_model(f"bench_{name}", rows, cols)

            # 注入测试流量
            num_requests = rows * cols * 2  # 每个节点2个请求
            for i in range(num_requests):
                source = i % model.config.num_nodes
                destination = (i + 1) % model.config.num_nodes
                req_type = "read" if i % 2 == 0 else "write"
                model.inject_test_traffic(source, destination, req_type)

            # 运行1000个周期
            for cycle in range(1000):
                model.step()
                if cycle % 200 == 0:
                    print(f"  周期 {cycle}, 活跃请求: {model.get_active_request_count()}")

            end_time = time.time()
            elapsed = end_time - start_time

            results[name] = {
                "nodes": rows * cols,
                "cycles": 1000,
                "elapsed_time": elapsed,
                "cycles_per_second": 1000 / elapsed,
                "final_active_requests": model.get_active_request_count(),
            }

            print(f"  完成: {elapsed:.2f}秒, {1000/elapsed:.0f} cycles/sec")

            # 清理
            model.cleanup()

        print(f"\n性能基准测试结果:")
        for name, result in results.items():
            print(f"  {name}: {result['nodes']}节点, {result['cycles_per_second']:.0f} cycles/sec")

        print("✓ 性能基准测试完成")
        return True

    except Exception as e:
        print(f"✗ 性能基准测试失败: {e}")
        return False


def main():
    """主函数"""
    print("CrossRing NoC演示脚本")
    print("=" * 50)

    setup_logging()

    # 执行所有测试
    tests = [
        ("CrossRing Flit类", test_crossring_flit),
        ("CrossRing配置类", test_crossring_config),
        ("CrossRing IP接口", test_crossring_ip_interface),
        ("CrossRing主模型", test_crossring_model),
        ("集成测试", test_integration),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            if test_func():
                passed += 1
        except Exception as e:
            print(f"测试 {test_name} 异常: {e}")

    print(f"\n{'='*50}")
    print(f"测试结果: {passed}/{total} 通过")

    if passed == total:
        print("🎉 所有基础测试通过！")

        # 如果基础测试通过，运行性能测试
        if input("\n是否运行性能基准测试？(y/N): ").lower() == "y":
            run_performance_benchmark()
    else:
        print(f"❌ {total - passed} 个测试失败")
        return 1

    print("\nCrossRing NoC实现验证完成！")
    print("\n实现摘要:")
    print("- ✓ CrossRing专用Flit类 (STI三通道协议)")
    print("- ✓ CrossRing专用IP接口 (时钟域转换、资源管理)")
    print("- ✓ CrossRing主模型类 (仿真循环、性能统计)")
    print("- ✓ 扩展配置类 (工作负载优化、规模调整)")
    print("- ✓ 完整的模块导出和便捷函数")

    return 0


if __name__ == "__main__":
    sys.exit(main())
