#!/usr/bin/env python3
"""
CrossRing重构测试脚本。

测试新实现的CrossRing功能：
- 真实环形拓扑
- 四方向系统
- 环形桥接
- 维度转换
"""

import sys, os
import logging
from pathlib import Path

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)


def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("crossring_test.log", encoding="utf-8")],
    )


def test_ring_direction_mapper():
    """测试环形方向映射器"""
    print("\n=== 测试环形方向映射器 ===")

    try:
        from src.noc.crossring.ring_directions import RingDirectionMapper, RingDirection

        # 创建3x3网格的方向映射器
        mapper = RingDirectionMapper(3, 3)

        # 测试环形连接验证
        print("验证环形连接...")
        if mapper.validate_ring_connectivity():
            print("✓ 环形连接验证通过")
        else:
            print("✗ 环形连接验证失败")
            return False

        # 测试路径计算
        print("测试路径计算...")
        source, destination = 0, 8  # 从左上角到右下角
        path = mapper.get_ring_path(source, destination)
        print(f"从节点{source}到节点{destination}的路径: {path}")

        # 测试方向确定
        h_dir, v_dir = mapper.determine_ring_direction(source, destination)
        print(f"路由方向: 水平={h_dir}, 垂直={v_dir}")

        return True

    except Exception as e:
        print(f"✗ 环形方向映射器测试失败: {e}")
        return False


def test_ring_bridge():
    """测试环形桥接组件"""
    print("\n=== 测试环形桥接组件 ===")

    try:
        from src.noc.crossring.ring_bridge import RingBridge
        from src.noc.crossring.config import CrossRingConfig

        # 创建配置
        config = CrossRingConfig(num_row=3, num_col=3, config_name="test")

        # 创建环形桥接
        bridge = RingBridge(config)

        print(f"✓ 环形桥接创建成功: {len(bridge.cross_points)}个交叉点")

        # 测试交叉点状态
        status = bridge.get_global_status()
        print(f"✓ 全局状态获取成功: {status['total_cross_points']}个交叉点")

        return True

    except Exception as e:
        print(f"✗ 环形桥接测试失败: {e}")
        return False


def test_crossring_model():
    """测试CrossRing模型"""
    print("\n=== 测试CrossRing模型 ===")

    try:
        from src.noc.crossring import create_crossring_model

        # 创建模型
        print("创建CrossRing模型...")
        model = create_crossring_model("test", 3, 3)

        print(f"✓ 模型创建成功: {model}")

        # 获取模型摘要
        summary = model.get_model_summary()
        print(f"✓ 模型摘要: {summary['topology']}, {summary['total_nodes']}个节点")

        # 测试环形连接验证
        if model.direction_mapper.validate_ring_connectivity():
            print("✓ 模型中的环形连接验证通过")
        else:
            print("✗ 模型中的环形连接验证失败")
            return False

        # 测试基本仿真步骤
        print("测试仿真步骤...")
        for i in range(5):
            model.step()
        print(f"✓ 仿真步骤测试完成，当前周期: {model.cycle}")

        # 清理资源
        model.cleanup()
        print("✓ 资源清理完成")

        return True

    except Exception as e:
        print(f"✗ CrossRing模型测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_flit_enhancements():
    """测试Flit增强功能"""
    print("\n=== 测试Flit增强功能 ===")

    try:
        from src.noc.crossring.flit import create_crossring_flit
        from src.noc.crossring.ring_directions import RingDirection

        # 创建测试flit
        flit = create_crossring_flit(0, 8, [0, 1, 2, 5, 8])

        print(f"✓ Flit创建成功: {flit.packet_id}")

        # 测试四方向系统属性
        flit.current_ring_direction = RingDirection.TR
        flit.remaining_directions = [RingDirection.TD]
        flit.dimension_turn_cycle = 10

        print(f"✓ 四方向属性设置成功: 当前方向={flit.current_ring_direction}")

        # 测试路由信息
        routing_info = flit.get_routing_info()
        print(f"✓ 路由信息获取成功: {routing_info}")

        return True

    except Exception as e:
        print(f"✗ Flit增强功能测试失败: {e}")
        return False


def test_integration():
    """集成测试"""
    print("\n=== 集成测试 ===")

    try:
        from src.noc.crossring import quick_start_simulation

        print("运行快速仿真测试...")

        # 运行小规模仿真
        results = quick_start_simulation(config_name="custom", max_cycles=100, num_test_requests=5)

        print(f"✓ 快速仿真完成")
        print(f"  - 总周期: {results['simulation_info']['total_cycles']}")
        print(f"  - 配置: {results['simulation_info']['config']['config_name']}")

        return True

    except Exception as e:
        print(f"✗ 集成测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("CrossRing重构功能测试")
    print("=" * 50)

    setup_logging()

    tests = [
        ("环形方向映射器", test_ring_direction_mapper),
        ("环形桥接组件", test_ring_bridge),
        ("Flit增强功能", test_flit_enhancements),
        ("CrossRing模型", test_crossring_model),
        ("集成测试", test_integration),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} 通过")
            else:
                print(f"✗ {test_name} 失败")
        except Exception as e:
            print(f"✗ {test_name} 异常: {e}")
            import traceback

            traceback.print_exc()

    print(f"\n{'='*50}")
    print(f"测试结果: {passed}/{total} 通过")

    if passed == total:
        print("🎉 所有测试通过！CrossRing重构成功！")
        return 0
    else:
        print("❌ 部分测试失败，需要进一步调试")
        return 1


if __name__ == "__main__":
    exit(main())
