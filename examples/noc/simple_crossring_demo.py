#!/usr/bin/env python3
"""
简化的CrossRing NoC演示
=====================

这是一个简化的CrossRing NoC仿真演示，专注于核心功能：
1. 创建CrossRing模型
2. 注入简单的测试流量
3. 运行仿真
4. 显示基本结果

Usage:
    python simple_crossring_demo.py
"""

import sys
import logging
from pathlib import Path

# 添加src路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.noc.crossring.model import CrossRingModel
from src.noc.crossring.config import CrossRingConfig


def setup_logging():
    """设置简单的日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def create_simple_config():
    """创建简单的CrossRing配置"""
    config = CrossRingConfig(
        num_row=2,
        num_col=3,
        config_name="simple_demo"
    )
    
    # 简单的IP配置
    config.gdma_send_position_list = [0, 1]
    config.ddr_send_position_list = [4, 5]
    config.l2m_send_position_list = [2, 3]
    
    return config


def run_simple_simulation():
    """运行简单的仿真"""
    logger = setup_logging()
    
    logger.info("开始简单的CrossRing NoC仿真演示")
    
    try:
        # 1. 创建配置
        logger.info("1. 创建CrossRing配置")
        config = create_simple_config()
        
        # 2. 创建模型
        logger.info("2. 创建CrossRing模型")
        model = CrossRingModel(config)
        logger.info(f"   模型创建成功，IP接口数量: {len(model.ip_interfaces)}")
        
        # 3. 运行演示仿真（内置流量生成和注入）
        logger.info("3. 运行演示仿真")
        demo_result = model.run_demo_simulation(
            traffic_pattern="basic",
            max_cycles=1000,
            warmup_cycles=100,
            stats_start_cycle=100
        )
        
        if not demo_result["success"]:
            logger.warning("演示仿真失败")
            return False
        
        # 4. 显示报告
        logger.info("4. 显示结果")
        print("\n" + demo_result["report"])
        
        logger.info("仿真完成！")
        return True
        
    except Exception as e:
        logger.error(f"仿真失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # 清理资源
        if 'model' in locals():
            model.cleanup()


def main():
    """主函数"""
    print("=" * 60)
    print("简化的CrossRing NoC仿真演示")
    print("=" * 60)
    
    success = run_simple_simulation()
    
    if success:
        print("\n✓ 演示成功完成！")
        print("\n演示功能:")
        print("- ✓ 创建CrossRing配置")
        print("- ✓ 初始化CrossRing模型")
        print("- ✓ 注入测试流量")
        print("- ✓ 运行仿真")
        print("- ✓ 显示结果")
        return 0
    else:
        print("\n✗ 演示失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())