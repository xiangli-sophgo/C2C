# -*- coding: utf-8 -*-
"""
增强的拓扑对比分析演示
展示Tree vs Torus等多种拓扑在不同场景下的全面性能对比
"""

import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.visualization.comparison import PerformanceComparator
from src.c2c.topology.tree import TreeTopologyLogic, evaluate_tree_performance
from src.c2c.topology.torus import TorusTopologyLogic, TorusRoutingLogic
import numpy as np
import seaborn as sns

# 导入跨平台字体配置
from src.utils.font_config import configure_matplotlib_fonts

# 配置跨平台字体支持
configure_matplotlib_fonts()

# 设置seaborn样式
sns.set_style("whitegrid")


class ScenarioAnalyzer:
    """场景化分析器"""

    def __init__(self):
        self.comparator = PerformanceComparator()
        self.tree_logic = TreeTopologyLogic()
        self.torus_logic = TorusTopologyLogic()
        self.torus_routing = TorusRoutingLogic()

    def analyze_small_scale_scenario(self):
        """分析小规模场景 (4-16芯片)"""
        print("=== 小规模场景分析 (4-16芯片) ===")

        chip_counts = [4, 8, 12, 16]

        for chip_count in chip_counts:
            print(f"\n分析 {chip_count} 芯片配置...")

            # Tree拓扑分析
            try:
                chip_ids = list(range(chip_count))
                tree_root, tree_nodes = self.tree_logic.calculate_tree_structure(chip_ids, switch_capacity=4)
                tree_structure = {"root": tree_root, "nodes": tree_nodes}

                self.comparator.add_topology_data("tree", chip_count, tree_structure)
                print(f"  ✓ Tree拓扑: {len(tree_nodes)}个节点")

            except Exception as e:
                print(f"  ✗ Tree拓扑分析失败: {e}")

            # Torus拓扑分析
            try:
                torus_structure = self.torus_logic.calculate_torus_structure(chip_count, dimensions=2)

                self.comparator.add_topology_data("torus", chip_count, torus_structure, self.torus_routing)
                dims = torus_structure["grid_dimensions"]
                print(f"  ✓ Torus拓扑: {dims[0]}x{dims[1]}网格")

            except Exception as e:
                print(f"  ✗ Torus拓扑分析失败: {e}")

        return "small_scale"

    def analyze_medium_scale_scenario(self):
        """分析中等规模场景 (32-64芯片)"""
        print("\n=== 中等规模场景分析 (32-64芯片) ===")

        chip_counts = [32, 48, 64]

        for chip_count in chip_counts:
            print(f"\n分析 {chip_count} 芯片配置...")

            # Tree拓扑分析
            try:
                chip_ids = list(range(chip_count))
                tree_root, tree_nodes = self.tree_logic.calculate_tree_structure(chip_ids, switch_capacity=8)
                tree_structure = {"root": tree_root, "nodes": tree_nodes}

                self.comparator.add_topology_data("tree", chip_count, tree_structure)
                print(f"  ✓ Tree拓扑: {len(tree_nodes)}个节点")

            except Exception as e:
                print(f"  ✗ Tree拓扑分析失败: {e}")

            # Torus 2D拓扑分析
            try:
                torus_structure = self.torus_logic.calculate_torus_structure(chip_count, dimensions=2)

                self.comparator.add_topology_data("torus", chip_count, torus_structure, self.torus_routing)
                dims = torus_structure["grid_dimensions"]
                print(f"  ✓ Torus拓扑: {dims[0]}x{dims[1]}网格")

            except Exception as e:
                print(f"  ✗ Torus拓扑分析失败: {e}")

        return "medium_scale"

    def analyze_large_scale_scenario(self):
        """分析大规模场景 (128+芯片)"""
        print("\n=== 大规模场景分析 (128+芯片) ===")

        chip_counts = [128, 256]

        for chip_count in chip_counts:
            print(f"\n分析 {chip_count} 芯片配置...")

            # Tree拓扑分析 (大规模时可能很慢)
            try:
                chip_ids = list(range(min(chip_count, 128)))  # 限制分析规模
                tree_root, tree_nodes = self.tree_logic.calculate_tree_structure(chip_ids, switch_capacity=16)
                tree_structure = {"root": tree_root, "nodes": tree_nodes}

                self.comparator.add_topology_data("tree", len(chip_ids), tree_structure)
                print(f"  ✓ Tree拓扑: {len(tree_nodes)}个节点")

            except Exception as e:
                print(f"  ✗ Tree拓扑分析失败: {e}")

            # Torus 3D拓扑分析
            try:
                torus_structure = self.torus_logic.calculate_torus_structure(chip_count, dimensions=3)

                self.comparator.add_topology_data("torus", chip_count, torus_structure, self.torus_routing)
                dims = torus_structure["grid_dimensions"]
                if len(dims) == 3:
                    print(f"  ✓ Torus拓扑: {dims[0]}x{dims[1]}x{dims[2]}网格")
                else:
                    print(f"  ✓ Torus拓扑: {dims}网格")

            except Exception as e:
                print(f"  ✗ Torus拓扑分析失败: {e}")

        return "large_scale"

    def analyze_cost_sensitive_scenario(self):
        """分析成本敏感场景"""
        print("\n=== 成本敏感场景分析 ===")

        # 对比不同规模下的成本效率
        scenarios = [("小型部署", [8, 16]), ("中型部署", [32, 64]), ("大型部署", [128, 256])]

        cost_analysis = {}

        for scenario_name, chip_counts in scenarios:
            print(f"\n{scenario_name}:")
            cost_analysis[scenario_name] = {}

            for chip_count in chip_counts:
                print(f"  {chip_count}芯片配置:")

                # 估算Tree拓扑成本
                try:
                    if chip_count <= 128:  # 限制计算规模
                        chip_ids = list(range(chip_count))
                        tree_root, tree_nodes = self.tree_logic.calculate_tree_structure(chip_ids, switch_capacity=8)

                        chip_nodes = len([n for n in tree_nodes.keys() if n.startswith("chip_")])
                        switch_nodes = len([n for n in tree_nodes.keys() if n.startswith("switch_")])

                        cost_analysis[scenario_name][f"tree_{chip_count}"] = {"chips": chip_nodes, "switches": switch_nodes, "total_hardware": chip_nodes + switch_nodes}

                        print(f"    Tree: {chip_nodes}芯片 + {switch_nodes}交换机 = {chip_nodes + switch_nodes}设备")

                except Exception as e:
                    print(f"    Tree分析失败: {e}")

                # 估算Torus拓扑成本
                try:
                    dimensions = 2 if chip_count <= 64 else 3
                    cost_analysis[scenario_name][f"torus_{chip_count}"] = {"chips": chip_count, "switches": 0, "total_hardware": chip_count}

                    print(f"    Torus: {chip_count}芯片 + 0交换机 = {chip_count}设备")

                except Exception as e:
                    print(f"    Torus分析失败: {e}")

        return cost_analysis

    def analyze_fault_tolerance_scenario(self):
        """分析故障容错场景"""
        print("\n=== 故障容错场景分析 ===")

        fault_scenarios = [("单节点故障", 1), ("多节点故障", 2), ("级联故障", 3)]

        for scenario_name, fault_count in fault_scenarios:
            print(f"\n{scenario_name} (故障节点数: {fault_count}):")

            # 分析Tree拓扑的故障影响
            print("  Tree拓扑故障影响:")
            print(f"    - 交换机故障可能影响多个芯片节点")
            print(f"    - 根节点故障影响最大")
            print(f"    - 单点故障可能分割网络")

            # 分析Torus拓扑的故障影响
            print("  Torus拓扑故障影响:")
            print(f"    - 具有多路径冗余")
            print(f"    - 单点故障影响局部")
            print(f"    - 可通过重路由恢复连通性")

        return fault_scenarios

    def generate_scenario_comparison_report(self):
        """生成场景对比报告"""
        print("\n=== 生成场景对比报告 ===")

        if not self.comparator.metrics_data:
            print("没有可用数据，请先运行场景分析")
            return

        # 生成综合报告
        report = self.comparator.generate_comprehensive_report()

        # 保存报告
        report_path = "../output"
        os.makedirs(report_path, exist_ok=True)

        report_file = os.path.join(report_path, "enhanced_topology_comparison_report.md")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"详细报告已保存到: {report_file}")

        return report

    def generate_comparison_visualizations(self):
        """生成对比可视化图表"""
        print("\n=== 生成对比可视化图表 ===")

        if not self.comparator.metrics_data:
            print("没有可用数据，请先运行场景分析")
            return

        try:
            # 生成全面对比图表
            fig = self.comparator.compare_topologies_comprehensive()

            # 保存图表
            charts_path = "../output"
            os.makedirs(charts_path, exist_ok=True)

            chart_file = os.path.join(charts_path, "enhanced_topology_comparison.png")
            fig.savefig(chart_file, dpi=300, bbox_inches="tight")
            plt.close(fig)

            print(f"对比图表已保存到: {chart_file}")

            return chart_file

        except Exception as e:
            print(f"生成图表失败: {e}")
            return None


def main():
    """主演示函数"""
    print("启动增强的拓扑对比分析演示...\n")

    analyzer = ScenarioAnalyzer()

    try:
        # 1. 分析不同规模场景
        analyzer.analyze_small_scale_scenario()
        analyzer.analyze_medium_scale_scenario()
        analyzer.analyze_large_scale_scenario()

        # 2. 分析特定应用场景
        cost_analysis = analyzer.analyze_cost_sensitive_scenario()
        fault_scenarios = analyzer.analyze_fault_tolerance_scenario()

        # 3. 生成综合报告
        report = analyzer.generate_scenario_comparison_report()

        # 4. 生成可视化图表
        chart_file = analyzer.generate_comparison_visualizations()

        # 5. 输出关键结论
        print("\n=== 关键分析结论 ===")

        if analyzer.comparator.metrics_data:
            best_config = analyzer.comparator._find_best_topology_overall()
            print(f"✓ 综合最优配置: {best_config['topology']}拓扑，{best_config['chip_count']}芯片")
            print(f"✓ 综合评分: {best_config['score']:.3f}")

        print("\n✓ 小规模系统(4-16芯片): Tree拓扑管理简单，适合开发测试")
        print("✓ 中等规模系统(32-64芯片): Torus拓扑性能成本平衡最佳")
        print("✓ 大规模系统(128+芯片): Torus 3D拓扑提供最高性能")
        print("✓ 成本敏感场景: Torus拓扑无需额外交换机，硬件成本更低")
        print("✓ 高可靠性要求: Torus拓扑具有更好的故障容错能力")

        print(f"\n🎉 增强的拓扑对比分析完成！")
        print(f"📊 详细报告和图表已生成")

        if chart_file:
            print(f"💡 建议查看生成的可视化图表以获得更详细的对比分析")

    except Exception as e:
        print(f"\n❌ 分析过程中发生异常: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
