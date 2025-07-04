# -*- coding: utf-8 -*-
"""
拓扑配置优化器
根据应用需求提供最优拓扑选择和配置建议
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import math


@dataclass
class ApplicationRequirements:
    """应用需求定义"""

    chip_count: int
    budget_constraint: float  # 预算约束 (相对值)
    latency_requirement: str  # 'low', 'medium', 'high'
    reliability_requirement: str  # 'low', 'medium', 'high'
    scalability_requirement: str  # 'low', 'medium', 'high'
    management_complexity: str  # 'simple', 'moderate', 'complex'
    power_constraint: float  # 功耗约束 (相对值)


@dataclass
class TopologyRecommendation:
    """拓扑推荐结果"""

    topology_type: str
    configuration: Dict[str, Any]
    score: float
    pros: List[str]
    cons: List[str]
    optimization_tips: List[str]
    estimated_performance: Dict[str, float]


class TopologyOptimizer:
    """拓扑配置优化器"""

    def __init__(self):
        # 拓扑特性配置
        self.topology_characteristics = {
            "tree": {"management_complexity": "simple", "cost_factor": "high", "fault_tolerance": "low", "scalability": "medium", "latency_base": "medium"},  # 需要额外交换机
            "torus": {"management_complexity": "moderate", "cost_factor": "low", "fault_tolerance": "high", "scalability": "high", "latency_base": "low"},  # 无需额外交换机
        }

        # 权重配置
        self.requirement_weights = {"budget": 0.25, "latency": 0.20, "reliability": 0.20, "scalability": 0.15, "management": 0.10, "power": 0.10}

    def analyze_requirements(self, requirements: ApplicationRequirements) -> Dict[str, TopologyRecommendation]:
        """分析应用需求并生成拓扑推荐"""

        recommendations = {}

        # 分析Tree拓扑适合度
        tree_recommendation = self._analyze_tree_topology(requirements)
        recommendations["tree"] = tree_recommendation

        # 分析Torus拓扑适合度
        torus_recommendation = self._analyze_torus_topology(requirements)
        recommendations["torus"] = torus_recommendation

        return recommendations

    def _analyze_tree_topology(self, req: ApplicationRequirements) -> TopologyRecommendation:
        """分析Tree拓扑的适合度"""

        # 计算最优Tree配置
        optimal_config = self._optimize_tree_configuration(req.chip_count)

        # 计算适合度评分
        score = self._calculate_tree_score(req, optimal_config)

        # 生成优缺点分析
        pros, cons = self._analyze_tree_pros_cons(req, optimal_config)

        # 生成优化建议
        optimization_tips = self._generate_tree_optimization_tips(req, optimal_config)

        # 预测性能指标
        estimated_performance = self._estimate_tree_performance(optimal_config)

        return TopologyRecommendation(
            topology_type="tree", configuration=optimal_config, score=score, pros=pros, cons=cons, optimization_tips=optimization_tips, estimated_performance=estimated_performance
        )

    def _analyze_torus_topology(self, req: ApplicationRequirements) -> TopologyRecommendation:
        """分析Torus拓扑的适合度"""

        # 计算最优Torus配置
        optimal_config = self._optimize_torus_configuration(req.chip_count)

        # 计算适合度评分
        score = self._calculate_torus_score(req, optimal_config)

        # 生成优缺点分析
        pros, cons = self._analyze_torus_pros_cons(req, optimal_config)

        # 生成优化建议
        optimization_tips = self._generate_torus_optimization_tips(req, optimal_config)

        # 预测性能指标
        estimated_performance = self._estimate_torus_performance(optimal_config)

        return TopologyRecommendation(
            topology_type="torus", configuration=optimal_config, score=score, pros=pros, cons=cons, optimization_tips=optimization_tips, estimated_performance=estimated_performance
        )

    def _optimize_tree_configuration(self, chip_count: int) -> Dict[str, Any]:
        """优化Tree拓扑配置"""

        # 计算最优交换机容量
        if chip_count <= 16:
            optimal_capacity = 4
        elif chip_count <= 64:
            optimal_capacity = 8
        else:
            optimal_capacity = 16

        # 估算树的层数和节点数
        levels = math.ceil(math.log(chip_count) / math.log(optimal_capacity))
        estimated_switches = max(0, (chip_count - 1) // (optimal_capacity - 1))
        total_nodes = chip_count + estimated_switches

        return {"switch_capacity": optimal_capacity, "estimated_levels": levels, "estimated_switches": estimated_switches, "total_nodes": total_nodes, "chip_count": chip_count}

    def _optimize_torus_configuration(self, chip_count: int) -> Dict[str, Any]:
        """优化Torus拓扑配置"""

        # 选择最优维度
        if chip_count <= 64:
            dimensions = 2
            # 找到最接近正方形的网格
            sqrt_n = int(math.sqrt(chip_count))
            for i in range(sqrt_n, 0, -1):
                if chip_count % i == 0:
                    grid_dims = [i, chip_count // i]
                    break
            else:
                # 如果无法整除，选择最接近的
                grid_dims = [sqrt_n, math.ceil(chip_count / sqrt_n)]
        else:
            dimensions = 3
            # 找到最接近立方体的网格
            cube_root = int(round(chip_count ** (1 / 3)))
            best_balance = float("inf")
            best_dims = [cube_root, cube_root, cube_root]

            for i in range(max(1, cube_root - 2), cube_root + 3):
                for j in range(max(1, cube_root - 2), cube_root + 3):
                    k = math.ceil(chip_count / (i * j))
                    if i * j * k >= chip_count:
                        balance = max(i, j, k) / min(i, j, k)
                        if balance < best_balance:
                            best_balance = balance
                            best_dims = [i, j, k]

            grid_dims = best_dims

        # 计算网格平衡度
        balance_factor = min(grid_dims) / max(grid_dims) if max(grid_dims) > 0 else 1.0

        return {"dimensions": dimensions, "grid_dimensions": grid_dims, "balance_factor": balance_factor, "total_nodes": chip_count, "chip_count": chip_count}

    def _calculate_tree_score(self, req: ApplicationRequirements, config: Dict) -> float:
        """计算Tree拓扑适合度评分"""
        score = 0.0

        # 预算适合度 (Tree需要更多硬件)
        cost_factor = config["estimated_switches"] / config["chip_count"]
        budget_score = max(0, 1 - cost_factor) * req.budget_constraint
        score += self.requirement_weights["budget"] * budget_score

        # 延迟适合度 (Tree在小规模时表现不错)
        latency_multiplier = {"low": 0.5, "medium": 0.7, "high": 1.0}[req.latency_requirement]
        latency_score = max(0, 1 - config["estimated_levels"] * 0.1) * latency_multiplier
        score += self.requirement_weights["latency"] * latency_score

        # 可靠性适合度 (Tree容错性较低)
        reliability_multiplier = {"low": 1.0, "medium": 0.6, "high": 0.3}[req.reliability_requirement]
        reliability_score = 0.4 * reliability_multiplier  # Tree固有的低可靠性
        score += self.requirement_weights["reliability"] * reliability_score

        # 可扩展性适合度
        scalability_multiplier = {"low": 1.0, "medium": 0.7, "high": 0.5}[req.scalability_requirement]
        scalability_score = max(0, 1 - math.log(config["total_nodes"]) * 0.1) * scalability_multiplier
        score += self.requirement_weights["scalability"] * scalability_score

        # 管理复杂度适合度 (Tree管理简单)
        management_multiplier = {"simple": 1.0, "moderate": 0.8, "complex": 0.6}[req.management_complexity]
        management_score = 0.9 * management_multiplier
        score += self.requirement_weights["management"] * management_score

        # 功耗适合度 (Tree功耗相对较高)
        power_score = max(0, 1 - cost_factor * 0.5) * req.power_constraint
        score += self.requirement_weights["power"] * power_score

        return min(1.0, score)

    def _calculate_torus_score(self, req: ApplicationRequirements, config: Dict) -> float:
        """计算Torus拓扑适合度评分"""
        score = 0.0

        # 预算适合度 (Torus无需额外硬件)
        budget_score = 1.0 * req.budget_constraint
        score += self.requirement_weights["budget"] * budget_score

        # 延迟适合度 (Torus延迟通常较低)
        latency_multiplier = {"low": 1.0, "medium": 0.8, "high": 0.6}[req.latency_requirement]
        avg_distance = sum(config["grid_dimensions"]) / len(config["grid_dimensions"]) / 4
        latency_score = max(0, 1 - avg_distance * 0.2) * latency_multiplier
        score += self.requirement_weights["latency"] * latency_score

        # 可靠性适合度 (Torus容错性高)
        reliability_multiplier = {"low": 0.7, "medium": 0.9, "high": 1.0}[req.reliability_requirement]
        reliability_base = 0.8 + config["dimensions"] * 0.1  # 维度越高可靠性越好
        reliability_score = min(1.0, reliability_base) * reliability_multiplier
        score += self.requirement_weights["reliability"] * reliability_score

        # 可扩展性适合度 (Torus扩展性好)
        scalability_multiplier = {"low": 0.7, "medium": 0.9, "high": 1.0}[req.scalability_requirement]
        scalability_score = (0.7 + config["balance_factor"] * 0.3) * scalability_multiplier
        score += self.requirement_weights["scalability"] * scalability_score

        # 管理复杂度适合度 (Torus管理中等复杂)
        management_multiplier = {"simple": 0.6, "moderate": 1.0, "complex": 0.8}[req.management_complexity]
        management_score = 0.7 * management_multiplier
        score += self.requirement_weights["management"] * management_score

        # 功耗适合度 (Torus功耗较低)
        power_score = 0.9 * req.power_constraint
        score += self.requirement_weights["power"] * power_score

        return min(1.0, score)

    def _analyze_tree_pros_cons(self, req: ApplicationRequirements, config: Dict) -> Tuple[List[str], List[str]]:
        """分析Tree拓扑的优缺点"""
        pros = ["管理和配置相对简单", "支持层次化控制结构", "适合集中式管理场景", "调试和故障定位容易"]

        cons = [f"需要额外的{config['estimated_switches']}个交换机设备", "单点故障影响较大", "随规模增长性能下降明显", "硬件成本较高"]

        # 根据具体需求调整
        if req.chip_count <= 16:
            pros.append("小规模时性能表现良好")
        else:
            cons.append("大规模时路径长度增加明显")

        if req.reliability_requirement == "high":
            cons.append("不满足高可靠性要求")

        return pros, cons

    def _analyze_torus_pros_cons(self, req: ApplicationRequirements, config: Dict) -> Tuple[List[str], List[str]]:
        """分析Torus拓扑的优缺点"""
        pros = ["无需额外交换机，硬件成本低", "具有多路径冗余，容错性好", "平均路径长度短，延迟低", "扩展性优秀，适合大规模部署"]

        cons = ["初始配置和调试较复杂", "需要规则的网格布局", "路由算法相对复杂"]

        # 根据具体配置调整
        if config["balance_factor"] > 0.8:
            pros.append(f"网格平衡度高({config['balance_factor']:.2f})，性能optimal")
        else:
            cons.append(f"网格不够平衡({config['balance_factor']:.2f})，可能影响性能")

        if config["dimensions"] == 3:
            pros.append("3D拓扑提供更好的可扩展性")
            cons.append("3D布局和管理更复杂")

        return pros, cons

    def _generate_tree_optimization_tips(self, req: ApplicationRequirements, config: Dict) -> List[str]:
        """生成Tree拓扑优化建议"""
        tips = []

        # 交换机配置优化
        if config["switch_capacity"] < 8:
            tips.append(f"考虑使用更大容量的交换机(当前{config['switch_capacity']})来减少层数")

        # 规模优化建议
        if req.chip_count > 64:
            tips.append("大规模部署时考虑分层Tree或混合拓扑")

        # 可靠性优化
        if req.reliability_requirement in ["medium", "high"]:
            tips.append("添加冗余交换机链路提高可靠性")
            tips.append("实施故障快速检测和恢复机制")

        # 成本优化
        if req.budget_constraint < 0.7:
            tips.append("选择性能价格比更好的交换机设备")
            tips.append("考虑分阶段部署策略")

        return tips

    def _generate_torus_optimization_tips(self, req: ApplicationRequirements, config: Dict) -> List[str]:
        """生成Torus拓扑优化建议"""
        tips = []

        # 网格优化建议
        if config["balance_factor"] < 0.7:
            dims_str = "x".join(map(str, config["grid_dimensions"]))
            tips.append(f"当前网格({dims_str})不够平衡，考虑调整为更方形的布局")

        # 维度选择优化
        if req.chip_count > 100 and config["dimensions"] == 2:
            tips.append("大规模部署建议使用3D Torus以获得更好性能")

        # 路由优化
        if req.latency_requirement == "low":
            tips.append("使用最短路径优先的路由算法")
            tips.append("实施自适应路由以避开拥塞链路")

        # 可靠性优化
        if req.reliability_requirement == "high":
            tips.append("启用故障链路的旁路机制")
            tips.append("实施分布式路由表更新")

        # 管理优化
        if req.management_complexity == "simple":
            tips.append("使用可视化工具简化拓扑管理")
            tips.append("实施自动化配置和监控")

        return tips

    def _estimate_tree_performance(self, config: Dict) -> Dict[str, float]:
        """预测Tree拓扑性能"""
        avg_path_length = config["estimated_levels"] * 0.7  # 估算平均路径长度
        max_path_length = config["estimated_levels"] * 2  # 估算最大路径长度

        return {
            "avg_path_length": avg_path_length,
            "max_path_length": max_path_length,
            "bandwidth_efficiency": config["chip_count"] / config["total_nodes"],
            "cost_factor": config["estimated_switches"] / config["chip_count"],
            "fault_tolerance": max(0.1, 1 - config["estimated_switches"] / config["total_nodes"]),
        }

    def _estimate_torus_performance(self, config: Dict) -> Dict[str, float]:
        """预测Torus拓扑性能"""
        # 估算平均路径长度
        avg_path_length = sum(dim / 4 for dim in config["grid_dimensions"])
        max_path_length = sum(dim // 2 for dim in config["grid_dimensions"])

        return {
            "avg_path_length": avg_path_length,
            "max_path_length": max_path_length,
            "bandwidth_efficiency": 1.0,  # 所有节点都是芯片
            "cost_factor": 0.0,  # 无额外交换机
            "fault_tolerance": min(0.95, 0.5 + config["dimensions"] * 0.15),
        }

    def generate_optimization_report(self, requirements: ApplicationRequirements) -> str:
        """生成优化建议报告"""
        recommendations = self.analyze_requirements(requirements)

        # 找到最佳推荐
        best_topology = max(recommendations.keys(), key=lambda k: recommendations[k].score)

        report = []
        report.append("# 拓扑配置优化建议报告\n")

        # 需求分析
        report.append("## 应用需求分析\n")
        report.append(f"- 芯片数量: {requirements.chip_count}\n")
        report.append(f"- 预算约束: {requirements.budget_constraint:.1f}\n")
        report.append(f"- 延迟要求: {requirements.latency_requirement}\n")
        report.append(f"- 可靠性要求: {requirements.reliability_requirement}\n")
        report.append(f"- 可扩展性要求: {requirements.scalability_requirement}\n")
        report.append(f"- 管理复杂度: {requirements.management_complexity}\n")
        report.append(f"- 功耗约束: {requirements.power_constraint:.1f}\n\n")

        # 最佳推荐
        best_rec = recommendations[best_topology]
        report.append(f"## 推荐方案: {best_topology.upper()}拓扑\n")
        report.append(f"**适合度评分: {best_rec.score:.3f}**\n\n")

        # 配置详情
        report.append("### 配置详情\n")
        for key, value in best_rec.configuration.items():
            report.append(f"- {key}: {value}\n")
        report.append("\n")

        # 优缺点分析
        report.append("### 优势\n")
        for pro in best_rec.pros:
            report.append(f"+ {pro}\n")
        report.append("\n")

        report.append("### 局限性\n")
        for con in best_rec.cons:
            report.append(f"- {con}\n")
        report.append("\n")

        # 优化建议
        report.append("### 优化建议\n")
        for tip in best_rec.optimization_tips:
            report.append(f"💡 {tip}\n")
        report.append("\n")

        # 性能预测
        report.append("### 性能预测\n")
        for metric, value in best_rec.estimated_performance.items():
            report.append(f"- {metric}: {value:.3f}\n")
        report.append("\n")

        # 替代方案
        alternative_topology = "torus" if best_topology == "tree" else "tree"
        alt_rec = recommendations[alternative_topology]
        report.append(f"## 替代方案: {alternative_topology.upper()}拓扑\n")
        report.append(f"**适合度评分: {alt_rec.score:.3f}**\n")
        report.append("如果主要方案不适用，可以考虑此替代方案。\n\n")

        return "".join(report)


def create_sample_requirements() -> List[Tuple[str, ApplicationRequirements]]:
    """创建示例应用需求"""
    scenarios = [
        (
            "开发测试环境",
            ApplicationRequirements(
                chip_count=8, budget_constraint=0.6, latency_requirement="medium", reliability_requirement="low", scalability_requirement="low", management_complexity="simple", power_constraint=0.7
            ),
        ),
        (
            "生产环境集群",
            ApplicationRequirements(
                chip_count=64, budget_constraint=0.8, latency_requirement="low", reliability_requirement="high", scalability_requirement="high", management_complexity="moderate", power_constraint=0.6
            ),
        ),
        (
            "高性能计算",
            ApplicationRequirements(
                chip_count=256, budget_constraint=0.9, latency_requirement="low", reliability_requirement="high", scalability_requirement="high", management_complexity="complex", power_constraint=0.5
            ),
        ),
    ]

    return scenarios


if __name__ == "__main__":
    # 演示优化器功能
    optimizer = TopologyOptimizer()
    scenarios = create_sample_requirements()

    for scenario_name, requirements in scenarios:
        print(f"\n{'='*50}")
        print(f"场景: {scenario_name}")
        print(f"{'='*50}")

        report = optimizer.generate_optimization_report(requirements)
        print(report)
