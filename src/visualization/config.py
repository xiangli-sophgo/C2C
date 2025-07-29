# -*- coding: utf-8 -*-
"""
可视化配置模块
处理字体配置和显示问题
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import sys
import warnings


def setup_chinese_fonts():
    """配置中文字体显示"""
    from src.utils.font_config import configure_matplotlib_fonts
    
    # 抑制字体相关警告
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    
    # 使用跨平台字体配置
    configure_matplotlib_fonts()
    
    # 验证字体配置
    try:
        # 创建一个测试图表来验证字体
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.text(0.5, 0.5, "测试中文", ha="center", va="center", fontsize=12)
        plt.close(fig)
        return True
    except Exception as e:
        print(f"字体配置验证失败: {e}")
        return False


def get_color_scheme(scheme_name="default"):
    """获取颜色方案"""

    color_schemes = {
        "default": {
            "chip": "#4CAF50",  # 绿色 - 芯片
            "switch": "#2196F3",  # 蓝色 - 交换机
            "host": "#FF9800",  # 橙色 - 主机
            "root": "#9C27B0",  # 紫色 - 根节点
            "c2c": "#F44336",  # 红色 - C2C链路
            "pcie": "#607D8B",  # 灰蓝色 - PCIe链路
            "default_link": "#757575",  # 灰色 - 默认链路
        },
        "modern": {
            "chip": "#00BCD4",  # 青色
            "switch": "#3F51B5",  # 靛蓝
            "host": "#FF5722",  # 深橙
            "root": "#E91E63",  # 粉红
            "c2c": "#FF4081",  # 粉红色链路
            "pcie": "#607D8B",  # 蓝灰色链路
            "default_link": "#9E9E9E",
        },
        "colorblind": {
            "chip": "#1f77b4",  # 蓝色
            "switch": "#ff7f0e",  # 橙色
            "host": "#2ca02c",  # 绿色
            "root": "#d62728",  # 红色
            "c2c": "#9467bd",  # 紫色
            "pcie": "#8c564b",  # 棕色
            "default_link": "#7f7f7f",
        },
    }

    return color_schemes.get(scheme_name, color_schemes["default"])


def configure_plot_style(style="default"):
    """配置绘图样式"""

    # 设置字体
    setup_chinese_fonts()

    # 设置基础样式
    if style == "paper":
        # 适合论文发表的样式
        plt.rcParams["figure.figsize"] = (10, 8)
        plt.rcParams["font.size"] = 12
        plt.rcParams["axes.linewidth"] = 1.2
        plt.rcParams["grid.alpha"] = 0.3
        plt.rcParams["lines.linewidth"] = 2
    elif style == "presentation":
        # 适合演示的样式
        plt.rcParams["figure.figsize"] = (12, 9)
        plt.rcParams["font.size"] = 14
        plt.rcParams["axes.linewidth"] = 1.5
        plt.rcParams["grid.alpha"] = 0.4
        plt.rcParams["lines.linewidth"] = 2.5
    else:
        # 默认样式
        plt.rcParams["figure.figsize"] = (10, 8)
        plt.rcParams["font.size"] = 10
        plt.rcParams["axes.linewidth"] = 1
        plt.rcParams["grid.alpha"] = 0.3
        plt.rcParams["lines.linewidth"] = 2


def create_bilingual_labels(chinese_labels, english_labels):
    """创建中英文双语标签"""
    if not isinstance(chinese_labels, list):
        chinese_labels = [chinese_labels]
    if not isinstance(english_labels, list):
        english_labels = [english_labels]

    # 检查字体是否支持中文
    try:
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.text(0.5, 0.5, chinese_labels[0], ha="center", va="center")
        plt.close(fig)
        # 如果能正常显示中文，返回中文标签
        return chinese_labels
    except:
        # 如果不能显示中文，返回英文标签
        return english_labels


def safe_chinese_text(chinese_text, english_fallback):
    """安全的中文文本显示"""
    try:
        # 尝试显示中文
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.text(0.5, 0.5, chinese_text, ha="center", va="center")
        plt.close(fig)
        return chinese_text
    except:
        return english_fallback


# 预定义的中英文对照标签
BILINGUAL_LABELS = {
    "topology_types": {"tree": ("树拓扑", "Tree Topology"), "torus": ("环面拓扑", "Torus Topology"), "mesh": ("网格拓扑", "Mesh Topology")},
    "node_types": {"chip": ("芯片", "Chip"), "switch": ("交换机", "Switch"), "host": ("主机", "Host"), "root": ("根节点", "Root")},
    "metrics": {
        "avg_path_length": ("平均路径长度", "Avg Path Length"),
        "max_path_length": ("最大路径长度", "Max Path Length"),
        "bandwidth_efficiency": ("带宽效率", "Bandwidth Efficiency"),
        "cost_factor": ("成本因子", "Cost Factor"),
        "fault_tolerance": ("故障容错", "Fault Tolerance"),
        "scalability": ("可扩展性", "Scalability"),
    },
    "ui_elements": {
        "chip_count": ("芯片数量", "Chip Count"),
        "performance_comparison": ("性能对比", "Performance Comparison"),
        "topology_visualization": ("拓扑可视化", "Topology Visualization"),
        "optimization_advisor": ("优化建议", "Optimization Advisor"),
    },
}


def get_label(category, key, language="auto"):
    """获取标签文本"""
    if category not in BILINGUAL_LABELS or key not in BILINGUAL_LABELS[category]:
        return key

    chinese, english = BILINGUAL_LABELS[category][key]

    if language == "chinese":
        return chinese
    elif language == "english":
        return english
    else:  # auto
        return safe_chinese_text(chinese, english)


def init_visualization_config():
    """初始化可视化配置"""
    success = setup_chinese_fonts()
    configure_plot_style()

    if success:
        print("✓ 可视化配置初始化成功")
    else:
        print("⚠️  可视化配置存在问题，可能影响中文显示")

    return success


# 自动初始化
if __name__ != "__main__":
    init_visualization_config()
