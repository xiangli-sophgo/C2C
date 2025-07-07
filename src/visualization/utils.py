# -*- coding: utf-8 -*-
"""
可视化工具模块
提供颜色管理、图形工具和数据处理函数
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from typing import Dict, List, Tuple, Any, Optional


class ColorManager:
    """颜色管理器"""
    
    def __init__(self):
        self.color_schemes = {
            'default': {
                'chip': '#4CAF50',      # 绿色
                'switch': '#2196F3',    # 蓝色
                'host': '#FF9800',      # 橙色
                'root': '#9C27B0',      # 紫色
                'c2c_link': '#F44336',  # 红色
                'pcie_link': '#607D8B', # 灰蓝色
            },
            'colorblind': {
                'chip': '#1f77b4',      # 蓝色
                'switch': '#ff7f0e',    # 橙色
                'host': '#2ca02c',      # 绿色
                'root': '#d62728',      # 红色
                'c2c_link': '#9467bd',  # 紫色
                'pcie_link': '#8c564b', # 棕色
            },
            'dark': {
                'chip': '#00E676',      # 亮绿色
                'switch': '#00B0FF',    # 亮蓝色
                'host': '#FF6D00',      # 亮橙色
                'root': '#E040FB',      # 亮紫色
                'c2c_link': '#FF1744',  # 亮红色
                'pcie_link': '#78909C', # 亮灰色
            }
        }
        self.current_scheme = 'default'
    
    def set_scheme(self, scheme_name: str):
        """设置颜色方案"""
        if scheme_name in self.color_schemes:
            self.current_scheme = scheme_name
        else:
            print(f"警告: 未知颜色方案 {scheme_name}, 使用默认方案")
    
    def get_color(self, element_type: str) -> str:
        """获取指定元素的颜色"""
        return self.color_schemes[self.current_scheme].get(element_type, '#666666')
    
    def get_scheme(self) -> Dict[str, str]:
        """获取当前颜色方案"""
        return self.color_schemes[self.current_scheme]
    
    def generate_gradient_colors(self, n_colors: int, base_color: str = '#2196F3') -> List[str]:
        """生成渐变色列表"""
        # 将基础颜色转换为RGB
        base_rgb = mcolors.hex2color(base_color)
        
        # 生成渐变
        colors = []
        for i in range(n_colors):
            # 调整亮度
            factor = 0.3 + 0.7 * (i / max(1, n_colors - 1))
            new_rgb = tuple(min(1.0, c * factor + (1 - factor) * 0.9) for c in base_rgb)
            colors.append(mcolors.rgb2hex(new_rgb))
        
        return colors


class GraphicsUtils:
    """图形工具类"""
    
    @staticmethod
    def set_chinese_font():
        """设置中文字体"""
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
    
    @staticmethod
    def apply_style(style_name: str = 'seaborn'):
        """应用图形样式"""
        try:
            plt.style.use(style_name)
        except OSError:
            print(f"警告: 样式 {style_name} 不可用，使用默认样式")
    
    @staticmethod
    def create_custom_colormap(colors: List[str], name: str = 'custom') -> mcolors.LinearSegmentedColormap:
        """创建自定义颜色映射"""
        return mcolors.LinearSegmentedColormap.from_list(name, colors)
    
    @staticmethod
    def add_grid(ax, alpha: float = 0.3, linestyle: str = '--'):
        """添加网格"""
        ax.grid(True, alpha=alpha, linestyle=linestyle)
    
    @staticmethod
    def format_axes(ax, title: str = '', xlabel: str = '', ylabel: str = '', 
                   title_size: int = 14, label_size: int = 12):
        """格式化坐标轴"""
        if title:
            ax.set_title(title, fontsize=title_size, fontweight='bold')
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=label_size)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=label_size)
    
    @staticmethod
    def save_high_quality(fig, filename: str, dpi: int = 300, format: str = 'png'):
        """保存高质量图片"""
        fig.savefig(filename, dpi=dpi, format=format, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')


class DataProcessor:
    """数据处理工具"""
    
    @staticmethod
    def normalize_data(data: List[float], method: str = 'minmax') -> List[float]:
        """数据归一化"""
        data_array = np.array(data)
        
        if method == 'minmax':
            min_val, max_val = data_array.min(), data_array.max()
            if max_val > min_val:
                return ((data_array - min_val) / (max_val - min_val)).tolist()
            else:
                return data_array.tolist()
        
        elif method == 'zscore':
            mean_val, std_val = data_array.mean(), data_array.std()
            if std_val > 0:
                return ((data_array - mean_val) / std_val).tolist()
            else:
                return data_array.tolist()
        
        else:
            return data
    
    @staticmethod
    def calculate_statistics(data: List[float]) -> Dict[str, float]:
        """计算基础统计信息"""
        data_array = np.array(data)
        
        return {
            'mean': float(data_array.mean()),
            'median': float(np.median(data_array)),
            'std': float(data_array.std()),
            'min': float(data_array.min()),
            'max': float(data_array.max()),
            'q25': float(np.percentile(data_array, 25)),
            'q75': float(np.percentile(data_array, 75))
        }
    
    @staticmethod
    def smooth_data(data: List[float], window_size: int = 3) -> List[float]:
        """数据平滑"""
        if len(data) < window_size:
            return data
        
        smoothed = []
        for i in range(len(data)):
            start = max(0, i - window_size // 2)
            end = min(len(data), i + window_size // 2 + 1)
            smoothed.append(np.mean(data[start:end]))
        
        return smoothed


class LayoutOptimizer:
    """布局优化器"""
    
    @staticmethod
    def optimize_node_positions(positions: Dict[str, Tuple[float, float]], 
                               adjacency: Dict[str, List[str]], 
                               iterations: int = 100) -> Dict[str, Tuple[float, float]]:
        """优化节点位置以减少边交叉"""
        # 简化的力导向优化
        nodes = list(positions.keys())
        pos_array = np.array([positions[node] for node in nodes])
        
        for _ in range(iterations):
            forces = np.zeros_like(pos_array)
            
            # 计算排斥力
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    diff = pos_array[i] - pos_array[j]
                    distance = np.linalg.norm(diff)
                    if distance > 0:
                        force = diff / distance / (distance + 1)
                        forces[i] += force
                        forces[j] -= force
            
            # 计算吸引力（连接的节点）
            for node, neighbors in adjacency.items():
                if node in nodes:
                    i = nodes.index(node)
                    for neighbor in neighbors:
                        if neighbor in nodes:
                            j = nodes.index(neighbor)
                            diff = pos_array[j] - pos_array[i]
                            distance = np.linalg.norm(diff)
                            if distance > 0:
                                force = diff * distance * 0.01
                                forces[i] += force
                                forces[j] -= force
            
            # 更新位置
            pos_array += forces * 0.1
        
        # 转换回字典格式
        optimized_positions = {}
        for i, node in enumerate(nodes):
            optimized_positions[node] = tuple(pos_array[i])
        
        return optimized_positions
    
    @staticmethod
    def minimize_edge_crossings(positions: Dict[str, Tuple[float, float]], 
                               edges: List[Tuple[str, str]]) -> int:
        """计算边交叉数量"""
        crossings = 0
        
        for i, (a1, b1) in enumerate(edges):
            for j, (a2, b2) in enumerate(edges):
                if i < j:  # 避免重复计算
                    if GraphicsUtils._lines_intersect(
                        positions[a1], positions[b1],
                        positions[a2], positions[b2]
                    ):
                        crossings += 1
        
        return crossings
    
    @staticmethod
    def _lines_intersect(p1: Tuple[float, float], p2: Tuple[float, float],
                        p3: Tuple[float, float], p4: Tuple[float, float]) -> bool:
        """检查两条线段是否相交"""
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)


class PerformanceProfiler:
    """性能分析器"""
    
    def __init__(self):
        self.metrics = {}
    
    def start_timer(self, name: str):
        """开始计时"""
        import time
        self.metrics[name] = {'start': time.time()}
    
    def end_timer(self, name: str):
        """结束计时"""
        import time
        if name in self.metrics:
            self.metrics[name]['end'] = time.time()
            self.metrics[name]['duration'] = self.metrics[name]['end'] - self.metrics[name]['start']
    
    def get_duration(self, name: str) -> Optional[float]:
        """获取持续时间"""
        if name in self.metrics and 'duration' in self.metrics[name]:
            return self.metrics[name]['duration']
        return None
    
    def print_summary(self):
        """打印性能摘要"""
        print("性能分析结果:")
        print("-" * 40)
        for name, data in self.metrics.items():
            if 'duration' in data:
                print(f"{name}: {data['duration']:.3f}秒")


# 全局实例
color_manager = ColorManager()
graphics_utils = GraphicsUtils()
data_processor = DataProcessor()
layout_optimizer = LayoutOptimizer()

# 初始化设置
graphics_utils.set_chinese_font()