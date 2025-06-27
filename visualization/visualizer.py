# -*- coding: utf-8 -*-
"""
主要拓扑可视化器
支持Tree、Torus等多种拓扑类型的可视化展示
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import math

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class TopologyVisualizer:
    """拓扑可视化器主类"""
    
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        self.fig = None
        self.ax = None
        self.node_colors = {
            'chip': '#4CAF50',      # 绿色 - 芯片
            'switch': '#2196F3',    # 蓝色 - 交换机  
            'host': '#FF9800',      # 橙色 - 主机
            'root': '#9C27B0'       # 紫色 - 根节点
        }
        self.link_colors = {
            'c2c': '#F44336',       # 红色 - C2C链路
            'pcie': '#607D8B',      # 灰蓝色 - PCIe链路
            'default': '#757575'    # 灰色 - 默认链路
        }
        
    def create_figure(self, title: str = "C2C拓扑图"):
        """创建绘图画布"""
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.ax.set_title(title, fontsize=16, fontweight='bold')
        self.ax.set_aspect('equal')
        return self.fig, self.ax
    
    def visualize_topology_graph(self, topology_graph, layout_type='auto', **kwargs):
        """可视化TopologyGraph对象"""
        if not self.fig:
            self.create_figure("拓扑图可视化")
            
        # 获取NetworkX图
        G = topology_graph._graph
        
        # 确定布局
        if layout_type == 'auto':
            layout_type = self._detect_layout_type(topology_graph)
            
        pos = self._calculate_layout(G, layout_type, **kwargs)
        
        # 绘制节点
        self._draw_nodes(G, pos)
        
        # 绘制边
        self._draw_edges(G, pos)
        
        # 添加标签
        self._draw_labels(G, pos)
        
        # 添加图例
        self._add_legend()
        
        # 调整布局
        self.ax.set_axis_off()
        plt.tight_layout()
        
        return self.fig
    
    def visualize_tree_topology(self, tree_root, all_nodes, **kwargs):
        """可视化树状拓扑"""
        self.create_figure("树状拓扑结构")
        
        # 构建NetworkX图
        G = self._build_tree_graph(tree_root, all_nodes)
        
        # 计算层次化布局
        pos = self._calculate_tree_layout(G, tree_root.node_id)
        
        # 绘制
        self._draw_nodes(G, pos)
        self._draw_edges(G, pos)
        self._draw_labels(G, pos)
        self._add_legend()
        
        self.ax.set_axis_off()
        plt.tight_layout()
        
        return self.fig
    
    def visualize_torus_topology(self, torus_structure, **kwargs):
        """可视化环形拓扑"""
        dimensions = torus_structure['dimensions']
        title = f"{dimensions}D Torus拓扑 ({torus_structure['grid_dimensions']})"
        self.create_figure(title)
        
        if dimensions == 2:
            self._visualize_2d_torus(torus_structure, **kwargs)
        elif dimensions == 3:
            self._visualize_3d_torus(torus_structure, **kwargs)
        else:
            raise ValueError(f"不支持的维度: {dimensions}")
            
        return self.fig
    
    def highlight_path(self, path_nodes: List[str], color='red', width=3):
        """高亮显示路径"""
        if not self.ax:
            return
            
        # 找到路径边并高亮
        for i in range(len(path_nodes) - 1):
            src, dst = path_nodes[i], path_nodes[i + 1]
            # 在现有图上添加高亮路径
            # 这需要访问之前绘制的线条对象
            pass  # 具体实现需要存储边的对象引用
    
    def animate_message_routing(self, path_nodes: List[str], message_info: Dict):
        """动画展示消息路由过程"""
        # 使用matplotlib.animation实现
        pass  # 后续实现
    
    def _detect_layout_type(self, topology_graph) -> str:
        """自动检测拓扑类型"""
        stats = topology_graph.get_topology_statistics()
        num_nodes = stats['num_nodes']
        avg_degree = stats.get('average_degree', 0)
        
        # 简单的启发式判断
        if avg_degree > 3:
            return 'torus'
        else:
            return 'tree'
    
    def _calculate_layout(self, G, layout_type: str, **kwargs):
        """计算节点布局位置"""
        if layout_type == 'tree':
            return self._calculate_tree_layout(G, kwargs.get('root'))
        elif layout_type == 'torus':
            return self._calculate_torus_layout(G, **kwargs)
        elif layout_type == 'spring':
            return nx.spring_layout(G, k=2, iterations=50)
        elif layout_type == 'circular':
            return nx.circular_layout(G)
        else:
            return nx.spring_layout(G)
    
    def _calculate_tree_layout(self, G, root_node: str):
        """计算树状布局"""
        if not root_node or root_node not in G.nodes:
            # 找到度数最大的节点作为根
            root_node = max(G.nodes, key=lambda x: G.degree(x))
        
        # 使用层次化布局
        pos = {}
        levels = self._get_tree_levels(G, root_node)
        
        for level, nodes in levels.items():
            y = -level  # 从上到下排列
            if len(nodes) == 1:
                pos[nodes[0]] = (0, y)
            else:
                x_positions = np.linspace(-len(nodes)/2, len(nodes)/2, len(nodes))
                for i, node in enumerate(nodes):
                    pos[node] = (x_positions[i], y)
        
        return pos
    
    def _calculate_torus_layout(self, G, grid_dims=None, **kwargs):
        """计算环形拓扑布局"""
        if not grid_dims:
            # 尝试从节点数推算网格尺寸
            num_nodes = len(G.nodes)
            if num_nodes == 16:
                grid_dims = [4, 4]
            elif num_nodes == 64:
                grid_dims = [8, 8]
            else:
                # 简单的平方根估算
                side = int(math.sqrt(num_nodes))
                grid_dims = [side, side]
        
        pos = {}
        node_list = sorted(G.nodes, key=lambda x: int(x.split('_')[-1]) if 'chip_' in x else 0)
        
        for i, node in enumerate(node_list):
            if len(grid_dims) >= 2:
                x = i % grid_dims[0]
                y = i // grid_dims[0]
                pos[node] = (x, y)
        
        return pos
    
    def _build_tree_graph(self, tree_root, all_nodes):
        """从树结构构建NetworkX图"""
        G = nx.Graph()
        
        # 添加节点
        for node_id, node in all_nodes.items():
            node_type = 'chip' if 'chip' in node_id else 'switch'
            if node == tree_root:
                node_type = 'root'
            G.add_node(node_id, node_type=node_type, node_obj=node)
        
        # 添加边（基于parent-child关系）
        def add_edges_recursive(node):
            if hasattr(node, 'children'):
                for child in node.children:
                    G.add_edge(node.node_id, child.node_id, link_type='tree')
                    add_edges_recursive(child)
        
        add_edges_recursive(tree_root)
        return G
    
    def _get_tree_levels(self, G, root):
        """获取树的各层节点"""
        levels = {}
        visited = set()
        
        def dfs(node, level):
            if node in visited:
                return
            visited.add(node)
            
            if level not in levels:
                levels[level] = []
            levels[level].append(node)
            
            for neighbor in G.neighbors(node):
                if neighbor not in visited:
                    dfs(neighbor, level + 1)
        
        dfs(root, 0)
        return levels
    
    def _draw_nodes(self, G, pos):
        """绘制节点"""
        for node_type in self.node_colors:
            nodes = [n for n, d in G.nodes(data=True) 
                    if d.get('node_type', '').startswith(node_type)]
            if nodes:
                nx.draw_networkx_nodes(
                    G, pos, nodelist=nodes,
                    node_color=self.node_colors[node_type],
                    node_size=800,
                    alpha=0.8,
                    ax=self.ax
                )
    
    def _draw_edges(self, G, pos):
        """绘制边"""
        for link_type in self.link_colors:
            edges = [(u, v) for u, v, d in G.edges(data=True)
                    if d.get('link_type', 'default') == link_type]
            if edges:
                nx.draw_networkx_edges(
                    G, pos, edgelist=edges,
                    edge_color=self.link_colors[link_type],
                    width=2,
                    alpha=0.6,
                    ax=self.ax
                )
    
    def _draw_labels(self, G, pos):
        """绘制节点标签"""
        labels = {node: node.replace('_', '\n') for node in G.nodes}
        nx.draw_networkx_labels(
            G, pos, labels,
            font_size=8,
            font_weight='bold',
            ax=self.ax
        )
    
    def _add_legend(self):
        """添加图例"""
        legend_elements = []
        
        # 节点类型图例
        for node_type, color in self.node_colors.items():
            legend_elements.append(
                patches.Patch(color=color, label=f'{node_type.title()}节点')
            )
        
        # 链路类型图例  
        for link_type, color in self.link_colors.items():
            if link_type != 'default':
                legend_elements.append(
                    patches.Patch(color=color, label=f'{link_type.upper()}链路')
                )
        
        self.ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    def _visualize_2d_torus(self, torus_structure, **kwargs):
        """2D环形拓扑可视化"""
        grid_dims = torus_structure['grid_dimensions']
        coord_map = torus_structure['coordinate_map']
        neighbors = torus_structure['neighbors']
        
        # 绘制网格节点
        for chip_id, (x, y) in coord_map.items():
            # 绘制芯片节点
            circle = patches.Circle((x, y), 0.3, 
                                  color=self.node_colors['chip'], 
                                  alpha=0.8)
            self.ax.add_patch(circle)
            
            # 添加芯片ID标签
            self.ax.text(x, y, str(chip_id), 
                        ha='center', va='center', 
                        fontsize=8, fontweight='bold')
        
        # 绘制连接
        drawn_edges = set()
        for chip_id, neighbor_dict in neighbors.items():
            chip_coord = coord_map[chip_id]
            
            for direction, neighbor_coord in neighbor_dict.items():
                neighbor_id = torus_structure['id_map'][neighbor_coord]
                
                # 避免重复绘制
                edge = tuple(sorted([chip_id, neighbor_id]))
                if edge in drawn_edges:
                    continue
                drawn_edges.add(edge)
                
                x1, y1 = coord_map[chip_id]
                x2, y2 = coord_map[neighbor_id]
                
                # 处理环形连接（跨边界）
                dx, dy = x2 - x1, y2 - y1
                if abs(dx) > grid_dims[0] // 2:
                    # X方向跨界连接
                    continue  # 暂时跳过，后续可以用曲线表示
                if abs(dy) > grid_dims[1] // 2:
                    # Y方向跨界连接  
                    continue
                
                # 绘制直线连接
                self.ax.plot([x1, x2], [y1, y2], 
                           color=self.link_colors['c2c'], 
                           linewidth=2, alpha=0.6)
        
        # 设置坐标轴
        self.ax.set_xlim(-1, grid_dims[0])
        self.ax.set_ylim(-1, grid_dims[1])
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        
        # 添加网格信息
        info_text = f"网格尺寸: {grid_dims[0]}×{grid_dims[1]}\n芯片数量: {len(coord_map)}"
        self.ax.text(0.02, 0.98, info_text, 
                    transform=self.ax.transAxes, 
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def _visualize_3d_torus(self, torus_structure, **kwargs):
        """3D环形拓扑可视化（投影到2D）"""
        # 这里先实现简化的2D投影版本
        # 后续可以使用plotly实现真正的3D可视化
        grid_dims = torus_structure['grid_dimensions']
        coord_map = torus_structure['coordinate_map']
        
        # 将3D坐标投影到2D
        pos_2d = {}
        for chip_id, (x, y, z) in coord_map.items():
            # 简单的3D到2D投影
            proj_x = x + z * 0.5
            proj_y = y + z * 0.3
            pos_2d[chip_id] = (proj_x, proj_y)
        
        # 使用2D方式绘制（简化版）
        for chip_id, (proj_x, proj_y) in pos_2d.items():
            x, y, z = coord_map[chip_id]
            
            # 根据Z坐标调整颜色深度
            alpha = 0.4 + 0.6 * (z / (grid_dims[2] - 1))
            circle = patches.Circle((proj_x, proj_y), 0.3,
                                  color=self.node_colors['chip'],
                                  alpha=alpha)
            self.ax.add_patch(circle)
            
            # 添加标签，包含3D坐标
            self.ax.text(proj_x, proj_y, f"{chip_id}\n({x},{y},{z})",
                        ha='center', va='center',
                        fontsize=6, fontweight='bold')
        
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        
        # 添加3D信息
        info_text = f"3D网格: {grid_dims[0]}×{grid_dims[1]}×{grid_dims[2]}\n芯片数量: {len(coord_map)}\n(2D投影显示)"
        self.ax.text(0.02, 0.98, info_text,
                    transform=self.ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    def save_figure(self, filename: str, dpi=300):
        """保存图形"""
        if self.fig:
            self.fig.savefig(filename, dpi=dpi, bbox_inches='tight')
            print(f"图形已保存到: {filename}")
    
    def show(self):
        """显示图形"""
        if self.fig:
            plt.show()
    
    def close(self):
        """关闭图形"""
        if self.fig:
            plt.close(self.fig)
            self.fig = None
            self.ax = None