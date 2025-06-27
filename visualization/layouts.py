# -*- coding: utf-8 -*-
"""
拓扑布局算法
专门针对不同拓扑类型优化的布局算法
"""

import numpy as np
import networkx as nx
import math
from typing import Dict, List, Tuple, Any


class BaseLayout:
    """布局算法基类"""
    
    def __init__(self):
        pass
    
    def calculate_positions(self, graph, **kwargs) -> Dict[str, Tuple[float, float]]:
        """计算节点位置"""
        raise NotImplementedError


class TreeLayout(BaseLayout):
    """树状拓扑专用布局算法"""
    
    def __init__(self, vertical_spacing=1.0, horizontal_spacing=1.0):
        super().__init__()
        self.vertical_spacing = vertical_spacing
        self.horizontal_spacing = horizontal_spacing
    
    def calculate_positions(self, graph, root_node=None, **kwargs) -> Dict[str, Tuple[float, float]]:
        """计算树状布局位置"""
        if not root_node:
            root_node = self._find_root(graph)
        
        # 构建树结构
        tree_levels = self._build_tree_levels(graph, root_node)
        
        # 计算每层的位置
        positions = {}
        for level, nodes in tree_levels.items():
            y = -level * self.vertical_spacing
            
            if len(nodes) == 1:
                positions[nodes[0]] = (0, y)
            else:
                # 计算水平分布
                total_width = (len(nodes) - 1) * self.horizontal_spacing
                start_x = -total_width / 2
                
                for i, node in enumerate(nodes):
                    x = start_x + i * self.horizontal_spacing
                    positions[node] = (x, y)
        
        return positions
    
    def calculate_hierarchical_positions(self, tree_root, all_nodes) -> Dict[str, Tuple[float, float]]:
        """基于实际树结构计算位置"""
        positions = {}
        
        # 计算每个节点的子树大小
        subtree_sizes = self._calculate_subtree_sizes(tree_root)
        
        # 递归布局
        self._layout_recursive(tree_root, 0, 0, subtree_sizes, positions)
        
        return positions
    
    def _find_root(self, graph):
        """寻找树的根节点"""
        # 寻找度数最大的节点，或者名字包含'root'的节点
        root_candidates = [n for n in graph.nodes if 'root' in n.lower()]
        if root_candidates:
            return root_candidates[0]
        
        # 否则选择度数最大的节点
        return max(graph.nodes, key=lambda x: graph.degree(x))
    
    def _build_tree_levels(self, graph, root):
        """构建树的层级结构"""
        levels = {}
        visited = set()
        
        def bfs_levels():
            queue = [(root, 0)]
            visited.add(root)
            
            while queue:
                node, level = queue.pop(0)
                
                if level not in levels:
                    levels[level] = []
                levels[level].append(node)
                
                for neighbor in graph.neighbors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, level + 1))
        
        bfs_levels()
        return levels
    
    def _calculate_subtree_sizes(self, node):
        """计算每个节点的子树大小"""
        sizes = {}
        
        def dfs_size(current):
            if not hasattr(current, 'children') or not current.children:
                sizes[current.node_id] = 1
                return 1
            
            size = 1
            for child in current.children:
                size += dfs_size(child)
            
            sizes[current.node_id] = size
            return size
        
        dfs_size(node)
        return sizes
    
    def _layout_recursive(self, node, x, y, subtree_sizes, positions, level_height=1.0):
        """递归计算节点位置"""
        positions[node.node_id] = (x, y)
        
        if not hasattr(node, 'children') or not node.children:
            return
        
        # 计算子节点的总宽度需求
        total_child_width = sum(subtree_sizes[child.node_id] for child in node.children)
        
        # 起始位置
        start_x = x - (total_child_width - 1) / 2
        current_x = start_x
        
        for child in node.children:
            child_width = subtree_sizes[child.node_id]
            child_x = current_x + (child_width - 1) / 2
            child_y = y - level_height
            
            self._layout_recursive(child, child_x, child_y, subtree_sizes, positions, level_height)
            current_x += child_width


class TorusLayout(BaseLayout):
    """环形拓扑专用布局算法"""
    
    def __init__(self, grid_spacing=1.0):
        super().__init__()
        self.grid_spacing = grid_spacing
    
    def calculate_positions(self, torus_structure, **kwargs) -> Dict[str, Tuple[float, float]]:
        """计算环形拓扑位置"""
        dimensions = torus_structure['dimensions']
        
        if dimensions == 2:
            return self._calculate_2d_positions(torus_structure)
        elif dimensions == 3:
            return self._calculate_3d_positions(torus_structure)
        else:
            raise ValueError(f"不支持的维度: {dimensions}")
    
    def _calculate_2d_positions(self, torus_structure) -> Dict[str, Tuple[float, float]]:
        """计算2D环形位置"""
        coord_map = torus_structure['coordinate_map']
        positions = {}
        
        for chip_id, (x, y) in coord_map.items():
            pos_x = x * self.grid_spacing
            pos_y = y * self.grid_spacing
            positions[chip_id] = (pos_x, pos_y)
        
        return positions
    
    def _calculate_3d_positions(self, torus_structure) -> Dict[str, Tuple[float, float]]:
        """计算3D环形位置（投影到2D）"""
        coord_map = torus_structure['coordinate_map']
        grid_dims = torus_structure['grid_dimensions']
        positions = {}
        
        for chip_id, (x, y, z) in coord_map.items():
            # 使用等距投影
            proj_x = x * self.grid_spacing + z * self.grid_spacing * 0.5
            proj_y = y * self.grid_spacing + z * self.grid_spacing * 0.3
            positions[chip_id] = (proj_x, proj_y)
        
        return positions
    
    def calculate_circular_positions(self, num_nodes: int, radius: float = 5.0) -> Dict[str, Tuple[float, float]]:
        """计算圆形排列位置"""
        positions = {}
        angle_step = 2 * math.pi / num_nodes
        
        for i in range(num_nodes):
            angle = i * angle_step
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            positions[f"chip_{i}"] = (x, y)
        
        return positions


class OptimizedLayout(BaseLayout):
    """优化的布局算法"""
    
    def __init__(self, algorithm='spring'):
        super().__init__()
        self.algorithm = algorithm
    
    def calculate_positions(self, graph, **kwargs) -> Dict[str, Tuple[float, float]]:
        """使用NetworkX优化算法计算位置"""
        if self.algorithm == 'spring':
            return nx.spring_layout(graph, k=2, iterations=100)
        elif self.algorithm == 'kamada_kawai':
            return nx.kamada_kawai_layout(graph)
        elif self.algorithm == 'circular':
            return nx.circular_layout(graph)
        elif self.algorithm == 'shell':
            return nx.shell_layout(graph)
        else:
            return nx.spring_layout(graph)
    
    def force_directed_layout(self, graph, iterations=100, k=1.0, temperature=1.0):
        """力导向布局算法"""
        # 实现自定义的力导向算法
        positions = {}
        nodes = list(graph.nodes())
        
        # 初始化随机位置
        for node in nodes:
            positions[node] = (np.random.random() * 10, np.random.random() * 10)
        
        for iteration in range(iterations):
            forces = {node: np.array([0.0, 0.0]) for node in nodes}
            
            # 计算排斥力
            for i, node1 in enumerate(nodes):
                for j, node2 in enumerate(nodes):
                    if i != j:
                        pos1 = np.array(positions[node1])
                        pos2 = np.array(positions[node2])
                        diff = pos1 - pos2
                        distance = np.linalg.norm(diff)
                        
                        if distance > 0:
                            repulsion = k * k / distance
                            forces[node1] += repulsion * diff / distance
            
            # 计算吸引力（仅对相邻节点）
            for edge in graph.edges():
                node1, node2 = edge
                pos1 = np.array(positions[node1])
                pos2 = np.array(positions[node2])
                diff = pos2 - pos1
                distance = np.linalg.norm(diff)
                
                if distance > 0:
                    attraction = distance * distance / k
                    force_direction = diff / distance
                    forces[node1] += attraction * force_direction
                    forces[node2] -= attraction * force_direction
            
            # 更新位置
            cooling = temperature * (1 - iteration / iterations)
            for node in nodes:
                force_magnitude = np.linalg.norm(forces[node])
                if force_magnitude > 0:
                    displacement = forces[node] / force_magnitude * min(force_magnitude, cooling)
                    positions[node] = tuple(np.array(positions[node]) + displacement)
        
        return positions


class LayoutManager:
    """布局管理器"""
    
    def __init__(self):
        self.layouts = {
            'tree': TreeLayout(),
            'torus': TorusLayout(),
            'spring': OptimizedLayout('spring'),
            'circular': OptimizedLayout('circular'),
            'shell': OptimizedLayout('shell')
        }
    
    def get_layout(self, layout_type: str) -> BaseLayout:
        """获取指定类型的布局算法"""
        return self.layouts.get(layout_type, self.layouts['spring'])
    
    def calculate_optimal_layout(self, graph, topology_type=None):
        """根据拓扑类型选择最优布局"""
        if topology_type == 'tree':
            return self.layouts['tree'].calculate_positions(graph)
        elif topology_type == 'torus':
            return self.layouts['torus'].calculate_positions(graph)
        else:
            # 根据图的特征自动选择
            num_nodes = len(graph.nodes)
            avg_degree = sum(dict(graph.degree()).values()) / num_nodes
            
            if avg_degree < 2.5:  # 稀疏图，可能是树
                return self.layouts['tree'].calculate_positions(graph)
            elif avg_degree > 3.5:  # 密集图，可能是网格
                return self.layouts['spring'].calculate_positions(graph)
            else:
                return self.layouts['spring'].calculate_positions(graph)