"""
邻接矩阵生成和验证工具。

本模块提供用于生成和验证各种网络拓扑邻接矩阵的工具函数。
"""

import numpy as np
from typing import List, Tuple, Optional, Set
from collections import deque

from src.noc.utils.types import AdjacencyMatrix, ValidationResult


def create_crossring_adjacency_matrix(num_rows: int, num_cols: int) -> AdjacencyMatrix:
    """
    创建CrossRing拓扑的邻接矩阵（实现为Mesh）。

    Mesh拓扑特点：
    - 节点按二维网格排列（num_rows × num_cols）
    - 节点与其上下左右的邻居连接（如果存在）
    - 边缘节点没有环形回绕连接

    Args:
        num_rows: 行数
        num_cols: 列数

    Returns:
        邻接矩阵

    Raises:
        ValueError: 如果拓扑参数无效
    """
    if num_rows < 1 or num_cols < 1:
        raise ValueError(f"拓扑至少需要1×1节点，给定: {num_rows}×{num_cols}")

    num_nodes = num_rows * num_cols
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    for i in range(num_nodes):
        row, col = divmod(i, num_cols)

        # 水平连接（左右邻居）
        if col > 0:
            left_neighbor = i - 1
            adj_matrix[i, left_neighbor] = 1
        if col < num_cols - 1:
            right_neighbor = i + 1
            adj_matrix[i, right_neighbor] = 1

        # 垂直连接（上下邻居）
        if row > 0:
            up_neighbor = i - num_cols
            adj_matrix[i, up_neighbor] = 1
        if row < num_rows - 1:
            down_neighbor = i + num_cols
            adj_matrix[i, down_neighbor] = 1

    return adj_matrix.tolist()


def validate_adjacency_matrix(adj_matrix: AdjacencyMatrix) -> ValidationResult:
    """
    验证邻接矩阵的有效性。

    Args:
        adj_matrix: 邻接矩阵

    Returns:
        ValidationResult: (是否有效, 错误消息)
    """
    if not adj_matrix:
        return False, "邻接矩阵不能为空"

    n = len(adj_matrix)

    # 检查矩阵是否为方阵
    for i, row in enumerate(adj_matrix):
        if len(row) != n:
            return False, f"邻接矩阵第{i}行长度不匹配，期望{n}，实际{len(row)}"

    # 检查矩阵元素是否为0或1
    for i, row in enumerate(adj_matrix):
        for j, val in enumerate(row):
            if val not in [0, 1]:
                return False, f"邻接矩阵元素({i},{j})必须为0或1，实际为{val}"

    # 检查对角线元素是否为0（不允许自环）
    for i in range(n):
        if adj_matrix[i][i] != 0:
            return False, f"邻接矩阵对角线元素({i},{i})必须为0，不允许自环"

    # 检查矩阵是否对称（无向图）
    for i in range(n):
        for j in range(n):
            if adj_matrix[i][j] != adj_matrix[j][i]:
                return False, f"邻接矩阵不对称，元素({i},{j})={adj_matrix[i][j]}，但({j},{i})={adj_matrix[j][i]}"

    return True, None


def check_connectivity(adj_matrix: AdjacencyMatrix) -> bool:
    """
    检查图的连通性。

    Args:
        adj_matrix: 邻接矩阵

    Returns:
        是否连通
    """
    if not adj_matrix:
        return False

    n = len(adj_matrix)
    if n == 0:
        return True

    # 使用BFS检查连通性
    visited = [False] * n
    queue = deque([0])
    visited[0] = True
    visited_count = 1

    while queue:
        node = queue.popleft()
        for neighbor in range(n):
            if adj_matrix[node][neighbor] == 1 and not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)
                visited_count += 1

    return visited_count == n


def analyze_node_degrees(adj_matrix: AdjacencyMatrix) -> Tuple[List[int], int, int, float]:
    """
    分析节点度数分布。

    Args:
        adj_matrix: 邻接矩阵

    Returns:
        度数列表，最小度数，最大度数，平均度数
    """
    if not adj_matrix:
        return [], 0, 0, 0.0

    degrees = []
    for i, row in enumerate(adj_matrix):
        degree = sum(row)
        degrees.append(degree)

    min_degree = min(degrees) if degrees else 0
    max_degree = max(degrees) if degrees else 0
    avg_degree = sum(degrees) / len(degrees) if degrees else 0.0

    return degrees, min_degree, max_degree, avg_degree


def get_node_neighbors(adj_matrix: AdjacencyMatrix, node_id: int) -> List[int]:
    """
    获取指定节点的邻居节点列表。

    Args:
        adj_matrix: 邻接矩阵
        node_id: 节点ID

    Returns:
        邻居节点ID列表

    Raises:
        ValueError: 如果节点ID无效
    """
    if not adj_matrix:
        raise ValueError("邻接矩阵不能为空")

    n = len(adj_matrix)
    if node_id < 0 or node_id >= n:
        raise ValueError(f"节点ID {node_id} 超出范围 [0, {n-1}]")

    neighbors = []
    for i, connected in enumerate(adj_matrix[node_id]):
        if connected == 1:
            neighbors.append(i)

    return neighbors


def calculate_graph_diameter(adj_matrix: AdjacencyMatrix) -> int:
    """
    计算图的直径（最大最短路径长度）。

    Args:
        adj_matrix: 邻接矩阵

    Returns:
        图的直径
    """
    if not adj_matrix:
        return 0

    n = len(adj_matrix)
    if n <= 1:
        return 0

    # 使用Floyd-Warshall算法计算所有节点对之间的最短距离
    dist = [[float("inf")] * n for _ in range(n)]

    # 初始化距离矩阵
    for i in range(n):
        dist[i][i] = 0
        for j in range(n):
            if adj_matrix[i][j] == 1:
                dist[i][j] = 1

    # Floyd-Warshall算法
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    # 计算直径
    diameter = 0
    for i in range(n):
        for j in range(n):
            if dist[i][j] != float("inf"):
                diameter = max(diameter, dist[i][j])

    return int(diameter)


def calculate_clustering_coefficient(adj_matrix: AdjacencyMatrix) -> float:
    """
    计算图的聚类系数。

    Args:
        adj_matrix: 邻接矩阵

    Returns:
        聚类系数
    """
    if not adj_matrix:
        return 0.0

    n = len(adj_matrix)
    if n <= 2:
        return 0.0

    total_coefficient = 0.0
    valid_nodes = 0

    for i in range(n):
        # 获取节点i的邻居
        neighbors = get_node_neighbors(adj_matrix, i)
        degree = len(neighbors)

        if degree < 2:
            continue

        # 计算邻居之间的边数
        edges_between_neighbors = 0
        for j in range(len(neighbors)):
            for k in range(j + 1, len(neighbors)):
                if adj_matrix[neighbors[j]][neighbors[k]] == 1:
                    edges_between_neighbors += 1

        # 计算节点i的聚类系数
        possible_edges = degree * (degree - 1) // 2
        node_coefficient = edges_between_neighbors / possible_edges
        total_coefficient += node_coefficient
        valid_nodes += 1

    return total_coefficient / valid_nodes if valid_nodes > 0 else 0.0


def export_adjacency_matrix(adj_matrix: AdjacencyMatrix, filename: str, format_type: str = "txt") -> None:
    """
    导出邻接矩阵到文件。

    Args:
        adj_matrix: 邻接矩阵
        filename: 文件名
        format_type: 导出格式（txt, csv, json）
    """
    if not adj_matrix:
        raise ValueError("邻接矩阵不能为空")

    if format_type == "txt":
        with open(filename, "w", encoding="utf-8") as f:
            for row in adj_matrix:
                f.write(" ".join(map(str, row)) + "\n")
    elif format_type == "csv":
        import csv

        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(adj_matrix)
    elif format_type == "json":
        import json

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(adj_matrix, f, ensure_ascii=False, indent=2)
    else:
        raise ValueError(f"不支持的导出格式: {format_type}")
