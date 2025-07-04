import configparser
import numpy as np
import networkx as nx

# import pygraphviz as pgv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from itertools import combinations


def create_crossring_adjacency_matrix(num_nodes: int, num_cols: int) -> np.ndarray:
    """
    创建 CrossRing 拓扑邻接矩阵 (更通用的环绕连接实现)
    不影响原 op.create_adjacency_matrix 接口
    """
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    num_rows = num_nodes // num_cols
    for i in range(num_nodes):
        row = i // num_cols
        col = i % num_cols
        # 水平连接
        if col < num_cols - 1:
            adjacency_matrix[i][i + 1] = 1
        if col > 0:
            adjacency_matrix[i][i - 1] = 1
        # 垂直连接
        if row < num_rows - 1:
            adjacency_matrix[i][i + num_cols] = 1
        if row > 0:
            adjacency_matrix[i][i - num_cols] = 1
    return adjacency_matrix


def create_ring_adjacency_matrix(num_nodes: int) -> np.ndarray:
    """
    创建标准 Ring 拓扑邻接矩阵
    """
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    for i in range(num_nodes):
        next_node = (i + 1) % num_nodes
        prev_node = (i - 1) % num_nodes
        adjacency_matrix[i][next_node] = 1
        adjacency_matrix[i][prev_node] = 1
    return adjacency_matrix


def create_adjacency_matrix_adv(topology_type: str, num_nodes: int, rows: int = 0) -> np.ndarray:
    """
    推荐使用的扩展邻接矩阵生成接口：
    对于 Ring_* 和 CrossRing 拓扑，调用专用实现；
    其他类型则调用原始模块 op.create_adjacency_matrix。
    """
    if topology_type.startswith("Ring"):
        # Ring_N 格式: "Ring_<节点数>"
        try:
            ring_nodes = int(topology_type.split("_")[1])
        except (IndexError, ValueError):
            ring_nodes = num_nodes
        return create_ring_adjacency_matrix(ring_nodes)
    elif topology_type == "CrossRing":
        return create_crossring_adjacency_matrix(num_nodes, rows)
    else:
        # 调用原始模块的 create_adjacency_matrix，不修改其内部逻辑
        return create_adjacency_matrix(topology_type, num_nodes, rows)


def plot_adjacency_matrix(adjacency_matrix):
    """
    绘制邻接矩阵的函数。

    参数:
    adjacency_matrix (numpy.ndarray): 要绘制的邻接矩阵。
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(adjacency_matrix, cmap="Greys", interpolation="nearest")
    plt.colorbar(label="Edge Weight")
    plt.title("Adjacency Matrix", fontsize=16)
    plt.xlabel("Nodes", fontsize=14)
    plt.ylabel("Nodes", fontsize=14)

    num_nodes = adjacency_matrix.shape[0]

    # 选择每隔一个节点显示标签
    tick_indices = np.arange(1, num_nodes, step=8)  # 可以调整步长以控制显示的标签数量
    tick_labels = [f"Node {i}" for i in range(num_nodes)]

    plt.xticks(ticks=tick_indices, fontsize=10, rotation=45)
    plt.yticks(ticks=tick_indices, fontsize=10)

    plt.grid(False)  # 关闭网格
    plt.tight_layout()  # 自动调整子图参数以给标签留出足够的空间
    plt.show()


# Generate adjacency matrix under different topological structures.
def create_adjacency_matrix(topology_type, num_nodes, rows=0):
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    if topology_type == "Half Ring":
        for node in range(num_nodes):
            if node == 0:
                adjacency_matrix[node][node + 1] = 1
            elif node == num_nodes - 1:
                adjacency_matrix[node][node - 1] = 1
            else:
                adjacency_matrix[node][(node + 1) % num_nodes] = 1
                adjacency_matrix[node][(node - 1) % num_nodes] = 1
    elif topology_type == "Full Ring":
        for node in range(num_nodes):
            adjacency_matrix[node][(node + 1) % num_nodes] = 1
            adjacency_matrix[node][(node - 1) % num_nodes] = 1
    elif topology_type == "Star":
        # adjacency_matrix = np.zeros((num_nodes + 1, num_nodes + 1), dtype=int)
        for node in range(1, num_nodes):
            adjacency_matrix[0][node] = 1
            adjacency_matrix[node][0] = 1
    elif topology_type == "Mesh":
        assert num_nodes % rows == 0, "This is not a valid 2D Mesh."
        cols = num_nodes // rows
        for node in range(num_nodes):
            node_row = node // rows
            node_col = node % rows
            if node_row > 0:
                adjacency_matrix[node][node - rows] = 1
                adjacency_matrix[node - rows][node] = 1
            if node_row < cols - 1:
                adjacency_matrix[node][node + rows] = 1
                adjacency_matrix[node + rows][node] = 1
            if node_col > 0:
                adjacency_matrix[node][node - 1] = 1
                adjacency_matrix[node - 1][node] = 1
            if node_col < rows - 1:
                adjacency_matrix[node][node + 1] = 1
                adjacency_matrix[node + 1][node] = 1
    elif topology_type == "CrossRing":
        for node in range(num_nodes):
            if (node // rows) % 2 == 0:
                if node < rows:
                    adjacency_matrix[node][node + rows * 2] = 1
                elif node >= num_nodes - rows * 2:
                    adjacency_matrix[node][node - rows * 2] = 1
                else:
                    adjacency_matrix[node][node + rows * 2] = 1
                    adjacency_matrix[node][node - rows * 2] = 1
            else:
                # connect vertically backward
                adjacency_matrix[node][node - rows] = 1
                # only add horizontal neighbors if more than one column
                if rows > 1:
                    # left neighbor (column +1)
                    if node % rows == 0:
                        if node + 1 < num_nodes:
                            adjacency_matrix[node][node + 1] = 1
                    # right neighbor (column -1)
                    elif node % rows == rows - 1:
                        if node - 1 >= 0:
                            adjacency_matrix[node][node - 1] = 1
                    else:
                        if node + 1 < num_nodes:
                            adjacency_matrix[node][node + 1] = 1
                        if node - 1 >= 0:
                            adjacency_matrix[node][node - 1] = 1
    elif topology_type == "Torus":
        assert num_nodes % rows == 0, "This is not a valid 2D Torus."
        cols = num_nodes // rows
        for node in range(num_nodes):
            node_row = node // rows
            node_col = node % rows
            up = ((node_row - 1) % cols) * rows + node_col
            down = ((node_row + 1) % cols) * rows + node_col
            left = node_row * rows + (node_col - 1) % rows
            right = node_row * rows + (node_col + 1) % rows
            adjacency_matrix[node][up] = 1
            adjacency_matrix[node][down] = 1
            adjacency_matrix[node][left] = 1
            adjacency_matrix[node][right] = 1
    elif topology_type == "Binary Tree":
        num_all = 2 * num_nodes - 1
        adjacency_matrix = np.zeros((num_all, num_all), dtype=int)
        for node in range(num_all - num_nodes):
            left_child = 2 * node + 1
            right_child = 2 * node + 2
            if left_child < num_all:
                adjacency_matrix[node][left_child] = 1
                adjacency_matrix[left_child][node] = 1
            if right_child < num_all:
                adjacency_matrix[node][right_child] = 1
                adjacency_matrix[right_child][node] = 1
    else:
        raise ValueError("Topology Error: ", topology_type)
    return adjacency_matrix


# Find the shortest path between all nodes in the graph based on the given adjacency matrix.
def find_shortest_paths(adj_matrix):
    G = nx.DiGraph()
    num_nodes = len(adj_matrix)

    # 构建图
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i][j] == 1:
                G.add_edge(i, j)

    shortest_paths = {}
    hop_count = 0
    max_hop = 0
    count = 0

    # 计算最短路径和性能指标
    for node in G.nodes():
        shortest_paths[node] = {}
        for target_node in G.nodes():
            if node == target_node:
                shortest_paths[node][target_node] = [node]  # 自己到自己
                continue

            try:
                shortest_path = nx.shortest_path(G, source=node, target=target_node)
                shortest_paths[node][target_node] = shortest_path

                hop = len(shortest_path) - 1
                hop_count += hop
                count += 1
                max_hop = max(max_hop, hop)

            except nx.NetworkXNoPath:
                shortest_paths[node][target_node] = []

    # 计算平均跳数
    avg_hop = hop_count / count if count > 0 else 0
    # visualize_paths(G, shortest_paths)
    return shortest_paths


def visualize_paths(G, shortest_paths):
    pos = nx.spring_layout(G)  # 计算节点位置
    plt.figure(figsize=(10, 8))

    # 绘制图的边和节点
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=700)
    nx.draw_networkx_edges(G, pos, arrowstyle="-|>", arrowsize=20, edge_color="gray")
    nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif")

    # 绘制最短路径
    for source, targets in shortest_paths.items():
        for target, path in targets.items():
            if path and len(path) > 1:  # 只绘制有效路径
                path_edges = list(zip(path[:-1], path[1:]))  # 获取路径的边
                nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color="red", width=2)

    print("he")
    plt.title("Directed Graph with Shortest Paths Highlighted", fontsize=16)
    plt.axis("off")  # 关闭坐标轴
    plt.show()


def xy_route_mesh(num_nodes, rows):
    xy_paths = {}
    for source in range(num_nodes):
        xy_paths[source] = {}
        for target in range(num_nodes):
            if source == target:
                xy_paths[source][target] = [target]
            else:
                source_x, source_y = source % rows, source // rows
                target_x, target_y = target % rows, target // rows

                path = [source]
                current_x, current_y = source_x, source_y

                while current_x != target_x:
                    if current_x < target_x:
                        current_x += 1
                    else:
                        current_x -= 1
                    path.append(current_y * rows + current_x)

                while current_y != target_y:
                    if current_y < target_y:
                        current_y += 1
                    else:
                        current_y -= 1
                    path.append(current_y * rows + current_x)
                xy_paths[source][target] = path

    return xy_paths


def throughput_cal(topology, routes, num_nodes, rows, num_ddr, num_sdma, num_l2m, num_gdma):
    ddr_bandwidth = 76
    sdma_bandwidth = 76
    l2m_bandwidth = 100
    nodes = list(range(num_nodes))
    # #CrossRing
    ddr_send_placements = [(6, 8, 15, 19, 25, 29, 45, 49)]
    sdma_send_placements = [(15, 18, 25, 28, 35, 38, 45, 48)]
    # sdma_nodes = [5, 6, 7, 8, 9, 15, 18, 19, 25, 28, 29, 35, 38, 39, 45, 48, 49]
    # sdma_send_placements = list(combinations(sdma_nodes, num_sdma))
    l2m_send_placements = [(16, 17, 26, 27, 36, 37, 46, 47)]
    gdma_send_placements = [(16, 17, 26, 27, 36, 37, 46, 47)]
    # ddr_send_placements =[(7, 9, 18, 23, 30, 35, 42, 47)]
    # sdma_send_placements =[(6, 8, 10, 11, 43, 44, 45, 46)]
    # # sdma_nodes = [6, 7, 8, 9, 10, 11, 18, 23, 30, 35, 42, 43, 44, 45, 46, 47]
    # # sdma_send_placements = list(combinations(sdma_nodes, num_sdma))
    # l2m_send_placements =[(19, 20, 21, 22, 31, 32, 33, 34)]
    # gdma_send_placements =[(19, 20, 21, 22, 31, 32, 33, 34)]
    # #mesh
    # ddr_send_placements =[(1, 3, 5, 9, 10, 14, 20, 24)]
    # # sdma_send_placements = [(5, 8, 10, 13, 15, 18, 20, 23)]
    # sdma_send_placements = list(combinations(nodes, num_sdma))
    # # sdma_send_placements = [(6, 7, 11, 12, 16, 17, 21, 22)]
    # l2m_send_placements = [(6, 7, 11, 12, 16, 17, 21, 22)]
    # gdma_send_placements = [(6, 7, 11, 12, 16, 17, 21, 22)]
    # #tree
    # ddr_send_placements =[(7, 8, 9, 10, 11, 12, 13, 14)]
    # sdma_send_placements = [(7, 8, 9, 10, 11, 12, 13, 14)]
    # l2m_send_placements = [(7, 8, 9, 10, 11, 12, 13, 14)]
    # gdma_send_placements = [(7, 8, 9, 10, 11, 12, 13, 14)]
    # #star
    # ddr_send_placements =[(1, 2, 3, 4, 5, 6, 7, 8)]
    # sdma_send_placements = [(1, 2, 3, 4, 5, 6, 7, 8)]
    # l2m_send_placements = [(1, 2, 3, 4, 5, 6, 7, 8)]
    # gdma_send_placements = [(1, 2, 3, 4, 5, 6, 7, 8)]
    # #others
    # ddr_send_placements = list(combinations(nodes, num_ddr))
    # sdma_send_placements = list(combinations(nodes, num_sdma))
    # l2m_send_placements = list(combinations(nodes, num_l2m))
    # gdma_send_placements = list(combinations(nodes, num_gdma))
    placements = []
    for ddr_send_placement in ddr_send_placements:
        for sdma_send_placement in sdma_send_placements:
            for l2m_send_placement in l2m_send_placements:
                for gdma_send_placement in gdma_send_placements:
                    placements.append((ddr_send_placement, sdma_send_placement, l2m_send_placement, gdma_send_placement))
    optimal_throughput = []
    optimal_placement = []
    flow = []
    back_flow = []
    max_flow = float("inf")
    for ddr_send_placement, sdma_send_placement, l2m_send_placement, gdma_send_placement in placements:
        if topology == "CrossRing":
            ddr_recv_placement = tuple(x - rows for x in ddr_send_placement)
            sdma_recv_placement = tuple(x - rows for x in sdma_send_placement)
            l2m_recv_placement = tuple(x - rows for x in l2m_send_placement)
            gdma_recv_placement = tuple(x - rows for x in gdma_send_placement)
        else:
            ddr_recv_placement = ddr_send_placement
            sdma_recv_placement = sdma_send_placement
            l2m_recv_placement = l2m_send_placement
            gdma_recv_placement = gdma_send_placement

        throughput = np.zeros((num_nodes, num_nodes), dtype=float)
        # ddr->sdma
        for ddr_node in ddr_send_placement:
            for sdma_node in sdma_recv_placement:
                src_node, dst_node = ddr_node, sdma_node
                # if src_node != dst_node:
                if (src_node != dst_node) and (src_node - dst_node != rows):
                    route = routes[src_node][dst_node]
                    for i in range(len(route) - 1):
                        throughput[route[i]][route[i + 1]] += ddr_bandwidth / num_sdma
        # sdma->l2m
        for sdma_node in sdma_send_placement:
            for l2m_node in l2m_recv_placement:
                src_node, dst_node = sdma_node, l2m_node
                # if src_node != dst_node:
                if (src_node != dst_node) and (src_node - dst_node != rows):
                    route = routes[src_node][dst_node]
                    for i in range(len(route) - 1):
                        throughput[route[i]][route[i + 1]] += sdma_bandwidth / num_l2m
        # l2m->gdma
        for l2m_node in l2m_send_placement:
            for gdma_node in gdma_recv_placement:
                src_node, dst_node = l2m_node, gdma_node
                # if src_node != dst_node:
                if (src_node != dst_node) and (src_node - dst_node != rows):
                    route = routes[src_node][dst_node]
                    for i in range(len(route) - 1):
                        throughput[route[i]][route[i + 1]] += l2m_bandwidth / num_gdma

        # back_throughput = np.zeros((num_nodes, num_nodes), dtype=float)
        # #gdma->l2m
        # for gdma_node in gdma_send_placement:
        #     for l2m_node in l2m_recv_placement:
        #         src_node, dst_node = gdma_node, l2m_node
        #         if src_node != dst_node:
        #             route = routes[src_node][dst_node]
        #             for i in range(len(route) - 1):
        #                 back_throughput[route[i]][route[i + 1]] += (l2m_bandwidth / num_l2m)
        # #l2m->sdma
        # for l2m_node in l2m_send_placement:
        #     for sdma_node in sdma_recv_placement:
        #         src_node, dst_node = l2m_node, sdma_node
        #         if src_node != dst_node:
        #             route = routes[src_node][dst_node]
        #             for i in range(len(route) - 1):
        #                 back_throughput[route[i]][route[i + 1]] += (sdma_bandwidth / num_l2m)
        # #sdma->ddr
        # for sdma_node in sdma_send_placement:
        #     for ddr_node in ddr_recv_placement:
        #         src_node, dst_node = sdma_node, ddr_node
        #         if src_node != dst_node:
        #             route = routes[src_node][dst_node]
        #             for i in range(len(route) - 1):
        #                 back_throughput[route[i]][route[i + 1]] += (ddr_bandwidth / num_sdma)

        flow.append(np.amax(throughput))
        # back_flow.append(np.amax(back_throughput))
        # current_flow = 1.0 * np.amax(throughput) + 0 * np.amax(back_throughput)
        current_flow = 1.0 * np.amax(throughput)
        if current_flow < max_flow:
            max_flow = current_flow
            optimal_throughput = throughput
            optimal_placement = [ddr_send_placement, sdma_send_placement, l2m_send_placement, gdma_send_placement]
    return optimal_throughput, optimal_placement


def data_analysis(optimal_throughput, num_nodes):
    mat = np.array(optimal_throughput)
    non_zero_elements = mat[mat != 0]
    max_value = non_zero_elements.max() if non_zero_elements.size > 0 else None
    min_value = non_zero_elements.min() if non_zero_elements.size > 0 else None
    mean_value = non_zero_elements.mean() if non_zero_elements.size > 0 else None
    variance_value = non_zero_elements.var() if non_zero_elements.size > 0 else None
    print(f"Max: {max_value}, Min: {min_value}, Mean: {mean_value}, Variance: {variance_value}")

    non_zero_throughput = {}
    for i in range(num_nodes):
        for j in range(num_nodes):
            if optimal_throughput[i][j] != 0:
                non_zero_throughput[(i, j)] = optimal_throughput[i][j]
    return non_zero_throughput


def visualize_graph(num_nodes, adjacency_matrix, weight_matrix, placement):
    # Draw a heat map
    fontdict = {"family": "Times New Roman", "color": "black", "weight": "normal", "size": 5}
    plt.figure(figsize=(20, 8))
    plt.imshow(weight_matrix, cmap="Reds", interpolation="nearest")
    for i in range(len(weight_matrix)):
        for j in range(len(weight_matrix[0])):
            plt.text(j, i, f"{weight_matrix[i, j]:.2f}", fontdict, ha="center", va="center")
    colorbar = plt.colorbar(pad=0.02)
    colorbar.set_label("Flow", fontproperties="Times New Roman")
    plt.title("Flow Distribution Heatmap", fontdict={"family": "Times New Roman", "color": "black", "weight": "bold", "size": 14})
    plt.xlabel("Node ID", fontdict)
    plt.ylabel("Node ID", fontdict)

    info_text = (
        f"Number of Nodes: {num_nodes}\n"
        f"DDR Optimal Placement: {placement[0]}\n"
        f"SDMA Optimal Placement: {placement[1]}\n"
        f"L2M Optimal Placement: {placement[2]}\n"
        f"GDMA Optimal Placement: {placement[3]}"
    )
    plt.text(
        len(weight_matrix) + 5,
        0,
        info_text,
        fontsize=10,
        fontproperties="Times New Roman",
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"),
    )
    save_path = "heatmap.png"
    plt.savefig(save_path, dpi=800)
    plt.show(block=True)

    # Draw graph
    G = nx.DiGraph(rankdir="LR")
    G.add_nodes_from(range(num_nodes), fontname="Times New Roman")
    for i in range(num_nodes):
        for j in range(num_nodes):
            weight = weight_matrix[i][j]
            if adjacency_matrix[i][j] == 1:
                G.add_edge(i, j, label=weight, fontname="Times New Roman")

    nx.drawing.nx_agraph.write_dot(G, "graph.dot")


def main():
    # config = configparser.ConfigParser()
    # config.read("config.ini")

    # num_nodes = int(config["Parameters"]["num_nodes"])
    # rows = int(config["Parameters"]["rows"])
    # num_ddr = int(config["Parameters"]["num_ddr"])
    # num_sdma = int(config["Parameters"]["num_sdma"])
    # num_l2m = int(config["Parameters"]["num_l2m"])
    # num_gdma = int(config["Parameters"]["num_gdma"])
    # topology = config["Parameters"]["topology"]
    num_nodes = 128
    topology = "CrossRing"
    rows = 8
    num_ddr = 64
    num_sdma = 64
    num_l2m = 64
    num_gdma = 64

    adjacency_matrix = create_adjacency_matrix(topology, num_nodes, rows)
    # np.savetxt('data.txt', adjacency_matrix, fmt='%.1f')
    routes = find_shortest_paths(adjacency_matrix)
    # routes = xy_route_mesh(num_nodes, rows)
    # if topology == "Star":
    #     num_nodes += 1
    if topology == "Binary Tree":
        num_nodes = 2 * num_nodes - 1
    optimal_throughput, optimal_placement = throughput_cal(topology, routes, num_nodes, rows, num_ddr, num_sdma, num_l2m, num_gdma)
    non_zero_throughput = data_analysis(optimal_throughput, num_nodes)
    print(non_zero_throughput)
    visualize_graph(num_nodes, adjacency_matrix, optimal_throughput, optimal_placement)


if __name__ == "__main__":
    main()
