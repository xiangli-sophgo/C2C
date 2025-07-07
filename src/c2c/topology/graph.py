import networkx as nx
from typing import Dict, Any, List
from src.c2c.topology.base import BaseNode, BaseLink, BaseTopology


class TopologyGraph(BaseTopology):
    """基于NetworkX的拓扑图表示"""

    def __init__(self, topology_id: str):
        super().__init__(topology_id)
        self._graph = nx.Graph()

    def add_node(self, node: BaseNode):
        """添加节点到拓扑图"""
        super().add_node(node)
        self._graph.add_node(node.node_id, obj=node, type=node.node_type, properties=node.properties)

    def add_link(self, link: BaseLink):
        """添加链路到拓扑图"""
        super().add_link(link)
        self._graph.add_edge(link.endpoint_a.node_id, link.endpoint_b.node_id, obj=link, type=link.link_type, properties=link.properties, bandwidth=link.bandwidth, latency=link.latency)

    def remove_node(self, node_id: str):
        """从拓扑图移除节点"""
        if node_id in self._nodes:
            del self._nodes[node_id]
            self._graph.remove_node(node_id)

    def remove_link(self, link_id: str):
        """从拓扑图移除链路"""
        if link_id in self._links:
            link = self._links[link_id]
            self._graph.remove_edge(link.endpoint_a.node_id, link.endpoint_b.node_id)
            del self._links[link_id]

    def find_path(self, start_node_id: str, end_node_id: str, weight: str = None) -> List[str]:
        """查找从起始节点到目标节点的路径 (节点ID列表)
        Args:
            start_node_id: 起始节点ID
            end_node_id: 目标节点ID
            weight: 路径权重，可选 'bandwidth' 或 'latency'，默认为最短路径（跳数）
        Returns:
            包含节点ID的列表，表示路径。如果不存在路径，则返回空列表。
        """
        try:
            if weight == "latency":
                return nx.shortest_path(self._graph, source=start_node_id, target=end_node_id, weight="latency")
            elif weight == "bandwidth":
                # For bandwidth, we want to maximize, so we can use 1/bandwidth as weight for shortest path
                # Or, more simply, find path with max product of bandwidths (not directly supported by shortest_path)
                # For now, we'll just use shortest path by hops if bandwidth is requested as weight
                return nx.shortest_path(self._graph, source=start_node_id, target=end_node_id)
            else:
                return nx.shortest_path(self._graph, source=start_node_id, target=end_node_id)
        except nx.NetworkXNoPath:
            return []
        except nx.NodeNotFound:
            print(f"Error: One or both nodes ({start_node_id}, {end_node_id}) not found in the graph.")
            return []

    def get_topology_statistics(self) -> Dict[str, Any]:
        """获取拓扑统计信息"""
        return {
            "num_nodes": self._graph.number_of_nodes(),
            "num_links": self._graph.number_of_edges(),
            "is_connected": nx.is_connected(self._graph) if self._graph.number_of_nodes() > 0 else False,
            "average_degree": sum(dict(self._graph.degree()).values()) / self._graph.number_of_nodes() if self._graph.number_of_nodes() > 0 else 0,
        }

    def draw_topology(self):
        """可视化拓扑图 (需要matplotlib)"""
        try:
            import matplotlib.pyplot as plt

            pos = nx.spring_layout(self._graph)  # positions for all nodes
            nx.draw(self._graph, pos, with_labels=True, node_color="skyblue", node_size=2000, edge_cmap=plt.cm.Blues, font_size=10)
            edge_labels = nx.get_edge_attributes(self._graph, "type")
            nx.draw_edge_labels(self._graph, pos, edge_labels=edge_labels)
            plt.title(f"Topology: {self.topology_id}")
            plt.show()
        except ImportError:
            print("Matplotlib not found. Please install it to draw the topology: pip install matplotlib")
