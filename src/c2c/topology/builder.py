from src.c2c.topology.graph import TopologyGraph
from src.c2c.topology.base import BaseNode, BaseLink
from typing import Dict, Any


class TopologyBuilder:
    """拓扑构建器"""

    def __init__(self, topology_id: str):
        self._topology_graph = TopologyGraph(topology_id)
        self._nodes: Dict[str, BaseNode] = {}
        self._links: Dict[str, BaseLink] = {}

    def add_node(self, node: BaseNode):
        """添加节点到构建器"""
        if node.node_id in self._nodes:
            raise ValueError(f"Node with ID {node.node_id} already exists.")
        self._nodes[node.node_id] = node
        self._topology_graph.add_node(node)

    def add_link(self, link: BaseLink):
        """添加链路到构建器"""
        if link.link_id in self._links:
            raise ValueError(f"Link with ID {link.link_id} already exists.")
        # Ensure endpoints exist before adding link
        if link.endpoint_a.node_id not in self._nodes or link.endpoint_b.node_id not in self._nodes:
            raise ValueError(f"Endpoints for link {link.link_id} not found in builder.")
        self._links[link.link_id] = link
        self._topology_graph.add_link(link)

    def build(self) -> TopologyGraph:
        """构建并返回拓扑图"""
        return self._topology_graph

    def get_node(self, node_id: str) -> BaseNode:
        """获取已添加的节点"""
        return self._nodes.get(node_id)

    def get_link(self, link_id: str) -> BaseLink:
        """获取已添加的链路"""
        return self._links.get(link_id)
