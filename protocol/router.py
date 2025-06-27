from typing import List, Dict, Any
from topology.graph import TopologyGraph
from topology.base import BaseNode
from protocol.cdma import CDMAMessage

class Router:
    """负责消息路由的路由器"""
    def __init__(self, router_id: str, topology_graph: TopologyGraph):
        self._router_id = router_id
        self._topology_graph = topology_graph
        self._routing_table: Dict[str, str] = {}

    def update_routing_table(self, destination_node_id: str, next_hop_node_id: str):
        """更新路由表"""
        self._routing_table[destination_node_id] = next_hop_node_id

    def get_next_hop(self, destination_node_id: str) -> str | None:
        """获取下一跳节点ID"""
        return self._routing_table.get(destination_node_id)

    def route_message(self, message: CDMAMessage, current_node: BaseNode) -> BaseNode | None:
        """路由消息到下一跳节点"""
        destination_id = message.destination_id
        
        # If the message is for the current node, it's the final destination
        if destination_id == current_node.node_id:
            print(f"Router {self._router_id}: Message for {destination_id} reached its destination.")
            return current_node

        # Try to find a path using the topology graph
        path = self._topology_graph.find_path(current_node.node_id, destination_id)
        if len(path) > 1:
            next_hop_id = path[1] # The next node in the shortest path
            next_hop_node = self._topology_graph.get_node(next_hop_id)
            if next_hop_node:
                print(f"Router {self._router_id}: Routing message from {current_node.node_id} to {destination_id} via {next_hop_id}")
                return next_hop_node
            else:
                print(f"Router {self._router_id}: Next hop node {next_hop_id} not found in topology.")
                return None
        else:
            print(f"Router {self._router_id}: No path found from {current_node.node_id} to {destination_id}.")
            return None

    def calculate_and_set_routes(self, source_node_id: str):
        """为所有可达节点计算并设置最短路径路由"""
        all_nodes = self._topology_graph.get_all_nodes()
        for dest_id, _ in all_nodes.items():
            if source_node_id == dest_id:
                continue
            path = self._topology_graph.find_path(source_node_id, dest_id)
            if len(path) > 1:
                self.update_routing_table(dest_id, path[1])
            else:
                print(f"Router {self._router_id}: No direct path from {source_node_id} to {dest_id}")

    def __repr__(self):
        return f"<Router(id='{self._router_id}')>"
