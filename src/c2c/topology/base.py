from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class BaseNode(ABC):
    """节点抽象基类"""
    def __init__(self, node_id: str, node_type: str, properties: Dict[str, Any] = None):
        self._node_id = node_id
        self._node_type = node_type
        self._properties = properties if properties is not None else {}
        self._neighbors: Dict[str, 'BaseNode'] = {} # node_id -> BaseNode

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def node_type(self) -> str:
        return self._node_type

    @property
    def properties(self) -> Dict[str, Any]:
        return self._properties

    def add_property(self, key: str, value: Any):
        """添加或更新节点属性"""
        self._properties[key] = value

    def get_property(self, key: str, default: Any = None) -> Any:
        """获取节点属性"""
        return self._properties.get(key, default)

    def add_neighbor(self, neighbor_node: 'BaseNode'):
        """添加邻居节点"""
        if neighbor_node.node_id not in self._neighbors:
            self._neighbors[neighbor_node.node_id] = neighbor_node

    def get_neighbors(self) -> Dict[str, 'BaseNode']:
        """获取所有邻居节点"""
        return self._neighbors

    @abstractmethod
    def send_message(self, destination_node: 'BaseNode', message: Any):
        """发送消息接口"""
        pass

    @abstractmethod
    def receive_message(self, sender_node: 'BaseNode', message: Any):
        """接收消息接口"""
        pass

    def __repr__(self):
        return f"<{self.__class__.__name__}(id='{self.node_id}', type='{self.node_type}')>"

class BaseLink(ABC):
    """链路抽象基类"""
    def __init__(self, link_id: str, endpoint_a: BaseNode, endpoint_b: BaseNode, link_type: str, properties: Dict[str, Any] = None):
        self._link_id = link_id
        self._endpoint_a = endpoint_a
        self._endpoint_b = endpoint_b
        self._link_type = link_type
        self._properties = properties if properties is not None else {}

    @property
    def link_id(self) -> str:
        return self._link_id

    @property
    def endpoint_a(self) -> BaseNode:
        return self._endpoint_a

    @property
    def endpoint_b(self) -> BaseNode:
        return self._endpoint_b

    @property
    def link_type(self) -> str:
        return self._link_type

    @property
    def properties(self) -> Dict[str, Any]:
        return self._properties

    def add_property(self, key: str, value: Any):
        """添加或更新链路属性"""
        self._properties[key] = value

    def get_property(self, key: str, default: Any = None) -> Any:
        """获取链路属性"""
        return self._properties.get(key, default)

    @property
    @abstractmethod
    def bandwidth(self) -> float:
        """获取链路带宽 (Gbps)"""
        pass

    @property
    @abstractmethod
    def latency(self) -> float:
        """获取链路延迟 (ns)"""
        pass

    @property
    @abstractmethod
    def status(self) -> str:
        """获取链路状态"""
        pass

    def __repr__(self):
        return f"<{self.__class__.__name__}(id='{self.link_id}', type='{self.link_type}', {self.endpoint_a.node_id}<->{self.endpoint_b.node_id})>"

class BaseTopology(ABC):
    """拓扑抽象基类"""
    def __init__(self, topology_id: str):
        self._topology_id = topology_id
        self._nodes: Dict[str, BaseNode] = {} # node_id -> BaseNode
        self._links: Dict[str, BaseLink] = {} # link_id -> BaseLink

    @property
    def topology_id(self) -> str:
        return self._topology_id

    def add_node(self, node: BaseNode):
        """添加节点到拓扑"""
        if node.node_id not in self._nodes:
            self._nodes[node.node_id] = node

    def get_node(self, node_id: str) -> Optional[BaseNode]:
        """根据ID获取节点"""
        return self._nodes.get(node_id)

    def get_all_nodes(self) -> Dict[str, BaseNode]:
        """获取所有节点"""
        return self._nodes

    def add_link(self, link: BaseLink):
        """添加链路到拓扑"""
        if link.link_id not in self._links:
            self._links[link.link_id] = link
            # 确保链路两端的节点互为邻居
            link.endpoint_a.add_neighbor(link.endpoint_b)
            link.endpoint_b.add_neighbor(link.endpoint_a)

    def get_link(self, link_id: str) -> Optional[BaseLink]:
        """根据ID获取链路"""
        return self._links.get(link_id)

    def get_all_links(self) -> Dict[str, BaseLink]:
        """获取所有链路"""
        return self._links

    @abstractmethod
    def find_path(self, start_node_id: str, end_node_id: str) -> List[str]:
        """查找从起始节点到目标节点的路径 (节点ID列表)"""
        pass

    @abstractmethod
    def get_topology_statistics(self) -> Dict[str, Any]:
        """获取拓扑统计信息"""
        pass

    def __repr__(self):
        return f"<{self.__class__.__name__}(id='{self.topology_id}', nodes={len(self._nodes)}, links={len(self._links)})>"
