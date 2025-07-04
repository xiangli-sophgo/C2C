# -*- coding: utf-8 -*-
"""
SG2260E C2C 树状拓扑核心逻辑实现
"""
import math
from collections import deque, defaultdict

# --- 核心数据结构 (辅助定义) ---


class TreeNode:
    """树节点，可以是芯片或Switch"""

    def __init__(self, node_id, node_type, parent=None):
        self.node_id = node_id
        self.node_type = node_type  # 'chip' or 'switch'
        self.parent = parent
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)

    def __repr__(self):
        return f"TreeNode({self.node_id}, {self.node_type})"


# --- 1. 树状拓扑逻辑核心 ---


class TreeTopologyLogic:
    """
    树状拓扑核心算法逻辑
    基于PCIe Switch构建多级树状结构，支持最多128芯片
    """

    def calculate_tree_structure(self, chip_ids, switch_capacity=16):
        """
        计算最优树状结构
        输入: 芯片ID列表, Switch端口数
        输出: 树结构 (根节点), 节点字典

        核心逻辑:
        1. 根据芯片数量计算需要的Switch数量和层级
        2. 从叶子节点（芯片）开始，逐层向上构建
        3. 平衡树结构，最小化最大路径跳数
        """
        if not chip_ids:
            return None, {}

        chip_count = len(chip_ids)
        nodes = {f"chip_{cid}": TreeNode(f"chip_{cid}", "chip") for cid in chip_ids}

        current_level_nodes = list(nodes.values())
        switch_counter = 0

        while len(current_level_nodes) > 1:
            next_level_nodes = []
            # 将当前层的节点分组，每组最多switch_capacity个
            for i in range(0, len(current_level_nodes), switch_capacity):
                group = current_level_nodes[i : i + switch_capacity]

                # 创建父Switch节点
                switch_id = f"switch_{switch_counter}"
                parent_switch = TreeNode(switch_id, "switch")
                nodes[switch_id] = parent_switch
                switch_counter += 1

                # 连接子节点到父Switch
                for child_node in group:
                    parent_switch.add_child(child_node)
                    child_node.parent = parent_switch

                next_level_nodes.append(parent_switch)
            current_level_nodes = next_level_nodes

        root = current_level_nodes[0] if current_level_nodes else None
        return root, nodes

    def compute_routing_table(self, tree_root, all_nodes):
        """
        计算完整路由表
        输入: 树根节点, 所有节点的字典
        输出: 每对芯片间的路由路径字典

        核心逻辑:
        1. 使用LCA(最近公共祖先)算法计算任意两芯片间路径
        2. 路径是 src -> ... -> lca -> ... -> dst
        """
        routing_table = defaultdict(dict)
        chip_nodes = [node for node in all_nodes.values() if node.node_type == "chip"]

        for i in range(len(chip_nodes)):
            for j in range(i, len(chip_nodes)):
                src_node = chip_nodes[i]
                dst_node = chip_nodes[j]

                path = self._find_path_between(src_node, dst_node)

                routing_table[src_node.node_id][dst_node.node_id] = path
                # 反向路径
                routing_table[dst_node.node_id][src_node.node_id] = path[::-1]

        return dict(routing_table)

    def _find_path_between(self, src_node, dst_node):
        """辅助函数：找到两个节点之间的唯一路径"""
        if src_node == dst_node:
            return [src_node.node_id]

        # 找到LCA
        src_path_to_root = self._get_path_to_root(src_node)
        dst_path_to_root = self._get_path_to_root(dst_node)

        lca = None
        i, j = len(src_path_to_root) - 1, len(dst_path_to_root) - 1
        while i >= 0 and j >= 0 and src_path_to_root[i] == dst_path_to_root[j]:
            lca = src_path_to_root[i]
            i -= 1
            j -= 1

        # 构建路径: src -> lca -> dst
        path_up = [node.node_id for node in src_path_to_root if src_path_to_root.index(node) <= src_path_to_root.index(lca)]
        path_down = [node.node_id for node in dst_path_to_root if dst_path_to_root.index(node) < dst_path_to_root.index(lca)]

        return path_up + path_down[::-1]

    def _get_path_to_root(self, node):
        """辅助函数：获取从当前节点到根节点的路径"""
        path = []
        curr = node
        while curr:
            path.append(curr)
            curr = curr.parent
        return path

    def generate_c2c_sys_mapping(self, chip_id, tree_structure_nodes):
        """
        为指定芯片生成c2c_sys映射
        输入: 芯片ID, 树结构节点字典
        输出: 该芯片各c2c_sys的连接配置

        核心逻辑:
        1. 确定该芯片在树中的位置 (父节点是哪个Switch)
        2. 分配c2c_sys到不同方向(向上)
        3. 设置RC/EP模式 (芯片作为EP，连接到Switch的RC)
        """
        chip_node_id = f"chip_{chip_id}"
        chip_node = tree_structure_nodes.get(chip_node_id)
        if not chip_node or not chip_node.parent:
            return {}  # 孤立芯片或根

        mapping = {}
        # 假设c2c_sys0用于上行连接
        mapping["c2c_sys0"] = {"direction": "up", "connects_to": chip_node.parent.node_id, "mode": "EP"}  # Endpoint连接到Switch
        # 其他c2c_sys可用于同级或保留
        # ...
        return mapping

    def calculate_atu_assignment(self, src_chip_id, all_chip_ids, tree_structure_nodes):
        """
        计算ATU分配策略
        输入: 源芯片ID, 所有目标芯片ID列表, 树结构
        输出: ATU配置表 (简化的)

        核心逻辑:
        1. 为每个目标芯片分配唯一ATU (0-127)
        2. 根据路由路径设置地址映射 (此处简化)
        """
        atu_table = {}
        # 假设ATU ID从0开始分配给除自己外的所有芯片
        atu_id = 0
        for target_chip_id in all_chip_ids:
            if src_chip_id == target_chip_id:
                continue
            if atu_id >= 128:
                break  # 超出ATU资源

            atu_table[f"OB_ATU_{atu_id}"] = {
                "target_chip_id": target_chip_id,
                "enabled": True,
                # 地址范围等配置需要更详细的地址规划
                "address_map": f"map_for_chip_{target_chip_id}",
            }
            atu_id += 1
        return atu_table


# --- 2. 地址路由逻辑 ---


class TreeAddressRoutingLogic:
    """
    树状拓扑地址路由核心逻辑
    """

    def route_address_decision(self, src_chip_id, dst_chip_id, routing_table):
        """
        路由决策算法
        输入: 源芯片ID, 目标芯片ID, 完整路由表
        输出: 路由决策（走哪个c2c_sys）

        核心逻辑:
        1. 查找预计算好的最短路径
        2. 确定下一跳节点
        3. 根据下一跳选择对应的c2c_sys (简化逻辑)
        """
        src_node_id = f"chip_{src_chip_id}"
        dst_node_id = f"chip_{dst_chip_id}"

        path = routing_table.get(src_node_id, {}).get(dst_node_id)
        if not path or len(path) < 2:
            return "Direct"  # 目标是自己或无法路由

        next_hop = path[1]  # 路径中的第二个节点是下一跳

        # 简化逻辑：假设c2c_sys0总是用于上行链路
        return {"next_hop": next_hop, "c2c_sys_out": "c2c_sys0"}

    def address_translation_logic(self, src_addr, src_format, dst_format, routing_path):
        """
        地址格式转换逻辑 (概念实现)
        输入: 源地址，源格式，目标格式，路由路径
        输出: 转换后的地址
        """
        print(f"Translating {src_addr} from {src_format} to {dst_format} via {routing_path}")
        # 实际转换逻辑依赖于详细的地址空间定义
        # 例如，当从C2C Fabric进入PCIe Switch时，需要将C2C地址格式转换为PCIe TLP格式
        if src_format == "c2c_fmt" and dst_format == "pcie_fmt":
            # 伪代码：解析c2c地址，并打包成PCIe地址
            # new_addr = pcie_repack(c2c_parse(src_addr))
            return f"translated_{src_addr}_as_{dst_format}"
        return f"不支持的转换"

    def optimize_for_all_reduce(self, participating_chips, tree_structure_nodes):
        """
        All Reduce操作路由优化
        输入: 参与芯片列表，树结构
        输出: 优化的通信模式 (reduce-scatter + all-gather)

        核心逻辑:
        1. 构建一个逻辑上的二叉树或环进行reduce
        2. 利用物理树拓扑的LCA进行聚合
        """
        # 找到所有参与芯片的最高LCA作为聚合点
        if not participating_chips:
            return None

        nodes = [tree_structure_nodes[f"chip_{cid}"] for cid in participating_chips]

        # 简化：假设根节点为聚合点
        lca = self._find_lca_for_group(nodes)

        reduce_plan = {
            "type": "Reduce-Scatter + All-Gather",
            "aggregation_point": lca.node_id,
            "stages": {"reduce": "所有芯片向聚合点上传数据", "broadcast": "聚合点向下广播结果"},
        }
        return reduce_plan

    def _find_lca_for_group(self, nodes):
        # 辅助函数：找到一组节点的LCA
        if not nodes:
            return None
        paths = [self._get_path_to_root(node) for node in nodes]

        lca = nodes[0]
        while not all(lca in path for path in paths):
            lca = lca.parent
            if not lca:
                return None  # Should not happen in a valid tree
        return lca

    def _get_path_to_root(self, node):
        path = []
        curr = node
        while curr:
            path.append(curr)
            curr = curr.parent
        return path


# --- 3. 配置生成逻辑 ---


class TreeConfigGenerationLogic:
    """
    树状拓扑配置生成核心逻辑
    """

    def generate_chip_c2c_config(self, chip_id, tree_structure_nodes):
        """
        生成单个芯片的c2c_sys配置
        输出: 每个c2c_sys的RC/EP设置，连接映射
        """
        chip_node = tree_structure_nodes.get(f"chip_{chip_id}")
        if not chip_node:
            return {}

        config = {}
        # 上行端口配置
        if chip_node.parent:
            config["c2c_sys0"] = {"mode": "EP", "connected_to": chip_node.parent.node_id, "link_speed": "PCIe Gen4 x8", "cdma_cascade": "disabled"}  # 作为端点连接到Switch

        # 其他c2c_sys可以配置为NA或用于其他目的
        for i in range(1, 5):
            config[f"c2c_sys{i}"] = {"mode": "NA"}

        return config

    def generate_atu_config_table(self, chip_id, all_chip_ids, tree_structure_nodes):
        """
        生成ATU配置表
        输出: 128个outbound ATU + 4个inbound ATU的完整配置
        """
        config = {"outbound": {}, "inbound": {}}
        atu_id = 0

        # Outbound ATU for each target chip
        for target_id in all_chip_ids:
            if target_id == chip_id:
                continue
            if atu_id >= 128:
                break

            # 假设每个芯片占用一个固定的地址空间
            addr_space_size = 2**32  # 4GB per chip
            ob_up_addr = (target_id << 32) + addr_space_size - 1
            ob_lw_addr = target_id << 32

            config["outbound"][f"OB_ATU_{atu_id}"] = {
                "target_chip_id": target_id,
                "ob_up_addr": hex(ob_up_addr),
                "ob_lw_addr": hex(ob_lw_addr),
                "ob_dst_pc": 0,  # 目标是另一个芯片
                "ob_pc_msi": 0,
            }
            atu_id += 1

        # Inbound ATU (通常用于从外部访问本芯片)
        # ...
        return config

    def generate_switch_config(self, switch_id, tree_structure_nodes):
        """
        生成PCIe Switch配置
        输出: Switch的端口映射，路由表
        """
        switch_node = tree_structure_nodes.get(switch_id)
        if not switch_node or switch_node.node_type != "switch":
            return {}

        port_mapping = {}
        port_num = 0

        # 上行端口
        if switch_node.parent:
            port_mapping[f"port_{port_num}"] = {"direction": "up", "connects_to": switch_node.parent.node_id}
            port_num += 1

        # 下行端口
        for child in switch_node.children:
            port_mapping[f"port_{port_num}"] = {"direction": "down", "connects_to": child.node_id}
            port_num += 1

        return {"switch_id": switch_id, "port_count": len(port_mapping), "port_mapping": port_mapping, "routing_logic": "Auto (Hardware Default)"}  # Switch通常有自己的路由逻辑


# --- 4. 故障处理逻辑 ---


class TreeFaultToleranceLogic:
    """
    树状拓扑故障容错逻辑
    """

    def detect_failed_components(self, health_status):
        """
        故障检测算法
        输入: 各组件健康状态 (例如: {'chip_0': 'OK', 'link_0_1': 'Failed'})
        输出: 故障组件列表
        """
        failed = []
        for component, status in health_status.items():
            if status != "OK":
                failed.append(component)
        return failed

    def calculate_recovery_topology(self, original_root, all_nodes, failed_components):
        """
        故障恢复拓扑计算
        输入: 原始树结构，故障组件
        输出: 重构后的树结构 (可能多个)

        核心逻辑:
        1. 从图中移除故障节点和相关边
        2. 寻找剩余的连通分量
        3. 每个连通分量都是一个新的（可能更小的）树
        """
        # 简单移除故障节点
        healthy_nodes = {nid: node for nid, node in all_nodes.items() if nid not in failed_components}

        # 如果Switch故障，其所有子节点连接断开
        for failed_comp in failed_components:
            if "switch" in failed_comp:
                for node in healthy_nodes.values():
                    if node.parent and node.parent.node_id == failed_comp:
                        node.parent = None  # 断开连接

        # 重新计算连通分量 (森林)
        forest = []
        visited = set()
        for node_id, node in healthy_nodes.items():
            if node_id not in visited and node.parent is None:
                # 这是一个新树的根
                forest.append(node)
                # 标记整个树的节点为已访问
                q = deque([node])
                visited.add(node_id)
                while q:
                    curr = q.popleft()
                    for child in curr.children:
                        if child.node_id in healthy_nodes and child.node_id not in visited:
                            visited.add(child.node_id)
                            q.append(child)
        return forest, healthy_nodes

    def generate_recovery_config(self, new_trees, affected_chips):
        """
        生成故障恢复配置
        输入: 新树结构(森林), 受影响芯片
        输出: 重配置命令序列
        """
        configs = {}
        # 对每个受影响的芯片，重新生成其拓扑、路由和ATU配置
        for chip_id in affected_chips:
            # 找到该芯片属于哪个新树
            # ... (此处逻辑需要更复杂的实现来处理森林)
            configs[chip_id] = "为新子树重新运行完整配置生成"
        return configs


# --- 5. 具体算法要求 ---


def optimize_tree_structure(chip_count, switch_capacity):
    """
    树结构优化核心算法
    目标: 最小化最大路径跳数, 平衡各Switch负载, 最小化Switch总数
    """
    if chip_count <= 0:
        return {}

    # 1. 计算理论最优层数
    levels = math.ceil(math.log(chip_count, switch_capacity))

    # 2. 计算各层Switch数量
    switches_per_level = []
    nodes_at_level = chip_count
    for i in range(levels):
        num_switches = math.ceil(nodes_at_level / switch_capacity)
        switches_per_level.append(num_switches)
        nodes_at_level = num_switches

    total_switches = sum(switches_per_level)

    return {
        "chip_count": chip_count,
        "switch_capacity": switch_capacity,
        "tree_levels": levels + 1,  # (芯片层 + switch层)
        "max_path_hops": 2 * levels,  # (chip -> root -> chip)
        "total_switches_needed": total_switches,
        "switches_per_level": switches_per_level[::-1],  # 从靠近根的层开始
    }


def shortest_path_in_tree(src_chip_id, dst_chip_id, tree_root, all_nodes):
    """
    树状结构最短路径算法
    """
    logic = TreeTopologyLogic()
    src_node = all_nodes.get(f"chip_{src_chip_id}")
    dst_node = all_nodes.get(f"chip_{dst_chip_id}")
    if not src_node or not dst_node:
        return None

    path_nodes = logic._find_path_between(src_node, dst_node)
    return {"path": path_nodes, "lca": logic._get_path_to_root(src_node)[-1].node_id, "c2c_sys_decision": "Path determines c2c_sys choice at each hop."}  # 简化LCA查找


def optimize_all_reduce_tree(participating_chips, tree_structure_nodes):
    """
    All Reduce操作优化算法
    """
    logic = TreeAddressRoutingLogic()
    return logic.optimize_for_all_reduce(participating_chips, tree_structure_nodes)


# --- 6. 测试验证逻辑 ---


def validate_tree_topology(tree_root, all_nodes, switch_capacity):
    """
    拓扑合法性验证
    """
    errors = []

    # 1. 检查连通性 (所有节点是否都在树中)
    q = deque([tree_root])
    visited = {tree_root.node_id}
    while q:
        node = q.popleft()
        for child in node.children:
            if child.node_id not in visited:
                visited.add(child.node_id)
                q.append(child)

    if len(visited) != len(all_nodes):
        errors.append(f"Connectivity error: Visited {len(visited)} nodes, but total is {len(all_nodes)}")

    # 2. 检查树特性 (无环) - 通过父指针确保
    for node in all_nodes.values():
        if node.parent and node not in node.parent.children:
            errors.append(f"Parent-child mismatch for node {node.node_id}")

    # 3. 资源约束
    for node in all_nodes.values():
        if node.node_type == "switch" and len(node.children) > switch_capacity:
            errors.append(f"Switch {node.node_id} exceeds capacity ({len(node.children)} > {switch_capacity})")

    # 4. 配置一致性 (RC/EP) - 逻辑上，chip(EP)连接到switch(RC)
    # ... (需要更详细的配置信息来验证)

    return {"is_valid": not errors, "errors": errors}


def evaluate_tree_performance(tree_root, all_nodes):
    """
    性能评估算法
    """
    chip_nodes = [node for node in all_nodes.values() if node.node_type == "chip"]
    if len(chip_nodes) < 2:
        return {"average_path_length": 0, "max_path_length": 0}

    total_path_length = 0
    max_path_length = 0
    path_count = 0

    logic = TreeTopologyLogic()

    for i in range(len(chip_nodes)):
        for j in range(i + 1, len(chip_nodes)):
            path = logic._find_path_between(chip_nodes[i], chip_nodes[j])
            # 路径长度是边的数量
            path_len = len(path) - 1
            total_path_length += path_len
            if path_len > max_path_length:
                max_path_length = path_len
            path_count += 1

    avg_path_length = total_path_length / path_count if path_count > 0 else 0

    return {
        "average_path_length": round(avg_path_length, 2),
        "max_path_length": max_path_length,
        "bandwidth_utilization": "Depends on traffic pattern",
        "hotspot_analysis": "Switches closer to the root are potential hotspots.",
    }


if __name__ == "__main__":
    # --- 使用示例 ---
    print("--- 1. 构建拓扑 ---")
    CHIP_COUNT = 128
    SWITCH_CAPACITY = 8
    chip_ids = list(range(CHIP_COUNT))

    topo_logic = TreeTopologyLogic()
    tree_root, all_nodes = topo_logic.calculate_tree_structure(chip_ids, SWITCH_CAPACITY)
    print(f"构建了一个 {CHIP_COUNT} 芯片, {SWITCH_CAPACITY} 端口Switch的树. 根节点: {tree_root.node_id}")

    print("--- 2. 验证拓扑 ---")
    validation_result = validate_tree_topology(tree_root, all_nodes, SWITCH_CAPACITY)
    print(f"拓扑是否有效: {validation_result['is_valid']}")

    print("--- 3. 性能评估 ---")
    perf_report = evaluate_tree_performance(tree_root, all_nodes)
    print(f"性能报告: {perf_report}")

    print("--- 4. 计算路由表 ---")
    routing_table = topo_logic.compute_routing_table(tree_root, all_nodes)
    src, dst = f"chip_{chip_ids[0]}", f"chip_{chip_ids[-1]}"
    print(f"从 {src} 到 {dst} 的路径: {routing_table[src][dst]}")

    print("--- 5. 生成芯片配置 ---")
    config_gen = TreeConfigGenerationLogic()
    chip_0_config = config_gen.generate_chip_c2c_config(0, all_nodes)
    print(f"芯片0的C2C配置: {chip_0_config}")

    print("--- 6. 生成Switch配置 ---")
    # 找到第一个switch并生成配置
    first_switch_id = next(nid for nid, n in all_nodes.items() if n.node_type == "switch")
    switch_config = config_gen.generate_switch_config(first_switch_id, all_nodes)
    print(f"Switch '{first_switch_id}' 的配置: {switch_config}")

    print("--- 7. All-Reduce优化 ---")
    participating = [0, 1, 8, 9, 16, 17, 24, 25]
    all_reduce_plan = optimize_all_reduce_tree(participating, all_nodes)
    print(f"为芯片 {participating} 生成的All-Reduce计划: {all_reduce_plan}")

    print("--- 8. 故障模拟与恢复 ---")
    fault_logic = TreeFaultToleranceLogic()
    # 模拟一个底层Switch故障
    failed_switch = "switch_0"
    health_status = {nid: "OK" for nid in all_nodes}
    if failed_switch in health_status:
        health_status[failed_switch] = "Failed"

    failed_components = fault_logic.detect_failed_components(health_status)
    print(f"检测到故障组件: {failed_components}")

    forest, healthy_nodes = fault_logic.calculate_recovery_topology(tree_root, all_nodes, failed_components)
    print(f"故障后形成 {len(forest)} 个独立的树(森林).")
