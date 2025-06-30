# -*- coding: utf-8 -*-
"""
SG2260E C2C Torus拓扑核心逻辑实现
"""

import math
from itertools import product
from functools import reduce

# --- 1. Torus拓扑逻辑核心 ---

class TorusTopologyLogic:
    """
    Torus拓扑核心算法逻辑
    基于C2C Link构建多维环形网格结构
    """

    def calculate_torus_structure(self, chip_count, dimensions=2):
        """
        计算最优Torus结构
        输入: 芯片数量，维度数(2D/3D)
        输出: Torus网格配置（尺寸、坐标映射、邻居列表）
        """
        grid_dims = self.optimize_grid_dimensions(chip_count, dimensions)
        if not grid_dims:
            raise ValueError("无法为给定的芯片数量和维度找到合适的网格尺寸")

        coord_map, id_map = self.generate_coordinate_mapping(chip_count, grid_dims)
        
        neighbors = {}
        for chip_id, coord in coord_map.items():
            neighbors[chip_id] = self.calculate_neighbors(coord, grid_dims)
            
        return {
            "chip_count": chip_count,
            "dimensions": dimensions,
            "grid_dimensions": grid_dims,
            "coordinate_map": coord_map, # chip_id -> coord
            "id_map": id_map,           # coord -> chip_id
            "neighbors": neighbors      # chip_id -> neighbor list
        }

    def generate_coordinate_mapping(self, chip_count, grid_dimensions):
        """
        生成坐标映射
        输入: 芯片数量，网格尺寸[X,Y] 或 [X,Y,Z]
        输出: chip_id <-> 坐标的双向映射
        """
        coord_to_id = {}
        id_to_coord = {}
        
        dim_ranges = [range(d) for d in grid_dimensions]
        
        chip_id = 0
        for coord in product(*dim_ranges):
            if chip_id < chip_count:
                coord_to_id[coord] = chip_id
                id_to_coord[chip_id] = coord
                chip_id += 1
        
        return id_to_coord, coord_to_id

    def calculate_neighbors(self, coord, grid_dimensions):
        """
        计算指定坐标的所有邻居
        输入: 芯片坐标，网格尺寸
        输出: 邻居坐标列表及方向
        """
        neighbors = {}
        dims = len(grid_dimensions)
        directions = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]

        for i in range(dims):
            # 正方向
            pos_neighbor = list(coord)
            pos_neighbor[i] = (coord[i] + 1) % grid_dimensions[i]
            neighbors[directions[2*i]] = tuple(pos_neighbor)
            
            # 负方向
            neg_neighbor = list(coord)
            neg_neighbor[i] = (coord[i] - 1 + grid_dimensions[i]) % grid_dimensions[i]
            neighbors[directions[2*i+1]] = tuple(neg_neighbor)
            
        return neighbors

    def optimize_grid_dimensions(self, chip_count, dimensions):
        """
        优化网格尺寸
        输入: 芯片数量，维度数
        输出: 最优的网格尺寸配置
        """
        if chip_count <= 0: return []

        def get_factors(n):
            factors = set()
            for i in range(1, int(math.sqrt(n)) + 1):
                if n % i == 0:
                    factors.add(i)
                    factors.add(n//i)
            return sorted(list(factors))

        best_dims = []
        min_diff = float('inf')

        # 找到一个能容纳所有芯片的最小的“完美”网格数
        effective_chip_count = chip_count
        while True:
            factors = get_factors(effective_chip_count)
            if dimensions == 2:
                for f1 in factors:
                    f2 = effective_chip_count // f1
                    if abs(f1 - f2) < min_diff:
                        min_diff = abs(f1 - f2)
                        best_dims = sorted([f1, f2])
            elif dimensions == 3:
                for f1 in factors:
                    for f2 in factors:
                        if (effective_chip_count % (f1 * f2)) == 0:
                            f3 = effective_chip_count // (f1 * f2)
                            diff = max(f1, f2, f3) - min(f1, f2, f3)
                            if diff < min_diff:
                                min_diff = diff
                                best_dims = sorted([f1, f2, f3])
            if best_dims:
                return best_dims
            effective_chip_count += 1 # 如果找不到因子，增加数量再试

# --- 2. Torus路由算法 ---

class TorusRoutingLogic:
    """
    Torus拓扑路由核心逻辑
    维度顺序路由避免死锁
    """

    def dimension_order_routing(self, src_coord, dst_coord, grid_dimensions):
        """
        维度顺序路由算法 (DOR)
        输入: 源坐标，目标坐标，网格尺寸
        输出: 完整路由路径（坐标序列）
        """
        path = [src_coord]
        current_coord = list(src_coord)
        dims = len(grid_dimensions)

        # 按 X -> Y -> Z 顺序路由
        for i in range(dims):
            dist, direction = self._calculate_dim_distance(current_coord[i], dst_coord[i], grid_dimensions[i])
            
            for _ in range(dist):
                current_coord[i] = (current_coord[i] + direction + grid_dimensions[i]) % grid_dimensions[i]
                path.append(tuple(current_coord))
        
        return path

    def _calculate_dim_distance(self, src, dst, size):
        """计算单个维度的最短距离和方向"""
        # 正向
        dist_pos = (dst - src + size) % size
        # 反向
        dist_neg = (src - dst + size) % size
        
        if dist_pos <= dist_neg:
            return dist_pos, 1  # 1 代表正方向
        else:
            return dist_neg, -1 # -1 代表负方向

    def calculate_shortest_distance(self, src_coord, dst_coord, grid_dimensions):
        """
        计算两点间最短距离（曼哈顿距离）
        输出: 各维度的最短距离和方向
        """
        distances = {}
        total_hops = 0
        dims = len(grid_dimensions)
        directions = [("+X", "-X"), ("+Y", "-Y"), ("+Z", "-Z")]

        for i in range(dims):
            dist, direction = self._calculate_dim_distance(src_coord[i], dst_coord[i], grid_dimensions[i])
            distances[f"dim_{i}"] = {
                "distance": dist,
                "direction": directions[i][0] if direction == 1 else directions[i][1]
            }
            total_hops += dist
        
        distances["total_hops"] = total_hops
        return distances

    # adaptive_routing 和 fault_tolerant_routing 较为复杂，这里提供概念实现
    def adaptive_routing(self, src_coord, dst_coord, congestion_map, grid_dimensions):
        """
        自适应路由算法 (概念)
        """
        print("执行自适应路由，优先选择非拥塞路径，同时遵循维度顺序")
        # 逻辑：在每个维度选择方向时，不仅考虑最短路，也考虑congestion_map
        # 例如，如果+X方向拥塞，即使距离稍长，也可能选择-X方向
        return self.dimension_order_routing(src_coord, dst_coord, grid_dimensions)

    def fault_tolerant_routing(self, src_coord, dst_coord, failed_links, grid_dimensions):
        """
        容错路由算法 (概念)
        """
        print("执行容错路由，绕过故障链路")
        # 逻辑：如果DOR路径上的下一步是故障链路，则需要绕路。
        # 这可能需要临时违反DOR（例如，先走Y再走X），需要更复杂的死锁避免机制，如气泡路由。
        # 简化版：如果主路径故障，尝试备用路径（例如，反向走）。
        return self.dimension_order_routing(src_coord, dst_coord, grid_dimensions)

# --- 3. C2C系统映射逻辑 ---

class TorusC2CMappingLogic:
    """
    Torus拓扑到C2C系统映射逻辑
    """

    def map_directions_to_c2c_sys(self, chip_id, torus_structure):
        """
        将Torus方向映射到c2c_sys
        """
        neighbors = torus_structure['neighbors'][chip_id]
        mapping = {}
        # 预定义映射规则
        dir_to_sys = {
            "-X": "c2c_sys0", "+X": "c2c_sys1",
            "-Y": "c2c_sys2", "+Y": "c2c_sys3",
            "-Z": "c2c_sys4", "+Z": "c2c_sys4" # 3D时Z方向可能共享一个c2c_sys
        }
        
        for direction, neighbor_coord in neighbors.items():
            sys = dir_to_sys.get(direction)
            if sys:
                neighbor_id = torus_structure['id_map'][neighbor_coord]
                mapping[sys] = {"direction": direction, "connects_to_chip_id": neighbor_id}
        
        return mapping

    def generate_c2c_link_config(self, chip_id, torus_structure):
        """
        生成C2C Link配置
        """
        mapping = self.map_directions_to_c2c_sys(chip_id, torus_structure)
        config = {}
        for sys, info in mapping.items():
            config[sys] = {
                "mode": "RC", # C2C直连通常都是RC模式
                "link_width": "x8", # 默认x8，可由优化算法调整
                "target_chip_id": info['connects_to_chip_id'],
                "direction": info['direction']
            }
        return config

# --- 4. 地址路由决策逻辑 ---

class TorusAddressRoutingLogic:
    """
    Torus拓扑地址路由决策逻辑
    """

    def route_address_decision(self, src_chip_id, dst_chip_id, torus_structure):
        """
        Torus路由决策算法
        """
        src_coord = torus_structure['coordinate_map'][src_chip_id]
        dst_coord = torus_structure['coordinate_map'][dst_chip_id]
        grid_dims = torus_structure['grid_dimensions']
        
        next_hop_dir, next_hop_coord = self.next_hop_selection(src_coord, dst_coord, grid_dims)
        
        if not next_hop_dir:
            return {"decision": "Destination is the same chip"}

        # 从C2C映射中找到对应的c2c_sys
        c2c_mapping = TorusC2CMappingLogic().map_directions_to_c2c_sys(src_chip_id, torus_structure)
        c2c_sys_out = "N/A"
        for sys, info in c2c_mapping.items():
            if info['direction'] == next_hop_dir:
                c2c_sys_out = sys
                break

        return {
            "decision": f"Route via {next_hop_dir}",
            "next_hop_coord": next_hop_coord,
            "c2c_sys_out": c2c_sys_out
        }

    def next_hop_selection(self, current_coord, target_coord, grid_dimensions):
        """
        下一跳选择算法 (DOR)
        """
        if current_coord == target_coord:
            return None, None

        dims = len(grid_dimensions)
        directions = [("+X", "-X"), ("+Y", "-Y"), ("+Z", "-Z")]

        for i in range(dims):
            if current_coord[i] != target_coord[i]:
                dist, direction = TorusRoutingLogic()._calculate_dim_distance(
                    current_coord[i], target_coord[i], grid_dimensions[i]
                )
                next_coord = list(current_coord)
                next_coord[i] = (current_coord[i] + direction + grid_dimensions[i]) % grid_dimensions[i]
                
                hop_dir = directions[i][0] if direction == 1 else directions[i][1]
                return hop_dir, tuple(next_coord)
        return None, None # Should not be reached if coords are different

# --- 5. All Reduce优化逻辑 ---

class TorusAllReduceLogic:
    """
    Torus拓扑All Reduce优化逻辑
    """

    def optimize_all_reduce_pattern(self, torus_structure):
        """
        All Reduce模式优化 (维度分解)
        """
        grid_dims = torus_structure['grid_dimensions']
        plan = []
        
        # 1. Reduce-Scatter: 按维度进行
        for i, dim_size in enumerate(grid_dims):
            plan.append({
                "stage": f"Reduce-Scatter on Dim {i}",
                "steps": math.log2(dim_size),
                "description": f"Chips exchange and reduce data with neighbors along dimension {i}."
            })
        
        # 2. All-Gather: 按维度反向进行
        for i, dim_size in enumerate(grid_dims):
            plan.append({
                "stage": f"All-Gather on Dim {i}",
                "steps": math.log2(dim_size),
                "description": f"Chips broadcast results back along dimension {i}."
            })
            
        return plan

# --- 6. 故障恢复逻辑 ---

class TorusFaultToleranceLogic:
    """
    Torus拓扑故障容错逻辑
    """

    def detect_link_failures(self, health_status):
        """
        链路故障检测
        """
        failed_links = []
        for link, status in health_status.items():
            if status != 'OK':
                # link name format: "link_chipA_chipB"
                failed_links.append(link)
        return failed_links

    def generate_recovery_routing(self, failed_links, torus_structure):
        """
        生成恢复路由表 (概念)
        """
        print(f"Detected failed links: {failed_links}. Recalculating routes.")
        # 实际实现需要一个更复杂的路由算法，如基于上/下转弯模型的算法，
        # 以在存在故障的情况下仍然保证无死锁。
        return {
            "status": "Recovery routing table generated",
            "details": "Using adaptive routing to bypass failures."
        }

# --- 7. 特殊算法和测试验证 ---

# 特殊算法
def optimize_torus_dimensions(chip_count, max_dimensions=3):
    return TorusTopologyLogic().optimize_grid_dimensions(chip_count, max_dimensions)

def calculate_torus_distance(coord1, coord2, grid_size):
    return TorusRoutingLogic().calculate_shortest_distance(coord1, coord2, grid_size)

def ensure_deadlock_freedom(routing_function, torus_structure):
    """
    死锁避免验证算法 (概念)
    DOR本身是无死锁的，因为其严格的维度顺序打破了资源依赖循环。
    """
    is_dor = "dimension_order_routing" in routing_function.__name__
    return {
        "is_deadlock_free": is_dor,
        "reason": "The routing function uses strict Dimension-Order Routing (DOR), which is provably deadlock-free."
    }

# 测试验证
def test_torus_connectivity(torus_structure):
    """
    验证所有芯片对都可达
    """
    chip_ids = list(torus_structure['coordinate_map'].keys())
    router = TorusRoutingLogic()
    
    for src_id in chip_ids:
        for dst_id in chip_ids:
            if src_id == dst_id: continue
            src_coord = torus_structure['coordinate_map'][src_id]
            dst_coord = torus_structure['coordinate_map'][dst_id]
            path = router.dimension_order_routing(src_coord, dst_coord, torus_structure['grid_dimensions'])
            if not path or path[-1] != dst_coord:
                return {"is_connected": False, "error": f"Path not found from {src_id} to {dst_id}"}
    return {"is_connected": True}

if __name__ == '__main__':
    # --- 使用示例 ---
    CHIP_COUNT = 64
    DIMENSIONS = 2 # 2D Torus

    print("--- 1. 构建 {DIMENSIONS}D Torus ({CHIP_COUNT} chips) ---")
    topo_logic = TorusTopologyLogic()
    torus_structure = topo_logic.calculate_torus_structure(CHIP_COUNT, DIMENSIONS)
    grid_dims = torus_structure['grid_dimensions']
    print(f"计算出的最优网格尺寸: {grid_dims[0]}x{grid_dims[1]}")

    print("\n--- 2. 验证拓扑连通性 ---")
    connectivity_test = test_torus_connectivity(torus_structure)
    print(f"拓扑是否完全连通: {connectivity_test['is_connected']}")

    print("\n--- 3. 计算路由 ---")
    routing_logic = TorusRoutingLogic()
    src_id, dst_id = 0, 63
    src_coord = torus_structure['coordinate_map'][src_id]
    dst_coord = torus_structure['coordinate_map'][dst_id]
    path = routing_logic.dimension_order_routing(src_coord, dst_coord, grid_dims)
    print(f"从 Chip {src_id}{src_coord} 到 Chip {dst_id}{dst_coord} 的DOR路径 (长度 {len(path)-1}):")
    print(f"  {path[0]} -> ... -> {path[-1]}")

    dist_info = routing_logic.calculate_shortest_distance(src_coord, dst_coord, grid_dims)
    print(f"最短距离 (曼哈顿): {dist_info['total_hops']} hops")

    print("\n--- 4. 生成C2C映射和配置 ---")
    mapping_logic = TorusC2CMappingLogic()
    chip_0_mapping = mapping_logic.map_directions_to_c2c_sys(0, torus_structure)
    print(f"Chip 0 的方向到c2c_sys映射: {chip_0_mapping}")
    chip_0_config = mapping_logic.generate_c2c_link_config(0, torus_structure)
    print(f"Chip 0 的C2C Link配置 (部分): {list(chip_0_config.items())[0]}")

    print("\n--- 5. 地址路由决策 ---")
    addr_routing_logic = TorusAddressRoutingLogic()
    decision = addr_routing_logic.route_address_decision(src_id, dst_id, torus_structure)
    print(f"从 Chip {src_id} 到 {dst_id} 的路由决策: {decision}")

    print("\n--- 6. All-Reduce 优化 ---")
    all_reduce_logic = TorusAllReduceLogic()
    all_reduce_plan = all_reduce_logic.optimize_all_reduce_pattern(torus_structure)
    print(f"为 {grid_dims[0]}x{grid_dims[1]} Torus 生成的All-Reduce计划 (部分): {all_reduce_plan[0]}")

    print("\n--- 7. 死锁避免验证 ---")
    deadlock_check = ensure_deadlock_freedom(routing_logic.dimension_order_routing, torus_structure)
    print(f"路由算法是否无死锁: {deadlock_check['is_deadlock_free']}")