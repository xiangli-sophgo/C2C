"""
CrossRing NoC仿真的标准轮询调度器。
实现真正的轮询调度，使用索引管理。
"""

from typing import List, Any, Optional, Callable, Tuple
from collections import defaultdict


class RoundRobinScheduler:
    """
    使用索引管理的标准轮询调度器。
    
    该调度器通过始终推进索引来确保真正的公平性，
    无论是否找到匹配项。这避免了原始remove+append方法
    中存在的饥饿问题。
    
    主要特性:
    - O(1)索引更新（相比O(n)的remove操作）
    - 真正的公平性 - 所有候选项都有均等机会
    - 跨调用的索引持久性
    - 支持条件选择
    """
    
    def __init__(self):
        """初始化轮询调度器。"""
        # 存储每个队列键的当前索引
        self.indices = defaultdict(int)
        
        # 统计信息（可选，如不需要可删除）
        self.stats = {
            "total_selections": 0,
            "successful_selections": 0,
            "queue_accesses": defaultdict(int)
        }
    
    def select(self, key: str, candidates: List[Any], 
               check_func: Optional[Callable[[Any], bool]] = None) -> Tuple[Optional[Any], Optional[int]]:
        """
        使用轮询调度选择下一个候选项。
        
        参数:
            key: 调度队列的唯一标识符（例如："IQ_TR_14_req"）
            candidates: 候选项列表
            check_func: 可选的候选项有效性检查函数
            
        返回:
            (选中的候选项, 选中索引) 或 (None, None) 如果未找到
            
        关键点是索引总是前进，确保公平性。
        """
        if not candidates:
            return None, None
            
        self.stats["total_selections"] += 1
        self.stats["queue_accesses"][key] += 1
        
        current_idx = self.indices[key]
        candidates_len = len(candidates)
        start_idx = current_idx
        
        # Try each candidate starting from current position
        for offset in range(candidates_len):
            idx = (current_idx + offset) % candidates_len
            candidate = candidates[idx]
            
            # Check if candidate meets conditions
            if check_func is None or check_func(candidate):
                # Found a valid candidate
                self.stats["successful_selections"] += 1
                
                # CRITICAL: Always advance index for next call
                # This ensures fairness - we don't start from the same position next time
                self.indices[key] = (idx + 1) % candidates_len
                
                return candidate, idx
        
        # No valid candidate found, but still advance index for fairness
        self.indices[key] = (current_idx + 1) % candidates_len
        return None, None
    
    def get_current_index(self, key: str) -> int:
        """
        获取给定队列键的当前索引。
        
        参数:
            key: 队列标识符
            
        返回:
            当前索引（如果键不存在则返回0）
        """
        return self.indices.get(key, 0)
    
    def set_index(self, key: str, index: int, candidates_len: int):
        """
        手动设置队列的索引。
        
        参数:
            key: 队列标识符
            index: 新的索引位置
            candidates_len: 候选项列表长度，用于边界检查
        """
        if candidates_len > 0:
            self.indices[key] = index % candidates_len
        else:
            self.indices[key] = 0
    
    def reset_index(self, key: Optional[str] = None):
        """
        重置调度索引。
        
        参数:
            key: 要重置的特定队列，或None重置所有
        """
        if key is None:
            self.indices.clear()
        else:
            self.indices[key] = 0
    
    def get_stats(self) -> dict:
        """
        获取调度统计信息。
        
        返回:
            包含调度统计信息的字典
        """
        success_rate = 0.0
        if self.stats["total_selections"] > 0:
            success_rate = self.stats["successful_selections"] / self.stats["total_selections"]
            
        return {
            "total_selections": self.stats["total_selections"],
            "successful_selections": self.stats["successful_selections"],
            "success_rate": success_rate,
            "active_queues": len(self.indices),
            "queue_accesses": dict(self.stats["queue_accesses"])
        }
    
    def reset_stats(self):
        """重置所有统计信息。"""
        self.stats = {
            "total_selections": 0,
            "successful_selections": 0,
            "queue_accesses": defaultdict(int)
        }


# 常用情况的辅助函数
def create_scheduler() -> RoundRobinScheduler:
    """工厂函数，创建新的轮询调度器。"""
    return RoundRobinScheduler()


# 不同队列类型的使用示例：
"""
# IQ Usage Example:
iq_scheduler = RoundRobinScheduler()
key = f"IQ_{direction}_{ip_pos}_{network_type}"
ip_types = ["sdma", "gdma", "cdma", "ddr", "l2m"]

def check_ip_conditions(ip_type):
    return (network.IQ_channel_buffer[ip_type][ip_pos] and
            flit_meets_direction_condition and
            resources_available)

selected_ip_type, idx = iq_scheduler.select(key, ip_types, check_ip_conditions)
if selected_ip_type:
    # Process the selected IP type
    pass

# EQ Usage Example:  
eq_scheduler = RoundRobinScheduler()
key = f"EQ_{ip_type}_{ip_pos}"
ports = [0, 1, 2, 3]  # TU, TD, IQ, RB

def check_eq_conditions(port_idx):
    return (eject_flits[port_idx] is not None and
            destination_matches and
            buffer_has_space)

selected_port, idx = eq_scheduler.select(key, ports, check_eq_conditions)
if selected_port is not None:
    # Process the selected port
    pass

# RB Usage Example:
rb_scheduler = RoundRobinScheduler() 
key = f"RB_{direction}_{pos}_{next_pos}"
slots = [0, 1, 2, 3]

def check_rb_conditions(slot_idx):
    return (station_flits[slot_idx] is not None and
            destination_check(station_flits[slot_idx].destination, next_pos))

selected_slot, idx = rb_scheduler.select(key, slots, check_rb_conditions)
if selected_slot is not None:
    # Process the selected slot
    pass
"""