"""
CrossRing主模型类实现。

基于C2C仓库的架构，提供完整的CrossRing NoC仿真模型，
包括IP接口管理、网络组件和仿真循环控制。
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple
import logging
from collections import defaultdict

from .config import CrossRingConfig
from .ip_interface import CrossRingIPInterface
from .flit import CrossRingFlit, get_crossring_flit_pool_stats
from ..types import NodeId


class CrossRingModel:
    """
    CrossRing主模型类。
    
    该类负责：
    1. 整体仿真循环控制
    2. IP接口实例管理
    3. CrossRing网络组件管理（骨架）
    4. 全局状态监控和调试
    5. 性能统计收集
    """
    
    def __init__(self, config: CrossRingConfig):
        """
        初始化CrossRing模型
        
        Args:
            config: CrossRing配置实例
        """
        self.config = config
        self.cycle = 0
        
        # IP接口管理
        self.ip_interfaces: Dict[str, CrossRingIPInterface] = {}
        self._ip_registry: Dict[str, CrossRingIPInterface] = {}  # 全局debug视图
        
        # CrossRing网络组件（暂时为骨架，后续实现）
        self.crossring_pieces: Dict[NodeId, Any] = {}  # {node_id: CrossRingPiece}
        self.networks = {
            "req": None,    # REQ网络（后续实现）
            "rsp": None,    # RSP网络（后续实现）
            "data": None    # DATA网络（后续实现）
        }
        
        # 性能统计
        self.stats = {
            "total_requests": 0,
            "total_responses": 0,
            "total_data_flits": 0,
            "read_retries": 0,
            "write_retries": 0,
            "average_latency": 0.0,
            "peak_active_requests": 0,
            "current_active_requests": 0,
        }
        
        # 仿真状态
        self.is_running = False
        self.is_finished = False
        
        # 日志配置
        self.logger = logging.getLogger(f"CrossRingModel_{id(self)}")
        
        # 初始化所有组件
        self._setup_ip_interfaces()
        self._setup_crossring_networks()
        
        self.logger.info(f"CrossRing模型初始化完成: {config.num_row}x{config.num_col}, {len(self.ip_interfaces)}个IP接口")
    
    def _setup_ip_interfaces(self) -> None:
        """设置所有IP接口"""
        ip_type_configs = [
            ("gdma", self.config.gdma_send_position_list),
            ("sdma", self.config.sdma_send_position_list),
            ("cdma", self.config.cdma_send_position_list),
            ("ddr", self.config.ddr_send_position_list),
            ("l2m", self.config.l2m_send_position_list),
        ]
        
        for ip_type, positions in ip_type_configs:
            for node_id in positions:
                key = f"{ip_type}_{node_id}"
                self.ip_interfaces[key] = CrossRingIPInterface(
                    config=self.config,
                    ip_type=ip_type,
                    node_id=node_id,
                    model=self
                )
                
                self.logger.debug(f"创建IP接口: {key} at node {node_id}")
    
    def _setup_crossring_networks(self) -> None:
        """设置CrossRing网络组件（骨架实现）"""
        # 这里是网络组件的骨架实现
        # 后续需要实现inject_queue, eject_queue, Ring Piece等
        
        for node_id in range(self.config.num_nodes):
            # 为每个节点创建CrossRing piece的占位符
            self.crossring_pieces[node_id] = {
                "node_id": node_id,
                "inject_queues": {"req": [], "rsp": [], "data": []},
                "eject_queues": {"req": [], "rsp": [], "data": []},
                "ring_buffers": {"horizontal": [], "vertical": []},
                "coordinates": self._get_node_coordinates(node_id),
            }
        
        self.logger.info(f"CrossRing网络组件骨架创建完成: {len(self.crossring_pieces)}个节点")
    
    def _get_node_coordinates(self, node_id: NodeId) -> Tuple[int, int]:
        """
        获取节点在CrossRing网格中的坐标
        
        Args:
            node_id: 节点ID
            
        Returns:
            (x, y)坐标
        """
        x = node_id % self.config.num_col
        y = node_id // self.config.num_col
        return x, y
    
    def register_ip_interface(self, ip_interface: CrossRingIPInterface) -> None:
        """
        注册IP接口（用于全局debug和管理）
        
        Args:
            ip_interface: IP接口实例
        """
        key = f"{ip_interface.ip_type}_{ip_interface.node_id}"
        self._ip_registry[key] = ip_interface
        
        self.logger.debug(f"注册IP接口到全局registry: {key}")
    
    def step(self) -> None:
        """执行一个仿真周期"""
        self.cycle += 1
        
        # 处理所有IP接口
        for ip_interface in self.ip_interfaces.values():
            ip_interface.step(self.cycle)
        
        # 处理CrossRing网络（后续实现）
        self._step_crossring_networks()
        
        # 更新统计信息
        self._update_statistics()
        
        # 定期输出调试信息
        if self.cycle % 1000 == 0:
            self.logger.debug(f"周期 {self.cycle}: {self.get_active_request_count()}个活跃请求")
    
    def _step_crossring_networks(self) -> None:
        """
        处理CrossRing网络（骨架实现）
        
        后续需要实现：
        1. inject_queue → ring网络
        2. ring网络内部传输
        3. ring网络 → eject_queue
        4. ETag/ITag处理
        5. 拥塞控制
        """
        # 骨架实现：简单地移动flits
        for node_id, piece in self.crossring_pieces.items():
            # 处理inject队列 → ring缓冲区
            for channel in ["req", "rsp", "data"]:
                inject_queue = piece["inject_queues"][channel]
                if inject_queue:
                    # 简化实现：直接移动到对应的eject队列
                    # 实际实现需要经过ring传输
                    flit = inject_queue.pop(0)
                    target_node = flit.destination
                    if target_node in self.crossring_pieces:
                        self.crossring_pieces[target_node]["eject_queues"][channel].append(flit)
                        flit.is_arrive = True
                        flit.arrival_network_cycle = self.cycle
    
    def _update_statistics(self) -> None:
        """更新性能统计"""
        # 更新活跃请求数
        current_active = self.get_active_request_count()
        self.stats["current_active_requests"] = current_active
        if current_active > self.stats["peak_active_requests"]:
            self.stats["peak_active_requests"] = current_active
        
        # 累计重试数
        total_read_retries = sum(ip.read_retry_num_stat for ip in self._ip_registry.values())
        total_write_retries = sum(ip.write_retry_num_stat for ip in self._ip_registry.values())
        self.stats["read_retries"] = total_read_retries
        self.stats["write_retries"] = total_write_retries
    
    def run_simulation(self, max_cycles: int = 10000, 
                      warmup_cycles: int = 1000,
                      stats_start_cycle: int = 1000) -> Dict[str, Any]:
        """
        运行完整仿真
        
        Args:
            max_cycles: 最大仿真周期数
            warmup_cycles: 热身周期数
            stats_start_cycle: 统计开始周期
            
        Returns:
            仿真结果字典
        """
        self.logger.info(f"开始CrossRing仿真: max_cycles={max_cycles}, warmup={warmup_cycles}")
        
        self.is_running = True
        stats_enabled = False
        
        try:
            for cycle in range(1, max_cycles + 1):
                self.step()
                
                # 启用统计收集
                if cycle == stats_start_cycle:
                    stats_enabled = True
                    self._reset_statistics()
                    self.logger.info(f"周期 {cycle}: 开始收集统计数据")
                
                # 检查仿真结束条件
                if self._should_stop_simulation():
                    self.logger.info(f"周期 {cycle}: 检测到仿真结束条件")
                    break
                
                # 定期输出进度
                if cycle % 5000 == 0:
                    self.logger.info(f"仿真进度: {cycle}/{max_cycles} 周期")
        
        except KeyboardInterrupt:
            self.logger.warning("仿真被用户中断")
        
        except Exception as e:
            self.logger.error(f"仿真过程中发生错误: {e}")
            raise
        
        finally:
            self.is_running = False
            self.is_finished = True
        
        # 生成仿真结果
        results = self._generate_simulation_results(stats_start_cycle)
        self.logger.info(f"CrossRing仿真完成: 总周期={self.cycle}")
        
        return results
    
    def _should_stop_simulation(self) -> bool:
        """检查是否应该停止仿真"""
        # 简单的停止条件：没有活跃请求
        active_requests = self.get_active_request_count()
        
        # 如果连续1000个周期没有活跃请求，则停止
        if not hasattr(self, '_idle_cycles'):
            self._idle_cycles = 0
        
        if active_requests == 0:
            self._idle_cycles += 1
        else:
            self._idle_cycles = 0
        
        return self._idle_cycles >= 1000
    
    def _reset_statistics(self) -> None:
        """重置统计计数器"""
        for ip in self._ip_registry.values():
            ip.read_retry_num_stat = 0
            ip.write_retry_num_stat = 0
            # 可以重置更多统计信息
    
    def _generate_simulation_results(self, stats_start_cycle: int) -> Dict[str, Any]:
        """生成仿真结果"""
        effective_cycles = self.cycle - stats_start_cycle
        
        # 汇总IP接口统计
        ip_stats = {}
        for key, ip in self._ip_registry.items():
            ip_stats[key] = ip.get_status()
        
        # 计算平均延迟等指标
        total_transactions = sum(ip.rn_tracker_pointer["read"] + ip.rn_tracker_pointer["write"] 
                               for ip in self._ip_registry.values())
        
        results = {
            "simulation_info": {
                "total_cycles": self.cycle,
                "effective_cycles": effective_cycles,
                "config": self.config.to_dict(),
            },
            "global_stats": self.stats.copy(),
            "ip_interface_stats": ip_stats,
            "network_stats": {
                "total_transactions": total_transactions,
                "peak_active_requests": self.stats["peak_active_requests"],
                "total_read_retries": self.stats["read_retries"],
                "total_write_retries": self.stats["write_retries"],
            },
            "memory_stats": {
                "flit_pool": get_crossring_flit_pool_stats(),
            }
        }
        
        return results
    
    def get_active_request_count(self) -> int:
        """获取当前活跃请求总数"""
        total = 0
        for ip in self._ip_registry.values():
            total += len(ip.rn_tracker["read"])
            total += len(ip.rn_tracker["write"])
            total += len(ip.sn_tracker)
        return total
    
    def get_global_tracker_status(self) -> Dict[str, Any]:
        """
        获取全局tracker状态
        
        Returns:
            包含所有IP接口tracker状态的字典
        """
        status = {}
        for key, ip_interface in self._ip_registry.items():
            status[key] = {
                "rn_read_active": len(ip_interface.rn_tracker["read"]),
                "rn_write_active": len(ip_interface.rn_tracker["write"]),
                "rn_read_available": ip_interface.rn_tracker_count["read"],
                "rn_write_available": ip_interface.rn_tracker_count["write"],
                "sn_active": len(ip_interface.sn_tracker),
                "sn_ro_available": ip_interface.sn_tracker_count.get("ro", 0),
                "sn_share_available": ip_interface.sn_tracker_count.get("share", 0),
                "read_retries": ip_interface.read_retry_num_stat,
                "write_retries": ip_interface.write_retry_num_stat,
            }
        return status
    
    def print_debug_status(self) -> None:
        """打印调试状态"""
        status = self.get_global_tracker_status()
        print(f"\n=== CrossRing Model Cycle {self.cycle} Debug Status ===")
        print(f"活跃请求总数: {self.get_active_request_count()}")
        print(f"统计信息: {self.stats}")
        
        print("\nIP接口状态:")
        for ip_key, ip_status in status.items():
            print(f"  {ip_key}: RN({ip_status['rn_read_active']}R+{ip_status['rn_write_active']}W), " +
                  f"SN({ip_status['sn_active']}), 重试({ip_status['read_retries']}R+{ip_status['write_retries']}W)")
    
    def inject_test_traffic(self, 
                           source: NodeId,
                           destination: NodeId, 
                           req_type: str,
                           count: int = 1,
                           burst_length: int = 4) -> List[str]:
        """
        注入测试流量
        
        Args:
            source: 源节点
            destination: 目标节点
            req_type: 请求类型
            count: 请求数量
            burst_length: 突发长度
            
        Returns:
            生成的packet_id列表
        """
        packet_ids = []
        
        # 找到源节点对应的IP接口
        source_ips = [ip for key, ip in self._ip_registry.items() 
                     if ip.node_id == source]
        
        if not source_ips:
            self.logger.warning(f"源节点 {source} 没有找到对应的IP接口")
            return packet_ids
        
        # 使用第一个找到的IP接口
        ip_interface = source_ips[0]
        
        for i in range(count):
            packet_id = f"test_{source}_{destination}_{req_type}_{self.cycle}_{i}"
            success = ip_interface.enqueue_request(
                source=source,
                destination=destination,
                req_type=req_type,
                burst_length=burst_length,
                packet_id=packet_id
            )
            
            if success:
                packet_ids.append(packet_id)
                self.logger.debug(f"注入测试请求: {packet_id}")
            else:
                self.logger.warning(f"测试请求注入失败: {packet_id}")
        
        return packet_ids
    
    def get_model_summary(self) -> Dict[str, Any]:
        """获取模型摘要信息"""
        return {
            "config_name": self.config.config_name,
            "topology": f"{self.config.num_row}x{self.config.num_col}",
            "total_nodes": self.config.num_nodes,
            "ip_interfaces": len(self.ip_interfaces),
            "current_cycle": self.cycle,
            "active_requests": self.get_active_request_count(),
            "simulation_status": {
                "is_running": self.is_running,
                "is_finished": self.is_finished,
            },
            "networks_ready": {
                "req": self.networks["req"] is not None,
                "rsp": self.networks["rsp"] is not None,
                "data": self.networks["data"] is not None,
            }
        }
    
    def cleanup(self) -> None:
        """清理资源"""
        self.logger.info("开始清理CrossRing模型资源")
        
        # 清理IP接口
        for ip in self.ip_interfaces.values():
            # 这里可以添加IP接口的清理逻辑
            pass
        
        # 清理网络组件
        self.crossring_pieces.clear()
        
        # 清理统计信息
        self.stats.clear()
        
        self.logger.info("CrossRing模型资源清理完成")
    
    def __del__(self):
        """析构函数"""
        if hasattr(self, 'logger'):
            self.logger.debug("CrossRing模型对象被销毁")
    
    @property
    def total_active_requests(self) -> int:
        """总活跃请求数（属性访问）"""
        return self.get_active_request_count()
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (f"CrossRingModel({self.config.config_name}, "
                f"{self.config.num_row}x{self.config.num_col}, "
                f"cycle={self.cycle}, "
                f"active_requests={self.get_active_request_count()})")


def create_crossring_model(config_name: str = "default",
                          num_row: int = 5,
                          num_col: int = 4,
                          **config_kwargs) -> CrossRingModel:
    """
    创建CrossRing模型的便捷函数
    
    Args:
        config_name: 配置名称
        num_row: 行数
        num_col: 列数
        **config_kwargs: 其他配置参数
        
    Returns:
        CrossRing模型实例
    """
    config = CrossRingConfig(
        num_col=num_col,
        num_row=num_row,
        config_name=config_name
    )
    
    # 应用额外的配置参数
    if config_kwargs:
        config.from_dict(config_kwargs)
    
    return CrossRingModel(config)