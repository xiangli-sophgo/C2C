"""
C2C仿真引擎模块
核心仿真引擎，集成现有拓扑和协议组件
"""

import heapq
from typing import Dict, List, Optional, Any, Tuple
from src.c2c.topology.builder import TopologyBuilder
from src.c2c.topology.graph import TopologyGraph
from src.c2c.topology.link import C2CDirectLink, PCIeLink
from src.c2c.protocol.cdma_system import CDMASystem
from .events import SimulationEvent, EventType, EventFactory
from .fake_chip import FakeChip
from .stats import SimulationStats
import logging
import time


class C2CSimulationEngine:
    """
    C2C仿真引擎，集成现有拓扑和协议组件
    提供事件驱动的芯片间通信仿真
    """

    def __init__(self, topology_builder: TopologyBuilder, log_level: int = logging.INFO):
        """
        初始化仿真引擎

        Args:
            topology_builder: 拓扑构建器实例
            log_level: 日志级别
        """
        self.topology_builder = topology_builder
        self.topology_graph = topology_builder._topology_graph

        # 仿真状态
        self.current_time_ns = 0
        self.simulation_running = False
        self.max_simulation_time_ns = 0

        # 事件队列（优先队列，按时间戳排序）
        self.event_queue: List[SimulationEvent] = []
        self.processed_events: List[SimulationEvent] = []

        # 芯片映射 - 将拓扑中的ChipNode映射到FakeChip
        self.chip_map: Dict[str, FakeChip] = {}
        self._initialize_chip_mapping()

        # 统计收集器
        self.stats = SimulationStats()

        # 日志配置
        logging.basicConfig(level=log_level)
        self.logger = logging.getLogger("C2CSimulationEngine")

        print(f"初始化C2C仿真引擎，拓扑ID: {self.topology_graph.topology_id}")
        print(f"发现 {len(self.chip_map)} 个芯片节点")

    def _initialize_chip_mapping(self):
        """初始化芯片映射，将TopologyBuilder中的ChipNode转换为FakeChip"""
        for node_id, node in self.topology_builder._nodes.items():
            if hasattr(node, "chip_id"):  # 检查是否为ChipNode
                fake_chip = FakeChip(chip_id=node.chip_id, board_id=node.board_id, cdma_engines=node.cdma_engines)
                self.chip_map[node_id] = fake_chip
                print(f"创建FakeChip映射: {node_id} -> {fake_chip.chip_id}")

        # 建立芯片间的C2C连接
        self._establish_c2c_connections()

    def _establish_c2c_connections(self):
        """根据拓扑图建立芯片间的C2C连接"""
        c2c_port_usage = {chip_id: 0 for chip_id in self.chip_map.keys()}

        for link_id, link in self.topology_builder._links.items():
            if isinstance(link, C2CDirectLink):
                source_id = link.endpoint_a.node_id
                target_id = link.endpoint_b.node_id

                if source_id in self.chip_map and target_id in self.chip_map:
                    source_chip = self.chip_map[source_id]
                    target_chip = self.chip_map[target_id]

                    # 为源芯片分配端口
                    source_port = c2c_port_usage[source_id]
                    if source_port < 5:
                        source_chip.connect_c2c_port(source_port, target_chip)
                        c2c_port_usage[source_id] += 1
                        print(f"建立C2C连接: {source_id}[端口{source_port}] <-> {target_id}")
                    else:
                        print(f"警告: 芯片 {source_id} 的C2C端口已满")

    def add_simulation_event(self, event: SimulationEvent):
        """
        添加仿真事件到事件队列

        Args:
            event: 要添加的仿真事件
        """
        heapq.heappush(self.event_queue, event)
        self.logger.debug(f"添加事件到队列: {event}")

    def add_cdma_send_event(self, timestamp_ns: int, source_chip_id: str, target_chip_id: str, data_size: int = 1024, priority: int = 0):
        """
        添加CDMA发送事件的便捷方法

        Args:
            timestamp_ns: 事件时间戳
            source_chip_id: 源芯片ID
            target_chip_id: 目标芯片ID
            data_size: 数据大小（字节）
            priority: 事件优先级
        """
        event = EventFactory.create_cdma_send_event(timestamp_ns, source_chip_id, target_chip_id, data_size, priority)
        self.add_simulation_event(event)
        print(f"添加CDMA发送事件: {source_chip_id} -> {target_chip_id}, " f"时间: {timestamp_ns}ns, 大小: {data_size}字节")

    def add_periodic_traffic(self, source_chip_id: str, target_chip_id: str, period_ns: int, data_size: int, start_time_ns: int = 0, end_time_ns: Optional[int] = None):
        """
        添加周期性流量模式

        Args:
            source_chip_id: 源芯片ID
            target_chip_id: 目标芯片ID
            period_ns: 周期（纳秒）
            data_size: 每次传输的数据大小
            start_time_ns: 开始时间
            end_time_ns: 结束时间（None表示持续到仿真结束）
        """
        if end_time_ns is None:
            end_time_ns = self.max_simulation_time_ns

        current_time = start_time_ns
        event_count = 0

        while current_time <= end_time_ns:
            self.add_cdma_send_event(current_time, source_chip_id, target_chip_id, data_size)
            current_time += period_ns
            event_count += 1

        print(f"添加周期性流量: {source_chip_id} -> {target_chip_id}, " f"周期: {period_ns}ns, 事件数: {event_count}")

    def run_simulation(self, simulation_time_ns: int) -> SimulationStats:
        """
        运行仿真

        Args:
            simulation_time_ns: 仿真时间长度（纳秒）

        Returns:
            仿真统计结果
        """
        self.max_simulation_time_ns = simulation_time_ns
        self.simulation_running = True
        self.current_time_ns = 0

        print(f"开始仿真，总时长: {simulation_time_ns}ns ({simulation_time_ns/1e9:.3f}秒)")

        # 重置统计信息
        self.stats.reset()
        for chip in self.chip_map.values():
            chip.reset_statistics()

        start_real_time = time.time()
        events_processed = 0

        try:
            while self.simulation_running and self.event_queue:
                # 获取下一个事件
                event = heapq.heappop(self.event_queue)

                # 检查是否超过仿真时间
                if event.timestamp_ns > simulation_time_ns:
                    print(f"事件时间戳 {event.timestamp_ns}ns 超过仿真时间，停止仿真")
                    break

                # 更新仿真时间
                self.current_time_ns = event.timestamp_ns

                # 处理事件
                self._process_event(event)
                self.processed_events.append(event)
                events_processed += 1

                # 更新统计信息
                self.stats.update_from_event(event)

                # 定期输出进度
                if events_processed % 1000 == 0:
                    progress = (self.current_time_ns / simulation_time_ns) * 100
                    print(f"仿真进度: {progress:.1f}%, 已处理事件: {events_processed}")

            # 添加仿真结束事件
            end_event = SimulationEvent(timestamp_ns=self.current_time_ns, event_type=EventType.SIMULATION_END, source_chip_id="engine", target_chip_id="all", event_id="simulation_end")
            self.processed_events.append(end_event)

        except KeyboardInterrupt:
            print("仿真被用户中断")
        except Exception as e:
            print(f"仿真过程中发生错误: {e}")
            self.logger.error(f"仿真错误: {e}")
        finally:
            self.simulation_running = False

        # 收集最终统计信息
        self._finalize_statistics()

        end_real_time = time.time()
        real_time_elapsed = end_real_time - start_real_time

        print(f"仿真完成！")
        print(f"仿真时间: {self.current_time_ns}ns ({self.current_time_ns/1e9:.3f}秒)")
        print(f"实际用时: {real_time_elapsed:.3f}秒")
        print(f"处理事件: {events_processed}")
        print(f"仿真加速比: {(self.current_time_ns/1e9)/real_time_elapsed:.2f}x")

        return self.stats

    def _process_event(self, event: SimulationEvent):
        """
        处理单个仿真事件

        Args:
            event: 要处理的事件
        """
        try:
            # 根据事件类型分发到相应的芯片
            if event.source_chip_id in self.chip_map:
                source_chip = self.chip_map[event.source_chip_id]
                source_chip.process_simulation_event(event)

            # 如果是接收事件，也要通知目标芯片
            if event.event_type == EventType.CDMA_RECEIVE and event.target_chip_id in self.chip_map:
                target_chip = self.chip_map[event.target_chip_id]
                target_chip.process_simulation_event(event)

            # 根据事件类型生成后续事件
            self._generate_follow_up_events(event)

        except Exception as e:
            self.logger.error(f"处理事件 {event.event_id} 时发生错误: {e}")

    def _generate_follow_up_events(self, event: SimulationEvent):
        """
        根据当前事件生成后续事件

        Args:
            event: 当前处理的事件
        """
        if event.event_type == EventType.CDMA_SEND:
            # CDMA发送事件后，生成对应的接收事件
            # 计算传输延迟
            transfer_delay_ns = self._calculate_transfer_delay(event.source_chip_id, event.target_chip_id, event.data_size)

            receive_event = EventFactory.create_cdma_receive_event(
                timestamp_ns=event.timestamp_ns + transfer_delay_ns, source_chip_id=event.source_chip_id, target_chip_id=event.target_chip_id, cdma_packet=event.cdma_packet
            )
            receive_event.data_size = event.data_size

            self.add_simulation_event(receive_event)

    def _calculate_transfer_delay(self, source_id: str, target_id: str, data_size: int) -> int:
        """
        计算两个芯片间的传输延迟

        Args:
            source_id: 源芯片ID
            target_id: 目标芯片ID
            data_size: 数据大小

        Returns:
            传输延迟（纳秒）
        """
        # 简化的延迟模型
        base_latency_ns = 1000  # 1微秒基础延迟

        # 根据数据大小计算传输时间（假设1GB/s带宽）
        transfer_time_ns = data_size  # 1 byte = 1 ns

        return base_latency_ns + transfer_time_ns

    def _finalize_statistics(self):
        """完成统计信息收集"""
        # 从所有芯片收集统计信息
        for chip_id, chip in self.chip_map.items():
            chip_stats = chip.get_statistics()
            self.stats.add_chip_stats(chip_id, chip_stats)

        # 设置总仿真时间
        self.stats.total_simulation_time_ns = self.current_time_ns
        self.stats.total_events_processed = len(self.processed_events)

    def get_simulation_results(self) -> Dict[str, Any]:
        """
        获取仿真结果摘要

        Returns:
            包含仿真结果的字典
        """
        results = {
            "simulation_time_ns": self.current_time_ns,
            "events_processed": len(self.processed_events),
            "chips": {chip_id: chip.get_statistics() for chip_id, chip in self.chip_map.items()},
            "topology_info": {"topology_id": self.topology_graph.topology_id, "node_count": len(self.topology_builder._nodes), "link_count": len(self.topology_builder._links)},
            "statistics": self.stats.get_summary(),
        }

        return results

    def stop_simulation(self):
        """停止仿真"""
        self.simulation_running = False
        print("仿真已停止")

    def clear_events(self):
        """清空事件队列"""
        self.event_queue.clear()
        self.processed_events.clear()
        print("事件队列已清空")
