import os
from typing import List, Dict, Tuple, Optional, Iterator
from collections import defaultdict, deque


class TrafficFileReader:
    """按需读取traffic文件的迭代器

    内部处理：
    - 输入：traffic文件中的时间（纳秒）+ time_offset（纳秒）
    - 输出：请求时间转换为网络周期数
    """

    def __init__(self, filename: str, traffic_file_path: str, config, time_offset: int, traffic_id: str, buffer_size: int = 10000):
        """
        Args:
            time_offset: 时间偏移量（纳秒）
        """
        self.filename = filename
        self.abs_path = os.path.join(traffic_file_path, filename)
        self.config = config
        self.time_offset = time_offset
        self.traffic_id = traffic_id
        self.buffer_size = buffer_size

        # 统计信息
        self.total_req = 0
        self.total_flit = 0
        self.read_req = 0
        self.write_req = 0
        self.read_flit = 0
        self.write_flit = 0

        # 文件状态
        self._file_handle = None
        self._buffer = deque()
        self._eof = False
        self._stats_calculated = False

        # 预先扫描文件获取统计信息
        self._calculate_file_stats()

    def _calculate_file_stats(self):
        """预先扫描文件计算总请求数和flit数"""
        if self._stats_calculated:
            return

        with open(self.abs_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 7:
                    continue

                op, burst = parts[5], int(parts[6])
                if op == "R":
                    self.read_req += 1
                    self.read_flit += burst
                else:
                    self.write_req += 1
                    self.write_flit += burst

        self.total_req = self.read_req + self.write_req
        self.total_flit = self.read_flit + self.write_flit
        self._stats_calculated = True

    def _open_file(self):
        """打开文件句柄"""
        if self._file_handle is None:
            self._file_handle = open(self.abs_path, "r")

    def _close_file(self):
        """关闭文件句柄"""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None

    def _fill_buffer(self):
        """填充缓冲区"""
        if self._eof or not self._file_handle:
            return

        count = 0
        lines_batch = []

        # Read in larger batches to reduce I/O overhead
        while count < self.buffer_size:
            if not lines_batch:
                # Read multiple lines at once
                batch_size = min(1000, self.buffer_size - count)
                for _ in range(batch_size):
                    line = self._file_handle.readline()
                    if not line:
                        self._eof = True
                        break
                    lines_batch.append(line)
                if not lines_batch:
                    break

            line = lines_batch.pop(0)
            parts = line.strip().split(",")
            if len(parts) < 7:
                continue

            # 解析请求
            t, src, src_t, dst, dst_t, op, burst = parts
            # t是纳秒，time_offset也是纳秒，都需要转换为网络周期数
            t = (int(t) + self.time_offset) * self.config.NETWORK_FREQUENCY
            src, dst, burst = int(src), int(dst), int(burst)

            req_tuple = (t, src, src_t, dst, dst_t, op, burst, self.traffic_id)
            self._buffer.append(req_tuple)
            count += 1

    def get_requests_until_cycle(self, max_cycle: int) -> List[Tuple]:
        """获取指定周期之前的所有请求

        Args:
            max_cycle: 最大网络周期数

        Returns:
            请求列表，每个请求的时间是网络周期数
        """
        self._open_file()
        results = []

        # 从缓冲区获取
        while self._buffer:
            req = self._buffer[0]
            if req[0] <= max_cycle:  # 比较网络周期数
                results.append(self._buffer.popleft())
            else:
                break

        # 如果缓冲区为空且文件未结束，填充更多数据
        while not self._eof and (not self._buffer or self._buffer[0][0] <= max_cycle):
            self._fill_buffer()

            while self._buffer:
                req = self._buffer[0]
                if req[0] <= max_cycle:  # 比较网络周期数
                    results.append(self._buffer.popleft())
                else:
                    break

        return results

    def peek_next_cycle(self) -> Optional[int]:
        """查看下一个请求的网络周期数，不消费

        Returns:
            下一个请求的网络周期数，如果没有更多请求返回None
        """
        self._open_file()

        if not self._buffer and not self._eof:
            self._fill_buffer()

        if self._buffer:
            return self._buffer[0][0]
        return None

    def has_more_requests(self) -> bool:
        """检查是否还有更多请求"""
        return len(self._buffer) > 0 or not self._eof

    def close(self):
        """关闭文件读取器"""
        self._close_file()
        self._buffer.clear()


class TrafficState:
    """跟踪单个traffic的执行状态"""

    def __init__(self, traffic_id: str, chain_id: str, total_req: int, total_flit: int):
        self.traffic_id = traffic_id
        self.chain_id = chain_id
        self.total_req = total_req
        self.total_flit = total_flit
        self.injected_req = 0
        self.completed_req = 0
        self.sent_flit = 0
        self.received_flit = 0
        self.start_time = 0
        self.end_time = 0
        self.actual_end_time = 0  # 纳秒

    def is_injection_completed(self) -> bool:
        """判断是否所有请求都已注入"""
        return self.injected_req >= self.total_req

    def is_completed(self) -> bool:
        """判断traffic是否完成"""
        # return self.injected_req >= self.total_req and self.sent_flit >= self.total_flit and self.received_flit >= self.total_flit
        return self.injected_req >= self.total_req and self.received_flit >= self.total_flit

    def update_injected_req(self):
        """更新已注入请求数"""
        self.injected_req += 1

    def update_sent_flit(self):
        """更新已发送flit数"""
        self.sent_flit += 1

    def update_received_flit(self):
        """更新已接收flit数"""
        self.received_flit += 1


class SerialChain:
    """表示一条串行的traffic链，每条链维护独立的请求队列"""

    def __init__(self, chain_id: str, traffic_files: List[str]):
        self.chain_id = chain_id
        self.traffic_files = traffic_files
        self.current_index = 0
        self.chain_time_offset = 0  # 纳秒
        self.is_completed = False
        self.start_time = 0
        # 每条链独立的请求队列
        self.pending_requests = deque()
        self.current_traffic_id = None

    def get_current_traffic_file(self) -> Optional[str]:
        """获取当前应该执行的traffic文件"""
        if self.current_index < len(self.traffic_files):
            return self.traffic_files[self.current_index]
        return None

    def advance_to_next(self):
        """推进到下一个traffic

        Args:
            gap_time: 间隔时间（纳秒）
        """
        self.current_index += 1
        self.pending_requests.clear()  # 清空当前队列
        self.current_traffic_id = None
        if self.current_index >= len(self.traffic_files):
            self.is_completed = True

    def has_next_traffic(self) -> bool:
        """检查是否还有下一个traffic"""
        return self.current_index < len(self.traffic_files)

    def has_pending_requests(self) -> bool:
        """检查是否还有待处理的请求"""
        return len(self.pending_requests) > 0

    def get_ready_requests(self, current_cycle: int) -> List[Tuple]:
        """获取所有准备就绪的请求（批量处理）

        Args:
            current_cycle: 当前网络周期数

        Returns:
            准备就绪的请求列表，每个请求的时间是网络周期数
        """
        ready_requests = []

        # 检查队列头部的所有准备就绪请求
        while self.pending_requests:
            next_req = self.pending_requests[0]
            if next_req[0] <= current_cycle:  # 网络周期数比较：请求时间 <= 当前周期
                ready_requests.append(self.pending_requests.popleft())
            else:
                break  # 遇到未到时间的请求，停止

        return ready_requests

    def load_traffic_requests(self, requests: List[Tuple], traffic_id: str):
        """加载traffic的请求到链的队列

        Args:
            requests: 请求列表，每个请求的时间是网络周期数
            traffic_id: traffic标识符
        """
        self.pending_requests.clear()
        self.pending_requests.extend(requests)
        self.current_traffic_id = traffic_id


class TrafficScheduler:
    """Traffic调度器，支持多条并行的串行链

    时间单位约定：
    - time_offset: 纳秒
    - 请求时间: 网络周期数（内部计算）
    - 显示输出: 纳秒（用户友好）
    """

    def __init__(self, config, traffic_file_path: str):
        self.config = config
        self.traffic_file_path = traffic_file_path
        self.parallel_chains: List[SerialChain] = []
        self.active_traffics: Dict[str, TrafficState] = {}
        self.current_cycle = 0  # 当前网络周期数
        self.verbose = False

        # Ring拓扑支持
        self.ring_config = None
        self.use_ring_mapping = False

    def setup_parallel_chains(self, chains_config: List[List[str]]):
        """设置并行的串行链"""
        self.parallel_chains.clear()
        self.active_traffics.clear()

        for i, traffic_files in enumerate(chains_config):
            if not traffic_files:
                continue
            chain_id = f"chain_{i}"
            chain = SerialChain(chain_id, traffic_files)
            self.parallel_chains.append(chain)

        if self.verbose:
            print(f"Setup {len(self.parallel_chains)} parallel chains")
            for chain in self.parallel_chains:
                print(f"  {chain.chain_id}: {chain.traffic_files}")

    def setup_single_chain(self, traffic_files: List[str]):
        """设置单条串行链（向后兼容）"""
        self.setup_parallel_chains([traffic_files])

    def start_initial_traffics(self):
        """启动每条链的第一个traffic"""
        for chain in self.parallel_chains:
            if chain.has_next_traffic():
                self._start_single_traffic(chain)

    def _start_single_traffic(self, chain: SerialChain):
        """启动链中的单个traffic"""
        traffic_file = chain.get_current_traffic_file()
        if not traffic_file:
            return

        traffic_id = f"{chain.chain_id}_t{chain.current_index}"

        # 解析traffic文件
        total_req, total_flit, requests = self._parse_traffic_file(traffic_file, chain.chain_time_offset, traffic_id)

        # 创建traffic状态
        traffic_state = TrafficState(traffic_id, chain.chain_id, total_req, total_flit)
        traffic_state.start_time = chain.chain_time_offset
        self.active_traffics[traffic_id] = traffic_state

        # 将请求加载到对应链的队列中
        chain.load_traffic_requests(requests, traffic_id)

        if self.verbose:
            print(f"Started traffic {traffic_file} on {chain.chain_id}")
            print(f"  Traffic ID: {traffic_id}, Requests: {total_req}, Flits: {total_flit}")
            print(f"  Time offset: {chain.chain_time_offset}ns")
            if requests:
                first_time_ns = requests[0][0] // self.config.NETWORK_FREQUENCY
                last_time_ns = requests[-1][0] // self.config.NETWORK_FREQUENCY
                print(f"  First request time: {first_time_ns}ns ({requests[0][0]} cycles)")
                print(f"  Last request time: {last_time_ns}ns ({requests[-1][0]} cycles)")

    def _parse_traffic_file(self, filename: str, time_offset: int, traffic_id: str) -> Tuple[int, int, List[Tuple]]:
        """解析traffic文件并添加时间偏移

        Args:
            time_offset: 时间偏移量（纳秒）

        Returns:
            (total_req, total_flit, requests)
            requests中的时间已转换为网络周期数
        """
        abs_path = os.path.join(self.traffic_file_path, filename)
        requests = []
        read_req = write_req = read_flit = write_flit = 0

        with open(abs_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 7:
                    continue

                # 解析原始数据
                t, src, src_t, dst, dst_t, op, burst = parts
                # t是纳秒，time_offset也是纳秒，都需要转换为网络周期数
                t = (int(t) + time_offset) * self.config.NETWORK_FREQUENCY
                src, dst, burst = int(src), int(dst), int(burst)

                # 创建带traffic_id的请求元组
                req_tuple = (t, src, src_t, dst, dst_t, op, burst, traffic_id)
                requests.append(req_tuple)

                # 统计
                if op == "R":
                    read_req += 1
                    read_flit += burst
                else:
                    write_req += 1
                    write_flit += burst

        total_req = read_req + write_req
        total_flit = read_flit + write_flit

        return total_req, total_flit, requests

    def get_ready_requests(self, current_cycle: int) -> List[Tuple]:
        """
        获取所有链中准备就绪的请求

        Args:
            current_cycle: 当前网络周期数

        Returns:
            准备就绪的请求列表，每个请求的时间是网络周期数
        """
        self.current_cycle = current_cycle
        ready_requests = []

        # 遍历所有链，获取所有准备就绪的请求
        for chain in self.parallel_chains:
            chain_requests = chain.get_ready_requests(current_cycle)
            ready_requests.extend(chain_requests)

            # 更新注入统计
            for req in chain_requests:
                traffic_id = req[7]  # traffic_id在索引7
                self.update_traffic_stats(traffic_id, "injected_req")

        return ready_requests

    def get_save_filename(self) -> str:
        """
        生成用于保存结果的文件名
        Returns:
            文件名，不包含扩展名
        """
        if not self.parallel_chains:
            return "no_traffic"

        chain_names = []
        for chain in self.parallel_chains:
            # 移除扩展名并连接链中的文件名
            files_without_ext = [f[:-4] if f.endswith(".txt") else f for f in chain.traffic_files]
            chain_name = "-".join(files_without_ext)
            chain_names.append(chain_name)

        # 单条链且只有一个文件：直接返回文件名
        if len(self.parallel_chains) == 1 and len(self.parallel_chains[0].traffic_files) == 1:
            return chain_names[0]

        # 单条链但有多个文件：串行，使用简短前缀
        elif len(self.parallel_chains) == 1:
            return f"s_{chain_names[0]}"

        # 多条链：并行，使用简短前缀
        else:
            combined = "_".join(chain_names)
            return f"p_{combined}"

    def has_pending_requests(self) -> bool:
        """检查是否还有等待处理的请求"""
        return any(chain.has_pending_requests() for chain in self.parallel_chains)

    def update_traffic_stats(self, traffic_id: str, stat_type: str):
        """更新traffic统计信息"""
        if traffic_id in self.active_traffics:
            state = self.active_traffics[traffic_id]
            if stat_type == "injected_req":
                state.update_injected_req()
            elif stat_type == "sent_flit":
                state.update_sent_flit()
            elif stat_type == "received_flit":
                state.update_received_flit()
                # 记录实际结束时间（纳秒）
                if state.received_flit >= state.total_flit:
                    state.actual_end_time = self.current_cycle // self.config.NETWORK_FREQUENCY

    def check_and_advance_chains(self, current_cycle: int) -> List[str]:
        """检查traffic完成情况并推进链

        Args:
            current_cycle: 当前网络周期数

        Returns:
            已完成的traffic_id列表
        """
        self.current_cycle = current_cycle
        completed_traffics = []

        for traffic_id, state in list(self.active_traffics.items()):
            # 检查对应的链是否还有请求要注入
            chain = self._find_chain_by_id(state.chain_id)
            injection_done = state.is_injection_completed() and (not chain or not chain.has_pending_requests())

            if injection_done and state.is_completed():
                completed_traffics.append(traffic_id)

                if chain:
                    # actual_end_time已经是纳秒时间
                    actual_end_ns = state.actual_end_time
                    state.end_time = actual_end_ns

                    # 更新链的时间偏移（实际结束时间 + 间隔）
                    gap_time = 0  # 两个traffic之间的间隔
                    chain.chain_time_offset = actual_end_ns + gap_time

                    if self.verbose:
                        print(f"Traffic {traffic_id} completed:")
                        print(f"  Actual end time: {actual_end_ns}ns")
                        print(f"  Next time offset: {chain.chain_time_offset}ns")

                    # 推进到下一个traffic
                    chain.advance_to_next()

                    # 如果链还有下一个traffic，立即启动
                    if chain.has_next_traffic():
                        self._start_single_traffic(chain)
                    else:
                        if self.verbose:
                            print(f"Chain {chain.chain_id} completed")

                del self.active_traffics[traffic_id]

        return completed_traffics

    def _find_chain_by_id(self, chain_id: str) -> Optional[SerialChain]:
        """根据chain_id查找链"""
        for chain in self.parallel_chains:
            if chain.chain_id == chain_id:
                return chain
        return None

    def is_all_completed(self) -> bool:
        """检查是否所有链都已完成"""
        return len(self.active_traffics) == 0 and not self.has_pending_requests() and all(chain.is_completed for chain in self.parallel_chains)

    def get_active_traffic_count(self) -> int:
        """获取当前活跃的traffic数量"""
        return len(self.active_traffics)

    def get_chain_status(self) -> Dict[str, Dict]:
        """获取所有链的状态信息

        Returns:
            链状态字典，time_offset单位为纳秒
        """
        status = {}
        for chain in self.parallel_chains:
            status[chain.chain_id] = {
                "current_index": chain.current_index,
                "total_traffics": len(chain.traffic_files),
                "current_file": chain.get_current_traffic_file(),
                "time_offset": chain.chain_time_offset,
                "pending_requests": len(chain.pending_requests),
                "current_traffic_id": chain.current_traffic_id,
                "is_completed": chain.is_completed,
            }
        return status

    def set_verbose(self, verbose: bool):
        """设置详细输出模式"""
        self.verbose = verbose

    def get_finish_time_stats(self) -> Dict[str, int]:
        """获取读写操作的结束时间统计

        Returns:
            结束时间统计字典，所有时间单位为纳秒
        """
        read_end_times = []
        write_end_times = []
        all_end_times = []

        # 从所有活跃和已完成的traffic中收集结束时间
        for traffic_state in self.active_traffics.values():
            if traffic_state.actual_end_time > 0:
                # 这里简化处理，实际中可能需要根据traffic文件内容来区分读写
                # 目前假设每个traffic都包含读写操作
                # actual_end_time已经是纳秒时间
                end_time_ns = traffic_state.actual_end_time
                read_end_times.append(end_time_ns)
                write_end_times.append(end_time_ns)
                all_end_times.append(end_time_ns)

        return {
            "R_finish_time": max(read_end_times) if read_end_times else 0,
            "W_finish_time": max(write_end_times) if write_end_times else 0,
            "Total_finish_time": max(all_end_times) if all_end_times else 0,
        }

    def reset(self):
        """重置调度器状态"""
        self.parallel_chains.clear()
        self.active_traffics.clear()
        self.current_cycle = 0  # 当前网络周期数
