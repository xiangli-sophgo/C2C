"""
Flit and TokenBucket classes for NoC simulation.
Contains the basic data unit (Flit) and rate limiting mechanism (TokenBucket).
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .network import Network  # 仅用于 IDE 类型提示

import numpy as np
from collections import deque
from typing import List, Optional, Union
import threading


class TokenBucket:
    """Simple token bucket for rate limiting with fractional tokens."""

    def __init__(self, rate: float = 1.0, bucket_size: float = 10.0):
        # rate: tokens added per cycle; bucket_size: max tokens stored
        self.rate = float(rate)
        self.bucket_size = float(bucket_size)
        self.tokens = self.bucket_size
        self.last_cycle = 0

    def consume(self, num: float = 1.0) -> bool:
        """Attempt to consume `num` tokens. Returns True if sufficient tokens."""
        # print(self.tokens)
        if self.tokens + 1e-8 >= num:
            self.tokens -= num
            return True
        return False

    def refill(self, cycle: int):
        """Refill tokens based on cycles elapsed since last refill."""
        # Calculate elapsed cycles
        dt = cycle - self.last_cycle
        if dt <= 0:
            return
        # Add fractional tokens
        added = dt * self.rate
        # Update last refill time
        self.last_cycle = cycle
        # Cap tokens to bucket size
        self.tokens = min(self.tokens + added, self.bucket_size)


class Flit:
    __slots__ = [
        "source",
        "source_original",
        "destination",
        "destination_original",
        "source_type",
        "destination_type",
        "original_source_type",
        "original_destination_type",
        "burst_length",
        "path",
        "flit_position",
        "is_finish",
        "packet_id",
        "traffic_id",
        "moving_direction",
        "moving_direction_v",
        "flit_type",
        "req_type",
        "req_attr",
        "req_state",
        "id",
        "flit_id",
        "is_last_flit",
        "circuits_completed_v",
        "circuits_completed_h",
        "wait_cycle_h",
        "wait_cycle_v",
        "path_index",
        "current_seat_index",
        "current_link",
        "rsp_type",
        "rn_tracker_type",
        "sn_tracker_type",
        "sn_rsp_generate_cycle",
        "early_rsp",
        "current_position",
        "station_position",
        "departure_cycle",
        "req_departure_cycle",
        "departure_network_cycle",
        "departure_inject_cycle",
        "arrival_cycle",
        "arrival_network_cycle",
        "arrival_eject_cycle",
        "entry_db_cycle",
        "leave_db_cycle",
        "start_inject",
        "is_injected",
        "is_ejected",
        "is_new_on_network",
        "is_on_station",
        "is_delay",
        "is_arrive",
        "predicted_duration",
        "actual_duration",
        "actual_ject_duration",
        "actual_network_duration",
        "itag_v",
        "itag_h",
        "is_tagged",
        "ETag_priority",
        "used_entry_level",
        "cmd_entry_cake0_cycle",
        "cmd_entry_noc_from_cake0_cycle",
        "cmd_entry_noc_from_cake1_cycle",
        "cmd_received_by_cake0_cycle",
        "cmd_received_by_cake1_cycle",
        "data_entry_noc_from_cake0_cycle",
        "data_entry_noc_from_cake1_cycle",
        "data_received_complete_cycle",
        "data_entry_network_cycle",
        "rsp_entry_network_cycle",
        "transaction_latency",
        "cmd_latency",
        "data_latency",
    ]

    last_id = 0

    def __init__(self, source, destination, path):
        self.source = source
        self.source_original = -1
        self.destination = destination
        self.destination_original = -1
        self.source_type = None
        self.destination_type = None
        self.original_source_type = None
        self.original_destination_type = None
        self.burst_length = -1
        self.path = path
        self.flit_position = ""
        self.is_finish = False
        Flit.last_id += 1
        self.packet_id = None
        # 标记该flit所属的traffic
        self.traffic_id = None
        self.moving_direction = self.calculate_direction(path)
        self.moving_direction_v = 1 if source < destination else -1
        self.flit_type = "flit"
        self.req_type = None
        self.req_attr = "new"
        self.req_state = "valid"
        self.id = Flit.last_id
        self.flit_id = -1
        self.is_last_flit = False
        self.circuits_completed_v = 0
        self.circuits_completed_h = 0
        self.wait_cycle_h = 0
        self.wait_cycle_v = 0
        self.path_index = 0
        self.current_seat_index = -1
        self.current_link = None
        self.rsp_type = None
        self.rn_tracker_type = None
        self.sn_tracker_type = None
        self.init_param()

    def init_param(self):
        self.early_rsp = False
        self.current_position = self.path[0]
        self.station_position = -1
        self.departure_cycle = np.inf
        self.req_departure_cycle = np.inf
        self.departure_network_cycle = np.inf
        self.departure_inject_cycle = np.inf
        self.arrival_cycle = np.inf
        self.arrival_network_cycle = np.inf
        self.arrival_eject_cycle = np.inf
        self.entry_db_cycle = np.inf
        self.leave_db_cycle = np.inf
        self.start_inject = False
        self.is_injected = False
        self.is_ejected = False
        self.is_new_on_network = True
        self.is_on_station = False
        self.is_delay = False
        self.is_arrive = False
        self.predicted_duration = 0
        self.actual_duration = 0
        self.actual_ject_duration = 0
        self.actual_network_duration = 0
        self.itag_v = False
        self.itag_h = False
        self.is_tagged = False
        self.ETag_priority = "T2"  # 默认优先级为 T2
        # 记录下环 / 弹出时实际占用的是哪一级 entry（"T0" / "T1" / "T2"）
        self.used_entry_level = None
        # Latency record
        self.cmd_entry_cake0_cycle = np.inf
        self.cmd_entry_noc_from_cake0_cycle = np.inf
        self.cmd_entry_noc_from_cake1_cycle = np.inf
        self.cmd_received_by_cake0_cycle = np.inf
        self.cmd_received_by_cake1_cycle = np.inf
        self.data_entry_noc_from_cake0_cycle = np.inf
        self.data_entry_noc_from_cake1_cycle = np.inf
        self.data_received_complete_cycle = np.inf
        self.data_entry_network_cycle = np.inf
        self.rsp_entry_network_cycle = np.inf
        self.transaction_latency = np.inf
        self.cmd_latency = np.inf
        self.data_latency = np.inf
        self.sn_rsp_generate_cycle = np.inf

    def sync_latency_record(self, flit):
        if flit.req_type == "read":
            self.cmd_entry_cake0_cycle = min(flit.cmd_entry_cake0_cycle, self.cmd_entry_cake0_cycle)
            self.cmd_entry_noc_from_cake0_cycle = min(flit.cmd_entry_noc_from_cake0_cycle, self.cmd_entry_noc_from_cake0_cycle)
            self.cmd_received_by_cake1_cycle = min(flit.cmd_received_by_cake1_cycle, self.cmd_received_by_cake1_cycle)
            self.data_entry_noc_from_cake1_cycle = min(flit.data_entry_noc_from_cake1_cycle, self.data_entry_noc_from_cake1_cycle)
            self.data_received_complete_cycle = min(flit.data_received_complete_cycle, self.data_received_complete_cycle)
        elif flit.req_type == "write":
            self.cmd_entry_cake0_cycle = min(flit.cmd_entry_cake0_cycle, self.cmd_entry_cake0_cycle)
            self.cmd_entry_noc_from_cake0_cycle = min(flit.cmd_entry_noc_from_cake0_cycle, self.cmd_entry_noc_from_cake0_cycle)
            self.cmd_received_by_cake1_cycle = min(flit.cmd_received_by_cake1_cycle, self.cmd_received_by_cake1_cycle)
            self.cmd_entry_noc_from_cake1_cycle = min(flit.cmd_entry_noc_from_cake1_cycle, self.cmd_entry_noc_from_cake1_cycle)
            self.cmd_received_by_cake0_cycle = min(flit.cmd_received_by_cake0_cycle, self.cmd_received_by_cake0_cycle)
            self.data_entry_noc_from_cake0_cycle = min(flit.data_entry_noc_from_cake0_cycle, self.data_entry_noc_from_cake0_cycle)
            self.data_received_complete_cycle = min(flit.data_received_complete_cycle, self.data_received_complete_cycle)

    def calculate_direction(self, path):
        if len(path) < 2:
            return 0  # Or handle this case appropriately
        return 1 if path[1] - path[0] == 1 else -1 if path[1] - path[0] == -1 else 0

    def inject(self, network: "Network"):  # 使用字符串类型标注
        if self.path_index == 0 and not self.is_injected:
            if len(self.path) > 1:  # Ensure there is a next position
                next_position = self.path[self.path_index + 1]
                if network.can_move_to_next(self, self.source, next_position):
                    self.current_position = self.source
                    self.is_injected = True
                    self.is_new_on_network = True
                    self.current_link = None
                    return True
        return False

    def __repr__(self):
        req_attr = "O" if self.req_attr == "old" else "N"
        type_display = self.rsp_type[:3] if self.rsp_type else self.req_type[0]
        flit_position = (
            f"{self.current_position}:{self.flit_position}"
            if self.flit_position != "Link"
            else f"({self.current_position}: {self.current_link[0]}->{self.current_link[1]}).{self.current_seat_index}, "
        )
        finish_status = "F" if self.is_finish else ""
        eject_status = "E" if self.is_ejected else ""
        ITag_H = "H" if self.itag_h else ""
        ITag_V = "V" if self.itag_v else ""

        return (
            f"{self.packet_id}.{self.flit_id} {self.source}.{self.source_type[0]}{self.source_type[-1]}->{self.destination}.{self.destination_type[0]}{self.destination_type[-1]}: "
            f"{flit_position}, "
            f"{req_attr}, {self.flit_type}, {type_display}, "
            f"{finish_status}{eject_status}, "
            f"{self.ETag_priority}, {ITag_H}, {ITag_V}"
        )

    def _reset_for_reuse(self):
        """Reset flit to clean state for object pool reuse"""
        # Reset critical fields but keep __slots__ structure
        self.is_finish = False
        self.is_arrive = False
        self.is_injected = False
        self.is_ejected = False
        self.is_new_on_network = True
        self.is_on_station = False
        self.is_delay = False
        self.wait_cycle_h = 0
        self.wait_cycle_v = 0
        self.circuits_completed_h = 0
        self.circuits_completed_v = 0
        self.path_index = 0
        self.current_seat_index = -1
        self.current_link = None
        self.traffic_id = None

        # Reset timing fields
        self.departure_cycle = np.inf
        self.req_departure_cycle = np.inf
        self.departure_network_cycle = np.inf
        self.departure_inject_cycle = np.inf
        self.arrival_cycle = np.inf
        self.arrival_network_cycle = np.inf
        self.arrival_eject_cycle = np.inf

        # Reset latency fields
        self.transaction_latency = np.inf
        self.cmd_latency = np.inf
        self.data_latency = np.inf

    @classmethod
    def create_flit(cls, source, destination, path):
        """Factory method to create Flit using object pool"""
        return _flit_pool.get_flit(source, destination, path)

    @classmethod
    def return_to_pool(cls, flit):
        """Return flit to object pool"""
        _flit_pool.return_flit(flit)

    @classmethod
    def clear_flit_id(cls):
        cls.last_id = 0

    @classmethod
    def get_pool_stats(cls):
        """Get object pool statistics"""
        return _flit_pool.get_stats()


class FlitPool:
    """Object pool for Flit instances to reduce GC pressure"""

    def __init__(self, initial_size=1000):
        self._pool = deque()
        self._lock = threading.Lock()
        self._created_count = 0

        # Pre-populate the pool
        for _ in range(initial_size):
            self._pool.append(self._create_new_flit())

    def _create_new_flit(self):
        """Create a new Flit instance"""
        self._created_count += 1
        return Flit.__new__(Flit)

    def get_flit(self, source, destination, path):
        """Get a Flit from the pool or create a new one"""
        with self._lock:
            if self._pool:
                flit = self._pool.popleft()
            else:
                flit = self._create_new_flit()

        # Initialize the flit
        flit.__init__(source, destination, path)
        return flit

    def return_flit(self, flit):
        """Return a Flit to the pool after use"""
        if flit is None:
            return

        # Reset the flit to a clean state
        flit._reset_for_reuse()

        with self._lock:
            if len(self._pool) < 2000:  # Limit pool size
                self._pool.append(flit)

    def get_stats(self):
        """Get pool statistics"""
        with self._lock:
            return {"pool_size": len(self._pool), "created_count": self._created_count}


# Global flit pool instance
_flit_pool = FlitPool()
