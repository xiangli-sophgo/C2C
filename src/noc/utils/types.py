"""
NoC types and enumerations.

This module defines common types, enumerations, and data structures used across
the NoC (Network-on-Chip) abstraction layer.
"""

from enum import Enum
from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass, field


class TopologyType(Enum):
    """NoC topology type enumeration."""

    CROSSRING = "crossring"
    RING = "ring"
    MESH = "mesh"
    TORUS = "torus"
    TREE = "tree"
    FAT_TREE = "fat_tree"
    BUTTERFLY = "butterfly"
    DRAGONFLY = "dragonfly"


class RoutingStrategy(Enum):
    """Routing strategy enumeration."""

    SHORTEST = "shortest"
    XY = "xy"  # XY路由（dimension-order routing）
    LOAD_BALANCED = "load_balanced"
    ADAPTIVE = "adaptive"
    DETERMINISTIC = "deterministic"
    OBLIVIOUS = "oblivious"
    MINIMAL = "minimal"
    CUSTOM = "custom"


class FlowControlType(Enum):
    """Flow control mechanism enumeration."""

    WORMHOLE = "wormhole"
    STORE_AND_FORWARD = "store_and_forward"
    VIRTUAL_CUT_THROUGH = "virtual_cut_through"
    CIRCUIT_SWITCHING = "circuit_switching"


class BufferType(Enum):
    """Buffer management type enumeration."""

    SHARED = "shared"
    DEDICATED = "dedicated"
    CREDIT_BASED = "credit_based"


class TrafficPattern(Enum):
    """Traffic pattern type enumeration."""

    UNIFORM_RANDOM = "uniform_random"
    HOTSPOT = "hotspot"
    TRANSPOSE = "transpose"
    BIT_COMPLEMENT = "bit_complement"
    BIT_REVERSAL = "bit_reversal"
    NEAREST_NEIGHBOR = "nearest_neighbor"
    TRACE_BASED = "trace_based"


class LinkType(Enum):
    """Link type enumeration."""

    BIDIRECTIONAL = "bidirectional"
    UNIDIRECTIONAL = "unidirectional"
    WIRELESS = "wireless"
    OPTICAL = "optical"


class Priority(Enum):
    """Priority level enumeration."""

    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class NoCMetrics:
    """NoC performance metrics data class."""

    # Latency metrics
    average_latency: float = 0.0
    max_latency: float = 0.0
    min_latency: float = 0.0

    # Throughput metrics
    throughput: float = 0.0
    effective_throughput: float = 0.0
    saturation_throughput: float = 0.0

    # Hop count metrics
    average_hop_count: float = 0.0
    max_hop_count: int = 0
    min_hop_count: int = 0

    # Link utilization metrics
    average_link_utilization: float = 0.0
    max_link_utilization: float = 0.0

    # Buffer metrics
    average_buffer_occupancy: float = 0.0
    max_buffer_occupancy: float = 0.0

    # Energy metrics
    total_energy: float = 0.0
    energy_per_bit: float = 0.0

    # Network diameter and bisection bandwidth
    network_diameter: int = 0
    bisection_bandwidth: float = 0.0

    # Fault tolerance metrics
    fault_tolerance: float = 0.0
    reliability: float = 0.0

    # Additional custom metrics
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NoCConfiguration:
    """NoC configuration parameters data class."""

    # Basic topology parameters
    topology_type: TopologyType = TopologyType.MESH
    num_nodes: int = 16
    dimensions: Tuple[int, ...] = (4, 4)

    # Routing configuration
    routing_strategy: RoutingStrategy = RoutingStrategy.SHORTEST
    adaptive_threshold: float = 0.7

    # Flow control
    flow_control: FlowControlType = FlowControlType.WORMHOLE
    buffer_type: BufferType = BufferType.SHARED

    # Buffer sizes
    input_buffer_size: int = 8
    output_buffer_size: int = 8

    # Link parameters
    link_bandwidth: float = 1.0  # GB/s
    link_latency: int = 1  # cycles

    # Packet/Flit parameters
    flit_size: int = 128  # bits
    packet_size: int = 512  # bits
    header_size: int = 64  # bits

    # Simulation parameters
    simulation_cycles: int = 10000
    warmup_cycles: int = 1000
    stats_collection_start: int = 1000

    # Traffic parameters
    traffic_pattern: TrafficPattern = TrafficPattern.UNIFORM_RANDOM
    injection_rate: float = 0.1  # flits per cycle per node

    # Clock and timing
    clock_frequency: float = 1.0  # GHz

    # Power and energy
    static_power: float = 0.0  # W
    dynamic_power_per_flit: float = 0.0  # W per flit

    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LinkMetrics:
    """Link-specific metrics data class."""

    utilization: float = 0.0
    bandwidth: float = 0.0
    latency: float = 0.0
    error_rate: float = 0.0
    power_consumption: float = 0.0
    total_flits: int = 0
    blocked_cycles: int = 0


@dataclass
class NodeMetrics:
    """Node-specific metrics data class."""

    # Buffer metrics
    input_buffer_occupancy: float = 0.0
    output_buffer_occupancy: float = 0.0

    # Processing metrics
    packets_processed: int = 0
    packets_generated: int = 0
    packets_received: int = 0

    # Latency metrics
    average_packet_latency: float = 0.0

    # Power metrics
    power_consumption: float = 0.0

    # Routing metrics
    routing_decisions: Dict[str, int] = field(default_factory=dict)

    # Error metrics
    packet_drops: int = 0
    routing_errors: int = 0


@dataclass
class TrafficFlow:
    """Traffic flow specification data class."""

    source: int
    destination: int
    priority: Priority = Priority.MEDIUM
    bandwidth_requirement: float = 0.0
    latency_requirement: float = float("inf")
    packet_size: int = 512
    flow_duration: int = -1  # -1 means infinite
    start_time: int = 0
    end_time: int = -1  # -1 means never end


# Type aliases for better code readability
NodeId = int
Position = Tuple[int, ...]  # Multi-dimensional position coordinates
Path = List[NodeId]  # Path represented as a list of node IDs
AdjacencyMatrix = List[List[int]]  # Adjacency matrix representation
LinkId = Tuple[NodeId, NodeId]  # Link identifier as (source, destination)
Coordinate = Tuple[int, int]  # 2D coordinate (x, y)
Coordinate3D = Tuple[int, int, int]  # 3D coordinate (x, y, z)

# Configuration type aliases
ConfigDict = Dict[str, Any]
MetricsDict = Dict[str, float]
ParameterDict = Dict[str, Any]

# Network state type aliases
BufferState = Dict[NodeId, Dict[str, int]]
LinkState = Dict[LinkId, Dict[str, Any]]
RoutingTable = Dict[NodeId, Dict[NodeId, Path]]

# Simulation type aliases
CycleCount = int
TimeStamp = float
EventList = List[Tuple[TimeStamp, str, Any]]

# Traffic type aliases
TrafficMatrix = List[List[float]]
InjectionRate = Dict[NodeId, float]
FlowSpecification = List[TrafficFlow]


class SimulationState(Enum):
    """Simulation state enumeration."""

    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    FINISHED = "finished"
    ERROR = "error"


class EventType(Enum):
    """Simulation event type enumeration."""

    PACKET_INJECTION = "packet_injection"
    PACKET_ARRIVAL = "packet_arrival"
    PACKET_ROUTING = "packet_routing"
    BUFFER_UPDATE = "buffer_update"
    LINK_TRANSMISSION = "link_transmission"
    STATS_COLLECTION = "stats_collection"
    SIMULATION_END = "simulation_end"


@dataclass
class SimulationEvent:
    """Simulation event data class."""

    timestamp: TimeStamp
    event_type: EventType
    node_id: Optional[NodeId] = None
    data: Optional[Any] = None
    priority: int = 0


@dataclass
class QoSRequirement:
    """Quality of Service requirement specification."""

    max_latency: float = float("inf")
    min_bandwidth: float = 0.0
    max_jitter: float = float("inf")
    max_packet_loss: float = 1.0
    priority: Priority = Priority.MEDIUM


@dataclass
class FaultModel:
    """Fault model specification."""

    fault_type: str = "transient"  # transient, permanent, intermittent
    affected_component: str = "link"  # link, node, buffer
    fault_rate: float = 0.0
    fault_duration: int = 1
    recovery_time: int = 0


# Utility type definitions
ValidationResult = Tuple[bool, Optional[str]]
OptimizationResult = Tuple[bool, Optional[ConfigDict], Optional[str]]
AnalysisResult = Dict[str, Any]

# Constants
DEFAULT_FLIT_SIZE = 128  # bits
DEFAULT_PACKET_SIZE = 512  # bits
DEFAULT_BUFFER_SIZE = 8  # flits
DEFAULT_LINK_BANDWIDTH = 1.0  # GB/s
DEFAULT_CLOCK_FREQUENCY = 1.0  # GHz

# Maximum values for validation
MAX_NODES = 10000
MAX_DIMENSIONS = 10
MAX_BUFFER_SIZE = 1024
MAX_PACKET_SIZE = 4096
