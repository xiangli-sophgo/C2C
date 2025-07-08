# CrossRing NoC Architecture Documentation

## 1. Overview

CrossRing is a Network-on-Chip (NoC) architecture designed for high-performance chip-to-chip communication, implementing the **Cross Ring Spec v2.0**. It features a specialized ring-based topology with horizontal and vertical rings, utilizing advanced slot-based transmission, I-Tag/E-Tag anti-starvation mechanisms, and efficient XY dimension-order routing.

### Key Features of Current Implementation

- **CrossRingSlot**: Compliant with Cross Ring Spec v2.0, containing Valid bit, I-Tag, E-Tag, and Flit data
- **RingSlice Pipeline**: Ring transmission basic units supporting pipeline operations
- **I-Tag Injection Reservation**: Prevents injection starvation through slot reservation
- **E-Tag Priority Upgrade**: Prevents ejection starvation through priority escalation
- **Simplified CrossPoint**: Standardized interfaces with clear responsibilities
- **Non-wrap-around Boundaries**: Edge nodes connect to themselves, not wrapping around

## 2. Core Concepts

### 2.1 CrossRingSlot Structure

The `CrossRingSlot` is the fundamental transmission unit in CrossRing, compliant with Cross Ring Spec v2.0:

```python
@dataclass
class CrossRingSlot:
    # Basic information
    slot_id: int
    cycle: int
    channel: str = "req"  # req/rsp/data
    
    # Slot content (per Cross Ring Spec v2.0)
    valid: bool = False              # Valid bit
    flit: Optional['CrossRingFlit'] = None  # Flit data
    
    # I-Tag information (injection reservation mechanism)
    itag_reserved: bool = False      # Whether reserved
    itag_direction: Optional[str] = None  # Reservation direction(TR/TL/TU/TD)
    itag_reserver_id: Optional[int] = None  # Reserver node ID
    
    # E-Tag information (ejection priority mechanism)
    etag_marked: bool = False        # Whether E-Tag marked
    etag_priority: PriorityLevel = PriorityLevel.T2  # T0/T1/T2 priority
    etag_direction: Optional[str] = None  # Mark direction
```

### 2.2 RingSlice Pipeline

The `RingSlice` is the basic transmission unit on rings, supporting pipeline operations:

- **Input Buffer**: Receives slots from previous ring segments
- **Current Slots**: Active slots being processed in current cycle
- **Output Buffer**: Completed slots ready for transmission to next segment
- **Pipeline Flow**: Input → Current → Output in successive cycles

## 3. CrossRing Topology Structure

### 3.1 Non-Wrap-Around Ring Topology

CrossRing implements a specialized ring topology with the following characteristics:

- **Bidirectional Rings**: Each ring supports both clockwise and counter-clockwise transmission
- **Non-Wrap-Around Boundaries**: **Edge nodes connect to themselves**, not wrapping around
- **Ring Intersection**: CrossPoint modules manage communication between horizontal and vertical rings

```
CrossRing Structure Example (4x4):
  0 ←→ 1 ←→ 2 ←→ 3 (3 connects to itself when going right)
  ↑    ↑    ↑    ↑
  ↓    ↓    ↓    ↓
  4 ←→ 5 ←→ 6 ←→ 7 (7 connects to itself when going right)
  ↑    ↑    ↑    ↑
  ↓    ↓    ↓    ↓
  8 ←→ 9 ←→ 10←→ 11 (11 connects to itself when going right)
  ↑    ↑    ↑    ↑
  ↓    ↓    ↓    ↓
  12←→ 13←→ 14←→ 15 (15 connects to itself when going right)
  (12,13,14,15 connect to themselves when going down)
```

**Key Difference**: Unlike traditional ring topologies, CrossRing boundary nodes connect to themselves, creating a unique transmission pattern optimized for the specific requirements of chip-to-chip communication.

### 3.2 Ring Direction System

The CrossRing topology uses four directional channels:

- **TL (Turn Left)**: Leftward horizontal movement
- **TR (Turn Right)**: Rightward horizontal movement  
- **TU (Turn Up)**: Upward vertical movement
- **TD (Turn Down)**: Downward vertical movement

### 3.3 CrossPoint Modules

CrossPoint modules are located at ring intersections with **simplified responsibilities**:

- **Ring Slice Interface**: Standardized interface to RingSlice chains
- **Injection Queue Management**: Managing local node injection queues
- **Basic Arbitration**: Simple priority-based arbitration
- **Status Reporting**: Providing visibility into crosspoint state

**Key Simplification**: The new CrossPoint design removes complex transmission control logic and focuses on clean interfaces between rings and local nodes.

## 4. Flow Control and Anti-Starvation Mechanisms

### 4.1 I-Tag Injection Reservation Mechanism

The I-Tag mechanism prevents injection starvation through slot reservation:

```python
# I-Tag trigger conditions
def should_trigger_itag(self, channel: str, direction: str, waiting_cycles: int) -> bool:
    threshold = self.itag_config['trigger_threshold']  # e.g., 100 cycles
    return waiting_cycles >= threshold

# I-Tag reservation process
def trigger_itag_reservation(self, channel: str, direction: str, ring_slice: RingSlice, cycle: int) -> bool:
    # Find available slot in ring
    available_slots = ring_slice.get_unreserved_slots(channel)
    if available_slots:
        slot = available_slots[0]
        slot.reserve_itag(reserver_id=self.node_id, direction=direction)
        return True
    return False
```

**Key Benefits**:
- Guarantees eventual injection for all nodes
- Prevents indefinite waiting in high-contention scenarios
- Maintains fairness across different traffic patterns

### 4.2 E-Tag Priority Upgrade Mechanism

The E-Tag mechanism prevents ejection starvation through priority escalation:

```python
# E-Tag priority levels
class PriorityLevel(Enum):
    T0 = "T0"  # Highest priority
    T1 = "T1"  # Medium priority  
    T2 = "T2"  # Lowest priority (default)

# Priority upgrade logic
def should_upgrade_etag(self, slot: CrossRingSlot, channel: str, direction: str, failed_attempts: int) -> Optional[PriorityLevel]:
    current_priority = slot.etag_priority
    
    # T2 -> T1 upgrade (any direction, 1 failed attempt)
    if current_priority == PriorityLevel.T2 and failed_attempts >= 1:
        return PriorityLevel.T1
    
    # T1 -> T0 upgrade (only TL/TU directions, 2 failed attempts)
    if (current_priority == PriorityLevel.T1 and 
        failed_attempts >= 2 and 
        direction in ["TL", "TU"]):
        return PriorityLevel.T0
    
    return None
```

**Priority-Based Ejection Control**:
- **T2**: Can eject if FIFO depth < 50% capacity
- **T1**: Can eject if FIFO depth < 93.75% capacity  
- **T0**: Uses round-robin polling mechanism

### 4.3 Tag Mechanism Coordination

I-Tag and E-Tag work together to provide end-to-end QoS guarantees:

1. **Injection Protection**: I-Tag ensures fair injection opportunities
2. **Transmission**: Normal ring transmission with slot-based flow
3. **Ejection Protection**: E-Tag ensures eventual successful ejection
4. **Performance**: Maintains high throughput while preventing starvation

## 5. Routing Algorithm

### 5.1 XY Dimension-Order Routing (DOR)

CrossRing implements XY dimension-order routing with boundary handling:

1. **Horizontal First**: Route horizontally to target column
2. **Vertical Second**: Route vertically to target row
3. **Boundary Handling**: Edge nodes connect to themselves when reaching boundaries

```python
# Current routing logic
def _get_next_node_in_direction(self, node_id: int, direction: RingDirection) -> int:
    x, y = self._get_node_coordinates(node_id)
    
    if direction == RingDirection.TL:
        # Leftward: if at left boundary, connect to self
        if x == 0:
            next_x = x  # Connect to self
        else:
            next_x = x - 1
        next_y = y
    elif direction == RingDirection.TR:
        # Rightward: if at right boundary, connect to self
        if x == self.config.num_col - 1:
            next_x = x  # Connect to self
        else:
            next_x = x + 1
        next_y = y
    # Similar logic for TU and TD...
    
    return next_y * self.config.num_col + next_x
```

**Key Difference**: Unlike wrap-around topologies, CrossRing's boundary nodes create self-loops, which optimizes for the specific communication patterns in chip-to-chip scenarios.

### 5.2 Transmission Decision Making

The system makes decisions between:
- **Ring Transmission**: Continue slot transmission along the ring
- **Ring Injection**: Inject new slots into ring (with I-Tag protection)
- **Ring Ejection**: Remove slots from ring at destination (with E-Tag protection)

### 5.3 Non-Wrap-Around Boundary Handling

Edge nodes implement self-connection instead of wrap-around:
- **Leftmost nodes**: Connect to themselves when moving left
- **Rightmost nodes**: Connect to themselves when moving right
- **Topmost nodes**: Connect to themselves when moving up
- **Bottommost nodes**: Connect to themselves when moving down

**Rationale**: This boundary behavior is optimized for CrossRing's specific use cases and avoids the complexity and potential performance issues of wrap-around connections.

*Content moved to section 4 - Flow Control and Anti-Starvation Mechanisms*

## 6. Node and Component Architecture

### 6.1 CrossRingNode Components

Each CrossRing node contains the following essential components:

```python
# Node architecture from current implementation
class CrossRingNode:
    def __init__(self, node_id: int, node_type: str):
        self.node_id = node_id
        self.node_type = node_type  # "processor", "memory", "io"
        
        # Core components
        self.crosspoint = CrossRingCrossPoint(node_id=node_id)
        self.ring_slices = {
            "TL": RingSlice(direction="TL", node_id=node_id),
            "TR": RingSlice(direction="TR", node_id=node_id),
            "TU": RingSlice(direction="TU", node_id=node_id),
            "TD": RingSlice(direction="TD", node_id=node_id)
        }
        
        # Tag management
        self.tag_manager = CrossRingTagManager(node_id=node_id)
        
        # Local interfaces
        self.injection_queues = {
            "req": [],
            "rsp": [], 
            "data": []
        }
        self.ejection_queues = {
            "req": [],
            "rsp": [],
            "data": []
        }
```

### 6.2 CrossRingCrossPoint Architecture

The simplified CrossPoint focuses on interface management:

```python
class CrossRingCrossPoint:
    def __init__(self, node_id: int):
        self.node_id = node_id
        self.injection_queues = {"req": [], "rsp": [], "data": []}
        self.ejection_queues = {"req": [], "rsp": [], "data": []}
        self.ring_interfaces = {"TL": None, "TR": None, "TU": None, "TD": None}
        
    def connect_ring_slice(self, direction: str, ring_slice: RingSlice):
        """Connect to ring slice in specified direction"""
        self.ring_interfaces[direction] = ring_slice
        
    def arbitrate_injection(self, channel: str) -> Optional[CrossRingFlit]:
        """Simple round-robin arbitration for injection"""
        if self.injection_queues[channel]:
            return self.injection_queues[channel].pop(0)
        return None
        
    def try_ejection(self, slot: CrossRingSlot, channel: str) -> bool:
        """Try to eject slot to local node"""
        if len(self.ejection_queues[channel]) < self.ejection_capacity:
            self.ejection_queues[channel].append(slot.flit)
            return True
        return False
```

### 6.3 RingSlice Pipeline Architecture

RingSlices implement pipelined transmission:

```python
class RingSlice:
    def __init__(self, direction: str, node_id: int, num_slots: int = 8):
        self.direction = direction
        self.node_id = node_id
        self.num_slots = num_slots
        
        # Pipeline stages
        self.input_buffer = []
        self.current_slots = [CrossRingSlot(i, 0) for i in range(num_slots)]
        self.output_buffer = []
        
        # Connections
        self.upstream_slice = None
        self.downstream_slice = None
        self.local_crosspoint = None
        
    def cycle_transmission(self, cycle: int):
        """Execute one cycle of pipeline transmission"""
        # Stage 3: Output slots to downstream
        if self.downstream_slice:
            self.downstream_slice.receive_slots(self.output_buffer)
        
        # Stage 2: Current slots become output
        self.output_buffer = self.current_slots.copy()
        
        # Stage 1: Input slots become current
        self.current_slots = self.input_buffer.copy()
        
        # Stage 0: Receive new input
        self.input_buffer = []
        
        # Update slot cycles
        for slot in self.current_slots:
            slot.cycle = cycle
```

### 6.4 CrossRingTagManager Integration

Tag management coordinates with node operations:

```python
class CrossRingTagManager:
    def __init__(self, node_id: int):
        self.node_id = node_id
        self.itag_config = {'trigger_threshold': 100, 'max_reservations': 4}
        self.etag_config = {'upgrade_threshold': 2, 'max_priority': 'T0'}
        
        # Tracking injection/ejection attempts
        self.injection_waiting = {'req': 0, 'rsp': 0, 'data': 0}
        self.ejection_failures = {'req': 0, 'rsp': 0, 'data': 0}
        
    def check_injection_starvation(self, channel: str, direction: str) -> bool:
        """Check if injection starvation protection needed"""
        waiting_cycles = self.injection_waiting[channel]
        return waiting_cycles >= self.itag_config['trigger_threshold']
        
    def check_ejection_starvation(self, channel: str, direction: str) -> Optional[str]:
        """Check if ejection priority upgrade needed"""
        failures = self.ejection_failures[channel]
        if failures >= 1:
            return "T1"  # Upgrade to T1
        if failures >= 2 and direction in ["TL", "TU"]:
            return "T0"  # Upgrade to T0 for specific directions
        return None
```

## 7. Protocol Integration

### 7.1 STI Three-Channel Protocol

CrossRing integrates with STI protocol supporting three channels:

- **REQ Channel**: Request packets (read/write commands)
- **RSP Channel**: Response packets (acknowledgments, data responses)
- **DATA Channel**: Data packets (write data, read data)

### 7.2 Packet Format Integration

```python
# STI packet integration
class CrossRingFlit:
    def __init__(self, packet_id: str, source: int, destination: int):
        # Basic routing information
        self.packet_id = packet_id
        self.source = source
        self.destination = destination
        
        # STI protocol fields
        self.channel = "req"  # req/rsp/data
        self.req_type = None  # read/write
        self.burst_length = 1
        
        # CrossRing specific fields
        self.current_position = -1
        self.path = []
        self.dest_xid = -1
        self.dest_yid = -1
        
        # Timing information
        self.injection_cycle = 0
        self.ejection_cycle = 0
```

### 7.3 Address Translation

CrossRing handles address translation between different domains:

```python
def translate_address(self, physical_addr: int) -> Tuple[int, int]:
    """Translate physical address to CrossRing node coordinates"""
    # Address mapping logic based on system configuration
    node_id = (physical_addr >> 20) & 0xFF  # Extract node ID from address
    x_coord = node_id % self.num_cols
    y_coord = node_id // self.num_cols
    return (x_coord, y_coord)
```

## 8. Performance Benefits

### 8.1 Anti-Starvation Guarantees

**I-Tag Benefits**:
- Prevents injection starvation in high-contention scenarios
- Maintains fairness across nodes with different traffic patterns
- Provides bounded injection latency guarantees

**E-Tag Benefits**:
- Prevents ejection starvation when destination buffers are full
- Enables priority-based ejection scheduling
- Reduces head-of-line blocking effects

### 8.2 Pipeline Efficiency

**RingSlice Pipeline**:
- Overlapped transmission and processing
- Improved throughput through parallelism
- Reduced per-hop latency

**Simplified CrossPoint**:
- Streamlined arbitration logic
- Reduced critical path delay
- Better scalability to larger topologies

### 8.3 Boundary Optimization

**Non-Wrap-Around Benefits**:
- Eliminates wrap-around wire complexity
- Reduces power consumption
- Optimizes for chip-to-chip communication patterns
- Simplifies physical layout constraints

## 9. Implementation Guidelines

### 9.1 Key Design Principles

1. **Cross Ring Spec v2.0 Compliance**: Strict adherence to specification
2. **Non-Wrap-Around Topology**: Edge nodes connect to themselves
3. **XY Dimension-Order Routing**: Deadlock-free routing algorithm
4. **I-Tag/E-Tag Integration**: Comprehensive anti-starvation mechanisms
5. **Pipeline Optimization**: Multi-stage pipeline for performance

### 9.2 Critical Implementation Areas

1. **Slot Structure**: Proper implementation of Valid/I-Tag/E-Tag/Flit fields
2. **Boundary Handling**: Correct self-connection for edge nodes
3. **Tag Mechanisms**: Proper I-Tag reservation and E-Tag priority logic
4. **Pipeline Timing**: Accurate cycle-by-cycle transmission simulation
5. **Protocol Integration**: Seamless STI three-channel protocol support

### 9.3 Validation Requirements

1. **Topology Validation**: Verify non-wrap-around connections
2. **Routing Validation**: Test XY dimension-order routing correctness
3. **Anti-Starvation Testing**: Validate I-Tag/E-Tag mechanisms under stress
4. **Protocol Compliance**: Ensure STI protocol compatibility
5. **Performance Testing**: Measure latency and throughput improvements

## 10. Conclusion

The updated CrossRing implementation based on Cross Ring Spec v2.0 provides a comprehensive NoC architecture with advanced anti-starvation mechanisms and optimized performance characteristics. The key innovations include:

- **Comprehensive Tag Mechanisms**: I-Tag and E-Tag systems working together to prevent both injection and ejection starvation
- **Non-Wrap-Around Topology**: Edge nodes connecting to themselves for optimized chip-to-chip communication
- **Pipeline Optimization**: Multi-stage RingSlice pipeline for improved throughput
- **Simplified CrossPoint**: Streamlined arbitration and interface management
- **STI Protocol Integration**: Full support for three-channel STI protocol

The architecture balances simplicity with sophistication, providing predictable performance guarantees while maintaining high throughput under varying load conditions. The implementation serves as a solid foundation for chip-to-chip communication in multi-chip systems.