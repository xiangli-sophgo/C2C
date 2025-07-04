# CrossRing NoC Architecture Documentation

## 1. Overview

CrossRing is a Network-on-Chip (NoC) architecture designed for high-performance chip-to-chip communication. It implements a true ring-based topology with bidirectional horizontal and vertical rings, providing efficient routing through XY dimension-order routing and advanced congestion control mechanisms.

## 2. CrossRing Topology Structure

### 2.1 True Ring Topology

CrossRing implements a genuine ring topology with the following characteristics:

- **Bidirectional Rings**: Each ring supports both clockwise and counter-clockwise transmission
- **Wrap-around Connections**: Edge nodes connect back to the beginning of the ring
- **Ring Intersection**: Cross Point modules manage communication between horizontal and vertical rings

```
Ring Structure Example (4x4):
  0 ←→ 1 ←→ 2 ←→ 3
  ↑    ↑    ↑    ↑
  ↓    ↓    ↓    ↓
  4 ←→ 5 ←→ 6 ←→ 7
  ↑    ↑    ↑    ↑
  ↓    ↓    ↓    ↓
  8 ←→ 9 ←→ 10←→ 11
  ↑    ↑    ↑    ↑
  ↓    ↓    ↓    ↓
  12←→ 13←→ 14←→ 15
```

### 2.2 Ring Direction System

The CrossRing topology uses four directional channels:

- **TL (Top-Left)**: Counter-clockwise horizontal, upward vertical
- **TR (Top-Right)**: Clockwise horizontal, upward vertical  
- **TU (Top-Up)**: Same as TL for vertical movement
- **TD (Top-Down)**: Same as TR for vertical movement

### 2.3 Cross Point Modules

Cross Point modules are located at ring intersections and handle:
- **Dimension Turning**: Converting horizontal movement to vertical and vice versa
- **Ring Arbitration**: Managing conflicts between horizontal and vertical rings
- **Flow Control**: Implementing backpressure and congestion control

## 3. Ring Bridge Architecture

### 3.1 Dimension Turning Functionality

The Ring Bridge is responsible for dimension turning when packets need to change from horizontal to vertical movement or vice versa:

```cpp
// From src_old implementation
if (current_direction == "horizontal" && needs_vertical_turn) {
    // Check vertical ring availability
    if (vertical_ring_available) {
        // Perform H→V turning
        transfer_to_vertical_ring(flit);
    } else {
        // Continue on horizontal ring
        continue_horizontal_transmission(flit);
    }
}
```

### 3.2 Ring Slice Pipeline Structure

Each Ring Bridge contains Ring Slices that implement a pipeline for packet processing:

1. **Input Stage**: Receive packets from rings
2. **Arbitration Stage**: Resolve conflicts between rings
3. **Turning Stage**: Perform dimension turning if needed
4. **Output Stage**: Forward packets to destination rings

### 3.3 ETag/ITag Priority Handling

The Ring Bridge implements priority-based arbitration:

- **ETag Priority**: T0 (highest) > T1 > T2 (lowest)
- **ITag Anti-starvation**: Prevents lower priority packets from being starved
- **Circuit Counting**: Tracks established circuits for flow control

## 4. Routing Algorithm

### 4.1 XY Dimension-Order Routing (DOR)

CrossRing implements XY dimension-order routing:

1. **Horizontal First**: Route horizontally to target column
2. **Vertical Second**: Route vertically to target row
3. **Ring Selection**: Choose appropriate ring direction based on distance

```cpp
// Simplified routing logic from src_old
void calculate_crossring_path(int source, int destination, int width, int height) {
    int src_x = source % width, src_y = source / width;
    int dst_x = destination % width, dst_y = destination / width;
    
    // Horizontal routing
    if (src_x != dst_x) {
        if (dst_x > src_x) {
            // Move right (clockwise)
            use_ring_direction("TR");
        } else {
            // Move left (counter-clockwise)
            use_ring_direction("TL");
        }
    }
    
    // Vertical routing
    if (src_y != dst_y) {
        if (dst_y > src_y) {
            // Move down
            use_ring_direction("TD");
        } else {
            // Move up
            use_ring_direction("TU");
        }
    }
}
```

### 4.2 Ring Transmission vs Injection

The system makes decisions between:
- **Ring Transmission**: Continue on current ring
- **Ring Injection**: Inject new packets into ring
- **Ring Ejection**: Remove packets from ring at destination

### 4.3 Wrap-around Edge Handling

Edge nodes implement wrap-around connections:
- **Leftmost nodes**: Connect to rightmost nodes
- **Rightmost nodes**: Connect to leftmost nodes
- **Topmost nodes**: Connect to bottommost nodes
- **Bottommost nodes**: Connect to topmost nodes

## 5. Congestion Control

### 5.1 ETag Priority System

Three priority levels for congestion control:

- **T0**: Highest priority, reserved for critical traffic
- **T1**: Medium priority, normal traffic
- **T2**: Lowest priority, best-effort traffic

```cpp
// From src_old Flit implementation
class Flit {
    string ETag_priority = "T2";  // Default priority
    
    void set_priority_based_on_congestion() {
        if (congestion_level > HIGH_THRESHOLD) {
            ETag_priority = "T0";  // Escalate to highest priority
        } else if (congestion_level > MEDIUM_THRESHOLD) {
            ETag_priority = "T1";
        }
    }
};
```

### 5.2 ITag Anti-starvation Mechanism

ITag prevents lower priority packets from being starved:

```cpp
// Anti-starvation logic from src_old
void check_itag_starvation() {
    for (auto& flit : waiting_queue) {
        if (flit.wait_cycles > STARVATION_THRESHOLD) {
            flit.itag_h = true;  // Set horizontal ITag
            flit.itag_v = true;  // Set vertical ITag
        }
    }
}
```

### 5.3 Circuit Establishment and Counting

The system tracks established circuits:

- **Circuit Counters**: Track active circuits per ring
- **Circuit Limits**: Prevent circuit overload
- **Circuit Release**: Release circuits when transmission completes

## 6. Protocol Integration

### 6.1 STI Three-Channel Protocol

CrossRing integrates with STI protocol using three channels:

- **REQ Channel**: Request packets (commands)
- **RSP Channel**: Response packets (acknowledgments)
- **DAT Channel**: Data packets (payload)

### 6.2 RN/SN Resource Management

#### 6.2.1 RN (Request Node) Resources

```cpp
// RN resource management from src_old
class RN_Resources {
    // Read Database
    map<string, vector<Flit>> rn_rdb;
    int rn_rdb_count;
    int rn_rdb_reserve;
    
    // Write Database
    map<string, vector<Flit>> rn_wdb;
    int rn_wdb_count;
    
    // Trackers
    map<string, vector<Flit>> rn_tracker; // read/write
    map<string, int> rn_tracker_count;
    map<string, int> rn_tracker_pointer;
};
```

#### 6.2.2 SN (Slave Node) Resources

```cpp
// SN resource management from src_old
class SN_Resources {
    // SN Tracker
    vector<Flit> sn_tracker;
    map<string, int> sn_tracker_count; // ro/share
    
    // Write Database
    map<string, vector<Flit>> sn_wdb;
    int sn_wdb_count;
    
    // Wait Queues
    map<string, vector<Flit>> sn_req_wait; // read/write
    
    // Delayed release
    map<int, vector<Flit>> sn_tracker_release_time;
};
```

### 6.3 Tracker Allocation and Release

#### 6.3.1 Read Request Processing

```cpp
void process_read_request(Flit& req) {
    if (req.req_attr == "new") {
        // Check SN resources
        if (sn_tracker_count["ro"] > 0) {
            // Allocate RO tracker
            req.sn_tracker_type = "ro";
            sn_tracker.push_back(req);
            sn_tracker_count["ro"]--;
            
            // Generate read data
            create_read_data_packet(req);
            
            // Release tracker immediately for read
            release_sn_tracker(req);
        } else {
            // Send negative response
            create_negative_response(req);
        }
    }
}
```

#### 6.3.2 Write Request Processing

```cpp
void process_write_request(Flit& req) {
    if (req.req_attr == "new") {
        // Check SN resources (tracker + wdb)
        if (sn_tracker_count["share"] > 0 && sn_wdb_count >= req.burst_length) {
            // Allocate resources
            req.sn_tracker_type = "share";
            sn_tracker.push_back(req);
            sn_tracker_count["share"]--;
            sn_wdb_count -= req.burst_length;
            
            // Send datasend response
            create_datasend_response(req);
        } else {
            // Send negative response
            create_negative_response(req);
        }
    }
}
```

### 6.4 Retry Mechanisms

#### 6.4.1 Read Retry Logic

```cpp
void handle_read_retry(Flit& rsp, Flit& req) {
    if (rsp.rsp_type == "negative") {
        // Retry logic
        if (req.req_attr != "old") {
            req.reset_for_retry();
            rn_rdb_count += req.burst_length;
            rn_rdb_reserve += 1;
        }
    } else if (rsp.rsp_type == "positive") {
        // Re-inject request
        req.req_attr = "old";
        req.reset_injection_state();
        inject_queue.push_front(req); // High priority retry
        rn_rdb_reserve -= 1;
    }
}
```

#### 6.4.2 Write Retry Logic

```cpp
void handle_write_retry(Flit& rsp, Flit& req) {
    if (rsp.rsp_type == "datasend") {
        // Send write data
        for (auto& data_flit : rn_wdb[req.packet_id]) {
            inject_data_queue.push(data_flit);
        }
        
        // Release RN resources
        release_rn_write_tracker(req);
        rn_wdb_count += req.burst_length;
    }
}
```

## 7. Clock Domain Conversion

### 7.1 Dual Clock System

CrossRing operates with dual clock domains:
- **1GHz Core Clock**: IP interface operations
- **2GHz NoC Clock**: Network transmission operations

### 7.2 Clock Ratio Handling

```cpp
class ClockDomainConverter {
    int clock_ratio = 2; // 2GHz / 1GHz
    
    void convert_l2h_timing(Flit& flit) {
        // Low speed to high speed conversion
        flit.departure_cycle *= clock_ratio;
    }
    
    void convert_h2l_timing(Flit& flit) {
        // High speed to low speed conversion
        flit.arrival_cycle /= clock_ratio;
    }
};
```

## 8. Performance Optimization

### 8.1 Pipeline Optimization

- **Ring Slice Pipelining**: Multi-stage pipeline for packet processing
- **Arbitration Pipelining**: Parallel arbitration for multiple rings
- **Resource Prefetching**: Proactive resource allocation

### 8.2 Congestion Avoidance

- **Adaptive Routing**: Dynamic path selection based on congestion
- **Circuit Throttling**: Limit circuit establishment under high load
- **Priority Escalation**: Automatic priority elevation for delayed packets

### 8.3 Resource Management

- **Tracker Pooling**: Efficient tracker allocation and reuse
- **Buffer Management**: Dynamic buffer allocation
- **Delayed Release**: Optimized resource release timing

## 9. Implementation Guidelines

### 9.1 Key Design Principles

1. **True Ring Topology**: Implement genuine ring connections with wrap-around
2. **XY Dimension-Order Routing**: Strictly follow XY routing for deadlock avoidance
3. **Priority-Based Arbitration**: Implement ETag/ITag priority system
4. **Resource Management**: Careful tracker and buffer management
5. **Protocol Integration**: Seamless STI protocol integration

### 9.2 Critical Implementation Areas

1. **Ring Bridge Logic**: Dimension turning and arbitration
2. **Wrap-around Connections**: Proper edge node handling
3. **Congestion Control**: ETag/ITag mechanisms
4. **Resource Allocation**: RN/SN tracker management
5. **Clock Domain Conversion**: Dual clock system handling

### 9.3 Validation Requirements

1. **Topology Validation**: Verify ring structure and connections
2. **Routing Validation**: Test XY dimension-order routing
3. **Congestion Testing**: Validate priority and anti-starvation mechanisms
4. **Protocol Compliance**: Ensure STI protocol compatibility
5. **Performance Testing**: Measure latency and throughput

## 10. Conclusion

CrossRing NoC provides a sophisticated ring-based communication architecture with advanced congestion control and efficient resource management. The key to successful implementation lies in maintaining the true ring topology, implementing proper dimension turning logic, and ensuring robust resource management throughout the system.

The architecture's strength lies in its combination of simple XY routing with sophisticated congestion control, providing both predictable performance and adaptive behavior under varying load conditions.