# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

C2C is a topology modeling framework chip-to-chip communication. It provides an object-oriented architecture for modeling various interconnect topologies (cascade, switch-based) and communication protocols (CDMA). The framework is designed to simulate chip-to-chip communication in multi-chip systems.

## Architecture

The codebase follows a layered architecture with clear separation of concerns:

### Core Components

1. **Topology Layer** (`topology/`):
   - `base.py`: Abstract base classes (BaseNode, BaseLink, BaseTopology) defining the fundamental interfaces
   - `node.py`: Concrete node implementations (ChipNode, SwitchNode, HostNode)
   - `link.py`: Link implementations (PCIeLink, C2CDirectLink)
   - `graph.py`: NetworkX-based topology graph representation with pathfinding and visualization
   - `builder.py`: Builder pattern for constructing topologies

2. **Protocol Layer** (`protocol/`):
   - `base.py`: Protocol state management and abstract interfaces
   - `cdma.py`: CDMA protocol implementation with message types and state transitions
   - `credit.py`: Credit-based flow control mechanism
   - `address.py`: Address space translation between different formats
   - `router.py`: Message routing with topology-aware path calculation

3. **Support Modules**:
   - `utils/`: Constants and custom exceptions
   - `config/`: Configuration loading utilities
   - `examples/`: Demonstration scripts

### Key Design Patterns

- **Abstract Base Classes**: Extensive use of ABC for defining interfaces
- **Builder Pattern**: TopologyBuilder for constructing complex topologies
- **Strategy Pattern**: Different node types and link types with common interfaces
- **State Machine**: Protocol state management in BaseProtocol

## Common Development Tasks

### Running the Basic Demo
```bash
# Install dependencies first
pip install networkx matplotlib pyyaml streamlit plotly seaborn pandas numpy

# Run the basic demonstration
python3 examples/basic_demo.py
```

### Using Visualization Tools
```bash
# Run static visualization demo
python3 examples/visualization_demo.py

# Launch interactive web interface
python3 run_webapp.py
# or
streamlit run visualization/interactive.py

# Run topology validation tests
python3 examples/tree_torus_validation.py
```

### Testing Topologies
The framework supports creating various topology types:
- Cascade topologies: Direct chip-to-chip connections
- Switch-based topologies: PCIe switches connecting multiple chips
- Mixed topologies: Combination of direct and switch connections

### Key Classes to Understand

- **TopologyGraph**: Central class for topology representation using NetworkX
- **TopologyBuilder**: Main interface for constructing topologies
- **CDMAProtocol**: Core communication protocol implementation
- **CreditManager**: Flow control mechanism
- **Router**: Message routing with path calculation

### Visualization Classes

- **TopologyVisualizer**: Main visualization engine for all topology types
- **TreeLayout/TorusLayout**: Specialized layout algorithms for different topologies
- **PerformanceComparator**: Tool for comparing topology performance metrics
- **Interactive Web App**: Streamlit-based GUI for real-time topology exploration

## Development Guidelines

### Module Dependencies
- Core dependencies: `networkx`, `matplotlib`, `pyyaml`
- Standard library: `abc`, `enum`, `typing`, `dataclasses`

### Code Style
- Follows PEP 8 conventions
- Uses type hints throughout
- Comprehensive docstrings for all public methods
- Abstract base classes for extensibility

### Testing
Run the demo script to verify core functionality:
```bash
python examples/basic_demo.py
```

The demo validates:
- Node and link creation
- Topology construction
- Path finding algorithms
- CDMA protocol state transitions
- Credit management
- Address translation
- Message routing

## Architecture Notes

- The framework is designed for extensibility - new node types, link types, and protocols can be added by inheriting from base classes
- NetworkX is used for graph operations and pathfinding algorithms
- State management is centralized in protocol classes
- The builder pattern ensures consistent topology construction
- Address translation supports multiple address formats for different communication contexts

## CrossRing NoC Architecture

### Overview
CrossRing is a Network-on-Chip (NoC) architecture based on Cross Ring Spec v2.0, featuring:
- Ring-based topology with horizontal and vertical rings
- Non-wrap-around boundary handling (edge nodes connect to themselves)
- Advanced I-Tag/E-Tag anti-starvation mechanisms
- XY dimension-order routing
- Slot-based transmission with pipeline optimization

### Key Components

#### 1. CrossRingModel (`src/noc/crossring/model.py`)
- Main model class for CrossRing simulation
- Manages network topology setup and simulation loops
- Coordinates IP interfaces, nodes, and links
- Handles traffic injection and collection

#### 2. CrossRingNode (`src/noc/crossring/node.py`)
- Contains two CrossPoints (horizontal and vertical)
- Manages IP injection/ejection queues
- Handles local arbitration and flow control
- Implements ring buffer and bridge logic

#### 3. CrossRingCrossPoint (`src/noc/crossring/node.py`)
- **Critical Architecture**: Each CrossPoint manages 4 slices (2 per direction)
  - Arrival slice: For ejection decisions (from upstream ring)
  - Departure slice: For injection decisions (to downstream ring)
- Horizontal CP: Manages TL/TR directions
- Vertical CP: Manages TU/TD directions
- Integrates with Tag management for anti-starvation

#### 4. CrossRingLink (`src/noc/crossring/crossring_link.py`)
- Manages RingSlice chains forming the ring links
- Handles slot transmission between slices
- Implements pipeline flow control

#### 5. CrossRingSlot (`src/noc/crossring/crossring_link.py`)
- Basic transmission unit containing:
  - Valid bit
  - I-Tag (injection reservation)
  - E-Tag (ejection priority)
  - Flit data

#### 6. Tag Mechanism (`src/noc/crossring/tag_mechanism.py`)
- **I-Tag**: Injection reservation to prevent starvation
- **E-Tag**: Priority-based ejection (T0/T1/T2 levels)
- **FifoEntryManager**: Manages tiered entry allocation
- **T0_Etag_Order_FIFO**: Global round-robin queue for T0 slots

### Data Flow Paths

#### Injection Flow
1. IP → l2h_fifo → node channel_buffer
2. Node channel_buffer → inject_direction_fifos  
3. inject_direction_fifos → CrossPoint → departure slice
4. Departure slice → ring transmission

#### Ejection Flow
1. Ring arrival slice → CrossPoint
2. CrossPoint → eject_input_fifos
3. eject_input_fifos → ip_eject_channel_buffers
4. ip_eject_channel_buffers → h2l_fifos → IP

### Key Architecture Rules

1. **CrossPoint Slice Management**:
   - Each CP has 4 slices: 2 directions × 2 types (arrival/departure)
   - Arrival slices: Used for ejection decisions
   - Departure slices: Used for injection decisions

2. **Tag Mechanisms**:
   - I-Tag: Triggered when injection waits exceed threshold (80-100 cycles)
   - E-Tag: Priority upgrade after ejection failures (T2→T1→T0)
   - T0 requires round-robin arbitration via global queue

3. **Non-Wrap-Around Topology**:
   - Edge nodes connect to themselves, not wrapping around
   - Optimized for chip-to-chip communication patterns

4. **Model Responsibilities**:
   - Model coordinates simulation, does NOT handle inject/eject directly
   - All inject/eject logic is in nodes and CrossPoints
   - Model manages link transmission and node step coordination

### Testing and Debugging

#### Run CrossRing Debug Demo
```bash
python3 examples/noc/crossring_debug_demo.py
```

#### Key Debug Information
- IP interface states and queue depths
- CrossPoint slice connections and status
- Ring transmission and slot movement
- Tag mechanism triggers and upgrades
- Request lifecycle tracking

### Common Issues and Solutions

1. **Flits not transmitting**: Check IP-to-node connections and CrossPoint slice connections
2. **Starvation issues**: Verify I-Tag/E-Tag mechanisms are properly integrated
3. **Simulation hangs**: Check for deadlocks in ring transmission or FIFO flow control
4. **Tag mechanism not working**: Ensure Tag managers are properly initialized in CrossPoints

### File Structure
```
src/noc/crossring/
├── model.py              # Main CrossRing model
├── node.py               # Node and CrossPoint implementation
├── crossring_link.py     # Link and RingSlice implementation
├── tag_mechanism.py      # I-Tag/E-Tag anti-starvation
├── ip_interface.py       # CrossRing IP interface
├── flit.py              # CrossRing flit definition
└── config.py            # Configuration classes
```

### Architecture Documentation
Refer to `/docs/CrossRing_Architecture_Documentation.md` for detailed specification compliance and implementation guidelines.

## CrossRing Working Context
When working with CrossRing:
1. Always refer to CrossRing_Architecture_Documentation.md for spec compliance
2. Remember that CrossPoint manages 4 slices (2 per direction: arrival/departure)
3. Tag mechanisms are critical for preventing starvation
4. Model coordinates but doesn't handle inject/eject directly
5. All logging should be in Chinese
6. Non-wrap-around topology means edge nodes connect to themselves

## Recent Major Fixes (2025-07-10)

### Ring_Bridge重新注入机制修复
**问题**: Flit卡在ring_bridge（N1.RB），无法从水平环转换到垂直环进行维度转换。

**根本原因**: Ring_bridge输出的flit没有重新注入机制，导致维度转换后flit无法继续传输。

**解决方案**:
1. **修改CrossPoint注入逻辑**: 在`process_injection_from_fifos`方法中添加ring_bridge输出检查
   - 水平CrossPoint处理ring_bridge的TR/TL输出
   - 垂直CrossPoint处理ring_bridge的TU/TD输出
   - Ring_bridge输出具有比普通inject_direction_fifos更高的优先级

2. **架构理解澄清**: 
   - CrossRing有两个独立的CrossPoint（水平和垂直）
   - 它们的输入源完全不同，不需要复杂的仲裁优先级
   - Ring_bridge作为维度转换桥梁连接两个CrossPoint

3. **验证的数据流路径**:
   ```
   IP → channel_buffer → inject_direction_fifos → CrossPoint → Ring → 
   CrossPoint → ring_bridge_input → ring_bridge仲裁 → ring_bridge_output → 
   CrossPoint重新注入 → Ring → 目标节点
   ```

**修复文件**: `/src/noc/crossring/node.py` - `CrossRingCrossPoint.process_injection_from_fifos`

**测试结果**: Flit成功从节点0(0,0)路由到节点4(1,1)，维度转换正常工作。

### 其他修复
1. **PipelinedFIFO属性错误**: `fifo.depth` → `fifo.max_depth`
2. **Flit坐标显示**: 修复`source_ip_type`/`dest_ip_type`属性不一致问题
3. **位置跟踪**: 改进flit状态显示，简化为`source->dest:slice_index`格式

### 调试建议
- 使用`examples/noc/crossring_debug_demo.py`进行flit传输跟踪
- 关注ring_bridge仲裁日志："🎯 节点X: 从RB_XX获取到flit"
- 验证CrossPoint重新注入："✅ CrossPoint node_X_vertical 从ring_bridge XX方向注入flit到环路"