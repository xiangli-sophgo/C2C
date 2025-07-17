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
- **ALL LOGGING AND DEBUG OUTPUT MUST BE IN CHINESE** - 所有日志和调试输出必须使用中文

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

## NoC模块架构 (src/noc/)

NoC模块采用分层架构设计，基于继承和组合模式实现功能复用和扩展性：

### 基础抽象层 (base/)
提供所有NoC拓扑的通用功能和抽象接口：

- **model.py**: NoC模型基类
  - 功能：仿真循环控制、IP接口管理、统计收集、TrafficScheduler集成
  - 扩展性：支持新的NoC拓扑类型继承
  
- **topology.py**: 拓扑基类  
  - 功能：路由算法基础实现（XY/YX路由）、邻接矩阵管理、路径计算
  - 扩展性：支持新的路由策略和拓扑类型
  
- **node.py**: 节点基类
  - 功能：节点接口定义、基本状态管理
  - 扩展性：支持不同类型节点实现（router, switch, endpoint）
  
- **ip_interface.py**: IP接口基类
  - 功能：FIFO管理、流控机制、统计收集
  - 扩展性：支持不同协议的IP接口
  
- **flit.py**: Flit基类和对象池
  - 功能：数据包抽象、内存管理
  - 扩展性：支持新的包格式和协议
  
- **link.py**: 链路基类
  - 功能：点对点连接抽象
  - 扩展性：支持不同带宽和延迟特性
  
- **config.py**: 配置管理基类
  - 功能：参数验证、配置加载
  - 扩展性：支持新拓扑的配置参数

### 拓扑实现层
基于继承机制实现具体拓扑：

#### CrossRing实现 (crossring/)
- **model.py**: 继承base/model.py，实现CrossRing特有的仿真控制
- **topology.py**: 继承base/topology.py，实现CrossRing路由算法和环形拓扑结构  
- **node.py**: 继承base/node.py，实现CrossPoint、环形缓冲区、维度转换
- **ip_interface.py**: 继承base/ip_interface.py，实现CrossRing协议适配、Tag机制
- **flit.py**: 继承base/flit.py，实现CrossRing包格式和Tag字段
- **link.py**: 继承base/link.py，实现环形slice链接、pipeline流控
- **cross_point.py**: CrossRing核心组件，实现环形路由决策、仲裁机制
- **config.py**: 继承base/config.py，定义CrossRing特有参数

#### Mesh实现 (mesh/)
- 类似继承结构，实现Mesh拓扑特有功能

### 功能模块层
独立的功能模块，通过组合方式提供服务：

- **analysis/**: 性能分析工具
  - 功能：延迟分析、吞吐量统计、拥塞检测
  - 扩展性：支持新的分析指标和报告格式
  
- **utils/**: 通用工具
  - 功能：Traffic调度、工厂模式、类型定义、邻接矩阵生成
  - 扩展性：支持新的调度策略和工具函数
  
- **debug/**: 调试工具
  - 功能：请求追踪、状态监控、生命周期管理
  - 扩展性：支持新的调试模式和日志格式
  
- **visualization/**: 可视化工具
  - 功能：实时状态显示、性能图表、拓扑渲染
  - 扩展性：支持新的可视化方式和交互模式

### 新拓扑添加指南
1. 继承base/模块创建新拓扑实现
2. 实现特有的路由算法和节点行为  
3. 使用现有的analysis、utils、debug、visualization模块
4. 在utils/factory.py中注册新拓扑类型

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

## ⚠️ 两阶段执行模型 (Two-Phase Execution Model) - 关键架构要求

**所有组件必须严格遵循两阶段执行模型，确保每个周期内flit只前进一个阶段。**

### 正确的Flit传输时序
```
cycle 0: 请求生成 → pending_requests
cycle 1: pending_requests → l2h_fifo 
cycle 2: l2h_fifo → channel_buffer
cycle 3: channel_buffer → inject_direction_fifos (IQ_TR/TL/TU/TD)
cycle 4: inject_direction_fifos → 环路slice S0
```

### 组件两阶段实现要求

#### 1. IP接口 (CrossRingIPInterface)
- **compute阶段**: 更新FIFO状态，不执行传输
- **update阶段**: 执行数据传输，每周期只执行一个传输操作
  - 优先级：pending → l2h > l2h → channel_buffer
  - 绝对禁止同一个flit在同一周期跳跃多个阶段

#### 2. 节点 (CrossRingNode)  
- **compute阶段** (`step_compute_phase`):
  - 更新所有FIFO组合逻辑
  - **计算注入仲裁**：确定要从channel_buffer转移到IQ的flit，但**不执行传输**
  - 处理CrossPoint计算阶段
- **update阶段** (`step_update_phase`):
  - 更新所有FIFO寄存器状态
  - **执行注入仲裁**：基于compute阶段计算结果执行channel_buffer → inject_direction_fifos传输
  - 处理CrossPoint更新阶段

#### 3. 模型 (CrossRingModel)
- **compute阶段**: 调用所有组件的compute阶段
- **update阶段**: 调用所有组件的update阶段

### 调试验证命令
```bash
python3 test_flit_flow.py 2>&1 | grep -E "(周期|🔄|RequestTracker)"
```

正确输出示例：
```
周期1: pending->L2H传输
周期2: L2H->channel_buffer注入成功 (flit显示在N0.channel)
周期3: channel_buffer->IQ_TR (flit显示在N0.IQ_TR)
周期4: flit显示在0->1:0 (环路slice)
```

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