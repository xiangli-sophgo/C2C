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

### CrossRing数据流路径（XY路由策略）

#### 注入流程
1. IP请求 → pending_by_channel → L2H FIFO → channel_buffer → inject_direction_fifos

#### 上环流程（分方向）
**水平方向（直接上环）**：
- IQ_TR/IQ_TL → 水平环CrossPoint → 水平环链路

**垂直方向（通过Ring Bridge）**：
- IQ_TU/IQ_TD → Ring Bridge → RB_TU/RB_TD → 垂直环CrossPoint → 垂直环链路

#### 维度转换（仅单向）
**水平→垂直转换**：
- 水平环链路 → RB_TR/RB_TL → Ring Bridge → RB_TU/RB_TD → 垂直环链路
- **注意：XY路由无垂直→水平转换**

#### 弹出流程
CrossRing采用双路径弹出机制，根据flit所在的环和目标位置选择不同的弹出路径：

##### 1. CrossPoint弹出判断阶段
每个到达slice的flit都会通过`should_eject_flit()`进行弹出决策：

**横向CrossPoint**（处理TR/TL方向）：
- 到达目标节点 → `return True, "RB"` (通过Ring Bridge下环)
- 需要维度转换 → `return True, "RB"` (通过Ring Bridge转换)
- 继续传输 → `return False, ""` (留在水平环)

**纵向CrossPoint**（处理TU/TD方向）：
- 到达目标节点 → `return True, "EQ"` (直接下环到IP)
- 继续传输 → `return False, ""` (留在垂直环)

##### 2. 弹出执行路径

**路径A：水平环弹出（通过Ring Bridge）**
```
水平环 arrival slice → 横向CrossPoint → Ring Bridge → EjectQueue → IP
```
1. CrossPoint将flit传输到Ring Bridge
2. Ring Bridge通过仲裁输出到EQ方向
3. EjectQueue从`ring_bridge_EQ`源读取flit
4. 分发到目标IP的channel_buffer

**路径B：垂直环弹出（直接到EjectQueue）**
```
垂直环 arrival slice → 纵向CrossPoint → EjectQueue → IP
```
1. CrossPoint直接将flit写入`eject_input_fifos[channel][direction]` (TU/TD)
2. EjectQueue从对应方向源读取flit
3. 分发到目标IP的channel_buffer

##### 3. EjectQueue仲裁与分发
**XY路由下的活跃源**：`["IQ_EQ", "ring_bridge_EQ", "TU", "TD"]`

**仲裁流程**：
- 计算阶段：轮询各源，决定读取和分发策略
- 执行阶段：从选定源读取flit，写入目标IP的`ip_eject_channel_buffers`

##### 4. IP接收阶段
```
ip_eject_channel_buffers → h2l_fifos → IP.get_eject_flit()
```

**双路径设计原因**：
- 水平环需要统一处理维度转换和本地弹出
- 垂直环作为最终传输维度，直接弹出效率更高

### Ring Bridge配置规则

#### XY路由策略下的Ring Bridge
**输入源**：`["IQ_TU", "IQ_TD", "RB_TR", "RB_TL"]`
- IQ_TU/IQ_TD：垂直方向的直接注入
- RB_TR/RB_TL：水平环转入的维度转换

**输出方向**：`["EQ", "TU", "TD"]`
- EQ：本地弹出
- TU/TD：垂直环输出

**不处理**：IQ_TR/IQ_TL（水平方向直接上环，不经过Ring Bridge）

### Key Architecture Rules

1. **CrossPoint职责分工**:
   - 水平CrossPoint：处理IQ_TR/IQ_TL的直接上环
   - 垂直CrossPoint：处理Ring Bridge输出的RB_TU/RB_TD

2. **Tag Mechanisms**:
   - I-Tag: Triggered when injection waits exceed threshold (80-100 cycles)
   - E-Tag: **下环entry分配机制**，基于优先级和entry可用性判断是否可以下环
     - **Entry分配**: T2只用T2 entry，T1优先用T1后用T2，T0依次降级使用
     - **防饥饿**: 下环失败时触发优先级升级T2→T1→T0，有方向限制
     - **T0特殊**: 使用T0专用entry时需要全局队列轮询仲裁
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

## CrossRing 带宽分析系统

### 核心带宽计算公式

所有带宽计算统一使用公式：
```python
bandwidth = burst_length * 128 / time  # 直接得到 GB/s
```

### 带宽分析组件

#### ResultAnalyzer (`src/noc/analysis/result_analyzer.py`)
主要的带宽分析工具，提供以下核心方法：

1. **`calculate_bandwidth_metrics(requests, operation_type=None, endpoint_type="network")`**
   - 统一的带宽计算方法，支持工作区间算法
   - `endpoint_type`: 支持 "network"、"rn"、"sn" 三种端点类型
   - 返回非加权和加权带宽指标

2. **`analyze_port_bandwidth(metrics)`**
   - 按IP类型分组的端口带宽分析
   - 使用统一工作区间算法，确保结果一致性

### 时间窗口类型

#### 1. 网络时间 vs 端口时间
- **网络结束时间**: 数据在整个网络中传输完毕
- **RN端口时间**: 数据到达/离开RN端口（按读写区分）
- **SN端口时间**: 数据到达/离开SN端口（按读写区分）

#### 2. 端口时间的读写区分
```python
if req_type == "read":
    # 读请求：RN收到数据，SN发出数据
    rn_end_time = data_received_complete_cycle
    sn_end_time = data_entry_noc_from_cake1_cycle
else:  # write
    # 写请求：RN发出数据，SN收到数据
    rn_end_time = data_entry_noc_from_cake0_cycle  
    sn_end_time = data_received_complete_cycle
```

### 工作区间算法

#### WorkingInterval 类
```python
@dataclass
class WorkingInterval:
    start_time: int
    end_time: int
    duration: int
    flit_count: int
    total_bytes: int
    request_count: int
    
    @property
    def bandwidth(self) -> float:
        """区间内平均带宽 (GB/s)"""
        return self.total_bytes / self.duration
```

#### 加权 vs 非加权算法
- **非加权带宽**: `total_bytes / total_network_time`
- **加权带宽**: `Σ(区间带宽 × 区间flit数量) / Σ(区间flit数量)`

### 可视化工具

1. **累积带宽曲线**: 显示带宽随时间变化
2. **端口带宽对比**: 按IP类型对比读写性能
3. **流量分布图**: 节点和链路带宽可视化

### 使用示例

```python
from noc.analysis.result_analyzer import ResultAnalyzer

analyzer = ResultAnalyzer()
metrics = analyzer.convert_tracker_to_request_info(request_tracker, config)

# 网络整体带宽
overall_bw = analyzer.calculate_bandwidth_metrics(metrics, endpoint_type="network")

# RN端口带宽
rn_bw = analyzer.calculate_bandwidth_metrics(metrics, endpoint_type="rn")

# 端口级别分析
port_analysis = analyzer.analyze_port_bandwidth(metrics)
```

### 关键修正 (2025-07-23)

1. **端口时间处理**: 按读写操作正确区分RN/SN端口结束时间
2. **统一算法**: 移除重复的端口带宽计算，统一使用工作区间算法
3. **单位一致性**: 确保所有计算直接得到GB/s，无需额外转换
4. **endpoint_type支持**: `calculate_bandwidth_metrics`支持网络、RN、SN三种端点类型

### 注意事项

- 所有带宽值差异源于时间计算方式不同（网络 vs 端口 vs 工作区间）
- 推荐使用加权带宽作为主要性能指标
- 端口带宽分析必须考虑读写操作的时间差异

## CrossRing Request-Response Retry机制

### 整体机制概述

NoC系统的retry机制基于请求-响应模式，当SN端（Server Node，目标节点）资源不足时，会发送negative响应给RN端（Request Node，请求节点），RN端收到后将请求标记为retry并重新注入网络，关键要求是retry请求必须放到IP内部inject_fifo的队首以获得优先处理。

### 核心逻辑流程

#### 1. 初始请求处理
- RN端发送读/写请求到SN端
- SN端检查资源状态（tracker、WDB等）
- 资源充足：正常处理并发送positive/datasend响应
- 资源不足：发送negative响应，并将请求放入本地等待队列

#### 2. Negative响应处理（触发retry）
- RN端收到negative响应时，retry计数器+1
- 将原请求标记为"old"属性（区别于新请求的"new"）
- 请求状态设为"invalid"，等待资源可用信号

#### 3. 资源释放与Positive响应
- SN端完成其他请求时释放资源
- 从等待队列按FIFO顺序取出等待的请求
- 为等待请求分配资源并发送positive响应

#### 4. Retry请求重新注入（关键优先级逻辑）
- RN端收到positive响应，表示SN端现在有资源了
- **核心要求：将retry请求插入到inject_fifo的队首而非队尾**
- 这确保retry请求优先于新请求被处理，避免饥饿现象

### 响应类型说明

#### Negative响应
- **含义**：资源不足，请求被拒绝
- **触发条件**：
  - 读请求：SN端tracker资源不足
  - 写请求：SN端tracker或WDB（写数据缓冲）资源不足
- **后续动作**：请求进入SN端等待队列，RN端准备retry

#### Positive响应
- **含义**：资源已分配，可以重新发送请求
- **触发条件**：SN端资源释放后，从等待队列分配资源给等待请求
- **后续动作**：RN端将retry请求放到inject_fifo队首重新注入

#### Datasend响应
- **含义**：写操作数据传输完成
- **用途**：仅用于写请求的正常完成流程

### 关键数据结构

#### SN端
- **等待队列**：`sn_req_wait[req_type][ip_type][ip_pos]` - 存储因资源不足的等待请求
- **资源计数器**：跟踪tracker、WDB等资源的可用数量

#### RN端
- **Inject FIFO**：IP内部的请求注入队列，retry请求需插入队首
- **请求状态跟踪**：区分新请求("new")与retry请求("old")

### 优先级处理逻辑

#### 队首插入的重要性
1. **避免饥饿**：确保retry请求不会被持续到来的新请求阻塞
2. **公平性**：已经等待过的请求应该获得更高优先级
3. **性能**：减少请求的总体延迟和重试次数

#### 实现要点
- 使用支持队首插入的数据结构（如deque）
- retry请求调用队首插入方法而非普通的队尾插入
- 在请求对象中标记retry属性以便识别和统计

### 状态管理逻辑

#### 请求状态转换
1. **新请求**：`req_attr="new", req_state="valid"`
2. **收到negative**：`req_attr="old", req_state="invalid"`
3. **收到positive**：`req_attr="old", req_state="valid"`，重新注入到队首

#### 防重复处理
- 检查req_attr避免同一请求多次进入retry流程
- 确保每个negative响应只触发一次retry操作

### 统计监控

#### 基本统计
- `read_retry_num_stat`：读请求retry次数
- `write_retry_num_stat`：写请求retry次数

#### 统计时机
- 每次收到negative响应时计数器+1
- 在系统级别汇总所有IP接口的retry统计

### 与网络仲裁的交互

Ring_Bridge_arbitration等网络仲裁机制处理所有inject_fifo中的请求，包括retry请求。由于retry请求被放在队首，它们会被优先选中进行仲裁和转发，从而实现端到端的优先级处理。

### 实现关键点
1. **SN端资源释放时必须发送positive响应**
2. **RN端收到positive响应后将retry请求插入inject_fifo队首**
3. **维护完整的三维等待队列结构**
4. **正确的请求状态管理和转换**
5. **防重复处理的保护机制**

## E-Tag下环机制详细规范

### E-Tag下环判断逻辑（正确规范）

#### 1. Entry分配基本原则
E-Tag下环必须基于**正确的entry分配**，不存在"强制下环"：

- **T2级**: 只能使用T2专用entry
  ```python
  return entry_manager.can_allocate_entry("T2")
  ```

- **T1级**: 优先使用T1专用entry，没有时使用T2 entry
  ```python
  return entry_manager.can_allocate_entry("T1")  # 内部会检查T1和T2
  ```

- **T0级**: 优先使用T0专用entry，然后依次降级使用T1、T2 entry
  ```python
  if entry_manager.can_allocate_entry("T0"):
      # 如果使用T0专用entry，需要检查全局队列排序
      if using_t0_dedicated_entry:
          return self._is_first_in_t0_queue(slot, channel)
      else:
          return True  # 使用降级entry无需队列检查
  ```

#### 2. Entry管理器配置要求
每个方向需要正确配置：
- `total_depth`: 对应FIFO的总深度
- `t2_max`: T2级最大可用entry数量  
- `t1_max`: T1级最大可用entry数量（包含T2）
- `has_dedicated_entries`: 是否有专用entry（TL/TU=True, TR/TD=False）

#### 3. E-Tag升级规则
- **T2→T1升级**: 
  - `ETAG_BOTHSIDE_UPGRADE=0`: 只有TL和TU能升级
  - `ETAG_BOTHSIDE_UPGRADE=1`: TL、TR、TU、TD都能升级
- **T1→T0升级**: 只有TL或TU能升级，TR和TD永远不能升级到T0

#### 4. 配置参数映射
- `TL_ETAG_T1_UE_MAX`: TL方向T1阈值
- `TU_ETAG_T1_UE_MAX`: TU方向T1阈值
- `TR_ETAG_T1_UE_MAX`: 使用 `RB_IN_FIFO_DEPTH`
- `TD_ETAG_T1_UE_MAX`: 使用 `EQ_IN_FIFO_DEPTH`

### E-Tag实现要点
1. **Entry分配优先**: 所有下环都必须先获得对应的entry
2. **正确配置**: Entry管理器必须使用正确的FIFO深度和阈值配置
3. **T0队列管理**: T0使用专用entry时需要全局队列排序
4. **调试关键**: 检查entry管理器初始化和entry分配计算逻辑