# C2C 拓扑建模与仿真框架

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

一个用于C2C（Chip-to-Chip）系统拓扑的建模、分析、可视化和仿真的综合框架。

## 核心功能

- **拓扑建模**: 支持多种拓扑结构，如树状（Tree）、环形（Torus）、CrossRing等，并提供灵活的节点和链路定义。
- **协议实现**: 完整的CDMA协议栈，包括消息同步、流量控制、错误处理和性能监控。
- **仿真引擎**: 事件驱动仿真系统，支持cycle级精确的C2C通信仿真。
- **性能分析**: 对不同的拓扑结构进行全面的性能评估和对比，包括路径长度、带宽、成本和容错能力。
- **可视化**: 提供静态和交互式的拓扑可视化工具，帮助用户直观地理解网络结构。
- **可扩展性**: 框架设计灵活，易于扩展，可以支持新的拓扑类型、协议和分析指标。

## 项目结构

```
.
├── src/
│   ├── c2c/                # C2C核心模块（拓扑、协议、工具）
│   │   ├── topology/       # 拓扑建模相关
│   │   ├── protocol/       # 协议实现（CDMA等）
│   │   └── utils/          # 工具和异常
│   ├── noc/                # NoC实现 - 分层架构设计
│   │   ├── base/           # 基础抽象层（所有NoC的通用接口）
│   │   │   ├── model.py    # NoC模型基类（仿真控制、统计收集）
│   │   │   ├── topology.py # 拓扑基类（路由算法、路径计算）
│   │   │   ├── node.py     # 节点基类（状态管理、接口定义）
│   │   │   ├── ip_interface.py # IP接口基类（FIFO、流控）
│   │   │   ├── flit.py     # Flit基类和对象池管理
│   │   │   ├── link.py     # 链路基类（连接抽象）
│   │   │   └── config.py   # 配置管理基类
│   │   ├── crossring/      # CrossRing拓扑实现（继承base/）
│   │   │   ├── model.py    # CrossRing模型实现
│   │   │   ├── topology.py # CrossRing路由和拓扑结构
│   │   │   ├── node.py     # CrossRing节点（CrossPoint实现）
│   │   │   ├── cross_point.py # 环形路由核心组件
│   │   │   ├── ip_interface.py # CrossRing IP接口
│   │   │   ├── link.py     # 环形链路和slice管理
│   │   │   ├── flit.py     # CrossRing包格式
│   │   │   └── config.py   # CrossRing配置
│   │   ├── mesh/           # Mesh拓扑实现（继承base/）
│   │   ├── analysis/       # 性能分析模块
│   │   │   ├── crossring_analyzer.py # CrossRing专用分析
│   │   │   └── fifo_analyzer.py # FIFO统计分析
│   │   ├── utils/          # 通用工具模块
│   │   │   ├── traffic_scheduler.py # Traffic调度管理
│   │   │   ├── factory.py  # 组件工厂模式
│   │   │   ├── types.py    # 类型定义和枚举
│   │   │   └── adjacency.py # 邻接矩阵工具
│   │   ├── debug/          # 调试工具模块
│   │   │   └── request_tracker.py # 请求生命周期追踪
│   │   └── visualization/  # 可视化模块
│   │       ├── realtime_visualizer.py # 实时仿真可视化
│   │       ├── crossring_*_visualizer.py # CrossRing专用可视化
│   │       └── network_topology_visualizer.py # 拓扑可视化
│   ├── simulation/         # 仿真引擎
│   ├── visualization/      # 可视化工具
│   └── config/             # 配置管理
├── examples/               # 示例与演示脚本
├── scripts/                # 辅助脚本
├── output/                 # 生成的报告和图表
├── README.md
├── setup.py
└── requirements.txt
```

## NoC架构设计

### 设计原则
- **分层架构**: 基础抽象层 + 拓扑实现层 + 功能模块层
- **继承复用**: 通过继承base/模块实现新拓扑，避免重复代码
- **组合模式**: 独立的功能模块（analysis, utils, debug, visualization）
- **扩展性**: 支持新拓扑类型、路由策略、分析方法的简单添加

### 核心组件
- **base/**: 所有NoC拓扑的通用抽象接口和基础功能
- **crossring/**: CrossRing拓扑的完整实现，继承并扩展base/功能
- **mesh/**: Mesh拓扑实现（类似继承结构）
- **analysis/**: 性能分析工具集
- **utils/**: 通用工具和辅助功能
- **debug/**: 调试和监控工具
- **visualization/**: 可视化和图表生成

### 新拓扑添加指南
1. 继承base/中的相应模块
2. 实现拓扑特有的功能（路由算法、节点行为等）
3. 复用existing的analysis、utils、debug、visualization模块
4. 在utils/factory.py中注册新拓扑类型

## 安装

1.  **克隆仓库**

    ```bash
    git clone http://10.129.4.209/xiang.li/C2C.git
    cd C2C
    ```

2.  **创建并激活虚拟环境** (推荐)

    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows下用 venv\Scripts\activate
    ```

3.  **安装依赖**

    ```bash
    pip install -r requirements.txt
    ```

4.  **以可编辑模式安装项目**

    ```bash
    pip install -e .
    ```

## 快速上手

### 典型用法

以C2C模块为例：

```python
from src.c2c.topology.builder import TopologyBuilder
from src.c2c.topology.node import ChipNode
from src.c2c.topology.link import C2CDirectLink
from src.c2c.protocol.cdma_system import CDMASystem

# 1. 创建拓扑
builder = TopologyBuilder("my_topo")
chip0 = ChipNode("chip_0", "board_A")
chip1 = ChipNode("chip_1", "board_A")
builder.add_node(chip0)
builder.add_node(chip1)
builder.add_link(C2CDirectLink("link_0_1", chip0, chip1))
topology = builder.build()

# 2. 创建CDMA系统并连接
sys0 = CDMASystem("chip_0")
sys1 = CDMASystem("chip_1")
sys0.connect_to_chip("chip_1", sys1)

# 3. 发送/接收CDMA事务
recv_result = sys1.cdma_receive(
    dst_addr=0x1000, dst_shape=(64,), dst_mem_type="GMEM", src_chip_id="chip_0", data_type="float32"
)
send_result = sys0.cdma_send(
    src_addr=0x2000, src_shape=(64,), dst_chip_id="chip_1", src_mem_type="GMEM", data_type="float32"
)
```

### 运行示例脚本

-   **基础演示**:

    ```bash
    python examples/basic_demo.py
    ```

-   **仿真功能演示**:

    ```bash
    python examples/simulation_demo.py
    ```

-   **CrossRing NoC演示**:

    ```bash
    python examples/crossring_noc_demo.py
    ```

-   **拓扑对比分析**:

    ```bash
    python examples/topology_comparison.py
    ```

-   **启动Web应用**:

    ```bash
    streamlit run src/visualization/interactive.py
    ```

## 主要模块说明

- `src/c2c/topology/`  拓扑建模、节点、链路、构建器、拓扑优化
- `src/c2c/protocol/`  CDMA协议、信用管理、地址转换、流控、性能监控、错误处理
- `src/c2c/utils/`     异常、类型定义、通用工具
- `src/simulation/`    仿真引擎、事件、芯片模型
- `src/visualization/` 可视化与交互式分析

## 贡献与反馈

欢迎提交issue、PR或建议！

---

如需详细API文档和进阶用法，请参考各子模块下的README或docstring。
