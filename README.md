# C2C 拓扑建模与仿真框架

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

一个用于C2C（Chip-to-Chip）系统拓扑的建模、分析、可视化和仿真的综合框架。

## 核心功能

- **拓扑建模**: 支持多种拓扑结构，如树状（Tree）和环形（Torus），并提供灵活的节点和链路定义。
- **协议实现**: 完整的CDMA协议栈，包括消息同步、流量控制、错误处理和性能监控。
- **仿真引擎**: 全新的事件驱动仿真系统，支持cycle级精确的C2C通信仿真。
- **性能分析**: 对不同的拓扑结构进行全面的性能评估和对比，包括路径长度、带宽、成本和容错能力。
- **可视化**: 提供静态和交互式的拓扑可视化工具，帮助用户直观地理解网络结构。
- **可扩展性**: 框架设计灵活，易于扩展，可以支持新的拓扑类型、协议和分析指标。

## 项目结构

```
.
├── src/
│   ├── topology/         # 拓扑层核心逻辑
│   ├── protocol/         # 协议层实现（CDMA等）
│   ├── simulation/       # ✨ 新增：仿真引擎
│   ├── visualization/    # 可视化工具
│   ├── utils/            # 工具和常量
│   └── config/           # 配置管理
├── examples/
│   ├── basic_demo.py         # 基础功能演示
│   ├── simulation_demo.py    # ✨ 新增：仿真功能演示
│   ├── enhanced_topology_comparison.py # 拓扑对比分析
│   ├── tree_torus_validation.py # 拓扑算法验证
│   └── visualization_demo.py  # 可视化功能演示
├── scripts/
│   └── run_webapp.py         # 启动Web应用的脚本
├── output/                   # 生成的报告和图表
├── README.md
├── setup.py                  # 项目安装脚本
├── pyproject.toml            # 项目构建配置
└── requirements.txt          # 依赖库列表
```

## 安装

1.  **克隆仓库**

    ```bash
    git clone https://github.com/your-org/C2C.git
    cd C2C
    ```

2.  **创建并激活虚拟环境** (推荐)

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **安装依赖**

    ```bash
    pip install -r requirements.txt
    ```

4.  **以可编辑模式安装项目**

    ```bash
    pip install -e .
    ```

## 如何使用

### 直接运行脚本

你也可以直接运行 `examples` 目录中的脚本：

-   **基础演示**:

    ```bash
    python examples/basic_demo.py
    ```

-   **✨ 仿真功能演示** (新增):

    ```bash
    python examples/simulation_demo.py
    ```

-   **拓扑对比分析**:

    ```bash
    python examples/enhanced_topology_comparison.py
    ```

-   **启动Web应用**:

    ```bash
    streamlit run src/visualization/interactive.py
    ```

## ✨ 仿真引擎 (新功能)

全新的事件驱动仿真系统，支持cycle级精确的C2C通信仿真：

### 核心组件

- **C2CSimulationEngine**: 事件驱动的仿真引擎核心
- **FakeChip**: 简化的芯片模型，继承现有ChipNode
- **SimulationEvent**: 完整的事件系统，支持多种事件类型
- **SimulationStats**: 全面的统计收集和性能分析

### 功能特性

- 🎯 **事件驱动仿真**: 支持CDMA发送/接收、链路传输等多种事件
- 📊 **性能统计**: 吞吐量、延迟、利用率等关键性能指标
- 🔄 **周期性流量**: 支持复杂的流量模式和负载测试
- 🌐 **多芯片拓扑**: 支持复杂的芯片间通信拓扑
- 📈 **实时监控**: 仿真过程中的实时性能监控

### 快速开始

```python
from src.simulation import C2CSimulationEngine, FakeChip
from src.topology.builder import TopologyBuilder
from src.topology.node import ChipNode
from src.topology.link import C2CDirectLink

# 1. 创建拓扑
builder = TopologyBuilder("my_simulation")
chip0 = ChipNode("chip_0", "board_A")
chip1 = ChipNode("chip_1", "board_A") 
builder.add_node(chip0)
builder.add_node(chip1)
builder.add_link(C2CDirectLink("link_0_1", chip0, chip1))

# 2. 创建仿真引擎
simulator = C2CSimulationEngine(builder)

# 3. 添加仿真事件
simulator.add_cdma_send_event(
    timestamp_ns=1000,
    source_chip_id="chip_0",
    target_chip_id="chip_1", 
    data_size=1024
)

# 4. 运行仿真
stats = simulator.run_simulation(simulation_time_ns=1_000_000)
stats.print_summary()
```

### 仿真示例

运行完整的仿真演示：

```bash
python examples/simulation_demo.py
```

示例包含：
- 基础双芯片通信仿真
- 复杂4芯片环形拓扑仿真
- 周期性流量模式测试
- 性能统计和分析

## 可视化与分析

-   **静态图表**: `visualization_demo.py` 演示了如何生成各种拓扑的静态图表。
-   **交互式Web应用**: `interactive.py` 提供了一个基于 `streamlit` 的Web界面，允许用户：
    -   动态配置拓扑参数。
    -   实时查看拓扑结构图。
    -   进行多维度性能对比分析。
    -   根据应用需求获取拓扑优化建议。

所有生成的图表和报告都将保存在 `output/` 目录下。
