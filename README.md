# C2C 拓扑建模基础框架

## 任务描述

本项目旨在实现SG2260E芯片间C2C通信的拓扑建模基础框架，包括拓扑层和协议层的核心抽象类和接口。这是一个芯片间互联系统的建模项目，需要支持不同的拓扑类型（如级联、Switch互联等）和CDMA协议。

## 项目结构

```
.
├── __init__.py
├── topology/
│   ├── __init__.py
│   ├── base.py          # 基础抽象类
│   ├── builder.py       # 拓扑构建器
│   ├── graph.py         # 图表示
│   ├── link.py          # 链路实现
│   ├── node.py          # 节点实现
│   ├── tree.py          # 树状拓扑核心逻辑
│   └── torus.py         # Torus拓扑核心逻辑
├── protocol/
│   ├── __init__.py
│   ├── address.py       # 地址转换
│   ├── base.py          # 协议基础类
│   ├── cdma.py          # CDMA协议
│   ├── credit.py        # Credit管理
│   └── router.py        # 路由器
├── config/
│   ├── __init__.py
│   └── loader.py        # 配置加载器
├── utils/
│   ├── __init__.py
│   ├── constants.py     # 常量定义
│   └── exceptions.py    # 异常定义
├── visualization/
│   ├── __init__.py
│   ├── visualizer.py    # 主可视化器
│   ├── layouts.py       # 布局算法
│   ├── comparison.py    # 性能对比工具
│   ├── interactive.py   # Web交互界面
│   └── utils.py         # 可视化工具
├── examples/
│   ├── __init__.py
│   ├── basic_demo.py    # 基础演示
│   ├── tree_torus_validation.py  # 拓扑验证脚本
│   └── visualization_demo.py     # 可视化演示
├── scripts/
│   └── fix_and_validate_topologies.py  # 修复验证脚本
├── output/              # 输出文件目录
│   └── .gitkeep        # 保持目录结构
└── run_webapp.py        # Web应用启动脚本
```

## 详细需求

### 1. 拓扑层基础类 (`topology/base.py`)

定义了 `BaseNode`, `BaseLink`, `BaseTopology` 三个抽象基类，为拓扑中的节点、链路和整体拓扑结构提供基础接口和属性管理。

### 2. 具体节点实现 (`topology/node.py`)

实现了 `ChipNode` (代表SG2260E芯片), `SwitchNode` (PCIe Switch), `HostNode` (Host PC) 等具体节点类型，继承自 `BaseNode` 并扩展了各自特有的属性。

### 3. 链路实现 (`topology/link.py`)

实现了 `PCIeLink` 和 `C2CDirectLink` 两种链路类型，继承自 `BaseLink` 并定义了各自的带宽、延迟等属性。

### 4. 图表示 (`topology/graph.py`)

`TopologyGraph` 类基于 NetworkX 库实现了拓扑图的表示，提供了节点的增删改查、路径查找（最短路径、最小跳数）以及图的可视化方法。

### 5. 树状拓扑核心逻辑 (`topology/tree.py`)

实现了SG2260E C2C系统的树状拓扑核心逻辑。包括：
-   **`TreeTopologyLogic`**: 计算最优树状结构、路由表、c2c_sys映射和ATU分配。
-   **`TreeAddressRoutingLogic`**: 基于树状拓扑的地址路由决策、地址格式转换和All Reduce操作优化。
-   **`TreeConfigGenerationLogic`**: 生成芯片c2c_sys配置、ATU配置表和PCIe Switch配置。
-   **`TreeFaultToleranceLogic`**: 实现故障检测、恢复拓扑计算和恢复配置生成。
-   **核心算法**: 包含树结构优化、最短路径和All Reduce优化算法。

### 6. Torus拓扑核心逻辑 (`topology/torus.py`)

实现了SG2260E C2C系统的Torus拓扑核心逻辑。包括：
-   **`TorusTopologyLogic`**: 计算最优Torus结构、坐标映射、邻居列表和网格尺寸优化。
-   **`TorusRoutingLogic`**: 实现维度顺序路由、最短距离计算、自适应路由和容错路由。
-   **`TorusC2CMappingLogic`**: 将Torus方向映射到c2c_sys、生成C2C Link配置和优化链路带宽。
-   **`TorusAddressRoutingLogic`**: Torus地址路由决策和下一跳选择。
-   **`TorusAllReduceLogic`**: 针对Torus拓扑的All Reduce模式优化、递归倍增和流水线优化。
-   **`TorusFaultToleranceLogic`**: 实现链路故障检测、故障影响分析和恢复路由生成。

### 7. 协议层基础 (`protocol/base.py`)

定义了 `BaseProtocol` 抽象基类和 `ProtocolState` 枚举，为协议层提供基础状态管理和消息处理接口。

### 8. CDMA协议 (`protocol/cdma.py`)

实现了 `CDMAProtocol`，处理 CDMA 消息的发送和接收指令，并定义了 `CDMAMessage` 的消息格式。

### 9. Credit管理 (`protocol/credit.py`)

`CreditManager` 类负责管理 Credit 机制，包括 Credit 的初始化、申请、授予以及状态查询。

### 10. 地址转换 (`protocol/address.py`)

`AddressTranslator` 类提供了 SG 地址空间与 PC 地址空间之间的转换功能，并定义了 `AddressFormat` 枚举。

### 11. 常量定义 (`utils/constants.py`)

定义了项目中使用的各种硬件参数、延迟参数和拓扑限制等常量。

### 12. 可视化模块 (`visualization/`)

提供了完整的拓扑可视化和分析工具：
- **`visualizer.py`**: 主可视化器，支持Tree/Torus/混合拓扑的静态可视化
- **`layouts.py`**: 专业布局算法，包括树状层次布局、环形网格布局等
- **`comparison.py`**: 性能对比分析工具，支持多拓扑对比、热点分析
- **`interactive.py`**: Streamlit Web界面，提供交互式拓扑配置和实时可视化
- **`utils.py`**: 可视化工具集，包括颜色管理、图形工具、数据处理

### 13. 基础演示 (`examples/basic_demo.py`)

提供了一个 `demo_cascade_topology` 函数，演示了如何创建4芯片级联拓扑、计算路径、模拟 CDMA 通信、Credit 管理和地址转换。

### 14. 验证和演示脚本

- **`tree_torus_validation.py`**: 专门验证Tree和Torus拓扑实现的详细测试
- **`visualization_demo.py`**: 展示所有可视化功能的演示脚本
- **`scripts/fix_and_validate_topologies.py`**: 综合验证和修复脚本

## 技术要求

1.  **使用Python 3.8+**，遵循PEP 8代码规范。
2.  **依赖库**：`networkx`, `matplotlib`, `pyyaml`, `dataclasses`, `enum`, `streamlit`, `plotly`, `seaborn`, `pandas`, `numpy`。
3.  **设计模式**：使用ABC抽象基类、工厂模式、策略模式。
4.  **文档**：每个类和方法都有清晰的docstring。
5.  **类型提示**：使用typing模块提供类型注解。
6.  **异常处理**：定义自定义异常类并合理使用。
7.  **可扩展性**：接口设计便于后续功能扩展。

## 验证要求

运行 `examples/basic_demo.py` 脚本，验证以下功能：

1.  能够创建不同类型的节点和链路。
2.  能够构建基础的4芯片级联拓扑。
3.  能够计算节点间的最短路径。
4.  CDMA协议的基础状态转换正常。
5.  Credit机制的申请/授予流程正确。

## 如何运行

### 基础安装和运行
1.  **安装依赖**：
    ```bash
    pip install networkx matplotlib pyyaml streamlit plotly seaborn pandas numpy
    ```
2.  **运行基础演示**：
    ```bash
    python examples/basic_demo.py
    ```

### 可视化工具使用

1.  **运行静态可视化演示**：
    ```bash
    python examples/visualization_demo.py
    ```

2.  **启动交互式Web界面**：
    ```bash
    python run_webapp.py
    ```
    或者
    ```bash
    streamlit run visualization/interactive.py
    ```

3.  **运行拓扑验证测试**：
    ```bash
    python examples/tree_torus_validation.py
    ```

### 可视化功能特性

- **📊 多拓扑可视化**: 支持Tree、Torus、混合拓扑的可视化展示
- **⚡ 性能对比分析**: Tree vs Torus性能指标对比图表
- **🛤️ 路径分析工具**: 交互式路径查找和高亮显示
- **🔥 热点分析**: 网络节点和链路的瓶颈分析
- **🎨 多样式支持**: 默认、色盲友好、暗色等多种颜色方案
- **💾 导出功能**: 支持高质量图片导出和数据报告生成
- **🌐 Web界面**: 基于Streamlit的交互式Web应用

### 输出文件管理

所有生成的图片、报告和分析结果都会保存在 `output/` 目录中，包括：
- 拓扑可视化图片 (PNG格式)
- 性能对比图表
- 分析报告 (Markdown格式)  
- 验证测试结果