# C2C 拓扑建模与分析框架

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

一个用于C2C（Chip-to-Chip）系统拓扑的建模、分析和可视化框架。

## 核心功能

- **拓扑建模**: 支持多种拓扑结构，如树状（Tree）和环形（Torus），并提供灵活的节点和链路定义。
- **性能分析**: 对不同的拓扑结构进行全面的性能评估和对比，包括路径长度、带宽、成本和容错能力。
- **可视化**: 提供静态和交互式的拓扑可视化工具，帮助用户直观地理解网络结构。
- **可扩展性**: 框架设计灵活，易于扩展，可以支持新的拓扑类型、协议和分析指标。

## 项目结构

```
.
├── src/
│   ├── topology/         # 拓扑层核心逻辑
│   ├── protocol/         # 协议层实现
│   ├── visualization/    # 可视化工具
│   ├── utils/            # 工具和常量
│   └── config/           # 配置管理
├── examples/
│   ├── basic_demo.py         # 基础功能演示
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
    python -m C2C.examples.basic_demo
    ```

-   **拓扑对比分析**:

    ```bash
    python -m C2C.examples.enhanced_topology_comparison
    ```

-   **启动Web应用**:

    ```bash
    streamlit run src/C2C/visualization/interactive.py
    ```

## 可视化与分析

本框架的核心优势之一是其强大的可视化和分析能力。

-   **静态图表**: `visualization_demo.py` 演示了如何生成各种拓扑的静态图表。
-   **交互式Web应用**: `interactive.py` 提供了一个基于 `streamlit` 的Web界面，允许用户：
    -   动态配置拓扑参数。
    -   实时查看拓扑结构图。
    -   进行多维度性能对比分析。
    -   根据应用需求获取拓扑优化建议。

所有生成的图表和报告都将保存在 `output/` 目录下。
