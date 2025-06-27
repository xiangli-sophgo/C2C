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
│   ├── node.py          # 节点实现
│   ├── link.py          # 链路实现
│   ├── graph.py         # 图表示
│   └── builder.py       # 拓扑构建器
├── protocol/
│   ├── __init__.py
│   ├── base.py          # 协议基础类
│   ├── cdma.py          # CDMA协议
│   ├── credit.py        # Credit管理
│   ├── address.py       # 地址转换
│   └── router.py        # 路由器
├── config/
│   ├── __init__.py
│   └── loader.py        # 配置加载器
├── utils/
│   ├── __init__.py
│   ├── constants.py     # 常量定义
│   └── exceptions.py    # 异常定义
└── examples/
    ├── __init__.py
    └── basic_demo.py    # 基础演示
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

### 5. 协议层基础 (`protocol/base.py`)

定义了 `BaseProtocol` 抽象基类和 `ProtocolState` 枚举，为协议层提供基础状态管理和消息处理接口。

### 6. CDMA协议 (`protocol/cdma.py`)

实现了 `CDMAProtocol`，处理 CDMA 消息的发送和接收指令，并定义了 `CDMAMessage` 的消息格式。

### 7. Credit管理 (`protocol/credit.py`)

`CreditManager` 类负责管理 Credit 机制，包括 Credit 的初始化、申请、授予以及状态查询。

### 8. 地址转换 (`protocol/address.py`)

`AddressTranslator` 类提供了 SG 地址空间与 PC 地址空间之间的转换功能，并定义了 `AddressFormat` 枚举。

### 9. 常量定义 (`utils/constants.py`)

定义了项目中使用的各种硬件参数、延迟参数和拓扑限制等常量。

### 10. 基础演示 (`examples/basic_demo.py`)

提供了一个 `demo_cascade_topology` 函数，演示了如何创建4芯片级联拓扑、计算路径、模拟 CDMA 通信、Credit 管理和地址转换。

## 技术要求

1.  **使用Python 3.8+**，遵循PEP 8代码规范。
2.  **依赖库**：`networkx`, `matplotlib`, `pyyaml`, `dataclasses`, `enum`。
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

1.  **安装依赖**：
    ```bash
    pip install networkx matplotlib pyyaml
    ```
2.  **运行演示**：
    ```bash
    python examples/basic_demo.py
    ```
