# NoC 分析框架使用指南

本指南介绍如何使用新的NoC分析框架进行性能分析、可视化和结果管理。

## 概述

NoC分析框架提供了完整的网络片上系统性能分析解决方案，包括：

- **性能指标分析**: 带宽、延迟、吞吐量、热点分析
- **可视化功能**: 多维度图表、性能仪表板、网络流量图
- **输出管理**: 自动化的文件管理和会话组织
- **结果导出**: 多格式数据导出支持

## 快速开始

### 1. 基本使用

```python
from src.noc.analysis import ResultProcessor, PerformanceVisualizer, SimulationContext

# 配置参数
config = {
    'topology_type': 'mesh_4x4',
    'node_count': 16,
    'gap_threshold_ns': 200,
    'window_size_ns': 1000
}

# 创建仿真会话
with SimulationContext(
    model_name="crossring",
    topology_type="4x4_mesh",
    config=config,
    session_name="my_experiment"
) as output_manager:
    
    # 初始化分析器
    processor = ResultProcessor(config, output_manager)
    
    # 收集数据（从你的仿真模型）
    processor.collect_simulation_data(simulation_model)
    
    # 执行分析
    network_metrics = processor.analyze_performance()
    
    # 生成可视化
    visualizer = PerformanceVisualizer(output_manager=output_manager)
    visualizer.create_performance_dashboard(network_metrics)
    
    # 导出结果
    processor.export_results('excel', 'complete_analysis')
```

### 2. 手动管理会话

```python
from src.noc.analysis import OutputManager, ResultProcessor

# 创建输出管理器
output_manager = OutputManager()

# 创建会话
session_id = output_manager.create_session(
    model_name="mesh_model",
    topology_type="8x8_mesh",
    config=your_config
)

# 进行分析...

# 保存自定义数据
output_manager.save_data(custom_data, "custom_analysis", "json")
output_manager.save_report(report_text, "detailed_report", "md")
```

## 输出文件结构

每次仿真都会创建一个独立的会话目录：

```
output/
└── {model}_{topology}_{session_name}_{timestamp}/
    ├── README.md                    # 会话说明文档
    ├── session_metadata.json       # 会话元数据
    ├── config/                      # 配置文件目录
    │   ├── main_config.yaml        # 主配置文件
    │   └── custom_config.yaml      # 自定义配置文件
    ├── logs/                        # 日志文件目录
    │   ├── session.log             # 会话日志
    │   └── simulation_run.log      # 仿真运行日志
    ├── figures/                     # 图片文件目录
    │   ├── bandwidth_analysis.png  # 带宽分析图
    │   ├── latency_analysis.png    # 延迟分析图
    │   ├── throughput_analysis.png # 吞吐量分析图
    │   ├── hotspot_analysis.png    # 热点分析图
    │   ├── performance_dashboard.png # 性能仪表板
    │   └── network_flow.png        # 网络流量图
    ├── data/                        # 数据文件目录
    │   ├── raw_simulation_data.json # 原始仿真数据
    │   ├── performance_summary.json # 性能摘要
    │   ├── request_details.csv     # 请求详细数据
    │   └── complete_analysis.xlsx  # 完整分析数据
    ├── analysis/                    # 分析结果目录
    │   └── detailed_results.json   # 详细分析结果
    └── reports/                     # 报告文件目录
        ├── performance_report.md   # 性能分析报告
        └── session_summary.md      # 会话摘要
```

## 功能详解

### 性能分析

#### 1. 带宽分析
- **平均带宽**: 基于总传输时间计算
- **峰值带宽**: 单个请求的最大带宽
- **有效带宽**: 基于工作区间计算，排除空闲时间
- **带宽效率**: 有效带宽与峰值带宽的比率
- **工作区间分析**: 识别活跃传输时段

#### 2. 延迟分析
- **总延迟**: 从请求开始到完成的时间
- **命令延迟**: 命令传输延迟
- **数据延迟**: 数据传输延迟
- **网络延迟**: 网络传输延迟
- **延迟分布**: P50, P95, P99百分位数

#### 3. 吞吐量分析
- **平均吞吐量**: 基于总处理时间
- **峰值吞吐量**: 时间窗口内的最大吞吐量
- **持续吞吐量**: 工作区间内的平均吞吐量
- **吞吐量稳定性**: 吞吐量变化的标准差

#### 4. 热点分析
- **节点流量统计**: 每个节点的进出流量
- **拥塞检测**: 基于流量和延迟的拥塞识别
- **负载均衡**: 节点间负载分布分析
- **热点标识**: 自动标识网络热点节点

### 可视化功能

#### 1. 性能图表
```python
visualizer = PerformanceVisualizer(output_manager=output_manager)

# 带宽分析图 (2x2子图)
visualizer.plot_bandwidth_analysis(network_metrics)

# 延迟分析图 (2x2子图)
visualizer.plot_latency_analysis(network_metrics)

# 吞吐量分析图 (2x2子图)
visualizer.plot_throughput_analysis(network_metrics)

# 热点分析图 (2x2子图)
visualizer.plot_hotspot_analysis(network_metrics)
```

#### 2. 性能仪表板
```python
# 创建综合性能仪表板 (4x4网格布局)
visualizer.create_performance_dashboard(network_metrics)
```

#### 3. 网络流量图
```python
flow_visualizer = NetworkFlowVisualizer(layout='grid')
topology_info = {'rows': 4, 'cols': 4}
flow_visualizer.visualize_network_flow(requests, topology_info)
```

### 数据导出

#### 1. JSON格式
```python
# 导出性能摘要和配置信息
processor.export_results('json', 'performance_summary')
```

#### 2. CSV格式
```python
# 导出请求详细数据
processor.export_results('csv', 'request_details')
```

#### 3. Excel格式
```python
# 导出多工作表详细分析
processor.export_results('excel', 'complete_analysis')
```

### 会话管理

#### 1. 列出历史会话
```python
output_manager = OutputManager()
sessions = output_manager.list_sessions()

for session in sessions:
    print(f"会话: {session['session_id']}")
    print(f"模型: {session['model_name']}")
    print(f"拓扑: {session['topology_type']}")
    print(f"时间: {session['created_time']}")
```

#### 2. 加载已有会话
```python
output_manager.load_session("session_id_here")
```

#### 3. 清理旧会话
```python
# 保留最近10个会话，删除其余
output_manager.cleanup_old_sessions(keep_count=10)
```

## 配置选项

### 分析配置
```python
config = {
    'topology_type': 'mesh_4x4',           # 拓扑类型
    'node_count': 16,                      # 节点数量
    'gap_threshold_ns': 200,               # 工作区间合并阈值
    'window_size_ns': 1000,                # 吞吐量分析时间窗口
    'congestion_threshold': 0.8,           # 拥塞检测阈值
    'network_frequency': 2.0,              # 网络频率 (GHz)
    'routing_algorithm': 'xy_routing',     # 路由算法
    'packet_size': 128                     # 包大小 (bytes)
}
```

### 可视化配置
```python
visualizer = PerformanceVisualizer(
    style='seaborn-v0_8',                  # 绘图样式
    figsize=(12, 8),                       # 图片尺寸
    output_manager=output_manager          # 输出管理器
)
```

## 高级用法

### 1. 自定义分析器
```python
# 使用独立的分析器
bandwidth_analyzer = BandwidthAnalyzer(gap_threshold_ns=500)
latency_analyzer = LatencyAnalyzer()
throughput_analyzer = ThroughputAnalyzer(window_size_ns=2000)
hotspot_analyzer = HotspotAnalyzer(congestion_threshold=0.9)

# 执行特定分析
bandwidth_metrics = bandwidth_analyzer.calculate_bandwidth_metrics(requests)
latency_metrics = latency_analyzer.calculate_latency_metrics(requests)
```

### 2. 按条件分析
```python
# 按请求类型分析
read_requests = [r for r in requests if r.request_type == RequestType.READ]
write_requests = [r for r in requests if r.request_type == RequestType.WRITE]

read_bandwidth = bandwidth_analyzer.calculate_bandwidth_metrics(read_requests)
write_bandwidth = bandwidth_analyzer.calculate_bandwidth_metrics(write_requests)

# 按跳数分析延迟
latency_by_distance = latency_analyzer.analyze_latency_by_distance(requests)

# 按节点对分析延迟
latency_by_node_pair = latency_analyzer.analyze_latency_by_node_pair(requests)
```

### 3. 自定义可视化
```python
import matplotlib.pyplot as plt

# 创建自定义图表
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制自定义数据
latencies = [r.total_latency for r in requests]
ax.hist(latencies, bins=50, alpha=0.7)
ax.set_xlabel('延迟 (ns)')
ax.set_ylabel('频次')
ax.set_title('延迟分布')

# 保存自定义图表
if output_manager:
    output_manager.save_figure(fig, "custom_latency_histogram")
```

### 4. 自定义数据保存
```python
# 保存自定义分析结果
custom_analysis = {
    'analysis_type': 'custom_hotspot_analysis',
    'hotspot_nodes': [1, 5, 9, 13],
    'congestion_levels': [0.8, 0.9, 0.7, 0.85],
    'recommendations': [
        '增加节点1的缓冲区大小',
        '优化节点5的路由算法',
        '减少节点9的负载'
    ]
}

output_manager.save_analysis_result(custom_analysis, "custom_hotspot_analysis")
```

## 故障排除

### 常见问题

#### 1. 中文字体显示问题
```python
# 在代码开头添加字体配置
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
```

#### 2. 输出目录权限问题
确保output目录有写权限：
```bash
chmod 755 output/
```

#### 3. Excel导出失败
安装openpyxl库：
```bash
pip install openpyxl
```

#### 4. 内存不足
对于大量请求数据，可以分批处理：
```python
# 分批处理请求
batch_size = 1000
for i in range(0, len(requests), batch_size):
    batch_requests = requests[i:i+batch_size]
    batch_metrics = processor.analyze_batch(batch_requests)
```

### 性能优化

#### 1. 减少内存使用
```python
# 只保留必要的请求数据
essential_requests = [
    RequestMetrics(
        packet_id=r.packet_id,
        request_type=r.request_type,
        start_time=r.start_time,
        end_time=r.end_time,
        total_bytes=r.total_bytes
    ) for r in full_requests
]
```

#### 2. 加速分析
```python
# 使用较大的工作区间阈值
config['gap_threshold_ns'] = 1000  # 默认200

# 使用较大的时间窗口
config['window_size_ns'] = 5000    # 默认1000
```

## 扩展开发

### 1. 添加新的性能指标
```python
@dataclass
class CustomMetrics:
    custom_value: float
    custom_data: List[float]
    
    @classmethod
    def from_requests(cls, requests: List[RequestMetrics]) -> 'CustomMetrics':
        # 自定义计算逻辑
        custom_value = sum(r.total_bytes for r in requests) / len(requests)
        custom_data = [r.bandwidth_gbps for r in requests]
        return cls(custom_value=custom_value, custom_data=custom_data)
```

### 2. 添加新的可视化类型
```python
class CustomVisualizer:
    def __init__(self, output_manager=None):
        self.output_manager = output_manager
    
    def plot_custom_analysis(self, data, filename="custom_plot"):
        fig, ax = plt.subplots(figsize=(10, 6))
        # 自定义绘图逻辑
        
        if self.output_manager:
            self.output_manager.save_figure(fig, filename)
        return fig
```

### 3. 添加新的分析器
```python
class CustomAnalyzer:
    def __init__(self, custom_param: float = 1.0):
        self.custom_param = custom_param
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def analyze_custom_metric(self, requests: List[RequestMetrics]) -> Dict[str, Any]:
        # 自定义分析逻辑
        return {'custom_metric': calculated_value}
```

## 最佳实践

### 1. 会话命名
使用描述性的会话名称：
```python
session_name = f"experiment_{experiment_id}_run_{run_number}"
```

### 2. 配置管理
为不同的实验创建配置模板：
```python
# configs/mesh_4x4_template.yaml
topology_type: mesh_4x4
node_count: 16
gap_threshold_ns: 200
window_size_ns: 1000
```

### 3. 批量分析
```python
experiments = [
    ('crossring', '8_ring', config_8_ring),
    ('mesh', '4x4_mesh', config_4x4),
    ('mesh', '8x8_mesh', config_8x8)
]

for model, topology, config in experiments:
    with SimulationContext(model, topology, config) as output_manager:
        # 执行分析
        pass
```

### 4. 结果比较
```python
# 比较不同配置的结果
results_comparison = {}

for session in output_manager.list_sessions():
    session_dir = output_manager.base_output_dir / session['session_id']
    summary_file = session_dir / 'data' / 'performance_summary.json'
    
    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)
        results_comparison[session['session_id']] = summary
```

## 示例代码

完整的分析示例请参考：
- `examples/noc_analysis_with_output_demo.py` - 完整演示
- `examples/test_output_management.py` - 输出管理测试
- `examples/noc_analysis_demo.py` - 基础分析演示

## 更新日志

### v1.0.0 (2023-12-01)
- 初始版本发布
- 基础分析功能
- 可视化支持
- 输出管理系统

### 未来计划
- 实时分析支持
- 更多拓扑类型支持
- 机器学习辅助热点预测
- Web界面支持