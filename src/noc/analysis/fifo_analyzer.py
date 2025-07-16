"""
FIFO统计分析器

用于收集、分析和导出PipelinedFIFO的统计信息，支持CSV格式导出。
"""

import csv
import os
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path

from src.noc.base.ip_interface import PipelinedFIFO


class FIFOStatsCollector:
    """FIFO统计信息收集器"""
    
    def __init__(self):
        self.fifo_stats = {}  # {fifo_identifier: fifo_object}
        self.collected_stats = []  # 最终统计数据列表
        
    def register_fifo(self, fifo: PipelinedFIFO, node_id: str = "", simplified_name: str = ""):
        """
        注册需要统计的FIFO
        
        Args:
            fifo: PipelinedFIFO实例
            node_id: 节点ID（用于分组）
            simplified_name: 简化的FIFO标识符（如：req_RB_IN_TR）
        """
        fifo_identifier = f"{node_id}_{simplified_name}"
        self.fifo_stats[fifo_identifier] = {
            "fifo": fifo,
            "node_id": node_id,
            "simplified_name": simplified_name
        }
        
    def collect_all_stats(self):
        """收集所有注册FIFO的统计信息"""
        self.collected_stats = []
        
        for identifier, info in self.fifo_stats.items():
            fifo = info["fifo"]
            node_id = info["node_id"]
            simplified_name = info["simplified_name"]
            
            # 获取FIFO统计数据
            stats = fifo.get_statistics()
            
            # 添加标识信息，使用简化的名称作为FIFO名称
            stats["节点ID"] = node_id
            stats["FIFO名称"] = simplified_name  # 覆盖原来的技术名称
            
            self.collected_stats.append(stats)
            
    def export_to_csv(self, filename: str = None, output_dir: str = "results") -> str:
        """
        导出统计数据到CSV文件
        
        Args:
            filename: 文件名（不包含扩展名），如果为None则自动生成
            output_dir: 输出目录
            
        Returns:
            导出的文件路径
        """
        # 确保输出目录存在
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 生成文件名
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fifo_statistics_{timestamp}"
            
        filepath = os.path.join(output_dir, f"{filename}.csv")
        
        if not self.collected_stats:
            self.collect_all_stats()
            
        if not self.collected_stats:
            print("⚠️ 没有收集到FIFO统计数据")
            return filepath
            
        # 定义CSV列标题（中文）- 移除冗余列
        headers = [
            "节点ID", "FIFO名称", "最大容量",
            "当前深度", "峰值深度", "平均深度", "利用率百分比",
            "空队列周期数", "满队列周期数",
            "总写入尝试", "成功写入次数", "总读取尝试", "成功读取次数",
            "写入效率", "读取效率", "写入阻塞次数", "读取阻塞次数",
            "溢出尝试次数", "下溢尝试次数",
            "平均停留时间", "最小停留时间", "最大停留时间",
            "高优先级写入", "总仿真周期", "活跃周期百分比"
        ]
        
        # 写入CSV文件
        with open(filepath, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            
            for stats in self.collected_stats:
                # 确保所有必需的字段都存在
                row = {}
                for header in headers:
                    row[header] = stats.get(header, 0)
                writer.writerow(row)
                
        print(f"📊 FIFO统计数据已导出到: {filepath}")
        print(f"📈 共导出 {len(self.collected_stats)} 个FIFO的统计信息")
        
        return filepath
        
    def get_summary_report(self) -> str:
        """生成统计摘要报告"""
        if not self.collected_stats:
            self.collect_all_stats()
            
        if not self.collected_stats:
            return "没有收集到FIFO统计数据"
            
        report = ["=" * 60]
        report.append("FIFO 统计摘要报告")
        report.append("=" * 60)
        
        # 基本统计
        total_fifos = len(self.collected_stats)
        report.append(f"总FIFO数量: {total_fifos}")
        
        # 按节点分组统计
        nodes = set(stats["节点ID"] for stats in self.collected_stats)
        report.append(f"涉及节点数: {len(nodes)}")
        
        # 按FIFO名称前缀分组统计（提取功能类型）
        fifo_types = set()
        for stats in self.collected_stats:
            fifo_name = stats["FIFO名称"]
            if '_' in fifo_name:
                # 提取功能类型部分（如req_RB_IN_TR中的RB_IN）
                parts = fifo_name.split('_')
                if len(parts) >= 3:
                    fifo_type = '_'.join(parts[1:-1])  # 去掉通道和方向，保留功能类型
                    fifo_types.add(fifo_type)
        report.append(f"FIFO功能类型: {', '.join(sorted(fifo_types))}")
        
        # 利用率统计
        utilizations = [stats["利用率百分比"] for stats in self.collected_stats if stats["利用率百分比"] > 0]
        if utilizations:
            avg_util = sum(utilizations) / len(utilizations)
            max_util = max(utilizations)
            min_util = min(utilizations)
            report.append(f"平均利用率: {avg_util:.2f}%")
            report.append(f"最高利用率: {max_util:.2f}%")
            report.append(f"最低利用率: {min_util:.2f}%")
            
        # 写入效率统计
        write_effs = [stats["写入效率"] for stats in self.collected_stats if stats["写入效率"] > 0]
        if write_effs:
            avg_write_eff = sum(write_effs) / len(write_effs)
            report.append(f"平均写入效率: {avg_write_eff:.2f}%")
            
        # 读取效率统计
        read_effs = [stats["读取效率"] for stats in self.collected_stats if stats["读取效率"] > 0]
        if read_effs:
            avg_read_eff = sum(read_effs) / len(read_effs)
            report.append(f"平均读取效率: {avg_read_eff:.2f}%")
            
        # 阻塞统计
        total_write_stalls = sum(stats["写入阻塞次数"] for stats in self.collected_stats)
        total_read_stalls = sum(stats["读取阻塞次数"] for stats in self.collected_stats)
        report.append(f"总写入阻塞次数: {total_write_stalls}")
        report.append(f"总读取阻塞次数: {total_read_stalls}")
        
        # 停留时间统计
        residence_times = [stats["平均停留时间"] for stats in self.collected_stats if stats["平均停留时间"] > 0]
        if residence_times:
            avg_residence = sum(residence_times) / len(residence_times)
            max_residence = max(residence_times)
            min_residence = min(residence_times)
            report.append(f"平均flit停留时间: {avg_residence:.2f} 周期")
            report.append(f"最长flit停留时间: {max_residence:.2f} 周期")
            report.append(f"最短flit停留时间: {min_residence:.2f} 周期")
            
        report.append("=" * 60)
        
        return "\n".join(report)
        
    def get_fifo_details(self, node_id: str = None, fifo_name_filter: str = None) -> List[Dict]:
        """
        获取特定条件下的FIFO详细信息
        
        Args:
            node_id: 过滤特定节点，None表示所有节点
            fifo_name_filter: 过滤FIFO名称（支持部分匹配），None表示所有类型
            
        Returns:
            符合条件的FIFO统计信息列表
        """
        if not self.collected_stats:
            self.collect_all_stats()
            
        filtered_stats = []
        for stats in self.collected_stats:
            if node_id is not None and stats["节点ID"] != node_id:
                continue
            if fifo_name_filter is not None and fifo_name_filter not in stats["FIFO名称"]:
                continue
            filtered_stats.append(stats)
            
        return filtered_stats


class FIFOVisualizer:
    """FIFO统计数据可视化器"""
    
    def __init__(self, stats_collector: FIFOStatsCollector):
        self.collector = stats_collector
        
    def plot_utilization_comparison(self, save_path: str = None):
        """绘制FIFO利用率对比图"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            
            # 设置中文字体
            matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
            matplotlib.rcParams['axes.unicode_minus'] = False
            
            if not self.collector.collected_stats:
                self.collector.collect_all_stats()
                
            # 提取数据
            labels = [f"N{s['节点ID']}_{s['FIFO名称']}" for s in self.collector.collected_stats]
            utilizations = [s["利用率百分比"] for s in self.collector.collected_stats]
            
            # 创建图表
            plt.figure(figsize=(12, 6))
            bars = plt.bar(range(len(labels)), utilizations)
            plt.xlabel('FIFO标识')
            plt.ylabel('利用率 (%)')
            plt.title('FIFO利用率对比')
            plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
            
            # 添加数值标签
            for i, (bar, util) in enumerate(zip(bars, utilizations)):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{util:.1f}%', ha='center', va='bottom')
                        
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"📊 利用率对比图已保存到: {save_path}")
            else:
                plt.show()
                
        except ImportError:
            print("⚠️ 需要安装matplotlib库才能使用可视化功能: pip install matplotlib")
            
    def plot_throughput_analysis(self, save_path: str = None):
        """绘制吞吐量分析图"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            
            # 设置中文字体
            matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
            matplotlib.rcParams['axes.unicode_minus'] = False
            
            if not self.collector.collected_stats:
                self.collector.collect_all_stats()
                
            # 提取数据
            labels = [f"N{s['节点ID']}_{s['FIFO名称']}" for s in self.collector.collected_stats]
            write_eff = [s["写入效率"] for s in self.collector.collected_stats]
            read_eff = [s["读取效率"] for s in self.collector.collected_stats]
            
            # 创建图表
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # 写入效率
            ax1.bar(range(len(labels)), write_eff, alpha=0.7, color='blue')
            ax1.set_xlabel('FIFO标识')
            ax1.set_ylabel('写入效率 (%)')
            ax1.set_title('FIFO写入效率')
            ax1.set_xticks(range(len(labels)))
            ax1.set_xticklabels(labels, rotation=45, ha='right')
            
            # 读取效率
            ax2.bar(range(len(labels)), read_eff, alpha=0.7, color='green')
            ax2.set_xlabel('FIFO标识')
            ax2.set_ylabel('读取效率 (%)')
            ax2.set_title('FIFO读取效率')
            ax2.set_xticks(range(len(labels)))
            ax2.set_xticklabels(labels, rotation=45, ha='right')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"📊 吞吐量分析图已保存到: {save_path}")
            else:
                plt.show()
                
        except ImportError:
            print("⚠️ 需要安装matplotlib库才能使用可视化功能: pip install matplotlib")