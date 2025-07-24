"""
FIFO统计分析器

基于ip_interface.py中的FIFOStatistics类，提供统计收集、分析和导出功能。
"""

import csv
import os
import time
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path

from src.noc.base.ip_interface import PipelinedFIFO


class FIFOStatsCollector:
    """FIFO统计信息收集器，使用PipelinedFIFO内置的FIFOStatistics"""
    
    def __init__(self):
        self.fifo_registry = {}  # {fifo_identifier: {"fifo": obj, "node_id": str, "name": str}}
        
    def register_fifo(self, fifo: PipelinedFIFO, node_id: str = "", simplified_name: str = ""):
        """注册需要统计的FIFO"""
        fifo_identifier = f"{node_id}_{simplified_name}"
        self.fifo_registry[fifo_identifier] = {
            "fifo": fifo,
            "node_id": node_id,
            "simplified_name": simplified_name
        }
        
    def collect_all_stats(self) -> List[Dict[str, Any]]:
        """收集所有注册FIFO的统计信息"""
        collected_stats = []
        
        for identifier, info in self.fifo_registry.items():
            fifo = info["fifo"]
            stats = fifo.get_statistics()  # 使用内置统计
            
            # 添加标识信息
            stats["节点ID"] = info["node_id"]
            stats["FIFO名称"] = info["simplified_name"]
            
            collected_stats.append(stats)
            
        return collected_stats
            
    def export_to_csv(self, filename: str = None, output_dir: str = "results") -> str:
        """导出统计数据到CSV文件"""
        # 确保输出目录存在
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 生成文件名（统一使用Unix时间戳格式）
        if filename is None:
            timestamp = int(time.time())
            filename = f"fifo_statistics_{timestamp}"
            
        filepath = os.path.join(output_dir, f"{filename}.csv")
        
        collected_stats = self.collect_all_stats()
        if not collected_stats:
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
            
            for stats in collected_stats:
                # 确保所有必需的字段都存在
                row = {}
                for header in headers:
                    row[header] = stats.get(header, 0)
                writer.writerow(row)
                
        print(f"📊 FIFO统计数据已导出到: {filepath}")
        print(f"📈 共导出 {len(collected_stats)} 个FIFO的统计信息")
        
        return filepath
        
    def get_summary_report(self) -> str:
        """生成统计摘要报告"""
        collected_stats = self.collect_all_stats()
        if not collected_stats:
            return "没有收集到FIFO统计数据"
            
        report = ["=" * 60]
        report.append("FIFO 统计摘要报告")
        report.append("=" * 60)
        
        # 基本统计
        total_fifos = len(collected_stats)
        report.append(f"总FIFO数量: {total_fifos}")
        
        # 按节点分组统计
        nodes = set(stats["节点ID"] for stats in collected_stats)
        report.append(f"涉及节点数: {len(nodes)}")
        
        # 利用率统计
        utilizations = [stats["利用率百分比"] for stats in collected_stats if stats["利用率百分比"] > 0]
        if utilizations:
            avg_util = sum(utilizations) / len(utilizations)
            max_util = max(utilizations)
            min_util = min(utilizations)
            report.append(f"平均利用率: {avg_util:.2f}%")
            report.append(f"最高利用率: {max_util:.2f}%")
            report.append(f"最低利用率: {min_util:.2f}%")
            
        # 效率统计
        write_effs = [stats["写入效率"] for stats in collected_stats if stats["写入效率"] > 0]
        read_effs = [stats["读取效率"] for stats in collected_stats if stats["读取效率"] > 0]
        if write_effs:
            avg_write_eff = sum(write_effs) / len(write_effs)
            report.append(f"平均写入效率: {avg_write_eff:.2f}%")
        if read_effs:
            avg_read_eff = sum(read_effs) / len(read_effs)
            report.append(f"平均读取效率: {avg_read_eff:.2f}%")
            
        # 阻塞统计
        total_write_stalls = sum(stats["写入阻塞次数"] for stats in collected_stats)
        total_read_stalls = sum(stats["读取阻塞次数"] for stats in collected_stats)
        report.append(f"总写入阻塞次数: {total_write_stalls}")
        report.append(f"总读取阻塞次数: {total_read_stalls}")
        
        report.append("=" * 60)
        return "\n".join(report)
        
    def get_fifo_details(self, node_id: str = None, fifo_name_filter: str = None) -> List[Dict]:
        """获取特定条件下的FIFO详细信息"""
        collected_stats = self.collect_all_stats()
        
        filtered_stats = []
        for stats in collected_stats:
            if node_id is not None and stats["节点ID"] != node_id:
                continue
            if fifo_name_filter is not None and fifo_name_filter not in stats["FIFO名称"]:
                continue
            filtered_stats.append(stats)
            
        return filtered_stats