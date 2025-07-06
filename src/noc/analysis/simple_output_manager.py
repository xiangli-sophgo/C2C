"""
简化的输出管理器 - 只保留核心功能
减少生成的文件数量，聚焦于最重要的结果
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


class SimpleOutputManager:
    """简化的输出管理器"""
    
    def __init__(self, base_output_dir: str = "output"):
        self.base_output_dir = Path(base_output_dir)
        self.current_session_dir: Optional[Path] = None
        self.session_id: Optional[str] = None
        
        # 确保基础输出目录存在
        self.base_output_dir.mkdir(exist_ok=True)
        
        # 设置日志
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def create_session(self, 
                      model_name: str,
                      topology_type: str, 
                      config: Dict[str, Any],
                      session_name: Optional[str] = None) -> str:
        """
        创建新的仿真会话目录
        只创建必要的文件夹和配置
        """
        # 生成会话ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if session_name:
            self.session_id = f"{model_name}_{topology_type}_{session_name}_{timestamp}"
        else:
            self.session_id = f"{model_name}_{topology_type}_{timestamp}"
        
        # 创建会话目录
        self.current_session_dir = self.base_output_dir / self.session_id
        self.current_session_dir.mkdir(exist_ok=True)
        
        # 只创建核心子目录
        (self.current_session_dir / "results").mkdir(exist_ok=True)    # 图片和数据
        
        # 保存基本配置
        config_file = self.current_session_dir / "config.json"
        session_info = {
            "session_id": self.session_id,
            "model_name": model_name,
            "topology_type": topology_type,
            "created_time": datetime.now().isoformat(),
            "config": config
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(session_info, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"创建会话: {self.session_id}")
        return self.session_id
    
    def save_figure(self, figure_data: Any, filename: str) -> str:
        """保存图片文件到results目录"""
        if not self.current_session_dir:
            raise RuntimeError("请先创建会话")
        
        figure_path = self.current_session_dir / "results" / f"{filename}.png"
        
        if hasattr(figure_data, 'savefig'):
            figure_data.savefig(figure_path, dpi=300, bbox_inches='tight', 
                              facecolor='white', edgecolor='none')
        
        self.logger.info(f"保存图片: {figure_path.name}")
        return str(figure_path)
    
    def save_data(self, data: Any, filename: str, format: str = 'json') -> str:
        """保存数据文件到results目录"""
        if not self.current_session_dir:
            raise RuntimeError("请先创建会话")
        
        data_path = self.current_session_dir / "results" / f"{filename}.{format}"
        
        if format == 'json':
            with open(data_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        elif format == 'csv':
            if hasattr(data, 'to_csv'):
                data.to_csv(data_path, index=False)
        elif format == 'xlsx':
            if hasattr(data, 'to_excel'):
                data.to_excel(data_path, index=False)
        
        self.logger.info(f"保存数据: {data_path.name}")
        return str(data_path)
    
    def save_summary_report(self, summary: Dict[str, Any]) -> str:
        """保存性能摘要报告"""
        if not self.current_session_dir:
            raise RuntimeError("请先创建会话")
        
        # 创建简化的摘要报告
        report_content = f"""# NoC 性能分析报告

## 基本信息
- 会话ID: {self.session_id}
- 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 性能指标
- 总带宽: {summary.get('bandwidth_gbps', 0):.3f} GB/s
- 平均延迟: {summary.get('latency_ns', 0):.1f} ns
- 总吞吐量: {summary.get('throughput_rps', 0):.0f} req/s
- 网络利用率: {summary.get('network_utilization', 0):.1%}
- 平均跳数: {summary.get('avg_hop_count', 0):.1f}

## 文件说明
- config.json: 仿真配置
- results/: 分析结果和图表
  - performance_summary.json: 详细性能数据
  - dashboard.png: 性能仪表板
  - data_export.xlsx: 完整数据导出
"""
        
        report_path = self.current_session_dir / "README.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return str(report_path)
    
    def get_session_dir(self) -> Optional[Path]:
        """获取当前会话目录"""
        return self.current_session_dir
    
    def get_results_dir(self) -> Optional[Path]:
        """获取结果目录"""
        if not self.current_session_dir:
            return None
        return self.current_session_dir / "results"


class SimpleSimulationContext:
    """简化的仿真上下文管理器"""
    
    def __init__(self, 
                 model_name: str,
                 topology_type: str,
                 config: Dict[str, Any],
                 session_name: Optional[str] = None,
                 base_output_dir: str = "output"):
        self.output_manager = SimpleOutputManager(base_output_dir)
        self.model_name = model_name
        self.topology_type = topology_type
        self.config = config
        self.session_name = session_name
        self.session_id: Optional[str] = None
    
    def __enter__(self):
        """进入上下文"""
        self.session_id = self.output_manager.create_session(
            self.model_name, 
            self.topology_type, 
            self.config,
            self.session_name
        )
        return self.output_manager
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文"""
        print(f"仿真完成: {self.session_id}")
        print(f"结果保存在: {self.output_manager.get_session_dir()}")


def get_simple_output_manager() -> SimpleOutputManager:
    """获取全局简化输出管理器实例"""
    if not hasattr(get_simple_output_manager, '_instance'):
        get_simple_output_manager._instance = SimpleOutputManager()
    return get_simple_output_manager._instance