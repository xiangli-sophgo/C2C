"""
输出管理器 - 管理仿真结果的输出文件夹结构和文件保存
为每次仿真创建独立的输出目录，保存配置、日志、图片和数据文件
"""

import os
import json
import time
import shutil
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import yaml


class OutputManager:
    """输出管理器"""
    
    def __init__(self, base_output_dir: str = "output"):
        self.base_output_dir = Path(base_output_dir)
        self.current_session_dir: Optional[Path] = None
        self.session_id: Optional[str] = None
        self.session_metadata: Dict[str, Any] = {}
        
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
        
        Args:
            model_name: 模型名称 (如 'crossring', 'mesh')
            topology_type: 拓扑类型 (如 '4x4_mesh', '8_ring')
            config: 配置字典
            session_name: 自定义会话名称
            
        Returns:
            会话ID
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
        
        # 创建子目录结构
        subdirs = [
            "config",      # 配置文件
            "logs",        # 日志文件
            "figures",     # 图片文件
            "data",        # 数据文件
            "reports",     # 报告文件
            "analysis"     # 分析结果
        ]
        
        for subdir in subdirs:
            (self.current_session_dir / subdir).mkdir(exist_ok=True)
        
        # 保存会话元数据
        self.session_metadata = {
            "session_id": self.session_id,
            "model_name": model_name,
            "topology_type": topology_type,
            "created_time": datetime.now().isoformat(),
            "config": config,
            "files": {
                "config": [],
                "logs": [],
                "figures": [],
                "data": [],
                "reports": [],
                "analysis": []
            }
        }
        
        # 保存配置文件
        self.save_config(config, "main_config")
        
        # 创建README文件
        self._create_session_readme()
        
        # 设置会话级日志
        self._setup_session_logging()
        
        self.logger.info(f"创建新会话: {self.session_id}")
        self.logger.info(f"会话目录: {self.current_session_dir}")
        
        return self.session_id
    
    def save_config(self, config: Dict[str, Any], filename: str) -> str:
        """保存配置文件"""
        if not self.current_session_dir:
            raise RuntimeError("请先创建会话")
        
        config_path = self.current_session_dir / "config" / f"{filename}.yaml"
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        self.session_metadata["files"]["config"].append(str(config_path.name))
        self._update_metadata()
        
        self.logger.info(f"保存配置文件: {config_path}")
        return str(config_path)
    
    def save_figure(self, figure_data: Any, filename: str, format: str = 'png') -> str:
        """保存图片文件"""
        if not self.current_session_dir:
            raise RuntimeError("请先创建会话")
        
        figures_dir = self.current_session_dir / "figures"
        figure_path = figures_dir / f"{filename}.{format}"
        
        # 保存matplotlib图片
        if hasattr(figure_data, 'savefig'):
            figure_data.savefig(figure_path, dpi=300, bbox_inches='tight', 
                              facecolor='white', edgecolor='none')
        else:
            # 如果是其他类型的图片数据，使用PIL保存
            try:
                from PIL import Image
                if isinstance(figure_data, Image.Image):
                    figure_data.save(figure_path)
                else:
                    raise ValueError("不支持的图片格式")
            except ImportError:
                raise ImportError("需要安装PIL/Pillow来保存图片: pip install Pillow")
        
        self.session_metadata["files"]["figures"].append(str(figure_path.name))
        self._update_metadata()
        
        self.logger.info(f"保存图片文件: {figure_path}")
        return str(figure_path)
    
    def save_data(self, data: Any, filename: str, format: str = 'json') -> str:
        """保存数据文件"""
        if not self.current_session_dir:
            raise RuntimeError("请先创建会话")
        
        data_dir = self.current_session_dir / "data"
        data_path = data_dir / f"{filename}.{format}"
        
        if format == 'json':
            with open(data_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        elif format == 'csv':
            if hasattr(data, 'to_csv'):
                data.to_csv(data_path, index=False)
            else:
                raise ValueError("CSV格式需要pandas DataFrame")
        elif format == 'xlsx':
            if hasattr(data, 'to_excel'):
                data.to_excel(data_path, index=False)
            else:
                raise ValueError("Excel格式需要pandas DataFrame")
        elif format == 'yaml':
            with open(data_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        else:
            raise ValueError(f"不支持的数据格式: {format}")
        
        self.session_metadata["files"]["data"].append(str(data_path.name))
        self._update_metadata()
        
        self.logger.info(f"保存数据文件: {data_path}")
        return str(data_path)
    
    def save_log(self, log_content: str, filename: str) -> str:
        """保存日志文件"""
        if not self.current_session_dir:
            raise RuntimeError("请先创建会话")
        
        logs_dir = self.current_session_dir / "logs"
        log_path = logs_dir / f"{filename}.log"
        
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(log_content)
        
        self.session_metadata["files"]["logs"].append(str(log_path.name))
        self._update_metadata()
        
        self.logger.info(f"保存日志文件: {log_path}")
        return str(log_path)
    
    def save_report(self, report_content: str, filename: str, format: str = 'md') -> str:
        """保存报告文件"""
        if not self.current_session_dir:
            raise RuntimeError("请先创建会话")
        
        reports_dir = self.current_session_dir / "reports"
        report_path = reports_dir / f"{filename}.{format}"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.session_metadata["files"]["reports"].append(str(report_path.name))
        self._update_metadata()
        
        self.logger.info(f"保存报告文件: {report_path}")
        return str(report_path)
    
    def save_analysis_result(self, analysis_data: Any, filename: str, format: str = 'json') -> str:
        """保存分析结果"""
        if not self.current_session_dir:
            raise RuntimeError("请先创建会话")
        
        analysis_dir = self.current_session_dir / "analysis"
        analysis_path = analysis_dir / f"{filename}.{format}"
        
        if format == 'json':
            with open(analysis_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, indent=2, ensure_ascii=False, default=str)
        elif format == 'pickle':
            import pickle
            with open(analysis_path, 'wb') as f:
                pickle.dump(analysis_data, f)
        else:
            raise ValueError(f"不支持的分析结果格式: {format}")
        
        self.session_metadata["files"]["analysis"].append(str(analysis_path.name))
        self._update_metadata()
        
        self.logger.info(f"保存分析结果: {analysis_path}")
        return str(analysis_path)
    
    def get_session_dir(self) -> Optional[Path]:
        """获取当前会话目录"""
        return self.current_session_dir
    
    def get_subdirectory(self, subdir: str) -> Optional[Path]:
        """获取指定子目录路径"""
        if not self.current_session_dir:
            return None
        return self.current_session_dir / subdir
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """列出所有历史会话"""
        sessions = []
        
        for session_dir in self.base_output_dir.iterdir():
            if session_dir.is_dir():
                metadata_file = session_dir / "session_metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        sessions.append(metadata)
                    except Exception as e:
                        self.logger.warning(f"无法读取会话元数据 {session_dir}: {e}")
        
        # 按创建时间排序
        sessions.sort(key=lambda x: x.get('created_time', ''), reverse=True)
        return sessions
    
    def load_session(self, session_id: str) -> bool:
        """加载已存在的会话"""
        session_dir = self.base_output_dir / session_id
        metadata_file = session_dir / "session_metadata.json"
        
        if not session_dir.exists() or not metadata_file.exists():
            self.logger.error(f"会话不存在: {session_id}")
            return False
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                self.session_metadata = json.load(f)
            
            self.session_id = session_id
            self.current_session_dir = session_dir
            
            # 重新设置日志
            self._setup_session_logging()
            
            self.logger.info(f"加载会话: {session_id}")
            return True
        except Exception as e:
            self.logger.error(f"加载会话失败 {session_id}: {e}")
            return False
    
    def cleanup_old_sessions(self, keep_count: int = 10):
        """清理旧的会话，只保留最近的几个"""
        sessions = self.list_sessions()
        
        if len(sessions) <= keep_count:
            return
        
        sessions_to_delete = sessions[keep_count:]
        
        for session in sessions_to_delete:
            session_id = session.get('session_id')
            if session_id:
                session_dir = self.base_output_dir / session_id
                if session_dir.exists():
                    try:
                        shutil.rmtree(session_dir)
                        self.logger.info(f"删除旧会话: {session_id}")
                    except Exception as e:
                        self.logger.error(f"删除会话失败 {session_id}: {e}")
    
    def generate_session_summary(self) -> str:
        """生成会话摘要报告"""
        if not self.session_metadata:
            return "没有活动的会话"
        
        summary = f"""
# 仿真会话摘要报告

## 基本信息
- **会话ID**: {self.session_metadata.get('session_id', 'N/A')}
- **模型名称**: {self.session_metadata.get('model_name', 'N/A')}
- **拓扑类型**: {self.session_metadata.get('topology_type', 'N/A')}
- **创建时间**: {self.session_metadata.get('created_time', 'N/A')}

## 配置参数
"""
        
        config = self.session_metadata.get('config', {})
        for key, value in config.items():
            summary += f"- **{key}**: {value}\n"
        
        summary += "\n## 生成的文件\n"
        
        files = self.session_metadata.get('files', {})
        for category, file_list in files.items():
            if file_list:
                summary += f"\n### {category.title()}\n"
                for filename in file_list:
                    summary += f"- {filename}\n"
        
        summary += f"\n## 会话目录\n{self.current_session_dir}\n"
        
        return summary
    
    def _create_session_readme(self):
        """创建会话说明文件"""
        readme_content = f"""# 仿真会话: {self.session_id}

## 会话信息
- 模型: {self.session_metadata.get('model_name')}
- 拓扑: {self.session_metadata.get('topology_type')}
- 创建时间: {self.session_metadata.get('created_time')}

## 目录结构
- `config/`: 配置文件
- `logs/`: 日志文件
- `figures/`: 图片和图表
- `data/`: 数据文件
- `reports/`: 分析报告
- `analysis/`: 分析结果

## 文件说明
此目录包含了完整的仿真运行记录，包括配置参数、运行日志、性能分析图表和原始数据。

## 如何使用
1. 查看 `config/main_config.yaml` 了解仿真配置
2. 查看 `reports/` 目录中的分析报告
3. 查看 `figures/` 目录中的性能图表
4. 原始数据在 `data/` 目录中

## 重现实验
使用 `config/main_config.yaml` 中的配置可以重现此次实验。
"""
        
        readme_path = self.current_session_dir / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
    
    def _setup_session_logging(self):
        """设置会话级日志"""
        if not self.current_session_dir:
            return
        
        log_dir = self.current_session_dir / "logs"
        log_file = log_dir / "session.log"
        
        # 创建文件处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 创建格式器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # 添加到logger
        self.logger.addHandler(file_handler)
        
        # 也为根logger添加文件处理器
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
    
    def _update_metadata(self):
        """更新会话元数据文件"""
        if not self.current_session_dir:
            return
        
        metadata_file = self.current_session_dir / "session_metadata.json"
        self.session_metadata["updated_time"] = datetime.now().isoformat()
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.session_metadata, f, indent=2, ensure_ascii=False, default=str)


class SimulationContext:
    """仿真上下文管理器，用于自动管理会话生命周期"""
    
    def __init__(self, 
                 model_name: str,
                 topology_type: str,
                 config: Dict[str, Any],
                 session_name: Optional[str] = None,
                 base_output_dir: str = "output"):
        self.output_manager = OutputManager(base_output_dir)
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
        if exc_type is not None:
            # 如果有异常，记录错误信息
            error_info = {
                "error_type": str(exc_type.__name__),
                "error_message": str(exc_val),
                "error_traceback": str(exc_tb)
            }
            self.output_manager.save_data(error_info, "error_info", "json")
        
        # 生成最终摘要报告
        summary = self.output_manager.generate_session_summary()
        self.output_manager.save_report(summary, "session_summary", "md")
        
        print(f"仿真会话完成: {self.session_id}")
        print(f"结果保存在: {self.output_manager.get_session_dir()}")


def get_output_manager() -> OutputManager:
    """获取全局输出管理器实例"""
    if not hasattr(get_output_manager, '_instance'):
        get_output_manager._instance = OutputManager()
    return get_output_manager._instance