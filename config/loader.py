import yaml
from typing import Dict, Any

class ConfigLoader:
    """配置加载器"""
    def __init__(self):
        pass

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """从YAML文件加载配置"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            print(f"Error: Config file not found at {config_path}")
            return {}
        except yaml.YAMLError as e:
            print(f"Error parsing YAML config file: {e}")
            return {}

    def save_config(self, config: Dict[str, Any], config_path: str):
        """将配置保存到YAML文件"""
        try:
            with open(config_path, 'w') as f:
                yaml.safe_dump(config, f, indent=2)
            print(f"Config saved to {config_path}")
        except IOError as e:
            print(f"Error saving YAML config file: {e}")
