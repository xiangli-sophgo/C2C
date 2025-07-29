#!/usr/bin/env python3
"""
字体配置模块
用于处理跨平台（Mac、Windows、Linux）的中文字体兼容性问题
"""

import platform
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from typing import List, Dict, Optional
import warnings


class FontConfig:
    """字体配置管理器"""
    
    # 各平台的中文字体优先级列表
    CHINESE_FONTS = {
        'Darwin': ['PingFang SC', 'Heiti SC', 'Arial Unicode MS', 'STHeiti'],  # Mac
        'Windows': ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi'],
        'Linux': ['WenQuanYi Micro Hei', 'DejaVu Sans', 'Noto Sans CJK SC', 'Droid Sans Fallback']
    }
    
    # 英文字体优先级列表
    ENGLISH_FONTS = ['Times New Roman', 'Times', 'serif', 'DejaVu Serif']
    
    # 数学字体
    MATH_FONTS = ['DejaVu Sans', 'Computer Modern', 'STIX']
    
    def __init__(self):
        self.system = platform.system()
        self.available_fonts = self._get_available_fonts()
        self.chinese_font = self._select_chinese_font()
        self.english_font = self._select_english_font()
        self.math_font = self._select_math_font()
    
    def _get_available_fonts(self) -> List[str]:
        """获取系统中所有可用的字体"""
        return [f.name for f in fm.fontManager.ttflist]
    
    def _select_font_from_list(self, font_list: List[str]) -> Optional[str]:
        """从字体列表中选择第一个可用的字体"""
        for font in font_list:
            if font in self.available_fonts:
                return font
        return None
    
    def _select_chinese_font(self) -> str:
        """选择合适的中文字体"""
        chinese_fonts = self.CHINESE_FONTS.get(self.system, self.CHINESE_FONTS['Linux'])
        selected = self._select_font_from_list(chinese_fonts)
        
        if not selected:
            # 如果没有找到预设的字体，尝试查找任何包含中文支持的字体
            for font in self.available_fonts:
                if any(keyword in font.lower() for keyword in ['chinese', 'cjk', 'sc', 'tc', '黑', '宋', '楷']):
                    selected = font
                    break
        
        if not selected:
            warnings.warn(f"未找到合适的中文字体，使用系统默认字体。当前系统: {self.system}")
            selected = 'sans-serif'
        
        return selected
    
    def _select_english_font(self) -> str:
        """选择合适的英文字体"""
        selected = self._select_font_from_list(self.ENGLISH_FONTS)
        return selected or 'serif'
    
    def _select_math_font(self) -> str:
        """选择合适的数学字体"""
        selected = self._select_font_from_list(self.MATH_FONTS)
        return selected or 'DejaVu Sans'
    
    def get_font_config(self) -> Dict[str, str]:
        """获取字体配置字典"""
        return {
            'font.sans-serif': [self.chinese_font] + ['sans-serif'],
            'font.serif': [self.english_font] + ['serif'],
            'font.monospace': ['Consolas', 'Monaco', 'Courier New', 'monospace'],
            'axes.unicode_minus': 'False',  # 解决负号显示问题
        }
    
    def print_font_info(self):
        """打印当前字体配置信息"""
        print(f"操作系统: {self.system}")
        print(f"中文字体: {self.chinese_font}")
        print(f"英文字体: {self.english_font}")
        print(f"数学字体: {self.math_font}")
        print(f"可用字体总数: {len(self.available_fonts)}")


# 全局字体配置实例
_font_config = None


def get_font_config() -> FontConfig:
    """获取全局字体配置实例（单例模式）"""
    global _font_config
    if _font_config is None:
        _font_config = FontConfig()
    return _font_config


def configure_matplotlib_fonts(verbose: bool = False) -> None:
    """
    配置matplotlib的字体设置
    
    Args:
        verbose: 是否打印字体配置信息
    """
    config = get_font_config()
    
    if verbose:
        config.print_font_info()
    
    # 更新matplotlib配置
    font_settings = config.get_font_config()
    for key, value in font_settings.items():
        if key == 'axes.unicode_minus':
            plt.rcParams[key] = False
        else:
            plt.rcParams[key] = value
    
    # 清除字体缓存，确保新设置生效
    # 注意：在较新版本的matplotlib中，_rebuild方法已被移除
    # 只需要清除缓存即可，matplotlib会自动重建
    try:
        # 尝试使用旧版本的方法
        fm.fontManager._rebuild()
    except AttributeError:
        # 新版本matplotlib不需要手动重建，会自动处理
        pass


def get_chinese_font() -> str:
    """获取当前系统的中文字体名称"""
    return get_font_config().chinese_font


def get_english_font() -> str:
    """获取当前系统的英文字体名称"""
    return get_font_config().english_font


def get_system_info() -> Dict[str, str]:
    """获取系统和字体信息"""
    config = get_font_config()
    return {
        'system': config.system,
        'chinese_font': config.chinese_font,
        'english_font': config.english_font,
        'math_font': config.math_font,
    }


def test_font_rendering():
    """测试字体渲染效果"""
    configure_matplotlib_fonts(verbose=True)
    
    # 创建测试图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 测试文本
    test_texts = [
        "English Text: Hello World!",
        "中文文本：你好，世界！",
        "数学符号：α β γ δ ∑ ∏ ∫",
        "混合文本：CrossRing拓扑结构 - 性能分析",
        "负数测试：-123.456",
    ]
    
    # 绘制测试文本
    for i, text in enumerate(test_texts):
        ax.text(0.1, 0.9 - i * 0.15, text, fontsize=14, transform=ax.transAxes)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("字体渲染测试 - Font Rendering Test")
    ax.set_xlabel("X轴标签 - X Axis Label")
    ax.set_ylabel("Y轴标签 - Y Axis Label")
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 运行字体测试
    test_font_rendering()