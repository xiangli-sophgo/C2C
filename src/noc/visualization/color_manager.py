#!/usr/bin/env python3
"""
颜色管理器

统一管理可视化系统中的颜色分配、高亮逻辑和调色板。
解决现有代码中颜色相关逻辑重复的问题。
"""

import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Any, Union


class ColorManager:
    """统一管理颜色分配和高亮逻辑"""

    def __init__(self, palette: Optional[List[str]] = None):
        """
        初始化颜色管理器
        
        Args:
            palette: 自定义调色板，为None时使用matplotlib默认调色板
        """
        # 默认调色板
        self.palette = palette or plt.rcParams["axes.prop_cycle"].by_key()["color"]
        
        # 预定义颜色常量
        self.HIGHLIGHT_COLOR = "red"
        self.DIMMED_COLOR = "lightgrey"
        self.DEFAULT_COLOR = "lightblue"
        self.TAGS_MODE_COLOR = "lightgray"
        
        # 包ID到颜色的缓存，避免重复计算
        self._pid_color_cache: Dict[Any, str] = {}
        
        # 当前模式状态
        self._current_theme = "default"

    def get_packet_color(self, packet_id: Any, highlight_mode: bool = False, 
                        target_pid: Any = None, show_tags_mode: bool = False) -> str:
        """
        获取包的显示颜色
        
        Args:
            packet_id: 包ID，可以是任意类型
            highlight_mode: 是否为高亮模式
            target_pid: 目标包ID（高亮模式时使用）
            show_tags_mode: 是否为标签显示模式
            
        Returns:
            颜色字符串（matplotlib颜色格式）
        """
        # 标签模式：统一使用浅色
        if show_tags_mode:
            return self.TAGS_MODE_COLOR
        
        # 高亮模式：目标包高亮，其他包变灰
        if highlight_mode and target_pid is not None:
            return self._get_highlight_color(packet_id, target_pid)
        
        # 正常模式：根据包ID分配颜色
        return self._get_normal_color(packet_id)

    def _get_highlight_color(self, packet_id: Any, target_pid: Any) -> str:
        """高亮模式下的颜色选择"""
        return self.HIGHLIGHT_COLOR if str(packet_id) == str(target_pid) else self.DIMMED_COLOR

    def _get_normal_color(self, packet_id: Any) -> str:
        """正常模式下根据包ID分配颜色"""
        if packet_id is None:
            return self.DEFAULT_COLOR
        
        # 使用缓存避免重复计算
        if packet_id not in self._pid_color_cache:
            try:
                color_index = int(packet_id) % len(self.palette)
                self._pid_color_cache[packet_id] = self.palette[color_index]
            except (ValueError, TypeError):
                # 如果packet_id无法转换为整数，使用默认颜色
                self._pid_color_cache[packet_id] = self.DEFAULT_COLOR
        
        return self._pid_color_cache[packet_id]

    def set_theme(self, theme_name: str):
        """
        设置颜色主题
        
        Args:
            theme_name: 主题名称 ("default", "dark", "high_contrast")
        """
        self._current_theme = theme_name
        
        if theme_name == "dark":
            self.palette = [
                "#ff6b6b", "#4ecdc4", "#45b7d1", "#96ceb4", "#feca57",
                "#ff9ff3", "#54a0ff", "#5f27cd", "#00d2d3", "#ff9f43"
            ]
            self.HIGHLIGHT_COLOR = "#ff3838"
            self.DIMMED_COLOR = "#404040"
            self.DEFAULT_COLOR = "#6c5ce7"
            self.TAGS_MODE_COLOR = "#636e72"
        elif theme_name == "high_contrast":
            self.palette = [
                "#000000", "#ffffff", "#ff0000", "#00ff00", "#0000ff",
                "#ffff00", "#ff00ff", "#00ffff", "#800000", "#008000"
            ]
            self.HIGHLIGHT_COLOR = "#ff0000"
            self.DIMMED_COLOR = "#808080"
            self.DEFAULT_COLOR = "#000000"
            self.TAGS_MODE_COLOR = "#c0c0c0"
        else:  # default theme
            self.palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            self.HIGHLIGHT_COLOR = "red"
            self.DIMMED_COLOR = "lightgrey"
            self.DEFAULT_COLOR = "lightblue"
            self.TAGS_MODE_COLOR = "lightgray"
        
        # 清除缓存，重新计算颜色
        self.clear_cache()

    def set_custom_color(self, packet_id: Any, color: str):
        """
        为特定包ID设置自定义颜色
        
        Args:
            packet_id: 包ID
            color: 自定义颜色
        """
        self._pid_color_cache[packet_id] = color

    def get_used_colors(self) -> List[str]:
        """获取当前已使用的颜色列表"""
        return list(set(self._pid_color_cache.values()))

    def get_color_mapping(self) -> Dict[Any, str]:
        """获取当前的包ID到颜色映射"""
        return self._pid_color_cache.copy()

    def clear_cache(self):
        """清除颜色缓存（用于重置或主题切换）"""
        self._pid_color_cache.clear()

    def get_palette_size(self) -> int:
        """获取当前调色板大小"""
        return len(self.palette)

    def get_theme_name(self) -> str:
        """获取当前主题名称"""
        return self._current_theme

    def __repr__(self) -> str:
        return (f"ColorManager(theme='{self._current_theme}', "
                f"palette_size={len(self.palette)}, "
                f"cached_colors={len(self._pid_color_cache)})")