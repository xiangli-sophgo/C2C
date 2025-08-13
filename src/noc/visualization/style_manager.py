#!/usr/bin/env python3
"""
可视化样式管理器

统一管理flit样式计算、E-Tag样式映射、透明度计算等。
解决现有代码中样式计算逻辑重复的问题。
"""

import matplotlib.colors as mcolors
from typing import Dict, Any, Tuple, Union, Optional
from .color_manager import ColorManager


class VisualizationStyleManager:
    """统一管理可视化样式计算"""

    def __init__(self, color_manager: Optional[ColorManager] = None):
        """
        初始化样式管理器
        
        Args:
            color_manager: 颜色管理器实例，为None时创建默认实例
        """
        self.color_manager = color_manager or ColorManager()
        
        # E-Tag样式映射 - 统一定义，避免重复
        self.ETAG_STYLES = {
            "T0": {"line_width": 2.0, "edge_color": "darkred"},
            "T1": {"line_width": 1.5, "edge_color": "darkblue"},
            "T2": {"line_width": 1.0, "edge_color": "black"}
        }
        
        # 透明度配置
        self.ALPHA_CONFIG = {
            "default": 1.0,          # 默认透明度
            "min": 0.4,              # 最小透明度
            "step": 0.2,             # 透明度递减步长
            "tags_mode": 0.3         # 标签模式透明度
        }

    def get_flit_style(self, flit: Any, use_highlight: bool = False, 
                       expected_packet_id: Any = None, highlight_color: Optional[str] = None,
                       show_tags_mode: bool = False) -> Tuple[Tuple[float, float, float, float], float, str]:
        """
        获取flit的完整样式
        
        Args:
            flit: flit对象或字典数据
            use_highlight: 是否启用高亮模式
            expected_packet_id: 期望高亮的包ID
            highlight_color: 自定义高亮颜色
            show_tags_mode: 是否为标签显示模式
            
        Returns:
            tuple: (face_color_rgba, line_width, edge_color)
                - face_color_rgba: 包含透明度的RGBA颜色元组
                - line_width: 边框线宽
                - edge_color: 边框颜色
        """
        # 提取flit属性
        attrs = self._extract_flit_attributes(flit)
        
        # 获取基础颜色
        base_color = self.color_manager.get_packet_color(
            attrs["packet_id"], 
            use_highlight, 
            expected_packet_id, 
            show_tags_mode
        )
        
        # 如果指定了自定义高亮颜色，使用它
        if use_highlight and highlight_color and str(attrs["packet_id"]) == str(expected_packet_id):
            base_color = highlight_color
        
        # 计算透明度
        alpha = self._calculate_alpha(attrs, show_tags_mode)
        
        # 获取E-Tag样式
        etag_style = self.ETAG_STYLES.get(attrs["etag"], self.ETAG_STYLES["T2"])
        
        # 将基础颜色转换为RGBA格式
        try:
            face_color_rgba = mcolors.to_rgba(base_color, alpha=alpha)
        except (ValueError, TypeError):
            # 如果颜色转换失败，使用默认颜色
            face_color_rgba = (0.5, 0.5, 1.0, alpha)  # 浅蓝色
        
        return (
            face_color_rgba,
            etag_style["line_width"],
            etag_style["edge_color"]
        )

    def _extract_flit_attributes(self, flit: Any) -> Dict[str, Any]:
        """
        提取flit属性的通用方法，兼容字典和对象格式
        
        Args:
            flit: flit对象或字典数据
            
        Returns:
            包含标准化属性的字典
        """
        if isinstance(flit, dict):
            return {
                "packet_id": flit.get("packet_id"),
                "flit_id": flit.get("flit_id", 0),
                "etag": flit.get("etag_priority", "T2"),
                "itag_h": flit.get("itag_h", False),
                "itag_v": flit.get("itag_v", False)
            }
        else:
            return {
                "packet_id": getattr(flit, "packet_id", None),
                "flit_id": getattr(flit, "flit_id", 0),
                "etag": getattr(flit, "etag_priority", "T2"),
                "itag_h": getattr(flit, "itag_h", False),
                "itag_v": getattr(flit, "itag_v", False)
            }

    def _calculate_alpha(self, attrs: Dict[str, Any], show_tags_mode: bool) -> float:
        """
        计算透明度
        
        Args:
            attrs: flit属性字典
            show_tags_mode: 是否为标签显示模式
            
        Returns:
            透明度值 (0.0-1.0)
        """
        if show_tags_mode:
            return self.ALPHA_CONFIG["tags_mode"]
        
        flit_id = attrs.get("flit_id", 0)
        if flit_id is not None:
            try:
                # 根据flit_id调整透明度：flit_id越大，透明度越低
                alpha = max(
                    self.ALPHA_CONFIG["min"], 
                    self.ALPHA_CONFIG["default"] - (int(flit_id) * self.ALPHA_CONFIG["step"])
                )
                return alpha
            except (ValueError, TypeError):
                pass
        
        return self.ALPHA_CONFIG["default"]

    def get_etag_style(self, etag_priority: str) -> Dict[str, Union[float, str]]:
        """
        获取E-Tag样式
        
        Args:
            etag_priority: E-Tag优先级 ("T0", "T1", "T2")
            
        Returns:
            样式字典，包含line_width和edge_color
        """
        return self.ETAG_STYLES.get(etag_priority, self.ETAG_STYLES["T2"]).copy()

    def set_etag_style(self, etag_priority: str, line_width: float, edge_color: str):
        """
        设置E-Tag样式
        
        Args:
            etag_priority: E-Tag优先级
            line_width: 线宽
            edge_color: 边框颜色
        """
        self.ETAG_STYLES[etag_priority] = {
            "line_width": line_width,
            "edge_color": edge_color
        }

    def set_alpha_config(self, config_key: str, value: float):
        """
        设置透明度配置
        
        Args:
            config_key: 配置键名
            value: 透明度值
        """
        if config_key in self.ALPHA_CONFIG:
            self.ALPHA_CONFIG[config_key] = max(0.0, min(1.0, value))

    def get_alpha_config(self) -> Dict[str, float]:
        """获取当前透明度配置"""
        return self.ALPHA_CONFIG.copy()

    def apply_style_to_patch(self, patch, flit: Any, use_highlight: bool = False,
                           expected_packet_id: Any = None, highlight_color: Optional[str] = None,
                           show_tags_mode: bool = False):
        """
        直接将样式应用到matplotlib patch对象
        
        Args:
            patch: matplotlib patch对象 (Rectangle, Circle等)
            flit: flit对象或字典数据
            use_highlight: 是否启用高亮模式
            expected_packet_id: 期望高亮的包ID
            highlight_color: 自定义高亮颜色
            show_tags_mode: 是否为标签显示模式
        """
        if flit is None:
            # 空flit：恢复默认样式
            patch.set_facecolor("none")
            patch.set_edgecolor("gray")
            patch.set_linewidth(0.8)
            patch.set_linestyle("--")
            return

        # 获取样式
        face_color, line_width, edge_color = self.get_flit_style(
            flit, use_highlight, expected_packet_id, highlight_color, show_tags_mode
        )
        
        # 应用样式
        patch.set_facecolor(face_color)
        patch.set_edgecolor(edge_color)
        patch.set_linewidth(max(line_width, 0.8))  # 确保最小线宽
        patch.set_linestyle("-")

    def create_empty_patch_style(self) -> Dict[str, Any]:
        """
        创建空patch的默认样式
        
        Returns:
            样式字典
        """
        return {
            "facecolor": "none",
            "edgecolor": "gray",
            "linewidth": 0.8,
            "linestyle": "--",
            "alpha": 0.7
        }

    def __repr__(self) -> str:
        return (f"VisualizationStyleManager("
                f"etag_styles={len(self.ETAG_STYLES)}, "
                f"color_manager={self.color_manager})")