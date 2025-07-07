# -*- coding: utf-8 -*-
"""
C2C拓扑可视化Web应用启动脚本
使用Streamlit创建交互式Web界面
"""

import subprocess
import sys
import os


def check_dependencies():
    """检查必需的依赖"""
    required_packages = ["streamlit", "matplotlib", "networkx", "numpy", "pandas", "plotly", "seaborn"]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("❌ 缺少以下依赖包:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\n请安装缺少的包:")
        print(f"pip install {' '.join(missing_packages)}")
        return False

    return True


def install_streamlit():
    """安装Streamlit（如果需要）"""
    try:
        import streamlit

        return True
    except ImportError:
        print("正在安装Streamlit...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
            print("✅ Streamlit安装成功")
            return True
        except subprocess.CalledProcessError:
            print("❌ Streamlit安装失败")
            return False


def run_webapp():
    """启动Web应用"""
    if not check_dependencies():
        return

    # 获取脚本目录
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 选择应用版本
    webapp_path = os.path.join(script_dir, r"../src/visualization", "interactive.py")
    app_name = "C2C拓扑性能对比分析"

    project_root = os.path.dirname(script_dir)

    print(f"🚀 启动{app_name}Web应用...")
    print("🌐 应用将在浏览器中自动打开")
    print("📝 使用 Ctrl+C 停止应用")

    try:
        # 启动Streamlit应用，设置正确的工作目录
        subprocess.run([sys.executable, "-m", "streamlit", "run", webapp_path, "--server.headless", "false", "--server.port", "8502", "--browser.gatherUsageStats", "false"], cwd=project_root)
    except KeyboardInterrupt:
        print("\n👋 应用已停止")
    except Exception as e:
        print(f"❌ 启动应用时发生错误: {str(e)}")


def show_usage():
    """显示使用说明"""
    print(
        """
🖥️ C2C拓扑性能对比分析Web应用

使用方法:
  python run_webapp.py            # 启动Web应用
  python run_webapp.py --help     # 显示帮助信息

功能特性:
  📊 交互式性能对比分析
  🎯 智能拓扑优化建议
  🌐 拓扑可视化
  📈 实时性能指标对比
  📋 详细分析报告生成
  ⚡ Plotly交互式图表
  🛤️ 路径分析工具
  🔥 网络热点分析

依赖要求:
  - Python 3.8+
  - streamlit
  - matplotlib
  - networkx
  - numpy
  - pandas
  - plotly (增强版)
  - seaborn (增强版)

首次使用:
  如果缺少依赖包，脚本会自动提示安装命令
  
"""
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] in ["--help", "-h"]:
            show_usage()
        else:
            print(f"未知参数: {sys.argv[1]}")
            show_usage()
    else:
        run_webapp()
