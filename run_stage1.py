"""
阶段1统一入口脚本

作为阶段1的统一入口，调用stage1/diff_main.py
"""
import sys
from pathlib import Path

# 添加stage1目录到路径
stage1_dir = Path(__file__).parent / 'stage1'
sys.path.insert(0, str(stage1_dir))

# 导入并运行main函数
from diff_main import main

if __name__ == '__main__':
    main()

