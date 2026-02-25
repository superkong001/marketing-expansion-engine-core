"""
阶段2统一入口脚本

作为阶段2的统一入口，调用stage2/stage2_main.py
"""
import sys
from pathlib import Path

# 添加stage2目录到路径
stage2_dir = Path(__file__).parent / 'stage2'
sys.path.insert(0, str(stage2_dir))

# 导入并运行main函数
from stage2_main import main

if __name__ == '__main__':
    main()

