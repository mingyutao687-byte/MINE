#!/usr/bin/env python
"""MINE 调度网关启动器。

设置正确的 PYTHONPATH 使 Mine 包可被导入，然后启动 FastAPI 网关。

用法:
  cd MINE
  python run_gateway.py

等价于:
  PYTHONPATH=. python -m Mine.api.gateway
"""

import sys
import os

# 将 MINE_core 加入 Python path（使得 Mine 包可被导入）
_MINE_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MINE_PARENT not in sys.path:
    sys.path.insert(0, _MINE_PARENT)

# 将 config_template 加入 Python path（兼容原代码的 import）
_CONFIG_TEMPLATE = os.path.join(os.path.dirname(__file__), 'config_template')
if _CONFIG_TEMPLATE not in sys.path:
    sys.path.insert(0, _CONFIG_TEMPLATE)

if __name__ == "__main__":
    import uvicorn
    from Mine.api.gateway import app

    print(f"Starting MINE Gateway on port 7000...")
    print(f"  PYTHONPATH root: {_MINE_PARENT}")
    print(f"  Config templates: {_CONFIG_TEMPLATE}")
    uvicorn.run(app, host="0.0.0.0", port=7000, log_level='warning')
