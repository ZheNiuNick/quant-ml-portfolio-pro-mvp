#!/bin/bash
# Railway 启动脚本
# 确保从项目根目录运行，这样 app.py 可以找到 src/ 和 config/ 目录

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 如果脚本在 zhe_trading_strategy 目录，则切换到项目根目录
if [[ "$SCRIPT_DIR" == *"zhe_trading_strategy"* ]]; then
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
    cd "$PROJECT_ROOT"
fi

# 进入 zhe_trading_strategy 目录运行 gunicorn
cd zhe_trading_strategy

# 运行 gunicorn
exec gunicorn app:app --bind 0.0.0.0:${PORT:-8080} --workers 2 --timeout 120 --access-logfile - --error-logfile -

