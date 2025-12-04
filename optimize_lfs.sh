#!/bin/bash
# 优化 Git LFS 配置：将小文件移出 LFS，只保留大文件

set -e

echo "🔧 开始优化 Git LFS 配置..."

# 小文件列表（这些文件将移出 LFS，直接提交到 Git）
SMALL_FILES=(
    "outputs/backtests/*.parquet"
    "outputs/backtests/*.json"
    "data/factors/factor_ic_ir.parquet"
    "outputs/portfolios/weights.parquet"
)

echo ""
echo "📋 将从 LFS 移除的小文件："
for file in "${SMALL_FILES[@]}"; do
    echo "  - $file"
done

echo ""
read -p "确认继续？(y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ 操作已取消"
    exit 1
fi

echo ""
echo "📝 更新 .gitattributes 文件..."

# 创建新的 .gitattributes，只对特定大文件使用 LFS
cat > .gitattributes << 'EOF'
# 大文件使用 LFS（不部署到 Railway，仅本地使用）
data/factors/factor_store.parquet filter=lfs diff=lfs merge=lfs -text
data/processed/prices.parquet filter=lfs diff=lfs merge=lfs -text
duckdb/*.duckdb filter=lfs diff=lfs merge=lfs -text

# 小文件直接提交到 Git（不使用 LFS）
# outputs/backtests/*.parquet
# outputs/backtests/*.json
# data/factors/factor_ic_ir.parquet
# outputs/portfolios/weights.parquet
EOF

echo "✅ .gitattributes 已更新"

echo ""
echo "🔄 从 LFS 跟踪中移除小文件..."

for pattern in "${SMALL_FILES[@]}"; do
    if git lfs untrack "$pattern" 2>/dev/null; then
        echo "  ✅ 已移除: $pattern"
    else
        echo "  ⚠️  未找到或已移除: $pattern"
    fi
done

echo ""
echo "📦 准备提交更改..."
echo ""
echo "下一步操作："
echo "1. 检查文件状态: git status"
echo "2. 查看 .gitattributes 内容确认无误"
echo "3. 提交更改: git add .gitattributes && git commit -m 'Optimize: Move small files out of LFS'"
echo "4. 推送到 GitHub: git push"
echo ""
echo "⚠️  注意：小文件需要从 LFS 迁移到普通 Git 存储"
echo "   这需要：git lfs migrate export --include=\"小文件模式\" --everything"

echo ""
echo "✅ 配置优化完成！"

