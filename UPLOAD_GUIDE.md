# 📤 上传文件到 Hugging Face 指南

## 快速开始

### 步骤 1: 获取访问令牌

1. 访问 [Hugging Face 令牌页面](https://huggingface.co/settings/tokens)
2. 点击 **"New token"**
3. 填写信息：
   - **Token name**: `quant-ml-upload`
   - **Type**: `Write` (需要写入权限)
4. 点击 **"Generate token"**
5. **复制令牌**（只显示一次，请保存好）

### 步骤 2: 创建数据集（如果还没有）

1. 访问 [创建数据集页面](https://huggingface.co/new-dataset)
2. 填写信息：
   - **Repository name**: `quant-ml-data`
   - **Owner**: 选择你的用户名
   - **Visibility**: `Public` (推荐) 或 `Private`
3. 点击 **"Create repository"**

### 步骤 3: 上传文件

有三种方法：

#### 方法 1: 使用环境变量（推荐）

```bash
# 设置令牌
export HF_TOKEN=your_token_here

# 运行上传脚本
python upload_to_hf.py
```

#### 方法 2: 命令行参数

```bash
python upload_to_hf.py your_token_here
```

#### 方法 3: 交互式输入

```bash
python upload_to_hf.py
# 脚本会提示输入令牌
```

## 上传过程

上传 317MB 的文件可能需要：
- **快速网络**: 2-5 分钟
- **普通网络**: 5-10 分钟
- **慢速网络**: 10-20 分钟

请耐心等待，上传过程中不要中断。

## 验证上传

上传成功后，访问：
https://huggingface.co/datasets/NickNiu/quant-ml-data

你应该能看到 `data/factors/factor_store.parquet` 文件。

## 故障排除

### 错误: 401 Unauthorized

**原因**: 令牌无效或权限不足

**解决**:
- 检查令牌是否正确
- 确保令牌有 `Write` 权限
- 重新生成令牌

### 错误: 404 Repository not found

**原因**: 数据集仓库不存在

**解决**:
1. 访问 https://huggingface.co/new-dataset
2. 创建名为 `quant-ml-data` 的数据集
3. 确保所有者是你的用户名

### 错误: 网络超时

**原因**: 网络连接不稳定或文件太大

**解决**:
- 检查网络连接
- 重试上传
- 如果持续失败，可以考虑分块上传（需要修改脚本）

## 下一步

上传成功后：

1. ✅ 提交代码更改
2. ✅ 推送到 GitHub
3. ✅ Railway 自动重新部署
4. ✅ 应用会自动下载文件

## 需要帮助？

如果遇到问题，请检查：
1. 令牌是否正确且有 `Write` 权限
2. 数据集是否存在
3. 网络连接是否正常
4. 文件是否存在: `data/factors/factor_store.parquet`

