# 🚀 快速上传指南

## 问题：`huggingface-cli` 命令找不到

没关系！我已经创建了一个 Python 脚本，不需要 CLI 命令。

## 三步上传文件

### 步骤 1: 获取 Hugging Face 令牌

1. 访问：https://huggingface.co/settings/tokens
2. 点击 **"New token"**
3. 选择权限：**"Write"**
4. 复制令牌

### 步骤 2: 选择一种方式运行上传

#### 方式 A: 环境变量（推荐）

```bash
export HF_TOKEN=your_token_here
python upload_to_hf.py
```

#### 方式 B: 命令行参数

```bash
python upload_to_hf.py your_token_here
```

#### 方式 C: 交互式输入

```bash
python upload_to_hf.py
# 脚本会提示你输入令牌
```

### 步骤 3: 等待上传完成

上传 317MB 文件需要 2-10 分钟，请耐心等待。

## 完成后

上传成功后，你可以：
1. ✅ 访问数据集：https://huggingface.co/datasets/NickNiu/quant-ml-data
2. ✅ 提交代码更改
3. ✅ Railway 会自动部署
4. ✅ 应用会自动下载文件

## 需要帮助？

如果遇到问题，请查看：`UPLOAD_GUIDE.md`

