# 部署到 GitHub 指南

## 📦 部署文件清单

```
deploy_github/
├── streamlit_app.py              # 主应用（29KB）
├── utils.py                      # CalibratedModel类（47KB）
├── requirements.txt              # 依赖列表
├── model_XGBoost_calibrated.pkl  # XGBoost模型（63KB）
├── lasso_scaler_params.pkl       # 标准化参数（2KB）
├── README.md                     # 项目说明
├── .gitignore                    # Git忽略配置
└── .streamlit/
    └── config.toml               # Streamlit配置
```

## 🚀 推送到 GitHub 步骤

### 方法1: 直接推送（如果当前仓库已配置）

```bash
# 1. 进入部署目录
cd deploy_github

# 2. 初始化git（如果是新目录）
git init

# 3. 添加远程仓库
git remote add origin https://github.com/Rusiansda/major-bleeding-prediction.git

# 4. 添加所有文件
git add .

# 5. 提交
git commit -m "feat: Deploy Major Bleeding Prediction Web App v2.0

- 基于XGBoost模型 (AUC=0.910)
- 19个输入特征
- 支持SHAP可解释性"

# 6. 推送到main分支
git push -u origin main --force
```

### 方法2: 覆盖更新（保留历史）

```bash
# 1. 先拉取最新代码
git pull origin main

# 2. 用deploy_github内容覆盖项目根目录
# 手动复制文件或：
cp deploy_github/* .
cp -r deploy_github/.streamlit .

# 3. 提交更改
git add .
git commit -m "Update deployment files"
git push origin main
```

### 方法3: 使用GitHub Desktop

1. 打开 GitHub Desktop
2. 选择 `major-bleeding-prediction` 仓库
3. 将 `deploy_github` 中的文件复制到本地仓库
4. 填写提交信息并提交
5. 点击 "Push origin"

## ⚙️ Streamlit Cloud 自动部署

GitHub推送后，Streamlit Cloud会自动重新部署：

1. 访问 https://share.streamlit.io
2. 选择 `major-bleeding-prediction` 仓库
3. 主文件选择 `streamlit_app.py`
4. 点击 Deploy

## 🔍 验证部署

推送后检查以下链接：
- GitHub仓库: https://github.com/Rusiansda/major-bleeding-prediction
- Streamlit应用: https://major-bleeding-prediction.streamlit.app

## 📁 文件大小检查

```bash
# 检查大文件（GitHub限制100MB）
ls -lh *.pkl
```

当前模型文件大小：
- model_XGBoost_calibrated.pkl: ~63KB ✅
- lasso_scaler_params.pkl: ~2KB ✅

## 🆘 常见问题

### 问题1: 模型文件无法加载
检查 `streamlit_app.py` 中的路径配置，支持以下查找顺序：
1. 同目录下的 `.pkl` 文件
2. 上级目录的 `生成的文件_major_bleeding/` 文件夹

### 问题2: 缺少依赖
确保 `requirements.txt` 包含：
```
streamlit
xgboost
shap
numpy
pandas
matplotlib
```

### 问题3: 中文显示乱码
已在代码中配置matplotlib中文字体：
```python
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
```

## 📞 技术支持

如有问题请联系开发团队。
