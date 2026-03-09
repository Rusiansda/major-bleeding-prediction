# 消化道大出血风险预测系统

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

基于 CatBoost 机器学习的消化道大出血风险预测 Web 应用。

## 🏥 项目简介

本项目是一个用于预测 ICU 患者消化道大出血风险的机器学习应用。模型基于 20 个临床特征，验证集 AUC 达到 0.886。

### 主要功能
- 🔮 风险概率预测
- 📊 SHAP 特征重要性解释
- 📈 可视化风险等级
- 📥 预测结果导出

## 🚀 在线访问

**Streamlit Cloud**: [点击访问](https://your-app-url.streamlit.app)

## 📋 本地运行

### 环境要求
- Python 3.9+
- pip

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/your-username/major-bleeding-prediction.git
cd major-bleeding-prediction

# 2. 安装依赖
pip install -r requirements.txt

# 3. 运行应用
streamlit run major_bleeding_web_deploy.py
```

应用将在 http://localhost:8501 启动

## 🏗️ 技术栈

- **前端**: Streamlit
- **机器学习**: CatBoost, scikit-learn
- **可解释性**: SHAP
- **数据处理**: pandas, numpy

## 📊 模型性能

| 指标 | 验证集 |
|------|--------|
| AUC | 0.886 (95% CI: 0.840-0.930) |
| 敏感度 | 90.2% |
| 特异度 | 72.3% |
| NPV | 99.7% |

## 🔬 输入特征

### 基础特征（18个）
- 生命体征：体温、收缩压、舒张压、呼吸频率、GCS评分
- 实验室指标：白细胞、血红蛋白、血小板、纤维蛋白原、ALT、总胆红素、球蛋白、肌酐、尿素氮、血钾
- 其他：吸入氧浓度、氧分压、尿路感染、恶性肿瘤/免疫抑制
- 血液指标：红细胞压积(HCT)

### 派生特征（2个，自动计算）
- 白球比 (AG_Ratio) = 白蛋白 / 球蛋白
- 脉压 (Pulse_Pressure) = 收缩压 - 舒张压

## ⚠️ 免责声明

本系统仅供学术研究使用，预测结果仅供参考，不构成医疗建议。临床决策应结合专业医生的判断。

## 📄 许可证

MIT License

## 👥 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系方式

如有问题，请联系：your-email@example.com
