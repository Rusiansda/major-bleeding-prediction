# 消化道大出血风险预测系统

基于XGBoost机器学习的消化道大出血风险预测Web应用。

## 🚀 快速开始

### 在线访问
部署在 Streamlit Cloud: [https://major-bleeding-prediction.streamlit.app](https://major-bleeding-prediction.streamlit.app)

### 本地运行

```bash
# 安装依赖
pip install -r requirements.txt

# 启动应用
streamlit run streamlit_app.py
```

## 📋 项目结构

```
.
├── streamlit_app.py          # 主应用文件
├── requirements.txt          # Python依赖
├── model_XGBoost_calibrated.pkl    # XGBoost校准模型
├── lasso_scaler_params.pkl   # 特征标准化参数
├── utils.py                  # 工具类（CalibratedModel）
└── .streamlit/
    └── config.toml          # Streamlit配置
```

## 🩺 风险分层

| 风险概率 | 等级 | 建议 |
|---------|------|------|
| <10% | 低风险 | 常规监测 |
| 10-30% | 中低风险 | 加强监测 |
| 30-50% | 中风险 | 预防措施 |
| ≥50% | 高风险 | 积极干预 |

## 👨‍⚕️ 免责声明

本工具仅供医学研究参考，不构成医疗建议。临床决策应由专业医生根据患者具体情况做出。

## 📧 联系方式

如有问题，请联系开发团队。
