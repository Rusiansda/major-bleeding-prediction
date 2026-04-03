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

## 🔬 模型信息

- **算法**: XGBoost (校准后)
- **特征数**: 19个
- **验证集AUC**: 0.910 (95% CI: 0.872-0.945)
- **验证集准确率**: 97.02%

### 输入特征

| 类别 | 特征 |
|------|------|
| 基本信息 | Age |
| 生命体征 | Temperature, Systolic_BP, Respiratory_Rate, GCS_Score |
| 血常规 | WBC, HGB, HCT, PLT |
| 凝血功能 | Fibrinogen |
| 肝功能 | ALT, Globulin, Albumin |
| 肾功能 | Creatinine, BUN |
| 呼吸支持 | Oxygen_Concentration, Partial_Pressure_Of_Oxygen |
| 电解质 | Serum_Calcium |
| 派生特征 | AG_Ratio (白球比) |

## 🩺 风险分层

| 风险概率 | 等级 | 建议 |
|---------|------|------|
| <10% | 低风险 | 常规监测 |
| 10-30% | 中低风险 | 加强监测 |
| 30-50% | 中风险 | 预防措施 |
| ≥50% | 高风险 | 积极干预 |

## 📝 更新日志

### v2.0 (2025-03-28)
- 模型升级: CatBoost → XGBoost
- 特征优化: 20个 → 19个
- 性能提升: AUC 0.886 → 0.910

## 👨‍⚕️ 免责声明

本工具仅供医学研究参考，不构成医疗建议。临床决策应由专业医生根据患者具体情况做出。

## 📧 联系方式

如有问题，请联系开发团队。
