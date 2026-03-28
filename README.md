# 消化道大出血风险预测系统 - v2.0 部署指南

## 更新说明

### 版本信息
- **版本**: v2.0
- **更新日期**: 2025-03-28
- **模型**: XGBoost (最佳模型)
- **验证集 AUC**: 0.910 (95% CI: 0.872-0.945)

### 主要更新
1. **模型升级**: 从 CatBoost 升级为 XGBoost（性能更优）
2. **特征更新**: 
   - 特征数从 20 个优化为 19 个
   - 新增 One-Hot 编码的感染类型变量
   - 新增年龄 (Age)、血钙 (Serum_Calcium)
   - 移除脉压 (Pulse_Pressure)
3. **性能提升**: AUC 从 0.886 提升至 0.910

---

## 部署步骤

### 1. 环境要求
```bash
Python 3.8+
Streamlit 1.28+
XGBoost 2.0+
SHAP 0.42+
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 文件准备
确保以下文件存在：
```
生成的文件_major_bleeding/
  ├── model_XGBoost_calibrated.pkl  # XGBoost校准模型
  
生成的文件/
  ├── robust_scaler.pkl             # 特征标准化器
  └── lasso_selected_features.pkl   # 选中的特征列表
```

### 4. 启动应用
```bash
# 方法1: 直接运行
streamlit run major_bleeding_web_deploy_v2.py

# 方法2: 指定端口
streamlit run major_bleeding_web_deploy_v2.py --server.port 8501
```

---

## 特征说明

### 输入特征 (17个基础特征)

| 类别 | 特征 | 说明 | 单位 |
|------|------|------|------|
| 基本信息 | Age | 年龄 | 岁 |
| 生命体征 | Temperature | 体温 | °C |
| | Systolic_BP | 收缩压 | mmHg |
| | Respiratory_Rate | 呼吸频率 | 次/分 |
| | GCS_Score | GCS评分 | 3-15 |
| 感染类型 | Infection_Type | 感染类型 | 单选 |
| 血常规 | WBC | 白细胞 | ×10⁹/L |
| | HGB | 血红蛋白 | g/L |
| | HCT | 红细胞压积 | % |
| | PLT | 血小板 | ×10⁹/L |
| 凝血功能 | Fibrinogen | 纤维蛋白原 | g/L |
| 肝功能 | ALT | 谷丙转氨酶 | U/L |
| | Globulin | 球蛋白 | g/L |
| | Albumin | 白蛋白 | g/L |
| 肾功能 | Creatinine | 肌酐 | μmol/L |
| | BUN | 尿素氮 | mmol/L |
| 呼吸 | Oxygen_Concentration | 吸入氧浓度 | % |
| | Partial_Pressure_Of_Oxygen | 氧分压 | mmHg |
| 电解质 | Serum_Calcium | 血钙 | mmol/L |

### 派生特征 (1个)
- **AG_Ratio**: 白球比 = 白蛋白 / 球蛋白

### One-Hot 编码特征 (2个)
- **Infection_Type_Urinary**: 尿路感染 (0/1)
- **Infection_Type_Gastrointestinal**: 消化道感染 (0/1)

**参考组**: 肺部感染 (所有哑变量=0时)

---

## 模型性能

### 验证集指标
| 指标 | 值 |
|------|-----|
| AUC | 0.9098 (95% CI: 0.872-0.945) |
| 准确率 | 97.02% |
| 敏感度 | 36.59% |
| 特异度 | 98.26% |
| PPV | 30.00% |
| NPV | 98.70% |
| F1分数 | 0.3297 |
| 最佳阈值 | 0.6852 |

### 风险分层
| 风险概率 | 等级 | 建议 |
|:--------:|:----:|------|
| <10% | 低风险 | 常规监测 |
| 10-30% | 中低风险 | 加强监测 |
| 30-50% | 中风险 | 预防措施 |
| ≥50% | 高风险 | 积极干预 |

---

## 注意事项

1. **特征顺序**: 模型对特征顺序敏感，请确保输入顺序与训练时一致
2. **标准化**: 连续变量会自动进行 RobustScaler 标准化
3. **缺失值**: 请确保所有输入特征都有有效值
4. **感染类型**: 选择后系统会自动进行 One-Hot 编码

---

## 技术细节

### 模型结构
```python
CalibratedModel(
    base_model=XGBClassifier(
        n_estimators=...,
        max_depth=...,
        learning_rate=...,
        ...
    ),
    calibration_method='isotonic'
)
```

### SHAP 解释
- 使用 TreeExplainer 进行特征贡献解释
- 显示前10个最重要的特征
- 红色：增加出血风险
- 蓝色：降低出血风险

---

## 联系方式

如有问题，请联系开发团队。
