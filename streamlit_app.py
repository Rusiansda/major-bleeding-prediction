"""
消化道大出血(Major Bleeding)风险预测 Web 应用
使用 Streamlit 部署 - 基于 XGBoost 机器学习模型
"""

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import os

# 设置 matplotlib 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 页面配置
st.set_page_config(
    page_title="消化道大出血风险预测系统",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============== 加载模型 ==============
@st.cache_resource
def load_model():
    """加载 XGBoost 模型（校准后用于预测，提取原始基学习器用于SHAP）"""
    # 加载校准模型（用于预测）
    # 尝试多个路径（本地开发和部署环境）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(current_dir, "model_XGBoost_calibrated.pkl"),  # 同一目录
        os.path.join(os.path.dirname(current_dir), "生成的文件_major_bleeding", "model_XGBoost_calibrated.pkl"),  # 本地开发
    ]
    
    cal_model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            cal_model_path = path
            break
    
    if cal_model_path is None:
        raise FileNotFoundError(f"找不到模型文件，尝试的路径: {possible_paths}")
    
    with open(cal_model_path, 'rb') as f:
        cal_model = pickle.load(f)
    
    # 提取原始 XGBoost 模型（用于SHAP解释）
    orig_model = None
    try:
        # 自定义 CalibratedModel 类 (from utils.py)
        if hasattr(cal_model, 'base_model'):
            orig_model = cal_model.base_model
            st.sidebar.info("使用 CalibratedModel.base_model 提取原始XGBoost模型")
        # sklearn CalibratedClassifierCV
        elif hasattr(cal_model, 'calibrated_classifiers_') and len(cal_model.calibrated_classifiers_) > 0:
            orig_model = cal_model.calibrated_classifiers_[0].estimator
            st.sidebar.info("使用 CalibratedClassifierCV 提取原始模型")
        # 直接就是原始模型
        elif hasattr(cal_model, 'predict_proba'):
            orig_model = cal_model
            st.sidebar.info("直接使用原始模型")
    except Exception as e:
        st.sidebar.warning(f"提取原始模型失败: {e}")
        orig_model = None
    
    return cal_model, orig_model


@st.cache_resource
def load_scaler():
    """加载特征标准化器"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(current_dir, "lasso_scaler_params.pkl"),  # LASSO专用scaler
            os.path.join(current_dir, "robust_scaler.pkl"),  # 备用
        ]
        
        scaler_path = None
        for path in possible_paths:
            if os.path.exists(path):
                scaler_path = path
                break
        
        if scaler_path is None:
            st.sidebar.warning("找不到标准化器文件")
            return None, []
        
        with open(scaler_path, 'rb') as f:
            scaler_data = pickle.load(f)
        
        # 如果是lasso_scaler_params，直接返回参数
        if 'lasso' in scaler_path.lower():
            return scaler_data, list(scaler_data.keys())
        else:
            return scaler_data['scaler'], scaler_data['continuous_vars']
    except Exception as e:
        st.sidebar.warning(f"加载标准化器失败: {e}")
        return None, []


def scale_features(feature_values, feature_names, scaler, continuous_vars):
    """对连续变量进行标准化"""
    if scaler is None or len(continuous_vars) == 0:
        return feature_values
    
    result = feature_values.copy()
    
    # 需要标准化的LASSO特征（根据训练时的设置）
    # 注意：Infection_Type_Urinary 和 Infection_Type_Gastrointestinal 是二元变量，不需要标准化
    lasso_continuous_vars = [
        'Age', 'Temperature', 'Systolic_BP', 'Respiratory_Rate', 'GCS_Score',
        'WBC', 'HGB', 'HCT', 'PLT', 'Fibrinogen', 'Creatinine', 'BUN',
        'Oxygen_Concentration', 'Partial_Pressure_Of_Oxygen', 'Serum_Calcium',
        'AG_Ratio', 'ALT'
    ]
    
    # 判断scaler类型
    is_lasso_scaler = isinstance(scaler, dict)
    
    # 手动标准化
    for col_name in lasso_continuous_vars:
        if col_name in feature_names:
            idx = feature_names.index(col_name)
            
            try:
                if is_lasso_scaler:
                    # 使用LASSO专用scaler参数
                    if col_name in scaler:
                        median = scaler[col_name]['median']
                        iqr = scaler[col_name]['iqr']
                        if iqr > 0:
                            result[idx] = (feature_values[idx] - median) / iqr
                else:
                    # 使用原始RobustScaler
                    if col_name in continuous_vars:
                        col_idx_in_scaler = continuous_vars.index(col_name)
                        if hasattr(scaler, 'center_'):
                            median = scaler.center_[col_idx_in_scaler]
                        else:
                            median = scaler.quantile_[col_idx_in_scaler]
                        scale = scaler.scale_[col_idx_in_scaler]
                        if scale > 0:
                            result[idx] = (feature_values[idx] - median) / scale
            except Exception as e:
                # 如果标准化失败，保持原值
                pass
    
    return result


# ============== 特征定义 ==============
# 基础输入特征（用户界面输入）
BASE_FEATURES = {
    # 基本信息
    "Age": {"label": "年龄 (岁)", "type": "numerical", "min": 18, "max": 100, "default": 65, "step": 1},
    
    # 生命体征
    "Temperature": {"label": "体温 (°C)", "type": "numerical", "min": 32.0, "max": 42.0, "default": 37.0, "step": 0.1},
    "Systolic_BP": {"label": "收缩压 (mmHg)", "type": "numerical", "min": 0, "max": 250, "default": 120, "step": 1},
    "Respiratory_Rate": {"label": "呼吸频率 (次/分)", "type": "numerical", "min": 0, "max": 80, "default": 20, "step": 1},
    "GCS_Score": {"label": "GCS评分 (3-15)", "type": "numerical", "min": 3, "max": 15, "default": 15, "step": 1},
    
    # 感染类型（新增 One-Hot 编码）
    "Infection_Type": {
        "label": "感染类型", 
        "type": "categorical", 
        "options": [0, 1, 2, 3, 4], 
        "labels": ["肺部感染", "尿路感染", "消化道感染", "其他/无明确", "混合感染"],
        "default": 0
    },
    
    # 血常规
    "WBC": {"label": "白细胞 (×10⁹/L)", "type": "numerical", "min": 0.1, "max": 100, "default": 8.0, "step": 0.1},
    "HGB": {"label": "血红蛋白 (g/L)", "type": "numerical", "min": 20, "max": 220, "default": 110, "step": 1},
    "HCT": {"label": "红细胞压积 (%)", "type": "numerical", "min": 10, "max": 60, "default": 35, "step": 0.1},
    "PLT": {"label": "血小板 (×10⁹/L)", "type": "numerical", "min": 1, "max": 1100, "default": 180, "step": 1},
    
    # 凝血功能
    "Fibrinogen": {"label": "纤维蛋白原 (g/L)", "type": "numerical", "min": 0.1, "max": 15, "default": 3.0, "step": 0.1},
    
    # 肝功能
    "ALT": {"label": "谷丙转氨酶 (U/L)", "type": "numerical", "min": 1, "max": 6500, "default": 30, "step": 1},
    "Globulin": {"label": "球蛋白 (g/L)", "type": "numerical", "min": 5, "max": 60, "default": 30, "step": 1},
    "Albumin": {"label": "白蛋白 (g/L)", "type": "numerical", "min": 10, "max": 60, "default": 35, "step": 1},
    
    # 肾功能
    "Creatinine": {"label": "肌酐 (μmol/L)", "type": "numerical", "min": 5, "max": 1800, "default": 80, "step": 1},
    "BUN": {"label": "尿素氮 (mmol/L)", "type": "numerical", "min": 0.5, "max": 50, "default": 6.0, "step": 0.1},
    
    # 呼吸支持
    "Oxygen_Concentration": {"label": "吸入氧浓度 (%)", "type": "numerical", "min": 21, "max": 100, "default": 21, "step": 1},
    "Partial_Pressure_Of_Oxygen": {"label": "氧分压 (mmHg)", "type": "numerical", "min": 20, "max": 530, "default": 90, "step": 1},
    
    # 电解质
    "Serum_Calcium": {"label": "血钙 (mmol/L)", "type": "numerical", "min": 0.5, "max": 5, "default": 2.3, "step": 0.1},
}

# 模型特征列表（固定顺序，19个特征）- 必须与训练时顺序一致
MODEL_FEATURES = [
    'Age', 'Temperature', 'Systolic_BP', 'Respiratory_Rate', 'GCS_Score',
    'WBC', 'HGB', 'HCT', 'PLT', 'Fibrinogen', 'Creatinine', 'BUN',
    'Oxygen_Concentration', 'Partial_Pressure_Of_Oxygen', 'Serum_Calcium',
    'Infection_Type_Urinary', 'Infection_Type_Gastrointestinal',
    'AG_Ratio', 'ALT'
]

# ============== 特征工程函数 ==============
def calculate_derived_features(base_values):
    """计算派生特征"""
    derived = {}
    
    # AG_Ratio: 白球比 = 白蛋白 / 球蛋白
    albumin = base_values.get("Albumin", 35)
    globulin = base_values.get("Globulin", 30)
    if globulin > 0:
        derived["AG_Ratio"] = albumin / globulin
    else:
        derived["AG_Ratio"] = 1.17  # 默认值
    
    return derived


def create_infection_dummy(infection_type):
    """创建感染类型哑变量（One-Hot 编码）"""
    # 编码规则：
    # 0 = 肺部感染（参考组）
    # 1 = 尿路感染
    # 2 = 消化道感染
    # 3 = 其他/无明确
    # 4 = 混合感染
    
    urinary = 1 if infection_type == 1 else 0
    gastrointestinal = 1 if infection_type == 2 else 0
    # 其他和混合感染不创建哑变量（系数被LASSO压缩至0）
    
    return {"Infection_Type_Urinary": urinary, 
            "Infection_Type_Gastrointestinal": gastrointestinal}


def prepare_features_for_model(base_values, derived_values):
    """准备模型输入特征（按MODEL_FEATURES顺序）"""
    # 创建感染类型哑变量
    infection_dummies = create_infection_dummy(base_values["Infection_Type"])
    
    features = {}
    features["Age"] = base_values["Age"]
    features["Temperature"] = base_values["Temperature"]
    features["Systolic_BP"] = base_values["Systolic_BP"]
    features["Respiratory_Rate"] = base_values["Respiratory_Rate"]
    features["GCS_Score"] = base_values["GCS_Score"]
    features["WBC"] = base_values["WBC"]
    features["HGB"] = base_values["HGB"]
    features["HCT"] = base_values["HCT"]
    features["PLT"] = base_values["PLT"]
    features["Fibrinogen"] = base_values["Fibrinogen"]
    features["Creatinine"] = base_values["Creatinine"]
    features["BUN"] = base_values["BUN"]
    features["Oxygen_Concentration"] = base_values["Oxygen_Concentration"]
    features["Partial_Pressure_Of_Oxygen"] = base_values["Partial_Pressure_Of_Oxygen"]
    features["Serum_Calcium"] = base_values["Serum_Calcium"]
    features["Infection_Type_Urinary"] = infection_dummies["Infection_Type_Urinary"]
    features["Infection_Type_Gastrointestinal"] = infection_dummies["Infection_Type_Gastrointestinal"]
    features["AG_Ratio"] = derived_values["AG_Ratio"]
    features["ALT"] = base_values["ALT"]
    
    return [features[f] for f in MODEL_FEATURES]


# ============== SHAP解释 ==============
def get_shap_explanation(orig_model, feature_values, feature_names):
    """获取 SHAP 解释"""
    if orig_model is None:
        return None, None
    
    try:
        # 创建 SHAP Explainer
        explainer = shap.TreeExplainer(orig_model)
        
        # 转换为 numpy array
        X = np.array([feature_values])
        
        # 计算 SHAP values
        shap_values = explainer.shap_values(X)
        
        # 处理二分类的情况
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # 取正类的 SHAP 值
        
        return shap_values, explainer.expected_value
    except Exception as e:
        st.warning(f"SHAP 解释生成失败: {e}")
        return None, None


def plot_shap_force(shap_values, feature_values, feature_names, expected_value):
    """绘制 SHAP force plot"""
    if shap_values is None:
        return None
    
    try:
        # 处理 expected_value
        if isinstance(expected_value, list):
            expected_value = expected_value[1]
        
        # 创建 force plot
        fig, ax = plt.subplots(figsize=(12, 3))
        
        # 使用瀑布图替代 force plot（更易于保存和显示）
        shap.waterfall_plot(shap.Explanation(
            values=shap_values[0],
            base_values=expected_value,
            data=feature_values,
            feature_names=feature_names
        ), max_display=10, show=False)
        
        plt.tight_layout()
        return fig
    except Exception as e:
        st.warning(f"SHAP 图生成失败: {e}")
        return None


# ============== 主应用 ==============
def main():
    # 标题
    st.title("🏥 消化道大出血风险预测系统")
    st.markdown("---")
    
    # 加载模型和标准化器
    try:
        cal_model, orig_model = load_model()
        scaler, continuous_vars = load_scaler()
        st.sidebar.success("✅ XGBoost 模型加载成功")
        if scaler is not None:
            st.sidebar.info(f"✅ 标准化器加载成功")
        if orig_model is not None:
            st.sidebar.info("✅ SHAP解释: 可用")
        else:
            st.sidebar.info("⚠️ SHAP解释: 仅预测可用")
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        st.info("请确保已运行训练代码并生成模型文件: 生成的文件_major_bleeding/model_XGBoost_calibrated.pkl")
        return
    
    # 侧边栏信息
    
    with st.sidebar.expander("📖 使用说明"):
        st.markdown("""
        **输入患者信息：**
        1. 基本信息：年龄
        2. 生命体征：体温、血压、呼吸频率、GCS评分
        3. 感染类型：肺部/尿路/消化道/其他/混合
        4. 实验室检查：血常规、肝肾功能、凝血功能等
        
        **系统自动处理：**
        - 计算派生特征（白球比）
        - One-Hot 编码感染类型
        - 特征标准化
        
        **输出结果：**
        - 出血风险概率
        - SHAP 特征贡献解释
        """)
    
    # 主面板 - 输入区域
    col1, col2 = st.columns(2)
    
    base_values = {}
    
    with col1:
        st.subheader("👤 基本信息")
        for feature, config in BASE_FEATURES.items():
            if feature == "Age":
                if config["type"] == "numerical":
                    base_values[feature] = st.number_input(
                        config["label"],
                        min_value=float(config["min"]),
                        max_value=float(config["max"]),
                        value=float(config["default"]),
                        step=float(config.get("step", 1)),
                        key=feature
                    )
        
        st.subheader("🫀 生命体征")
        for feature, config in BASE_FEATURES.items():
            if feature in ["Temperature", "Respiratory_Rate", "GCS_Score", "Systolic_BP"]:
                if config["type"] == "numerical":
                    base_values[feature] = st.number_input(
                        config["label"],
                        min_value=float(config["min"]),
                        max_value=float(config["max"]),
                        value=float(config["default"]),
                        step=float(config.get("step", 1)),
                        key=feature
                    )
        
        st.subheader("🦠 感染类型")
        for feature, config in BASE_FEATURES.items():
            if feature == "Infection_Type":
                base_values[feature] = st.selectbox(
                    config["label"],
                    options=config["options"],
                    format_func=lambda x: config["labels"][x],
                    index=config["default"],
                    key=feature
                )
        
        st.subheader("🩸 血常规")
        for feature, config in BASE_FEATURES.items():
            if feature in ["WBC", "HGB", "HCT", "PLT"]:
                if config["type"] == "numerical":
                    base_values[feature] = st.number_input(
                        config["label"],
                        min_value=float(config["min"]),
                        max_value=float(config["max"]),
                        value=float(config["default"]),
                        step=float(config.get("step", 1)),
                        key=feature
                    )
    
    with col2:
        st.subheader("🧪 凝血功能")
        for feature, config in BASE_FEATURES.items():
            if feature == "Fibrinogen":
                if config["type"] == "numerical":
                    base_values[feature] = st.number_input(
                        config["label"],
                        min_value=float(config["min"]),
                        max_value=float(config["max"]),
                        value=float(config["default"]),
                        step=float(config.get("step", 1)),
                        key=feature
                    )
        
        st.subheader("🧬 肝功能")
        for feature, config in BASE_FEATURES.items():
            if feature in ["ALT", "Globulin", "Albumin"]:
                if config["type"] == "numerical":
                    base_values[feature] = st.number_input(
                        config["label"],
                        min_value=float(config["min"]),
                        max_value=float(config["max"]),
                        value=float(config["default"]),
                        step=float(config.get("step", 1)),
                        key=feature
                    )
        
        st.subheader("🧪 肾功能")
        for feature, config in BASE_FEATURES.items():
            if feature in ["Creatinine", "BUN"]:
                if config["type"] == "numerical":
                    base_values[feature] = st.number_input(
                        config["label"],
                        min_value=float(config["min"]),
                        max_value=float(config["max"]),
                        value=float(config["default"]),
                        step=float(config.get("step", 1)),
                        key=feature
                    )
        
        st.subheader("🫁 呼吸支持")
        for feature, config in BASE_FEATURES.items():
            if feature in ["Oxygen_Concentration", "Partial_Pressure_Of_Oxygen"]:
                if config["type"] == "numerical":
                    base_values[feature] = st.number_input(
                        config["label"],
                        min_value=float(config["min"]),
                        max_value=float(config["max"]),
                        value=float(config["default"]),
                        step=float(config.get("step", 1)),
                        key=feature
                    )
        
        st.subheader("🧪 电解质")
        for feature, config in BASE_FEATURES.items():
            if feature == "Serum_Calcium":
                if config["type"] == "numerical":
                    base_values[feature] = st.number_input(
                        config["label"],
                        min_value=float(config["min"]),
                        max_value=float(config["max"]),
                        value=float(config["default"]),
                        step=float(config.get("step", 1)),
                        key=feature
                    )
    
    st.markdown("---")
    
    # 预测按钮
    if st.button("🔍 开始预测", type="primary", use_container_width=True):
        # 计算派生特征
        derived_values = calculate_derived_features(base_values)
        
        # 准备模型特征
        feature_values = prepare_features_for_model(base_values, derived_values)
        
        # 显示原始特征值（调试用）
        with st.expander("🔧 调试信息（原始特征值）"):
            st.write("原始特征值（未标准化）：")
            for i, (feat, val) in enumerate(zip(MODEL_FEATURES, feature_values)):
                st.write(f"  {i+1:2d}. {feat}: {val}")
        
        # 特征标准化
        if scaler is not None:
            feature_values_scaled = scale_features(feature_values, MODEL_FEATURES, scaler, continuous_vars)
            
            # 显示标准化后的特征值
            with st.expander("🔧 调试信息（标准化后特征值）"):
                st.write("标准化后特征值：")
                for i, (feat, val) in enumerate(zip(MODEL_FEATURES, feature_values_scaled)):
                    st.write(f"  {i+1:2d}. {feat}: {val:.4f}")
                
                # 检查是否有异常值
                nan_count = sum(1 for v in feature_values_scaled if np.isnan(v))
                inf_count = sum(1 for v in feature_values_scaled if np.isinf(v))
                if nan_count > 0 or inf_count > 0:
                    st.error(f"警告: 发现 {nan_count} 个 NaN, {inf_count} 个 Inf!")
            
            feature_values = feature_values_scaled
        
        # 预测
        try:
            X_input = np.array([feature_values])
            
            # 使用基模型预测（校准模型有问题，暂时使用基模型）
            if hasattr(cal_model, 'base_model'):
                # 使用基模型（XGBoost）
                proba = cal_model.base_model.predict_proba(X_input)[0]
                risk_proba = proba[1]
            elif hasattr(cal_model, 'predict_proba'):
                # 备用：使用校准模型
                proba = cal_model.predict_proba(X_input)[0]
                risk_proba = proba[1] if len(proba) > 1 else proba[0]
            else:
                risk_proba = 0.0
            
            # 显示结果
            result_col1, result_col2, result_col3 = st.columns([1, 2, 1])
            
            with result_col2:
                st.subheader("📊 预测结果")
                
                # 风险等级判断
                if risk_proba < 0.1:
                    risk_level = "低风险"
                    risk_color = "green"
                    recommendation = "常规监测"
                elif risk_proba < 0.3:
                    risk_level = "中低风险"
                    risk_color = "blue"
                    recommendation = "加强监测"
                elif risk_proba < 0.5:
                    risk_level = "中风险"
                    risk_color = "orange"
                    recommendation = "预防措施"
                else:
                    risk_level = "高风险"
                    risk_color = "red"
                    recommendation = "积极干预"
                
                # 显示风险概率
                st.markdown(f"""
                <div style="text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;">
                    <h3>出血风险概率</h3>
                    <h1 style="color: {risk_color}; font-size: 48px;">{risk_proba:.1%}</h1>
                    <h4 style="color: {risk_color};">{risk_level}</h4>
                    <p>建议: {recommendation}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # 显示特征信息
                with st.expander("📋 特征详情"):
                    st.write("**基础特征:**")
                    for k, v in base_values.items():
                        if k == "Infection_Type":
                            labels = ["肺部感染", "尿路感染", "消化道感染", "其他/无明确", "混合感染"]
                            st.write(f"  - {k}: {labels[v]} ({v})")
                        else:
                            st.write(f"  - {k}: {v}")
                    
                    st.write("**派生特征:**")
                    for k, v in derived_values.items():
                        st.write(f"  - {k}: {v:.3f}")
                    
                    # 显示 One-Hot 编码后的感染类型
                    infection_dummies = create_infection_dummy(base_values["Infection_Type"])
                    st.write("**感染类型哑变量:**")
                    for k, v in infection_dummies.items():
                        st.write(f"  - {k}: {v}")
                
                # SHAP 解释
                if orig_model is not None:
                    st.subheader("🔍 SHAP 特征贡献解释")
                    
                    shap_values, expected_value = get_shap_explanation(
                        orig_model, feature_values, MODEL_FEATURES
                    )
                    
                    if shap_values is not None:
                        # 创建瀑布图
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # 准备数据
                        if isinstance(expected_value, list):
                            base_val = expected_value[1]
                        else:
                            base_val = expected_value
                        
                        # 使用绝对值排序获取前10个特征
                        shap_abs = np.abs(shap_values[0])
                        top_idx = np.argsort(shap_abs)[-10:][::-1]
                        
                        top_features = [MODEL_FEATURES[i] for i in top_idx]
                        top_shap = [shap_values[0][i] for i in top_idx]
                        top_values = [feature_values[i] for i in top_idx]
                        
                        # 绘制条形图
                        colors = ['red' if x > 0 else 'blue' for x in top_shap]
                        bars = ax.barh(range(len(top_features)), top_shap, color=colors, alpha=0.7)
                        
                        ax.set_yticks(range(len(top_features)))
                        ax.set_yticklabels(top_features)
                        ax.set_xlabel('SHAP Value (impact on model output)')
                        ax.set_title('Top 10 Features by SHAP Value')
                        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                        
                        # 添加数值标签
                        for i, (bar, val) in enumerate(zip(bars, top_values)):
                            width = bar.get_width()
                            ax.text(width, bar.get_y() + bar.get_height()/2, 
                                   f' {val:.2f}', ha='left' if width > 0 else 'right', 
                                   va='center', fontsize=8)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # 解释文本
                        st.info(f"""
                        **SHAP解释说明:**
                        - 红色条形：增加出血风险的特征（正值）
                        - 蓝色条形：降低出血风险的特征（负值）
                        - 条形长度：特征影响的绝对值大小
                        - 基线概率：{base_val:.3f}
                        """)
                    else:
                        st.warning("SHAP 解释不可用，仅显示预测结果")
                        
        except Exception as e:
            st.error(f"预测失败: {e}")
            import traceback
            st.error(traceback.format_exc())


if __name__ == "__main__":
    main()
