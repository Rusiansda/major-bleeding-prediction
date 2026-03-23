"""
消化道大出血(Major Bleeding)风险预测 Web 应用
使用 Streamlit 部署 - 基于 CatBoost 机器学习模型
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
    """加载 CatBoost 模型（校准后用于预测，提取原始基学习器用于SHAP）"""
    # 加载校准模型（用于预测）
    cal_model_path = "model_CatBoost_calibrated.pkl"
    with open(cal_model_path, 'rb') as f:
        cal_model = pickle.load(f)
    
    # 提取原始 CatBoost 模型（用于SHAP解释）
    orig_model = None
    try:
        # 方式1: 自定义 CalibratedModel 类 (from utils.py)
        if hasattr(cal_model, 'base_model'):
            orig_model = cal_model.base_model
            st.sidebar.info("使用 CalibratedModel.base_model 提取原始模型")
        # 方式2: sklearn CalibratedClassifierCV
        elif hasattr(cal_model, 'calibrated_classifiers_') and len(cal_model.calibrated_classifiers_) > 0:
            orig_model = cal_model.calibrated_classifiers_[0].estimator
            st.sidebar.info("使用 CalibratedClassifierCV 提取原始模型")
        # 方式3: 直接就是原始模型 (CatBoost)
        elif hasattr(cal_model, 'predict_proba') and hasattr(cal_model, 'get_params'):
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
        scaler_path = "robust_scaler.pkl"
        with open(scaler_path, 'rb') as f:
            scaler_data = pickle.load(f)
        return scaler_data['scaler'], scaler_data['continuous_vars']
    except Exception as e:
        st.sidebar.warning(f"加载标准化器失败: {e}")
        return None, []


def scale_features(feature_values, feature_names, scaler, continuous_vars):
    """对连续变量进行标准化"""
    if scaler is None or len(continuous_vars) == 0:
        return feature_values
    
    # 创建包含所有原始特征的 DataFrame（用0填充缺失的特征）
    # scaler 是在更多特征上训练的，我们需要提供完整的特征集
    all_features_df = pd.DataFrame([feature_values], columns=feature_names)
    
    # 找出需要标准化的列（当前20个特征中存在的连续变量）
    cols_to_scale = [c for c in continuous_vars if c in feature_names]
    
    if len(cols_to_scale) == 0:
        return feature_values
    
    # 提取需要标准化的列的值
    values_to_scale = all_features_df[cols_to_scale].values
    
    # 获取这些列在原始 continuous_vars 中的索引
    col_indices = [continuous_vars.index(c) for c in cols_to_scale]
    
    # 手动标准化：使用 scaler 的中心和缩放参数
    # RobustScaler: transformed = (X - median) / IQR
    medians = scaler.center_[col_indices] if hasattr(scaler, 'center_') else scaler.quantile_[col_indices]
    scales = scaler.scale_[col_indices]
    
    # 避免除以0
    scales = np.where(scales == 0, 1, scales)
    
    scaled_values = (values_to_scale - medians) / scales
    
    # 将标准化后的值放回原数组
    result = feature_values.copy()
    for i, col_name in enumerate(cols_to_scale):
        idx = feature_names.index(col_name)
        result[idx] = scaled_values[0][i]
    
    return result

# ============== 特征定义 ==============
# 基础输入特征（15个）- 对应最终模型的输入变量
BASE_FEATURES = {
    "Age": {"label": "年龄 (岁)", "type": "numerical", "min": 18, "max": 100, "default": 60, "step": 1},
    "Temperature": {"label": "体温 (°C)", "type": "numerical", "min": 32.0, "max": 42.0, "default": 37.0, "step": 0.1},
    "Systolic_BP": {"label": "收缩压 (mmHg)", "type": "numerical", "min": 0, "max": 250, "default": 120, "step": 1},
    "GCS_Score": {"label": "GCS评分 (3-15)", "type": "numerical", "min": 3, "max": 15, "default": 15, "step": 1},
    "WBC": {"label": "白细胞 (×10⁹/L)", "type": "numerical", "min": 0.1, "max": 100, "default": 8.0, "step": 0.1},
    "HGB": {"label": "血红蛋白 (g/L)", "type": "numerical", "min": 20, "max": 220, "default": 110, "step": 1},
    "HCT": {"label": "红细胞压积 (%)", "type": "numerical", "min": 10, "max": 60, "default": 35, "step": 0.1},
    "PLT": {"label": "血小板 (×10⁹/L)", "type": "numerical", "min": 1, "max": 1100, "default": 180, "step": 1},
    "INR": {"label": "国际标准化比值 (INR)", "type": "numerical", "min": 0.5, "max": 15, "default": 1.0, "step": 0.1},
    "Fibrinogen": {"label": "纤维蛋白原 (g/L)", "type": "numerical", "min": 0.1, "max": 15, "default": 3.0, "step": 0.1},
    "ALT": {"label": "谷丙转氨酶 (U/L)", "type": "numerical", "min": 1, "max": 6500, "default": 30, "step": 1},
    "Creatinine": {"label": "肌酐 (μmol/L)", "type": "numerical", "min": 5, "max": 1800, "default": 80, "step": 1},
    "BUN": {"label": "尿素氮 (mmol/L)", "type": "numerical", "min": 0.5, "max": 50, "default": 6.0, "step": 0.1},
    "Oxygen_Concentration": {"label": "吸入氧浓度 (%)", "type": "numerical", "min": 21, "max": 100, "default": 21, "step": 1},
    "Partial_Pressure_Of_Oxygen": {"label": "氧分压 (mmHg)", "type": "numerical", "min": 20, "max": 530, "default": 90, "step": 1},
    "Serum_Calcium": {"label": "血钙 (mmol/L)", "type": "numerical", "min": 0.5, "max": 5, "default": 2.3, "step": 0.1},
}

# 模型特征列表（固定顺序，18个特征）- 必须与训练时顺序一致
# 最终模型特征: Age, Temperature, Systolic_BP, GCS_Score, WBC, HGB, HCT, PLT, 
#               INR, Fibrinogen, ALT, Creatinine, BUN, Oxygen_Concentration, 
#               Partial_Pressure_Of_Oxygen, Serum_Calcium, AG_Ratio, PT_INR_Product
MODEL_FEATURES = [
    'Age', 'Temperature', 'Systolic_BP', 'GCS_Score', 'WBC', 'HGB', 
    'HCT', 'PLT', 'INR', 'Fibrinogen', 'ALT', 'Creatinine', 'BUN', 
    'Oxygen_Concentration', 'Partial_Pressure_Of_Oxygen', 'Serum_Calcium', 
    'AG_Ratio', 'PT_INR_Product'
]

# ============== 特征工程函数 ==============
def calculate_derived_features(base_values):
    """计算派生特征"""
    derived = {}
    
    # AG_Ratio: 白球比 = 白蛋白 / 球蛋白
    # 使用默认值计算，因为最终模型中包含AG_Ratio
    albumin = 35  # 白蛋白默认值
    globulin = 30  # 球蛋白默认值
    if globulin > 0:
        derived["AG_Ratio"] = albumin / globulin
    else:
        derived["AG_Ratio"] = 1.17  # 默认值
    
    # PT_INR_Product: PT * INR 的乘积特征
    # PT (凝血酶原时间) 使用默认值
    pt_default = 12  # PT 正常值约 11-14 秒
    inr = base_values.get("INR", 1.0)
    derived["PT_INR_Product"] = pt_default * inr
    
    return derived

def prepare_features_for_model(base_values, derived_values):
    """准备模型输入特征（按MODEL_FEATURES顺序）"""
    features = {}
    # 按训练时的特征顺序准备（18个特征）
    features["Age"] = base_values["Age"]
    features["Temperature"] = base_values["Temperature"]
    features["Systolic_BP"] = base_values["Systolic_BP"]
    features["GCS_Score"] = base_values["GCS_Score"]
    features["WBC"] = base_values["WBC"]
    features["HGB"] = base_values["HGB"]
    features["HCT"] = base_values["HCT"]
    features["PLT"] = base_values["PLT"]
    features["INR"] = base_values["INR"]
    features["Fibrinogen"] = base_values["Fibrinogen"]
    features["ALT"] = base_values["ALT"]
    features["Creatinine"] = base_values["Creatinine"]
    features["BUN"] = base_values["BUN"]
    features["Oxygen_Concentration"] = base_values["Oxygen_Concentration"]
    features["Partial_Pressure_Of_Oxygen"] = base_values["Partial_Pressure_Of_Oxygen"]
    features["Serum_Calcium"] = base_values["Serum_Calcium"]
    features["AG_Ratio"] = derived_values["AG_Ratio"]
    features["PT_INR_Product"] = derived_values["PT_INR_Product"]
    
    return [features[f] for f in MODEL_FEATURES]

# ============== 主应用 ==============
def main():
    # 标题
    st.title("🏥 消化道大出血风险预测系统")
    st.markdown("基于 CatBoost 机器学习模型 | 验证集 AUC = 0.834 (95% CI: 0.782-0.884)")
    st.markdown("---")
    
    # 加载模型和标准化器
    try:
        cal_model, orig_model = load_model()
        scaler, continuous_vars = load_scaler()
        st.sidebar.success("✅ 模型加载成功")
        if scaler is not None:
            st.sidebar.info(f"✅ 标准化器加载成功 ({len(continuous_vars)}个连续变量)")
        if orig_model is not None:
            st.sidebar.info("SHAP解释: 可用")
        else:
            st.sidebar.info("SHAP解释: 仅预测可用")
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        st.info("请确保已运行训练代码并生成模型文件: 生成的文件_major_bleeding/model_CatBoost_calibrated.pkl")
        return
    
    # 侧边栏信息
    st.sidebar.header("⚙️ 模型信息")
    st.sidebar.info("当前模型: CatBoost")
    st.sidebar.info("验证集性能:\n- AUC: 0.834\n- 敏感度: 34.1%\n- 特异度: 95.5%\n- NPV: 98.6%")
    
    with st.sidebar.expander("📖 使用说明"):
        st.markdown("""
        1. 输入患者的生命体征和实验室检查结果
        2. 系统自动计算派生特征（白球比、脉压）
        3. 点击"开始预测"获取风险概率
        4. 查看 SHAP 图了解特征贡献
        
        **注意**: 本模型基于20个临床特征构建，包括：
        - 基础特征：体温、血压、呼吸频率、GCS评分等
        - 实验室指标：血常规、肝肾功能、凝血功能等
        - 派生特征：白球比、脉压（自动计算）
        """)
    
    # 主面板 - 输入区域
    col1, col2 = st.columns(2)
    
    base_values = {}
    
    with col1:
        st.subheader("🫀 生命体征")
        for feature, config in BASE_FEATURES.items():
            if feature in ["Temperature", "Systolic_BP", "GCS_Score", "Age", "Oxygen_Concentration"]:
                if config["type"] == "numerical":
                    base_values[feature] = st.number_input(
                        config["label"],
                        min_value=float(config["min"]),
                        max_value=float(config["max"]),
                        value=float(config["default"]),
                        step=float(config.get("step", 1)),
                        key=feature
                    )
        
        st.subheader("🧬 合并症")
        for feature, config in BASE_FEATURES.items():
            if feature in ["Urinary_Tract_Infection", "Malignancy_Or_Immunosuppression"]:
                options = config["options"]
                labels = config.get("labels", [str(o) for o in options])
                idx = options.index(config["default"]) if config["default"] in options else 0
                selected_label = st.selectbox(
                    config["label"],
                    options=labels,
                    index=idx,
                    key=feature
                )
                base_values[feature] = options[labels.index(selected_label)]
    
    with col2:
        st.subheader("🧪 实验室检查")
        for feature, config in BASE_FEATURES.items():
            if feature not in base_values:
                if config["type"] == "numerical":
                    base_values[feature] = st.number_input(
                        config["label"],
                        min_value=float(config["min"]),
                        max_value=float(config["max"]),
                        value=float(config["default"]),
                        step=float(config.get("step", 1)),
                        key=feature
                    )
    
    # 显示派生特征
    st.markdown("---")
    derived_values = calculate_derived_features(base_values)
    
    with st.expander("📊 派生特征（自动计算）"):
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.metric("白球比 (AG Ratio)", f"{derived_values['AG_Ratio']:.2f}",
                     help="白蛋白/球蛋白比值（默认值计算）")
        with col_d2:
            st.metric("PT-INR乘积", f"{derived_values['PT_INR_Product']:.1f}",
                     help="凝血酶原时间 × INR（PT默认12秒）")
    
    # 预测按钮
    st.markdown("---")
    if st.button("🚀 开始预测", type="primary", use_container_width=True):
        # 准备特征
        feature_values = prepare_features_for_model(base_values, derived_values)
        
        # 对特征进行标准化（与训练时一致）
        feature_values_scaled = scale_features(feature_values, MODEL_FEATURES, scaler, continuous_vars)
        
        features_array = np.array([feature_values_scaled])
        input_df = pd.DataFrame([feature_values_scaled], columns=MODEL_FEATURES)
        
        # 调试信息：显示输入特征值
        with st.expander("🔍 调试信息：输入特征值"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**原始值**")
                debug_df_raw = pd.DataFrame({
                    '特征名': MODEL_FEATURES,
                    '值': feature_values
                })
                st.dataframe(debug_df_raw, use_container_width=True)
            with col2:
                st.markdown("**标准化后**")
                debug_df_scaled = pd.DataFrame({
                    '特征名': MODEL_FEATURES,
                    '值': feature_values_scaled
                })
                st.dataframe(debug_df_scaled, use_container_width=True)
            st.write(f"特征数组形状: {features_array.shape}")
            st.write(f"原始值范围: [{min(feature_values):.2f}, {max(feature_values):.2f}]")
            st.write(f"标准化后范围: [{min(feature_values_scaled):.2f}, {max(feature_values_scaled):.2f}]")
        
        with st.spinner("正在进行预测分析..."):
            try:
                # ========== 预测（使用校准后模型） ==========
                prediction_proba = cal_model.predict_proba(features_array)[0]
                prediction_class = cal_model.predict(features_array)[0]
                
                # 调试信息：显示原始预测概率
                with st.expander("🔍 调试信息：原始预测输出"):
                    st.write(f"predict_proba 输出: {prediction_proba}")
                    st.write(f"predict 输出: {prediction_class}")
                
                risk_probability = prediction_proba[1] * 100
                
                # ========== 显示预测结果 ==========
                col_r1, col_r2, col_r3 = st.columns([1, 2, 1])
                
                with col_r2:
                    st.markdown("### 🎯 预测结果")
                    
                    # 风险等级判断
                    if risk_probability < 5:
                        risk_level = "低风险"
                        risk_color = "green"
                        bg_color = "#e8f5e9"
                    elif risk_probability < 15:
                        risk_level = "中风险"
                        risk_color = "orange"
                        bg_color = "#fff3e0"
                    else:
                        risk_level = "高风险"
                        risk_color = "red"
                        bg_color = "#ffebee"
                    
                    # 显示概率
                    st.markdown(f"""
                    <div style="text-align: center; padding: 30px; background-color: {bg_color}; border-radius: 15px; border: 2px solid {risk_color};">
                        <h1 style="color: {risk_color}; margin: 0; font-size: 48px;">{risk_probability:.2f}%</h1>
                        <p style="font-size: 20px; color: {risk_color}; margin-top: 10px;">消化道大出血风险概率</p>
                        <p style="font-size: 24px; font-weight: bold; color: {risk_color};">{risk_level}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 风险解读
                    if risk_color == "red":
                        st.error("⚠️ **高风险提示**: 该患者发生消化道大出血的风险较高，建议立即采取预防措施并密切监测。")
                    elif risk_color == "orange":
                        st.warning("⚡ **中风险提示**: 该患者存在一定程度的出血风险，建议加强观察和评估。")
                    else:
                        st.success("✅ **低风险**: 该患者发生消化道大出血的风险相对较低，但仍需常规监测。")
                
                # ========== SHAP 解释 ==========
                st.markdown("---")
                st.subheader("🧠 SHAP 特征重要性解释")
                
                try:
                    # 使用原始 CatBoost 模型计算 SHAP（TreeExplainer 支持树模型）
                    if orig_model is not None:
                        explainer = shap.TreeExplainer(orig_model)
                        shap_values = explainer.shap_values(input_df)
                    else:
                        raise ValueError("原始模型不可用")
                    
                    # 对于二分类，shap_values 是列表 [class_0_shap, class_1_shap]
                    if isinstance(shap_values, list):
                        shap_values_class = shap_values[1][0]  # 正类的 SHAP 值
                        expected_value = float(explainer.expected_value[1]) if isinstance(explainer.expected_value, (list, np.ndarray)) else float(explainer.expected_value)
                    else:
                        shap_values_class = shap_values[0]
                        expected_value = float(explainer.expected_value)
                    
                    # 保存到session_state防止重绘
                    st.session_state['shap_values'] = shap_values_class
                    st.session_state['expected_value'] = expected_value
                    st.session_state['feature_values'] = feature_values
                    
                except Exception as e:
                    st.error(f"SHAP 计算失败: {e}")
                    st.info("继续显示预测结果...")
                    st.session_state.pop('shap_values', None)
                
                # 使用保存的SHAP值绘制图表
                if 'shap_values' in st.session_state:
                    shap_values_class = st.session_state['shap_values']
                    expected_value = st.session_state['expected_value']
                    feature_values = st.session_state['feature_values']
                    
                    # 创建 SHAP 力图
                    st.markdown("**SHAP 力图 (Force Plot)**")
                    st.info("🔴 红色 = 增加风险 | 🔵 蓝色 = 降低风险")
                    
                    # 使用 matplotlib 绘制力图（避免 HTML 组件问题）
                    fig_force, ax_force = plt.subplots(figsize=(16, 3))
                    
                    # 排序 SHAP 值
                    sorted_indices = np.argsort(np.abs(shap_values_class))[::-1]
                    top_n = min(10, len(shap_values_class))  # 显示前10个特征
                    
                    # 获取基线值和最终值
                    base_val = expected_value
                    final_val = base_val + np.sum(shap_values_class)
                    
                    # 绘制基线
                    ax_force.axvline(x=base_val, color='black', linestyle='--', linewidth=1, label=f'Base: {base_val:.3f}')
                    ax_force.axvline(x=final_val, color='red', linestyle='--', linewidth=1, label=f'Output: {final_val:.3f}')
                    
                    # 绘制特征贡献
                    y_pos = np.arange(top_n)
                    colors = ['#ff0051' if shap_values_class[i] > 0 else '#008bfb' for i in sorted_indices[:top_n]]
                    contributions = [shap_values_class[i] for i in sorted_indices[:top_n]]
                    feature_names_short = [MODEL_FEATURES[i] for i in sorted_indices[:top_n]]
                    
                    # 绘制水平条形图
                    bars = ax_force.barh(y_pos, contributions, color=colors, alpha=0.8)
                    ax_force.set_yticks(y_pos)
                    ax_force.set_yticklabels(feature_names_short)
                    ax_force.set_xlabel('SHAP Value (impact on risk)')
                    ax_force.set_title('SHAP Force Plot - Top Contributing Features')
                    ax_force.invert_yaxis()
                    ax_force.legend(loc='lower right')
                    ax_force.grid(axis='x', alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig_force)
                    plt.close()
                    
                    # 创建 SHAP 条形图
                    st.markdown("**特征贡献度排序 (Bar Plot)**")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.bar_plot(
                        shap_values_class,
                        feature_values_scaled,  # 使用标准化后的特征值
                        MODEL_FEATURES,
                        show=False
                    )
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    # 创建 SHAP 瀑布图
                    st.markdown("**瀑布图 (Waterfall Plot)**")
                    fig2, ax2 = plt.subplots(figsize=(12, 8))
                    explanation = shap.Explanation(
                        values=shap_values_class,
                        base_values=expected_value,
                        data=feature_values_scaled,  # 使用标准化后的特征值
                        feature_names=MODEL_FEATURES
                    )
                    shap.waterfall_plot(explanation, show=False)
                    plt.tight_layout()
                    st.pyplot(fig2)
                    plt.close()
                    
                    st.caption("从基线值（模型平均预测）到最终预测值的累积贡献过程")
                else:
                    st.info("SHAP解释不可用，但预测结果已生成。")
                    
                # ========== 保存结果 ==========
                result_data = {
                    "预测时间": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "大出血风险": f"{risk_probability:.2f}%",
                    "风险等级": risk_level,
                    **{k: v for k, v in zip(MODEL_FEATURES, feature_values)}
                }
                
                result_df = pd.DataFrame([result_data])
                st.download_button(
                    label="📥 下载预测结果",
                    data=result_df.to_csv(index=False, encoding='utf-8-sig'),
                    file_name=f"major_bleeding_prediction_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"❌ 预测过程中出现错误: {str(e)}")
                st.exception(e)
    
    # 页脚
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🏥 消化道大出血风险预测系统 | 基于 CatBoost 机器学习模型</p>
        <p style="font-size: 12px;">⚠️ 本系统仅供学术研究使用，预测结果仅供参考，不构成医疗建议</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
