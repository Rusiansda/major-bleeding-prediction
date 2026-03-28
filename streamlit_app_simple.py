"""
消化道大出血(Major Bleeding)风险预测 Web 应用
简化版 - 适用于 Streamlit Cloud 部署
"""

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import os

# 设置 matplotlib 字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="消化道大出血风险预测系统", page_icon="🏥", layout="wide")

# ============== 加载模型和标准化器 ==============
@st.cache_resource
def load_model_and_scaler():
    """加载模型和标准化器 - 使用绝对路径"""
    
    # 获取当前文件目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 尝试多个可能的路径
    model_paths = [
        os.path.join(current_dir, "model_XGBoost_calibrated.pkl"),
        "/mount/src/major-bleeding-prediction/deploy/model_XGBoost_calibrated.pkl",
        "/app/deploy/model_XGBoost_calibrated.pkl",
        "./model_XGBoost_calibrated.pkl",
    ]
    
    scaler_paths = [
        os.path.join(current_dir, "lasso_scaler_params.pkl"),
        "/mount/src/major-bleeding-prediction/deploy/lasso_scaler_params.pkl",
        "/app/deploy/lasso_scaler_params.pkl",
        "./lasso_scaler_params.pkl",
    ]
    
    # 加载模型
    model = None
    for path in model_paths:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                cal_model = pickle.load(f)
                model = cal_model.base_model if hasattr(cal_model, 'base_model') else cal_model
            break
    
    if model is None:
        raise FileNotFoundError(f"找不到模型文件，尝试: {model_paths}")
    
    # 加载标准化器
    scaler = None
    for path in scaler_paths:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                scaler = pickle.load(f)
            break
    
    return model, scaler

# ============== 特征定义 ==============
MODEL_FEATURES = [
    'Age', 'Temperature', 'Systolic_BP', 'Respiratory_Rate', 'GCS_Score',
    'WBC', 'HGB', 'HCT', 'PLT', 'Fibrinogen', 'Creatinine', 'BUN',
    'Oxygen_Concentration', 'Partial_Pressure_Of_Oxygen', 'Serum_Calcium',
    'Infection_Type_Urinary', 'Infection_Type_Gastrointestinal',
    'AG_Ratio', 'ALT'
]

# ============== 主应用 ==============
def main():
    st.title("🏥 消化道大出血风险预测系统")
    st.markdown("---")
    
    # 加载模型
    try:
        model, scaler = load_model_and_scaler()
        st.sidebar.success("✅ 模型加载成功")
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        st.stop()
    
    # 侧边栏
    st.sidebar.header("⚙️ 模型信息")
    st.sidebar.info("模型: XGBoost")
    
    # 输入区域
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 基本信息")
        age = st.number_input("年龄 (岁)", 18, 100, 65)
        
        st.subheader("🫀 生命体征")
        temp = st.number_input("体温 (°C)", 32.0, 42.0, 37.0, 0.1)
        sbp = st.number_input("收缩压 (mmHg)", 0, 250, 120)
        rr = st.number_input("呼吸频率 (次/分)", 0, 80, 20)
        gcs = st.number_input("GCS评分", 3, 15, 15)
        
        st.subheader("🦠 感染类型")
        infection = st.selectbox("感染类型", 
            ["肺部感染", "尿路感染", "消化道感染", "其他/无明确", "混合感染"],
            index=0)
        infection_map = {"肺部感染": 0, "尿路感染": 1, "消化道感染": 2, 
                        "其他/无明确": 3, "混合感染": 4}
        infection_type = infection_map[infection]
    
    with col2:
        st.subheader("🩸 血常规")
        wbc = st.number_input("白细胞 (×10⁹/L)", 0.1, 100.0, 8.0, 0.1)
        hgb = st.number_input("血红蛋白 (g/L)", 20, 220, 110)
        hct = st.number_input("红细胞压积 (%)", 10.0, 60.0, 35.0, 0.1)
        plt = st.number_input("血小板 (×10⁹/L)", 1, 1100, 180)
        
        st.subheader("🧪 其他检查")
        fibrinogen = st.number_input("纤维蛋白原 (g/L)", 0.1, 15.0, 3.0, 0.1)
        creatinine = st.number_input("肌酐 (μmol/L)", 5, 1800, 80)
        bun = st.number_input("尿素氮 (mmol/L)", 0.5, 50.0, 6.0, 0.1)
        o2_conc = st.number_input("吸入氧浓度 (%)", 21, 100, 21)
        pao2 = st.number_input("氧分压 (mmHg)", 20, 530, 90)
        ca = st.number_input("血钙 (mmol/L)", 0.5, 5.0, 2.3, 0.1)
        
        # 派生特征
        albumin = st.number_input("白蛋白 (g/L)", 10, 60, 35)
        globulin = st.number_input("球蛋白 (g/L)", 5, 60, 30)
        ag_ratio = albumin / globulin if globulin > 0 else 1.17
        alt = st.number_input("谷丙转氨酶 (U/L)", 1, 6500, 30)
    
    # 创建哑变量
    inf_urinary = 1 if infection_type == 1 else 0
    inf_gi = 1 if infection_type == 2 else 0
    
    # 准备特征
    feature_values = [
        age, temp, sbp, rr, gcs, wbc, hgb, hct, plt, fibrinogen,
        creatinine, bun, o2_conc, pao2, ca, inf_urinary, inf_gi,
        ag_ratio, alt
    ]
    
    # 标准化
    if scaler is not None:
        for i, feat in enumerate(MODEL_FEATURES):
            if feat in scaler and feat not in ['Infection_Type_Urinary', 'Infection_Type_Gastrointestinal']:
                median = scaler[feat]['median']
                iqr = scaler[feat]['iqr']
                if iqr > 0:
                    feature_values[i] = (feature_values[i] - median) / iqr
    
    # 预测按钮
    if st.button("🔍 开始预测", type="primary", use_container_width=True):
        X = np.array([feature_values])
        proba = model.predict_proba(X)[0]
        risk = proba[1]
        
        # 显示结果
        col_r1, col_r2, col_r3 = st.columns([1, 2, 1])
        with col_r2:
            if risk < 0.1:
                level, color, advice = "低风险", "green", "常规监测"
            elif risk < 0.3:
                level, color, advice = "中低风险", "blue", "加强监测"
            elif risk < 0.5:
                level, color, advice = "中风险", "orange", "预防措施"
            else:
                level, color, advice = "高风险", "red", "积极干预"
            
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;">
                <h3>出血风险概率</h3>
                <h1 style="color: {color}; font-size: 48px;">{risk:.1%}</h1>
                <h4 style="color: {color};">{level}</h4>
                <p>建议: {advice}</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
