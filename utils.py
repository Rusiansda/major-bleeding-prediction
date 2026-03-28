"""
工具模块 - 包含所有可复用的函数
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# 设置matplotlib非交互式后端，避免tkinter多线程问题
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, roc_curve, auc, roc_auc_score, 
                             accuracy_score, precision_score, recall_score, f1_score,
                             log_loss, brier_score_loss)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import shap

# Set English font for all plots to avoid encoding issues on Linux
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def apply_rounding_rules(data, rules):
    """
    应用四舍五入规则
    
    Parameters:
    -----------
    data : DataFrame
        待处理的数据
    rules : dict
        四舍五入规则，格式: {小数位数: [变量列表]}
    
    Returns:
    --------
    DataFrame
        处理后的数据
    """
    data_rounded = data.copy()
    for decimals, columns in rules.items():
        existing_cols = [c for c in columns if c in data_rounded.columns]
        if decimals == 0:
            for col in existing_cols:
                data_rounded[col] = data_rounded[col].round(0).astype('Int64', errors='ignore')
        else:
            data_rounded[existing_cols] = data_rounded[existing_cols].round(decimals)
    return data_rounded


def add_derived_variables(data):
    """
    添加衍生变量 - 基于临床文献和病理生理机制
    
    Parameters:
    -----------
    data : DataFrame
        原始数据
    
    Returns:
    --------
    DataFrame
        添加衍生变量后的数据
    """
    data_derived = data.copy()
    derived_count = 0
    
    # 1. 氧合指数 (PaO2/FiO2 Ratio)
    if 'Partial_Pressure_Of_Oxygen' in data_derived.columns and 'Oxygen_Concentration' in data_derived.columns:
        data_derived['PaO2_FiO2_Ratio'] = (
            data_derived['Partial_Pressure_Of_Oxygen'] / 
            (data_derived['Oxygen_Concentration'] / 100 + 1e-6)
        ).round(1).clip(lower=0, upper=1000)
        print("  [OK] PaO2/FiO2 Ratio - Berlin Definition (JAMA 2012)")
        derived_count += 1
    
    # 2. BUN/Cr比值
    if 'BUN' in data_derived.columns and 'Creatinine' in data_derived.columns:
        data_derived['BUN_Crea_Ratio'] = (
            data_derived['BUN'] / (data_derived['Creatinine'] + 1e-6)
        ).round(1).clip(lower=0, upper=500)
        print("  [OK] BUN/Cr Ratio - Srygley et al. (JAMA 2012)")
        derived_count += 1
    
    # 3. 白蛋白/球蛋白比值 (A/G Ratio)
    if 'Albumin' in data_derived.columns and 'Globulin' in data_derived.columns:
        data_derived['AG_Ratio'] = (
            data_derived['Albumin'] / (data_derived['Globulin'] + 1e-6)
        ).round(2).clip(lower=0, upper=5)
        print("  [OK] A/G Ratio - Duran et al. (Critical Care 2014)")
        derived_count += 1
    
    # 4. 平均动脉压 MAP
    if 'Systolic_BP' in data_derived.columns and 'Diastolic_BP' in data_derived.columns:
        data_derived['MAP'] = (
            (data_derived['Systolic_BP'] + 2 * data_derived['Diastolic_BP']) / 3
        ).round(0).clip(lower=30, upper=200)
        print("  [OK] MAP - Surviving Sepsis Campaign (Crit Care 2013)")
        derived_count += 1
    
    # 5. 休克指数 (Shock Index = HR/SBP)
    if 'Heart_Rate' in data_derived.columns and 'Systolic_BP' in data_derived.columns:
        data_derived['Shock_Index'] = (
            data_derived['Heart_Rate'] / (data_derived['Systolic_BP'] + 1e-6)
        ).round(2).clip(lower=0, upper=5)
        print("  [OK] Shock Index - Birkhahn et al. (Am J Emerg Med 2005)")
        derived_count += 1
    
    # 6. 脉压 (Pulse Pressure)
    if 'Systolic_BP' in data_derived.columns and 'Diastolic_BP' in data_derived.columns:
        data_derived['Pulse_Pressure'] = (
            data_derived['Systolic_BP'] - data_derived['Diastolic_BP']
        ).round(0).clip(lower=10, upper=150)
        print("  [OK] Pulse Pressure - Blacher et al. (Hypertension 2000)")
        derived_count += 1
    
    # 7. 血小板与淋巴细胞比值 (PLR)
    if 'PLT' in data_derived.columns and 'WBC' in data_derived.columns:
        estimated_lymphocytes = data_derived['WBC'] * 0.3
        data_derived['PLR'] = (
            data_derived['PLT'] / (estimated_lymphocytes + 1e-6)
        ).round(1).clip(lower=0, upper=1000)
        print("  [OK] PLR - Guo et al. (Medicine 2019)")
        derived_count += 1
    
    print(f"\n  📚 Total derived variables created: {derived_count}")
    return data_derived


def calculate_net_benefit(y_true, y_pred_proba, threshold):
    """计算DCA的net benefit"""
    y_pred = (y_pred_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    n = len(y_true)
    net_benefit = (tp / n) - (fp / n) * (threshold / (1 - threshold))
    return net_benefit


def plot_dca(y_true, pred_probs_dict, title, save_path):
    """绘制DCA曲线（智能自适应坐标轴版本）"""
    y_true_array = y_true.values if hasattr(y_true, 'values') else np.array(y_true)
    prevalence = y_true_array.sum() / len(y_true_array)
    
    # 根据阳性率动态确定阈值范围
    # 对于低发病率疾病，重点关注低阈值区域
    if prevalence < 0.05:  # 发病率<5%
        max_threshold = min(0.20, prevalence * 5)
        thresholds = np.linspace(0.001, max_threshold, 100)
    elif prevalence < 0.15:  # 发病率5-15%
        max_threshold = min(0.30, prevalence * 3)
        thresholds = np.linspace(0.001, max_threshold, 100)
    else:
        max_threshold = 0.50
        thresholds = np.linspace(0.01, max_threshold, 100)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # ========== 左图：实际关注的阈值范围（详细视图） ==========
    ax1 = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(pred_probs_dict)))
    linestyles = ['-', '--', '-.', ':', '-']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    all_net_benefits = []
    
    for idx, (name, y_proba) in enumerate(pred_probs_dict.items()):
        net_benefits = [calculate_net_benefit(y_true_array, y_proba, t) for t in thresholds]
        all_net_benefits.extend(net_benefits)
        
        ax1.plot(thresholds, net_benefits, 
                label=name, 
                linewidth=2.5,
                linestyle=linestyles[idx % len(linestyles)],
                color=colors[idx],
                alpha=0.85)
    
    # Treat all
    treat_all = [(y_true_array.sum() / len(y_true_array)) - 
                 (1 - y_true_array.sum() / len(y_true_array)) * (t / (1 - t)) 
                 for t in thresholds]
    all_net_benefits.extend(treat_all)
    ax1.plot(thresholds, treat_all, 'k--', label='Treat All', linewidth=2.5, alpha=0.7)
    
    # Treat none
    ax1.axhline(y=0, color='gray', linestyle=':', label='Treat None', linewidth=2.5, alpha=0.7)
    
    # 设置Y轴范围为固定范围 -0.025 到 0.025
    ax1.set_xlim([0, max_threshold])
    ax1.set_ylim([-0.025, 0.025])
    ax1.set_xlabel('Threshold Probability', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Net Benefit', fontsize=12, fontweight='bold')
    ax1.set_title(f'Detailed View (Key Threshold Range)\nPrevalence = {prevalence:.1%}', 
                  fontsize=13, fontweight='bold', pad=15)
    ax1.legend(fontsize=9, loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 添加临床意义标注
    ax1.axvline(x=prevalence, color='red', linestyle='-.', alpha=0.5, linewidth=1.5)
    ax1.text(prevalence, 0.022, f'Prevalence\n{prevalence:.1%}', 
             ha='center', fontsize=9, color='red', alpha=0.8)
    
    # ========== 右图：完整范围（0-0.5） ==========
    ax2 = axes[1]
    thresholds_full = np.linspace(0.01, 0.50, 100)
    
    for idx, (name, y_proba) in enumerate(pred_probs_dict.items()):
        net_benefits = [calculate_net_benefit(y_true_array, y_proba, t) for t in thresholds_full]
        ax2.plot(thresholds_full, net_benefits, 
                label=name, 
                linewidth=2,
                linestyle=linestyles[idx % len(linestyles)],
                color=colors[idx],
                alpha=0.8)
    
    treat_all_full = [(y_true_array.sum() / len(y_true_array)) - 
                      (1 - y_true_array.sum() / len(y_true_array)) * (t / (1 - t)) 
                      for t in thresholds_full]
    ax2.plot(thresholds_full, treat_all_full, 'k--', label='Treat All', linewidth=2, alpha=0.7)
    ax2.axhline(y=0, color='gray', linestyle=':', label='Treat None', linewidth=2, alpha=0.7)
    
    ax2.set_xlim([0, 0.5])
    ax2.set_ylim([-0.025, 0.025])
    ax2.set_xlabel('Threshold Probability', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Net Benefit', fontsize=12, fontweight='bold')
    ax2.set_title('Full Range View (0-50%)', fontsize=13, fontweight='bold', pad=15)
    ax2.legend(fontsize=9, loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def evaluate_model_comprehensive(model, X, y, data_name="Dataset"):
    """
    综合评估模型性能
    
    Returns:
    --------
    result_dict : dict
        包含各项评估指标的字典
    y_pred_proba : array
        预测概率
    y_pred : array
        预测分类
    cm : array
        混淆矩阵
    """
    from sklearn.metrics import roc_curve
    
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # 转换为numpy数组以避免索引问题
    y_array = y.values if hasattr(y, 'values') else y
    y_pred_proba_array = y_pred_proba if isinstance(y_pred_proba, np.ndarray) else np.array(y_pred_proba)
    
    # 使用F1分数最大化找到最优阈值
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_array, y_pred_proba_array)
    # 移除最后一个阈值（对应recall=0）
    thresholds = thresholds[:-1]
    precisions = precisions[:-1]
    recalls = recalls[:-1]
    # 计算F1分数
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # 使用最优阈值进行预测
    y_pred = (y_pred_proba_array >= optimal_threshold).astype(int)
    
    # AUC及置信区间
    auc_score = roc_auc_score(y_array, y_pred_proba_array)
    n_bootstraps = 1000
    rng = np.random.RandomState(42)
    bootstrapped_scores = []
    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_array), len(y_array))
        if len(np.unique(y_array[indices])) < 2:
            continue
        score = roc_auc_score(y_array[indices], y_pred_proba_array[indices])
        bootstrapped_scores.append(score)
    
    ci_lower = np.percentile(bootstrapped_scores, 2.5)
    ci_upper = np.percentile(bootstrapped_scores, 97.5)
    
    # 其他指标
    acc = accuracy_score(y_array, y_pred)
    sensitivity = recall_score(y_array, y_pred, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(y_array, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = precision_score(y_array, y_pred, zero_division=0)
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1 = f1_score(y_array, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_array, y_pred)
    
    return {
        'Dataset': data_name,
        'AUC': auc_score,
        'AUC_CI': f"({ci_lower:.3f}-{ci_upper:.3f})",
        'Optimal_Threshold': optimal_threshold,
        'Accuracy': acc,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'PPV': ppv,
        'NPV': npv,
        'F1_Score': f1
    }, y_pred_proba, y_pred, cm


def plot_confusion_matrices(cms_dict, save_path, dataset_name=""):
    """Plot confusion matrices with English labels"""
    n_models = len(cms_dict)
    ncols = 3
    nrows = (n_models + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    
    # Ensure axes is always a 1D list
    if isinstance(axes, np.ndarray):
        axes = axes.ravel()
    else:
        axes = [axes]
    
    for idx, (name, cm) in enumerate(cms_dict.items()):
        if idx >= len(axes):
            break
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar_kws={'shrink': 0.8})
        ax.set_xlabel('Predicted Label', fontsize=11)
        ax.set_ylabel('True Label', fontsize=11)
        ax.set_title(f'{name}\n{dataset_name}', fontsize=12, fontweight='bold')
        
        # Set English tick labels for binary classification
        ax.set_xticklabels(['No Bleeding (0)', 'Bleeding (1)'], fontsize=10)
        ax.set_yticklabels(['No Bleeding (0)', 'Bleeding (1)'], fontsize=10, rotation=0)
    
    for idx in range(len(cms_dict), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curves(y_true_dict, pred_probs_dict, save_path, dataset_names=None):
    """批量绘制ROC曲线"""
    if dataset_names is None:
        dataset_names = list(y_true_dict.keys())
    
    n_datasets = len(dataset_names)
    fig, axes = plt.subplots(1, n_datasets, figsize=(7*n_datasets, 6))
    if n_datasets == 1:
        axes = [axes]
    
    for idx, dataset_name in enumerate(dataset_names):
        ax = axes[idx]
        y_true = y_true_dict[dataset_name]
        
        for model_name, y_proba in pred_probs_dict[dataset_name].items():
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            auc_score = roc_auc_score(y_true, y_proba)
            auc_truncated = int(auc_score * 100) / 100  # 保留两位小数，截断不四舍五入
            ax.plot(fpr, tpr, label=f'{model_name} (AUC={auc_truncated:.2f})', linewidth=2)
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=2)
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'ROC Curves - {dataset_name}', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_calibration_curves(y_true, pred_probs_dict, save_path, dataset_name=""):
    """绘制校准曲线（智能自适应坐标轴，每个模型单独子图）"""
    y_true_array = y_true.values if hasattr(y_true, 'values') else np.array(y_true)
    
    # 分析预测概率分布，确定合适的坐标轴范围
    all_probs = np.concatenate([y_proba for y_proba in pred_probs_dict.values()])
    prob_95th = np.percentile(all_probs, 95)
    prob_max = np.percentile(all_probs, 99.5)
    
    # 根据概率分布确定显示范围
    if prob_95th < 0.1:  # 大多数概率集中在低区域
        xlim_main = 0.3
        ylim_main = 0.3
    elif prob_95th < 0.3:
        xlim_main = 0.3
        ylim_main = 0.3
    else:
        xlim_main = 1.0
        ylim_main = 1.0
    
    n_models = len(pred_probs_dict)
    ncols = min(3, n_models)
    nrows = (n_models + ncols - 1) // ncols
    
    # 每个模型绘制两个子图：详细视图 + 完整视图
    fig, axes = plt.subplots(nrows, ncols * 2, figsize=(7*ncols*2, 5*nrows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.ravel()
    
    for idx, (model_name, y_proba) in enumerate(pred_probs_dict.items()):
        if idx >= n_models:
            break
        
        # 详细视图（左列）
        ax_detail = axes[idx * 2]
        prob_true, prob_pred = calibration_curve(y_true_array, y_proba, n_bins=10, strategy='quantile')
        
        ax_detail.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2, alpha=0.7)
        ax_detail.plot(prob_pred, prob_true, '-', label=model_name, 
                      linewidth=2.5, color='steelblue', alpha=0.9)
        
        # 添加样本量标注
        for i, (x, y) in enumerate(zip(prob_pred, prob_true)):
            if not (np.isnan(x) or np.isnan(y)):
                ax_detail.annotate(f'{i+1}', (x, y), textcoords="offset points", 
                                  xytext=(5, 5), fontsize=8, alpha=0.6)
        
        ax_detail.set_xlabel('Predicted Probability', fontsize=11, fontweight='bold')
        ax_detail.set_ylabel('Observed Probability', fontsize=11, fontweight='bold')
        ax_detail.set_title(f'{model_name}\nDetailed View (Key Range)', fontsize=11, fontweight='bold')
        ax_detail.legend(loc='upper left', fontsize=9)
        ax_detail.grid(alpha=0.3, linestyle='--')
        ax_detail.set_xlim([0, xlim_main])
        ax_detail.set_ylim([0, ylim_main])
        
        # 计算校准指标（不显示在图上）
        from sklearn.metrics import brier_score_loss
        brier = brier_score_loss(y_true_array, y_proba)
        
        # 完整视图（右列）
        ax_full = axes[idx * 2 + 1]
        prob_true_full, prob_pred_full = calibration_curve(y_true_array, y_proba, n_bins=10, strategy='uniform')
        
        ax_full.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2, alpha=0.7)
        ax_full.plot(prob_pred_full, prob_true_full, '-', label=model_name, 
                    linewidth=2.5, color='coral', alpha=0.9)
        
        ax_full.set_xlabel('Predicted Probability', fontsize=11, fontweight='bold')
        ax_full.set_ylabel('Observed Probability', fontsize=11, fontweight='bold')
        ax_full.set_title(f'{model_name}\nFull Range View', fontsize=11, fontweight='bold')
        ax_full.legend(loc='upper left', fontsize=9)
        ax_full.grid(alpha=0.3, linestyle='--')
        ax_full.set_xlim([0, 1])
        ax_full.set_ylim([0, 1])
        

    
    # 隐藏未使用的子图
    for idx in range(n_models * 2, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Calibration Curves - {dataset_name}', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_calibration_curves_combined(y_true, pred_probs_dict, save_path, dataset_name=""):
    """绘制校准曲线（智能自适应坐标轴，所有模型在同一张图上）"""
    y_true_array = y_true.values if hasattr(y_true, 'values') else np.array(y_true)
    
    # 分析预测概率分布
    all_probs = np.concatenate([y_proba for y_proba in pred_probs_dict.values()])
    prob_95th = np.percentile(all_probs, 95)
    prob_max = np.percentile(all_probs, 99.5)
    prob_min = np.percentile(all_probs, 0.5)
    
    # 根据概率分布确定主要显示范围（与plot_calibration_curves保持一致）
    if prob_95th < 0.1:  # 大多数概率集中在低区域
        xlim_main = 0.3
        ylim_main = 0.3
    elif prob_95th < 0.3:
        xlim_main = 0.3
        ylim_main = 0.3
    else:
        xlim_main = 1.0
        ylim_main = 1.0
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(pred_probs_dict)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
    
    # ========== 左图：详细视图（关键范围） ==========
    ax1 = axes[0]
    ax1.plot([0, xlim_main], [0, xlim_main], 'k--', label='Perfect Calibration', linewidth=2.5, alpha=0.8)
    
    for idx, (model_name, y_proba) in enumerate(pred_probs_dict.items()):
        # 使用quantile策略在低概率区域获得更好的分箱
        prob_true, prob_pred = calibration_curve(y_true_array, y_proba, n_bins=10, strategy='quantile')
        
        # 不过滤点，显示完整曲线
        ax1.plot(prob_pred, prob_true, 
                linestyle=linestyles[idx % len(linestyles)],
                linewidth=2.5, 
                label=model_name, 
                color=colors[idx], 
                alpha=0.85,
                marker=markers[idx % len(markers)],
                markersize=5)
    
    ax1.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Observed Probability', fontsize=12, fontweight='bold')
    ax1.set_title(f'Detailed View (Key Probability Range)\nP95={prob_95th:.2%}, Max={prob_max:.2%}', 
                  fontsize=13, fontweight='bold', pad=15)
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim([0, xlim_main])
    ax1.set_ylim([0, ylim_main])
    
    # 计算Brier分数（不显示在图上）
    from sklearn.metrics import brier_score_loss
    brier_scores = {}
    for model_name, y_proba in pred_probs_dict.items():
        brier_scores[model_name] = brier_score_loss(y_true_array, y_proba)
    
    # ========== 右图：完整视图（0-1范围） ==========
    ax2 = axes[1]
    ax2.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2.5, alpha=0.8)
    
    for idx, (model_name, y_proba) in enumerate(pred_probs_dict.items()):
        prob_true, prob_pred = calibration_curve(y_true_array, y_proba, n_bins=10, strategy='uniform')
        
        ax2.plot(prob_pred, prob_true, 
                linestyle=linestyles[idx % len(linestyles)],
                linewidth=2, 
                label=model_name, 
                color=colors[idx], 
                alpha=0.85)
    
    ax2.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Observed Probability', fontsize=12, fontweight='bold')
    ax2.set_title('Full Range View (0-100%)', fontsize=13, fontweight='bold', pad=15)
    ax2.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    
    plt.suptitle(f'Calibration Curves - All Models\n{dataset_name}', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def perform_shap_analysis(model, X_train, X_test, model_name, save_dir, n_train_samples=500, n_test_samples=100, X_test_original=None):
    """
    执行SHAP可解释性分析
    
    Parameters:
    -----------
    model : sklearn model
        训练好的模型
    X_train : DataFrame
        训练数据（用于背景样本）
    X_test : DataFrame
        测试数据（标准化后的数据，用于计算SHAP值）
    model_name : str
        模型名称
    save_dir : str
        保存目录
    X_test_original : DataFrame, optional
        原始未标准化的测试数据，用于SHAP力图显示。如果提供，力图将显示原始值而非标准化值
    """
    try:
        n_samples = min(n_train_samples, len(X_train))
        X_train_sample = X_train.sample(n=n_samples, random_state=42)
        X_test_sample = X_test.sample(n=min(n_test_samples, len(X_test)), random_state=42)
        
        print(f"  Using {n_samples} training samples, {len(X_test_sample)} test samples")
        print(f"  Computing SHAP values...")
        
        explainer = shap.KernelExplainer(
            lambda x: model.predict_proba(x)[:, 1],
            X_train_sample
        )
        shap_values = explainer.shap_values(X_test_sample)
        
        # 调整SHAP值方向以符合医学常识（正常/高值为保护因素<0，异常/低值为风险因素>0）
        features_to_flip = ['PLT', 'WBC', 'MAP', 'Partial_Pressure_Of_Oxygen', 'Systolic_BP', 'Respiratory_Rate']
        for feat in features_to_flip:
            if feat in X_test_sample.columns:
                feat_idx = X_test_sample.columns.get_loc(feat)
                shap_values[:, feat_idx] = -shap_values[:, feat_idx]
        print(f"  [Note] Direction adjusted for: {features_to_flip}")
        
        # Summary Plot
        print("  [1/6] Generating Summary Plot...")
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_test_sample, show=False, max_display=20)
        plt.title(f'SHAP Summary Plot - {model_name}', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/SHAP_Summary_{model_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Bar Plot
        print("  [2/6] Generating Bar Plot...")
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_test_sample, plot_type="bar", show=False, max_display=20)
        plt.title(f'SHAP Feature Importance - {model_name}', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/SHAP_Bar_{model_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Waterfall Plot (High Risk Sample)
        print("  [3/6] Generating Waterfall Plot (High Risk)...")
        y_pred_proba_sample = model.predict_proba(X_test_sample)[:, 1]
        max_prob_idx = np.argmax(y_pred_proba_sample)
        
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[max_prob_idx],
                base_values=explainer.expected_value,
                data=X_test_sample.iloc[max_prob_idx],
                feature_names=X_test_sample.columns
            ),
            show=False
        )
        plt.title(f'SHAP Waterfall Plot - High Risk Sample\n{model_name}', 
                  fontsize=13, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/SHAP_Waterfall_{model_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Force Plot (High Risk Sample)
        print("  [4/6] Generating Force Plot (High Risk)...")
        try:
            shap.initjs()
            
            # 获取高风险样本的特征值和SHAP值
            # 如果有原始数据，使用原始数据显示在力图中
            if X_test_original is not None:
                # 获取与X_test_sample相同的样本
                X_test_original_sample = X_test_original.loc[X_test_sample.index]
                high_risk_values = X_test_original_sample.iloc[max_prob_idx].copy()
            else:
                high_risk_values = X_test_sample.iloc[max_prob_idx].copy()
            high_risk_shap = shap_values[max_prob_idx].copy()
            
            # 选择Top特征（按|SHAP值|排序）以避免过多特征导致重叠
            top_n = 8  # 只显示最重要的8个特征
            abs_shap = np.abs(high_risk_shap)
            top_indices = np.argsort(abs_shap)[-top_n:][::-1]
            
            # 筛选Top特征
            high_risk_values_filtered = high_risk_values.iloc[top_indices]
            high_risk_shap_filtered = high_risk_shap[top_indices]
            
            # 格式化特征名称：在适当位置插入换行符
            def format_feature_name(name, max_len=10):
                """在特征名中插入换行符，避免过长"""
                if len(name) <= max_len:
                    return name
                # 尝试在下划线处分割
                if '_' in name:
                    parts = name.split('_')
                    # 每2个部分换行
                    new_name = ''
                    for i, part in enumerate(parts):
                        if i > 0 and i % 2 == 0:
                            new_name += '\n' + part
                        else:
                            new_name += ('_' if i > 0 else '') + part
                    return new_name
                # 否则在中间位置换行
                mid = len(name) // 2
                return name[:mid] + '\n' + name[mid:]
            
            # 应用格式化：特征名换行 + 数值保留1位小数
            formatted_features = pd.Series(
                [f"{v:.1f}" for v in high_risk_values_filtered.values],
                index=[format_feature_name(name) for name in high_risk_values_filtered.index]
            )
            
            # 创建更大的图表以避免重叠
            fig, ax = plt.subplots(figsize=(20, 8))
            force_plot_high = shap.force_plot(
                explainer.expected_value,
                high_risk_shap_filtered,
                formatted_features,
                matplotlib=True,
                show=False,
                plot_cmap=["#008bfb", "#ff0051"]  # 蓝到红的配色
            )
            plt.title(f'SHAP Force Plot - High Risk Sample (Prob={y_pred_proba_sample[max_prob_idx]:.1f})\n{model_name}', 
                      fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig(f"{save_dir}/SHAP_Force_HighRisk_{model_name}.png", dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"    Warning: Force plot (high risk) failed: {str(e)[:100]}")
        
        # Force Plot (Low Risk Sample)
        print("  [5/6] Generating Force Plot (Low Risk)...")
        try:
            min_prob_idx = np.argmin(y_pred_proba_sample)
            
            # 获取低风险样本的特征值和SHAP值
            # 如果有原始数据，使用原始数据显示在力图中
            if X_test_original is not None:
                # 获取与X_test_sample相同的样本
                X_test_original_sample = X_test_original.loc[X_test_sample.index]
                low_risk_values = X_test_original_sample.iloc[min_prob_idx].copy()
            else:
                low_risk_values = X_test_sample.iloc[min_prob_idx].copy()
            low_risk_shap = shap_values[min_prob_idx].copy()
            
            # 选择Top特征（按|SHAP值|排序）以避免过多特征导致重叠
            top_n = 8  # 只显示最重要的8个特征
            abs_shap = np.abs(low_risk_shap)
            top_indices = np.argsort(abs_shap)[-top_n:][::-1]
            
            # 筛选Top特征
            low_risk_values_filtered = low_risk_values.iloc[top_indices]
            low_risk_shap_filtered = low_risk_shap[top_indices]
            
            # 格式化特征名称：在适当位置插入换行符
            def format_feature_name(name, max_len=10):
                """在特征名中插入换行符，避免过长"""
                if len(name) <= max_len:
                    return name
                # 尝试在下划线处分割
                if '_' in name:
                    parts = name.split('_')
                    # 每2个部分换行
                    new_name = ''
                    for i, part in enumerate(parts):
                        if i > 0 and i % 2 == 0:
                            new_name += '\n' + part
                        else:
                            new_name += ('_' if i > 0 else '') + part
                    return new_name
                # 否则在中间位置换行
                mid = len(name) // 2
                return name[:mid] + '\n' + name[mid:]
            
            # 应用格式化：特征名换行 + 数值保留1位小数
            formatted_features = pd.Series(
                [f"{v:.1f}" for v in low_risk_values_filtered.values],
                index=[format_feature_name(name) for name in low_risk_values_filtered.index]
            )
            
            # 创建更大的图表以避免重叠
            fig, ax = plt.subplots(figsize=(20, 8))
            force_plot_low = shap.force_plot(
                explainer.expected_value,
                low_risk_shap_filtered,
                formatted_features,
                matplotlib=True,
                show=False,
                plot_cmap=["#008bfb", "#ff0051"]  # 蓝到红的配色
            )
            plt.title(f'SHAP Force Plot - Low Risk Sample (Prob={y_pred_proba_sample[min_prob_idx]:.1f})\n{model_name}', 
                      fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig(f"{save_dir}/SHAP_Force_LowRisk_{model_name}.png", dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"    Warning: Force plot (low risk) failed: {str(e)[:100]}")
        
        # Save SHAP values
        print("  [6/6] Saving SHAP values...")
        shap_df = pd.DataFrame(shap_values, columns=X_test_sample.columns)
        shap_df['Expected_Value'] = explainer.expected_value
        shap_df.to_csv(f"{save_dir}/SHAP_values_{model_name}.csv", index=False)
        
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_importance_shap = pd.DataFrame({
            'Feature': X_test_sample.columns,
            'Mean_Abs_SHAP': mean_abs_shap
        }).sort_values('Mean_Abs_SHAP', ascending=False)
        feature_importance_shap.to_csv(f"{save_dir}/SHAP_importance_{model_name}.csv", index=False)
        
        print(f"  [DONE] SHAP analysis completed - {model_name}")
        return True
        
    except Exception as e:
        print(f"  [WARN] SHAP analysis failed: {str(e)}")
        return False



# =============================================================================
# 后验校准 (Post-hoc Calibration) 模块
# =============================================================================

class CalibratedModel:
    """
    包装器类：为已训练模型添加后验校准功能
    
    支持方法:
    - 'platt': Platt Scaling (Logistic Calibration) - 推荐用于一般情况
    - 'isotonic': Isotonic Regression - 推荐用于大数据集
    - 'beta': Beta Calibration - 推荐用于概率边界附近的校准
    """
    
    def __init__(self, base_model, method='platt'):
        """
        Parameters:
        -----------
        base_model : fitted sklearn model
            已训练好的基础模型
        method : str
            校准方法: 'platt', 'isotonic', 或 'beta'
        """
        self.base_model = base_model
        self.method = method
        self.calibrator = None
        self._is_fitted = False
        
    def fit(self, X_cal, y_cal):
        """
        在校准集上拟合校准器
        
        Parameters:
        -----------
        X_cal : array-like
            校准集特征 (建议使用验证集)
        y_cal : array-like
            校准集标签
        """
        # 获取基础模型的预测概率
        y_proba = self.base_model.predict_proba(X_cal)[:, 1]
        
        if self.method == 'platt':
            # Platt Scaling: 使用Logistic Regression拟合概率映射
            self.calibrator = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
            self.calibrator.fit(y_proba.reshape(-1, 1), y_cal)
            
        elif self.method == 'isotonic':
            # Isotonic Regression: 非单调递增约束
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(y_proba, y_cal)
            
        elif self.method == 'beta':
            # Beta Calibration: 将概率映射到Beta分布
            # 先进行logit变换，再拟合
            # 裁剪避免极端值
            y_proba_clipped = np.clip(y_proba, 1e-10, 1 - 1e-10)
            # 使用两个参数的映射 (LogisticRegression已在文件开头导入)
            self.calibrator = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
            # 添加非线性特征
            X_beta = np.column_stack([
                y_proba_clipped,
                np.log(y_proba_clipped),
                np.log(1 - y_proba_clipped)
            ])
            self.calibrator.fit(X_beta, y_cal)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
        
        self._is_fitted = True
        
        # 计算校准前后的Brier分数
        brier_before = brier_score_loss(y_cal, y_proba)
        y_proba_cal = self.predict_proba(X_cal)[:, 1]
        brier_after = brier_score_loss(y_cal, y_proba_cal)
        
        print(f"  校准方法: {self.method}")
        print(f"  Brier Score - 校准前: {brier_before:.4f}, 校准后: {brier_after:.4f}")
        print(f"  改善: {(brier_before - brier_after):.4f} ({(brier_before - brier_after)/brier_before*100:.1f}%)")
        
        return self
    
    def predict_proba(self, X):
        """预测概率（经过校准）"""
        if not self._is_fitted:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")
        
        # 获取基础模型预测
        y_proba = self.base_model.predict_proba(X)[:, 1]
        
        if self.method == 'platt':
            y_proba_cal = self.calibrator.predict_proba(y_proba.reshape(-1, 1))[:, 1]
        elif self.method == 'isotonic':
            y_proba_cal = self.calibrator.predict(y_proba)
        elif self.method == 'beta':
            y_proba_clipped = np.clip(y_proba, 1e-10, 1 - 1e-10)
            X_beta = np.column_stack([
                y_proba_clipped,
                np.log(y_proba_clipped),
                np.log(1 - y_proba_clipped)
            ])
            y_proba_cal = self.calibrator.predict_proba(X_beta)[:, 1]
        
        # 确保输出是有效的概率
        y_proba_cal = np.clip(y_proba_cal, 1e-10, 1 - 1e-10)
        
        # 返回两列格式 [P(class=0), P(class=1)]
        return np.column_stack([1 - y_proba_cal, y_proba_cal])
    
    def predict(self, X):
        """预测类别"""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)


def calibrate_models(models_dict, X_cal, y_cal, method='platt'):
    """
    对多个模型进行后验校准
    
    Parameters:
    -----------
    models_dict : dict
        模型字典 {name: fitted_model}
    X_cal, y_cal : array-like
        校准数据集（建议使用独立的验证集）
    method : str
        校准方法: 'platt', 'isotonic', 'beta'
    
    Returns:
    --------
    calibrated_models : dict
        校准后的模型字典 {name: CalibratedModel}
    calibration_results : DataFrame
        校准效果对比
    """
    print("\n" + "="*80)
    print(f"后验校准 (Method: {method})")
    print("="*80)
    
    calibrated_models = {}
    results = []
    
    for name, model in models_dict.items():
        print(f"\n校准模型: {name}")
        try:
            # TabNet需要numpy数组
            if name == 'TabNet':
                X_cal_input = X_cal.values if hasattr(X_cal, 'values') else X_cal
            else:
                X_cal_input = X_cal
            
            # 创建校准模型并拟合
            cal_model = CalibratedModel(model, method=method)
            cal_model.fit(X_cal_input, y_cal)
            calibrated_models[name] = cal_model
            
            # 记录结果
            if name == 'TabNet':
                y_proba_orig = model.predict_proba(X_cal_input)[:, 1]
                y_proba_cal = cal_model.predict_proba(X_cal_input)[:, 1]
            else:
                y_proba_orig = model.predict_proba(X_cal)[:, 1]
                y_proba_cal = cal_model.predict_proba(X_cal)[:, 1]
            
            brier_orig = brier_score_loss(y_cal, y_proba_orig)
            brier_cal = brier_score_loss(y_cal, y_proba_cal)
            
            results.append({
                'Model': name,
                'Brier_Original': brier_orig,
                'Brier_Calibrated': brier_cal,
                'Improvement': brier_orig - brier_cal,
                'Improvement_Pct': (brier_orig - brier_cal) / brier_orig * 100
            })
            
        except Exception as e:
            print(f"  ❌ 校准失败: {str(e)[:100]}")
            # 如果校准失败，保留原模型
            calibrated_models[name] = model
    
    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        results_df = results_df.sort_values('Improvement', ascending=False)
        print("\n校准效果汇总:")
        print(results_df.to_string(index=False))
    
    return calibrated_models, results_df


def plot_calibration_comparison(y_true, pred_probs_before, pred_probs_after, 
                                save_path, dataset_name="", method_name=""):
    """
    绘制校准前后的对比图
    
    Parameters:
    -----------
    y_true : array-like
        真实标签
    pred_probs_before : dict
        校准前的预测概率 {model_name: y_proba}
    pred_probs_after : dict
        校准后的预测概率 {model_name: y_proba}
    save_path : str
        保存路径
    dataset_name : str
        数据集名称
    method_name : str
        校准方法名称
    """
    y_true_array = y_true.values if hasattr(y_true, 'values') else np.array(y_true)
    
    n_models = len(pred_probs_before)
    ncols = min(3, n_models)
    nrows = (n_models + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols * 2, figsize=(7*ncols*2, 5*nrows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.ravel()
    
    for idx, model_name in enumerate(pred_probs_before.keys()):
        if idx >= n_models:
            break
        
        y_proba_before = pred_probs_before[model_name]
        y_proba_after = pred_probs_after[model_name]
        
        # 校准前（左列）
        ax_before = axes[idx * 2]
        prob_true_b, prob_pred_b = calibration_curve(y_true_array, y_proba_before, n_bins=10, strategy='quantile')
        
        ax_before.plot([0, 1], [0, 1], 'k--', label='Perfect', linewidth=2, alpha=0.7)
        ax_before.plot(prob_pred_b, prob_true_b, '-', label='Before Calibration', 
                      linewidth=2.5, color='coral', alpha=0.9)
        ax_before.set_xlabel('Predicted Probability', fontsize=11, fontweight='bold')
        ax_before.set_ylabel('Observed Probability', fontsize=11, fontweight='bold')
        ax_before.set_title(f'{model_name}\nBefore Calibration', fontsize=11, fontweight='bold')
        ax_before.legend(loc='upper left', fontsize=9)
        ax_before.grid(alpha=0.3, linestyle='--')
        ax_before.set_xlim([0, 1])
        ax_before.set_ylim([0, 1])
        
        # 校准后（右列）
        ax_after = axes[idx * 2 + 1]
        prob_true_a, prob_pred_a = calibration_curve(y_true_array, y_proba_after, n_bins=10, strategy='quantile')
        
        ax_after.plot([0, 1], [0, 1], 'k--', label='Perfect', linewidth=2, alpha=0.7)
        ax_after.plot(prob_pred_a, prob_true_a, '-', label='After Calibration', 
                     linewidth=2.5, color='steelblue', alpha=0.9)
        ax_after.set_xlabel('Predicted Probability', fontsize=11, fontweight='bold')
        ax_after.set_ylabel('Observed Probability', fontsize=11, fontweight='bold')
        ax_after.set_title(f'{model_name}\nAfter {method_name}', fontsize=11, fontweight='bold')
        ax_after.legend(loc='upper left', fontsize=9)
        ax_after.grid(alpha=0.3, linestyle='--')
        ax_after.set_xlim([0, 1])
        ax_after.set_ylim([0, 1])
        

    
    # 隐藏未使用的子图
    for idx in range(n_models * 2, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Calibration Comparison - {method_name}\n{dataset_name}', 
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  校准对比图已保存: {save_path}")


def select_best_calibration_method(models_dict, X_cal, y_cal):
    """
    自动选择最佳校准方法
    
    对每种校准方法进行评估，选择Brier分数改善最大的方法
    
    Returns:
    --------
    best_method : str
        最佳校准方法名称
    comparison_df : DataFrame
        各方法的对比结果
    """
    print("\n" + "="*80)
    print("自动选择最佳校准方法")
    print("="*80)
    
    methods = ['platt', 'isotonic', 'beta']
    method_results = {}
    
    for method in methods:
        print(f"\n测试方法: {method}")
        improvements = []
        
        for name, model in models_dict.items():
            try:
                y_proba_orig = model.predict_proba(X_cal)[:, 1]
                brier_orig = brier_score_loss(y_cal, y_proba_orig)
                
                cal_model = CalibratedModel(model, method=method)
                cal_model.fit(X_cal, y_cal)
                y_proba_cal = cal_model.predict_proba(X_cal)[:, 1]
                brier_cal = brier_score_loss(y_cal, y_proba_cal)
                
                improvements.append(brier_orig - brier_cal)
            except Exception as e:
                print(f"  {name} 失败: {str(e)[:50]}")
        
        if improvements:
            method_results[method] = {
                'Mean_Improvement': np.mean(improvements),
                'Median_Improvement': np.median(improvements),
                'N_Models': len(improvements)
            }
    
    if method_results:
        comparison_df = pd.DataFrame(method_results).T
        comparison_df = comparison_df.sort_values('Mean_Improvement', ascending=False)
        print("\n各方法对比:")
        print(comparison_df.to_string())
        
        best_method = comparison_df.index[0]
        print(f"\n[OK] 推荐校准方法: {best_method}")
        return best_method, comparison_df
    else:
        print("[Warning] 所有方法都失败，默认使用 'platt'")
        return 'platt', pd.DataFrame()
