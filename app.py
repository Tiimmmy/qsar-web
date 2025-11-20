import os
import uuid
import re
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, MolSurf, Crippen
from rdkit.Chem.QED import qed
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, balanced_accuracy_score, roc_auc_score, roc_curve
)
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for



# ----------------------------
# 初始化 Flask 应用
# ----------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PLOT_FOLDER'] = 'static/plots'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PLOT_FOLDER'], exist_ok=True)


# 描述符计算函数（完整版）
def compute_all_descriptors(smiles_list, y_list):
    """
    计算所有 RDKit 内置 + 扩展描述符
    
    参数:
        smiles_list: List[str] - SMILES 字符串列表
        y_list: List[float] - 对应活性值（连续或离散）
    
    返回:
        X: np.ndarray (n_samples, n_features) - 描述符矩阵
        y: np.ndarray (n_samples,) - 对齐后的活性值
        valid_smiles: List[str] - 有效 SMILES
        desc_names: List[str] - 描述符名称
    """
    # 1. 标准描述符（来自 Descriptors.descList）
    desc_names = []
    desc_funcs = []
    for name, func in Descriptors.descList:
        desc_names.append(name)
        desc_funcs.append(func)

    # 2. 额外描述符（去重）
    extra_descs = {
        'NumRotatableBonds': Lipinski.NumRotatableBonds,
        'FractionCSP3': Lipinski.FractionCSP3,
        'TPSA': MolSurf.TPSA,
        'MolMR': Crippen.MolMR,
        'QED': lambda m: qed(m)
    }

    for name, func in extra_descs.items():
        if name not in desc_names:
            desc_names.append(name)
            desc_funcs.append(func)

    print(f"🧪 使用 {len(desc_names)} 个 RDKit 描述符")

    # 3. 计算描述符，过滤无效分子
    X = []
    valid_smiles = []
    valid_y = []

    for smi, y_val in zip(smiles_list, y_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            # 计算所有描述符
            desc_vals = [func(mol) for func in desc_funcs]
            # 清洗：替换 None / NaN / Inf 为 0.0
            cleaned_vals = []
            for v in desc_vals:
                if v is None or np.isnan(v) or np.isinf(v):
                    cleaned_vals.append(0.0)
                else:
                    cleaned_vals.append(float(v))
            X.append(cleaned_vals)
            valid_smiles.append(smi)
            valid_y.append(y_val)
        except Exception as e:
            # 可选：记录错误分子
            # print(f"⚠️ 跳过分子 {smi}: {e}")
            continue

    if len(X) == 0:
        raise ValueError("❌ 没有有效分子可用于建模")

    X = np.array(X, dtype=np.float32)
    y = np.array(valid_y, dtype=np.float32)

    print(f"✅ 最终数据集: {X.shape[0]} 分子 × {X.shape[1]} 描述符")

    if X.shape[0] < 5:
        raise ValueError("❌ 有效样本太少（<5），无法建模")

    return X, y, valid_smiles, desc_names


# ----------------------------
# GPR 回归模型定义
# ----------------------------
def create_conservative_gpr():
    kernel = (ConstantKernel(1.0, constant_value_bounds="fixed") *
              RBF(length_scale=22.0, length_scale_bounds="fixed") +
              WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-2, 10)))
    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-2,
        normalize_y=True,
        random_state=42
    )

def train_gpr_regression(X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    gpr_train_r2s, gpr_val_r2s = [], []
    gpr_val_maes, gpr_val_rmses = [], []
    avg_uncertainties = []
    
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)
        
        model = create_conservative_gpr()
        model.fit(X_tr_scaled, y_tr)
        
        y_pred_tr, sigma_tr = model.predict(X_tr_scaled, return_std=True)
        y_pred_val, sigma_val = model.predict(X_val_scaled, return_std=True)
        
        gpr_train_r2s.append(r2_score(y_tr, y_pred_tr))
        gpr_val_r2s.append(r2_score(y_val, y_pred_val))
        gpr_val_maes.append(mean_absolute_error(y_val, y_pred_val))
        gpr_val_rmses.append(np.sqrt(mean_squared_error(y_val, y_pred_val)))
        avg_uncertainties.append(np.mean(sigma_val))

    # 全量训练用于绘图
    scaler_full = StandardScaler()
    X_scaled_full = scaler_full.fit_transform(X)
    final_model = create_conservative_gpr()
    final_model.fit(X_scaled_full, y)
    y_pred_full, sigma_full = final_model.predict(X_scaled_full, return_std=True)

    return {
        'val_r2': np.mean(gpr_val_r2s),
        'mae': np.mean(gpr_val_maes),
        'rmse': np.mean(gpr_val_rmses),
        'avg_uncertainty': np.mean(avg_uncertainties),
        'y_true_plot': y,
        'y_pred_plot': y_pred_full,
        'sigma_plot': sigma_full,
        'n_samples': len(X)
    }


# ----------------------------
# 分类模型
# ----------------------------
def train_classification(X, y_binary, desc_names):
    Cs = [0.1, 1.0, 10.0, 100.0]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_bacc = -1
    best_C = 1.0

    for C in Cs:
        scores = []
        for train_idx, val_idx in skf.split(X, y_binary):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y_binary[train_idx], y_binary[val_idx]
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_val = scaler.transform(X_val)
            model = LogisticRegression(penalty='l1', solver='liblinear', C=C, class_weight='balanced', max_iter=1000)
            model.fit(X_tr, y_tr)
            scores.append(balanced_accuracy_score(y_val, model.predict(X_val)))
        if np.mean(scores) > best_bacc:
            best_bacc = np.mean(scores)
            best_C = C

    all_y_true, all_y_proba = [], []
    accs, baccs, aucs = [], [], []

    for train_idx, val_idx in skf.split(X, y_binary):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y_binary[train_idx], y_binary[val_idx]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_val = scaler.transform(X_val)
        model = LogisticRegression(penalty='l1', solver='liblinear', C=best_C, class_weight='balanced', max_iter=1000)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]

        accs.append(accuracy_score(y_val, y_pred))
        baccs.append(balanced_accuracy_score(y_val, y_pred))
        aucs.append(roc_auc_score(y_val, y_proba))

        all_y_true.extend(y_val)
        all_y_proba.extend(y_proba)

    scaler_full = StandardScaler()
    X_scaled = scaler_full.fit_transform(X)
    final_model = LogisticRegression(penalty='l1', solver='liblinear', C=best_C, class_weight='balanced', max_iter=1000)
    final_model.fit(X_scaled, y_binary)
    coef = final_model.coef_[0]

    return {
        'acc': np.mean(accs),
        'bacc': np.mean(baccs),
        'auc': np.mean(aucs),
        'y_true': all_y_true,
        'y_proba': all_y_proba,
        'coef': coef,
        'desc_names': desc_names
    }


# ----------------------------
# 主路由
# ----------------------------
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    task = request.form.get('task_type', 'regression')
    if not file or not file.filename.endswith('.csv'):
        return "❌ 请上传 CSV 文件", 400

    filename = f"{uuid.uuid4().hex}.csv"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # 👇 读取完整 CSV（不再限制 10 行）
        df_full = pd.read_csv(filepath)
        if len(df_full) > 5000:
            df_preview = df_full.head(5000)
            flash_message = "⚠️ 文件较大，仅预览前 5000 行"
        else:
            df_preview = df_full
            flash_message = None
    except Exception as e:
        return f"❌ 无法读取 CSV: {e}", 400

    # 传递给模板
    return render_template('file_preview.html',
                        filename=filename,
                        task=task,
                        columns=df_preview.columns.tolist(),
                        rows=df_preview.values.tolist(),
                        flash_message=flash_message)



@app.route('/start_training', methods=['POST'])
def start_training():
    filename = request.form['filename']
    task = request.form['task']

    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        df = pd.read_csv(filepath)

        # 自动识别列
        smiles_col = activity_col = None
        for col in df.columns:
            c = col.lower()
            if 'smile' in c: smiles_col = col
            if 'act' in c or 'value' in c or c in ['y', 'target', 'pic50']:
                activity_col = col
        if not smiles_col or not activity_col:
            return "❌ 列名需包含 'smile' 和 'act'", 400

        # 解析数据
        data = []
        for _, row in df.iterrows():
            smi = str(row[smiles_col]).strip()
            act_str = str(row[activity_col]).strip()
            if smi in ('nan', '') or act_str in ('nan', ''): continue
            match = re.search(r'[-+]?\d*\.?\d+', act_str)
            if match:
                try:
                    y_val = float(match.group())
                    mol = Chem.MolFromSmiles(smi)
                    if mol is not None:
                        data.append((smi, y_val))
                except:
                    continue
        if len(data) < 5:
            return "❌ 有效分子太少（至少需要 5 个）", 400

        smiles_list, y_all = zip(*data)
        y_all = list(y_all)

        # 计算描述符
        X, y_all, valid_smiles, desc_names = compute_all_descriptors(smiles_list, y_all)

        # 训练
        if task == 'regression':
            if len(set(np.round(y_all, 6))) < 2:
                return "❌ 回归任务需要变化的活性值", 400
            results = train_gpr_regression(X, y_all)

            plot_id = uuid.uuid4().hex
            sorted_idx = np.argsort(results['y_true_plot'])
            x_sorted = np.array(results['y_true_plot'])[sorted_idx]
            y_sorted = np.array(results['y_pred_plot'])[sorted_idx]
            sigma_sorted = np.array(results['sigma_plot'])[sorted_idx]

            plt.figure(figsize=(7, 6))
            plt.errorbar(x_sorted, y_sorted, yerr=sigma_sorted,
                         fmt='o', alpha=0.6, capsize=2, elinewidth=0.8, markersize=4)
            plt.plot([x_sorted.min(), x_sorted.max()], [x_sorted.min(), x_sorted.max()], 'r--', lw=1.5)
            plt.xlabel('True Activity')
            plt.ylabel('Predicted Activity')
            plt.title(f'GPR Regression (Val R² = {results["val_r2"]:.2f})')
            plt.tight_layout()
            plot_path = f"{plot_id}_gpr_regression.png"
            plt.savefig(os.path.join(app.config['PLOT_FOLDER'], plot_path), dpi=150)
            plt.close()

            return render_template('regression_result.html',
                r2=results['val_r2'],
                mae=results['mae'],
                rmse=results['rmse'],
                avg_uncertainty=results['avg_uncertainty'],
                n_samples=results['n_samples'],
                plot_url=url_for('static', filename=f'plots/{plot_path}')
            )

        else:  # classification
            threshold = np.median(y_all)
            y_binary = (y_all >= threshold).astype(int)
            if len(set(y_binary)) < 2:
                return "❌ 分类任务需要两类样本", 400
            results = train_classification(X, y_binary, desc_names)

            plot_id = uuid.uuid4().hex

            # 权重图
            plt.figure(figsize=(6, 4))
            colors = ['green' if w > 0 else 'red' for w in results['coef']]
            plt.barh(results['desc_names'], results['coef'], color=colors)
            plt.xlabel('Weight')
            plt.title('Key Descriptors (Green ↑ / Red ↓)')
            plt.tight_layout()
            weight_plot = f"{plot_id}_weights.png"
            plt.savefig(os.path.join(app.config['PLOT_FOLDER'], weight_plot))
            plt.close()

            # ROC 图
            fpr, tpr, _ = roc_curve(results['y_true'], results['y_proba'])
            plt.figure(figsize=(5, 5))
            plt.plot(fpr, tpr, label=f'AUC = {results["auc"]:.2f}')
            plt.plot([0,1], [0,1], 'k--', alpha=0.5)
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.title('ROC Curve')
            plt.legend()
            plt.tight_layout()
            roc_plot = f"{plot_id}_roc.png"
            plt.savefig(os.path.join(app.config['PLOT_FOLDER'], roc_plot))
            plt.close()

            return render_template('classification_result.html',
                acc=results['acc'],
                bacc=results['bacc'],
                auc=results['auc'],
                threshold=threshold,
                n_samples=len(X),
                weight_plot=url_for('static', filename=f'plots/{weight_plot}'),
                roc_plot=url_for('static', filename=f'plots/{roc_plot}'),
                features=[{'desc': d, 'weight': w} for d, w in zip(results['desc_names'], results['coef'])]
            )

    except Exception as e:
        return f"❌ 训练失败: {str(e)}", 500



if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)