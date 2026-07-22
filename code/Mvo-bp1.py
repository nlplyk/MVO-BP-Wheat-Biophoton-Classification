# -*- coding: utf-8 -*-
import warnings
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, \
    confusion_matrix
from sklearn.model_selection import train_test_split, KFold  # 加入KFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
import seaborn as sns
from itertools import cycle

warnings.filterwarnings("ignore")

# 1. 数据加载与预处理
# ---------------------------------------------------------
data = loadmat('matlab_normal.mat')
keys = ['feng_normal', 'ping_normal', 'jin_normal', 'zheng129_normal', 'zheng136_normal', 'yunhan_normal']

all_dfs = []
for key in keys:
    if key in data:
        df = pd.DataFrame(data[key])
        df['target'] = key
        all_dfs.append(df)
    else:
        print(f"Warning: Key {key} not found in .mat file")

combined_df = pd.concat(all_dfs, ignore_index=True)
label_encoder = LabelEncoder()
combined_df['target'] = label_encoder.fit_transform(combined_df['target'])

X = combined_df.drop('target', axis=1)
y = combined_df['target']

# 划分数据集（保留原测试集）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. 原始 BP 神经网络训练
# ---------------------------------------------------------
bp_model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    alpha=0.0005,
    max_iter=1000,
    learning_rate_init=0.001,
    batch_size=32,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42
)
bp_model.fit(X_train_scaled, y_train)

def calculate_metrics(y_true, y_pred, average='macro'):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average=average),
        'Recall': recall_score(y_true, y_pred, average=average),
        'F1-Score': f1_score(y_true, y_pred, average=average)
    }

bp_pred = bp_model.predict(X_test_scaled)
bp_metrics = calculate_metrics(y_test, bp_pred)
print("原始 BP 结果:", bp_metrics)

# 3. MVO 优化算法定义
# ---------------------------------------------------------
class MVO_Optimizer:
    def __init__(self, objective_func, lb, ub, dim, N_pop=30, max_iter=50):
        self.objective_func = objective_func
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.dim = dim
        self.N_pop = N_pop
        self.max_iter = max_iter

        self.universes = np.zeros((N_pop, dim))
        for i in range(dim):
            self.universes[:, i] = np.random.uniform(self.lb[i], self.ub[i], N_pop)

        self.sorted_universes = np.zeros((N_pop, dim))
        self.fitness = np.zeros(N_pop)
        self.best_universe = np.zeros(dim)
        self.best_fitness = float('inf')

    def optimize(self):
        print(f"MVO 开始优化 (Pop={self.N_pop}, Iter={self.max_iter})...")

        for i in range(self.N_pop):
            self.fitness[i] = self.objective_func(self.universes[i, :])

        sorted_indices = np.argsort(self.fitness)
        self.sorted_universes = self.universes[sorted_indices, :]

        self.best_fitness = self.fitness[sorted_indices[0]]
        self.best_universe = self.sorted_universes[0, :]

        for Time in range(1, self.max_iter + 1):
            WEP_Min = 0.2
            WEP_Max = 1.0
            WEP = WEP_Min + Time * ((WEP_Max - WEP_Min) / self.max_iter)
            p = 6
            TDR = 1 - (Time ** (1 / p) / self.max_iter ** (1 / p))
            for i in range(self.N_pop):
                inv_fitness = 1.0 / (self.fitness + 1e-10)
                prob = inv_fitness / np.sum(inv_fitness)
                for j in range(self.dim):
                    r1 = np.random.random()
                    if r1 < 0.5:
                        k = np.random.choice(range(self.N_pop), p=prob)
                        self.universes[i, j] = self.sorted_universes[k, j]
                    r2 = np.random.random()
                    if r2 < WEP:
                        r3 = np.random.random()
                        r4 = np.random.random()
                        if r3 < 0.5:
                            self.universes[i, j] = self.best_universe[j] + TDR * (
                                    (self.ub[j] - self.lb[j]) * r4 + self.lb[j])
                        else:
                            self.universes[i, j] = self.best_universe[j] - TDR * (
                                    (self.ub[j] - self.lb[j]) * r4 + self.lb[j])
            self.universes = np.clip(self.universes, self.lb, self.ub)
            for i in range(self.N_pop):
                self.fitness[i] = self.objective_func(self.universes[i, :])
            sorted_indices = np.argsort(self.fitness)
            self.sorted_universes = self.universes[sorted_indices, :]

            current_best_val = self.fitness[sorted_indices[0]]
            if current_best_val < self.best_fitness:
                self.best_fitness = current_best_val
                self.best_universe = self.sorted_universes[0, :]

            if Time % 5 == 0 or Time == 1:
                print(f"Iter {Time}/{self.max_iter}, Best Loss: {self.best_fitness:.4f}")
        return self.best_universe

def bp_objective_function(params):
    n_hidden = int(params[0])
    alpha_val = params[1]
    lr_val = params[2]

    clf = MLPClassifier(
        hidden_layer_sizes=(n_hidden,),
        activation='relu',
        solver='adam',
        alpha=alpha_val,
        learning_rate_init=lr_val,
        max_iter=200,
        random_state=42,
        early_stopping=True
    )
    clf.fit(X_train_scaled, y_train)
    pred = clf.predict(X_test_scaled)
    error = 1.0 - accuracy_score(y_test, pred)
    return error

# 4. 执行 MVO 优化
# ---------------------------------------------------------
print("-" * 30)
print("正在运行 MVO 优化 BP 超参数...")
lb = [20, 0.00001, 0.0001]
ub = [200, 0.01, 0.01]

mvo = MVO_Optimizer(bp_objective_function, lb, ub, dim=3, N_pop=10, max_iter=15)
best_params = mvo.optimize()

print(f"\n最佳参数找到: 节点数={int(best_params[0])}, Alpha={best_params[1]:.6f}, LR={best_params[2]:.6f}")

# 5. 使用最佳参数训练最终模型
# ---------------------------------------------------------
print("使用最佳参数训练最终 MVO-BP 模型...")
mvo_bp_model = MLPClassifier(
    hidden_layer_sizes=(int(best_params[0]),),
    activation='relu',
    solver='adam',
    alpha=best_params[1],
    learning_rate_init=best_params[2],
    max_iter=1000,
    batch_size=32,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42
)
mvo_bp_model.fit(X_train_scaled, y_train)
mvo_pred = mvo_bp_model.predict(X_test_scaled)
mvo_metrics = calculate_metrics(y_test, mvo_pred)
print("MVO-BP 结果:", mvo_metrics)

print("\n" + "=" * 40)
print("最终对比:")
results_df = pd.DataFrame([bp_metrics, mvo_metrics], index=['Standard BP', 'MVO-BP'])
print(results_df)

# ===================== 【新增：5折交叉验证】 =====================
print("\n" + "=" * 50)
print("              5折交叉验证（MVO-BP）")
print("=" * 50)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_scaled)):
    X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    model_cv = MLPClassifier(
        hidden_layer_sizes=(int(best_params[0]),),
        alpha=best_params[1],
        learning_rate_init=best_params[2],
        max_iter=800, random_state=42, early_stopping=True
    )
    model_cv.fit(X_tr, y_tr)
    y_pred_cv = model_cv.predict(X_val)
    acc = accuracy_score(y_val, y_pred_cv)
    prec = precision_score(y_val, y_pred_cv, average='macro')
    rec = recall_score(y_val, y_pred_cv, average='macro')
    f1 = f1_score(y_val, y_pred_cv, average='macro')

    cv_scores.append([acc, prec, rec, f1])
    print(f"第 {fold+1} 折 | 准确率: {acc:.4f} | 精确率: {prec:.4f} | 召回率: {rec:.4f} | F1: {f1:.4f}")

# 输出交叉验证平均结果
cv_mean = np.mean(cv_scores, axis=0)
print("\n【5折交叉验证平均结果】")
print(f"平均准确率: {cv_mean[0]:.4f}")
print(f"平均精确率: {cv_mean[1]:.4f}")
print(f"平均召回率: {cv_mean[2]:.4f}")
print(f"平均F1:    {cv_mean[3]:.4f}")

# ===================== ROC 绘图 =====================
y_score = mvo_bp_model.predict_proba(X_test_scaled)
n_classes = len(label_encoder.classes_)
y_test_bin = label_binarize(y_test, classes=range(n_classes))

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = roc_auc_score(y_test_bin[:, i], y_score[:, i])

fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
roc_auc["micro"] = roc_auc_score(y_test_bin, y_score, average="micro")

plt.figure(figsize=(10, 8))
lw = 2
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple'])
class_names = [label.replace('_normal', '') for label in label_encoder.classes_]

for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC {0} (area = {1:0.2f})'.format(class_names[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC Curve for MVO-BP')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('roc_curve_mvo_bp.png')
plt.show()