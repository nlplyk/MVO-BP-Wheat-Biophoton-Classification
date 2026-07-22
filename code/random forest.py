# -*- coding: utf-8 -*-

import warnings
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from itertools import cycle
from scipy import stats

warnings.filterwarnings("ignore")

# ===================== 数据加载与预处理 =====================
data = loadmat('matlab_normal.mat')
keys = ['feng_normal', 'ping_normal', 'jin_normal', 'zheng129_normal', 'zheng136_normal', 'yunhan_normal']

all_dfs = []
for key in keys:
    if key in data:
        df = pd.DataFrame(data[key])
        df['target'] = key
        all_dfs.append(df)

combined_df = pd.concat(all_dfs, ignore_index=True)
label_encoder = LabelEncoder()
combined_df['target'] = label_encoder.fit_transform(combined_df['target'])

X = combined_df.drop('target', axis=1)
y = combined_df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===================== 随机森林模型 =====================
print("=" * 50)
print("          Random Forest 结果")
print("=" * 50)

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
rf_model.fit(X_train_scaled, y_train)
y_pred = rf_model.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro')
rec = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print(f"准确率: {acc:.4f}")
print(f"精确率: {prec:.4f}")
print(f"召回率: {rec:.4f}")
print(f"F1分数: {f1:.4f}")

# ===================== 5折交叉验证 =====================
print("\n" + "=" * 50)
print("      Random Forest 5折交叉验证")
print("=" * 50)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (tr_idx, val_idx) in enumerate(kfold.split(X_train_scaled)):
    X_tr, X_val = X_train_scaled[tr_idx], X_train_scaled[val_idx]
    y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_tr, y_tr)
    yp = model.predict(X_val)

    a = accuracy_score(y_val, yp)
    p = precision_score(y_val, yp, average='macro')
    r = recall_score(y_val, yp, average='macro')
    f = f1_score(y_val, yp, average='macro')

    cv_scores.append([a, p, r, f])
    print(f"第{fold+1}折 | 准确率:{a:.4f} | 精确率:{p:.4f} | 召回率:{r:.4f} | F1:{f:.4f}")

# 均值 ± 标准差
cv_mean = np.mean(cv_scores, axis=0)
cv_std = np.std(cv_scores, axis=0)

print("\n【5折交叉验证平均结果】")
print(f"平均准确率: {cv_mean[0]:.4f} ± {cv_std[0]:.4f}")
print(f"平均精确率: {cv_mean[1]:.4f} ± {cv_std[1]:.4f}")
print(f"平均召回率: {cv_mean[2]:.4f} ± {cv_std[2]:.4f}")
print(f"平均F1:    {cv_mean[3]:.4f} ± {cv_std[3]:.4f}")

# 95% 置信区间
t_val = stats.t.ppf((1 + 0.95) / 2, 4)
ci_lower = cv_mean[0] - t_val * (cv_std[0] / np.sqrt(5))
ci_upper = cv_mean[0] + t_val * (cv_std[0] / np.sqrt(5))
print(f"准确率 95% 置信区间: [{ci_lower:.4f}, {ci_upper:.4f}]")

# ===================== ROC 曲线 =====================
y_score = rf_model.predict_proba(X_test_scaled)
n_classes = len(label_encoder.classes_)
y_test_bin = label_binarize(y_test, classes=range(n_classes))

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = roc_auc_score(y_test_bin[:, i], y_score[:, i])

plt.figure(figsize=(10, 8))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple'])
class_names = [label.replace('_normal', '') for label in label_encoder.classes_]

for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC {class_names[i]} (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest Multi-class ROC Curve')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('roc_rf.png')
plt.show()