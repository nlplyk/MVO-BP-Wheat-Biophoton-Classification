# -*- coding: utf-8 -*-

import warnings
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from itertools import cycle

warnings.filterwarnings("ignore")

# 1. 数据加载与预处理（完全保留原有逻辑，确保数据适配）
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

# 划分数据集（保持原有拆分比例和随机种子，确保结果可复现）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 标准化（与原有代码完全一致，避免数据预处理差异）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. 原始 BP 神经网络训练（保留原有设置，用于后续对比）
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

# 指标计算函数（完全保留，确保指标一致性）
def calculate_metrics(y_true, y_pred, average='macro'):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average=average),
        'Recall': recall_score(y_true, y_pred, average=average),
        'F1-Score': f1_score(y_true, y_pred, average=average)
    }

# 原始BP模型测试与结果输出
bp_pred = bp_model.predict(X_test_scaled)
bp_metrics = calculate_metrics(y_test, bp_pred)
print("原始 BP 结果:", bp_metrics)

# 3. GA 优化算法定义（替换原有PSO，适配BP超参数优化）
# ---------------------------------------------------------
class GA_Optimizer:
    def __init__(self, objective_func, lb, ub, dim, pop_size=30, max_iter=50, crossover_rate=0.8, mutation_rate=0.05):
        self.objective_func = objective_func  # 目标函数（最小化误差）
        self.lb = np.array(lb)               # 超参数下界
        self.ub = np.array(ub)               # 超参数上界
        self.dim = dim                       # 超参数维度（节点数、alpha、学习率）
        self.pop_size = pop_size             # 种群规模
        self.max_iter = max_iter             # 最大迭代次数
        self.crossover_rate = crossover_rate # 交叉概率
        self.mutation_rate = mutation_rate   # 变异概率

        # 初始化种群（位置=超参数组合）
        self.population = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        # 计算初始种群适应度（误差越小，适应度越高，此处用1/(误差+1e-10)）
        self.fitness = np.array([1/(self.objective_func(ind) + 1e-10) for ind in self.population])
        # 初始化最优个体
        self.best_ind = self.population[np.argmax(self.fitness)]
        self.best_fitness = np.max(self.fitness)
        self.best_error = 1/self.best_fitness - 1e-10

    def selection(self):
        # 轮盘赌选择，适应度越高，被选中概率越大
        fitness_sum = np.sum(self.fitness)
        selection_prob = self.fitness / fitness_sum
        # 选择种群索引
        selected_indices = np.random.choice(range(self.pop_size), size=self.pop_size, p=selection_prob)
        selected_pop = self.population[selected_indices]
        return selected_pop

    def crossover(self, pop):
        # 单点交叉
        for i in range(0, self.pop_size, 2):
            if np.random.random() < self.crossover_rate:
                # 随机选择交叉点
                cross_point = np.random.randint(1, self.dim)
                # 交换交叉点后的基因
                pop[i, cross_point:], pop[i+1, cross_point:] = pop[i+1, cross_point:].copy(), pop[i, cross_point:].copy()
        return pop

    def mutation(self, pop):
        # 变异操作，随机改变个体的部分基因
        for i in range(self.pop_size):
            if np.random.random() < self.mutation_rate:
                # 随机选择变异位置
                mut_point = np.random.randint(self.dim)
                # 在上下界内随机变异
                pop[i, mut_point] = np.random.uniform(self.lb[mut_point], self.ub[mut_point])
        return pop

    def optimize(self):
        print(f"GA 开始优化 (Pop={self.pop_size}, Iter={self.max_iter})...")
        for iter in range(1, self.max_iter + 1):
            # 1. 选择操作
            selected_pop = self.selection()
            # 2. 交叉操作
            crossover_pop = self.crossover(selected_pop)
            # 3. 变异操作
            new_pop = self.mutation(crossover_pop)
            # 4. 更新种群和适应度
            self.population = new_pop
            self.fitness = np.array([1/(self.objective_func(ind) + 1e-10) for ind in self.population])
            # 5. 更新最优个体
            current_best_idx = np.argmax(self.fitness)
            current_best_fitness = self.fitness[current_best_idx]
            current_best_error = 1/current_best_fitness - 1e-10

            if current_best_fitness > self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_ind = self.population[current_best_idx].copy()
                self.best_error = current_best_error

            # 减少打印频率，避免刷屏，与原有优化逻辑一致
            if iter % 5 == 0 or iter == 1:
                print(f"Iter {iter}/{self.max_iter}, Best Loss: {self.best_error:.4f}")
        
        return self.best_ind  # 返回最优超参数组合

# 4. BP超参数目标函数（完全保留原有逻辑，适配GA优化）
# ---------------------------------------------------------
def bp_objective_function(params):
    # 解码参数（与原有一致：隐藏层节点数、alpha正则化系数、学习率）
    n_hidden = int(params[0])  # 节点数必须为整数
    alpha_val = params[1]
    lr_val = params[2]

    # 构建BP模型（搜索时减少迭代次数，加快优化速度）
    clf = MLPClassifier(
        hidden_layer_sizes=(n_hidden,),
        activation='relu',
        solver='adam',
        alpha=alpha_val,
        learning_rate_init=lr_val,
        max_iter=200,  # 优化阶段迭代次数，与原有一致
        random_state=42,
        early_stopping=True
    )
    clf.fit(X_train_scaled, y_train)
    pred = clf.predict(X_test_scaled)
    error = 1.0 - accuracy_score(y_test, pred)  # 目标：最小化分类误差
    return error

# 5. 执行 GA 优化（替换原有PSO优化，超参数上下界不变）
# ---------------------------------------------------------
print("-" * 30)
print("正在运行 GA 优化 BP 超参数...")
# 超参数上下界（与原有一致，确保公平对比）
lb = [20, 0.00001, 0.0001]  # 隐藏层节点数：20-200，alpha：1e-5-0.01，学习率：1e-4-0.01
ub = [200, 0.01, 0.01]

# 实例化GA优化器（参数设置合理，兼顾优化速度和效果）
ga = GA_Optimizer(
    objective_func=bp_objective_function,
    lb=lb,
    ub=ub,
    dim=3,  # 3个超参数：节点数、alpha、学习率
    pop_size=10,  # 种群规模，与原有一致
    max_iter=15,  # 迭代次数，与原有一致
    crossover_rate=0.8,
    mutation_rate=0.05
)
best_params = ga.optimize()  # 得到GA优化后的最佳超参数

print(f"\n最佳参数找到: 节点数={int(best_params[0])}, Alpha={best_params[1]:.6f}, LR={best_params[2]:.6f}")

# 6. 使用最佳参数训练最终 GA-BP 模型（与原有逻辑一致）
# ---------------------------------------------------------
print("使用最佳参数训练最终 GA-BP 模型...")
ga_bp_model = MLPClassifier(
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

ga_bp_model.fit(X_train_scaled, y_train)
ga_pred = ga_bp_model.predict(X_test_scaled)
ga_metrics = calculate_metrics(y_test, ga_pred)
print("GA-BP 结果:", ga_metrics)

# 输出原始BP与GA-BP对比结果（替换原有PSO-BP对比）
print("\n" + "=" * 40)
print("最终对比:")
results_df = pd.DataFrame([bp_metrics, ga_metrics], index=['Standard BP', 'GA-BP'])
print(results_df)

# 7. 5折交叉验证（保留原有逻辑，适配GA-BP模型）
# ---------------------------------------------------------
print("\n" + "=" * 50)
print("              5折交叉验证（GA-BP）")
print("=" * 50)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_scaled)):
    X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # 使用GA优化后的最佳参数构建交叉验证模型
    model_cv = MLPClassifier(
        hidden_layer_sizes=(int(best_params[0]),),
        alpha=best_params[1],
        learning_rate_init=best_params[2],
        max_iter=800,  # 交叉验证模型迭代次数，与原有一致
        random_state=42,
        early_stopping=True
    )
    model_cv.fit(X_tr, y_tr)
    y_pred_cv = model_cv.predict(X_val)
    
    # 计算各折指标（与原有一致）
    acc = accuracy_score(y_val, y_pred_cv)
    prec = precision_score(y_val, y_pred_cv, average='macro')
    rec = recall_score(y_val, y_pred_cv, average='macro')
    f1 = f1_score(y_val, y_pred_cv, average='macro')

    cv_scores.append([acc, prec, rec, f1])
    print(f"第 {fold+1} 折 | 准确率: {acc:.4f} | 精确率: {prec:.4f} | 召回率: {rec:.4f} | F1: {f1:.4f}")

# 输出交叉验证平均结果
cv_mean = np.mean(cv_scores, axis=0)
cv_std = np.std(cv_scores, axis=0)  # 补充标准差，呼应审稿人要求
print("\n【5折交叉验证平均结果】")
print(f"平均准确率: {cv_mean[0]:.4f} ± {cv_std[0]:.4f}")
print(f"平均精确率: {cv_mean[1]:.4f} ± {cv_std[1]:.4f}")
print(f"平均召回率: {cv_mean[2]:.4f} ± {cv_std[2]:.4f}")
print(f"平均F1:    {cv_mean[3]:.4f} ± {cv_std[3]:.4f}")

# 计算准确率95%置信区间（呼应审稿人要求）
from scipy import stats
confidence_level = 0.95
n_folds = 5
t_val = stats.t.ppf((1 + confidence_level) / 2, n_folds - 1)
ci_lower = cv_mean[0] - t_val * (cv_std[0] / np.sqrt(n_folds))
ci_upper = cv_mean[0] + t_val * (cv_std[0] / np.sqrt(n_folds))
print(f"准确率95%置信区间: [{ci_lower:.4f}, {ci_upper:.4f}]")

# 8. ROC 绘图（保留原有逻辑，适配GA-BP模型）
# ---------------------------------------------------------
y_score = ga_bp_model.predict_proba(X_test_scaled)
n_classes = len(label_encoder.classes_)
y_test_bin = label_binarize(y_test, classes=range(n_classes))

# 计算每一类的 ROC 和 AUC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = roc_auc_score(y_test_bin[:, i], y_score[:, i])

# 计算微平均 ROC 曲线
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
roc_auc["micro"] = roc_auc_score(y_test_bin, y_score, average="micro")

# 绘制ROC曲线（与原有样式一致）
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
plt.title('Multi-class ROC Curve for GA-BP')  # 标题改为GA-BP
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('roc_curve_ga_bp.png')  # 保存文件名改为GA-BP
plt.show()  # 弹出绘图窗口
