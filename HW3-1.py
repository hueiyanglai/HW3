import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score  # 用於計算正確率

# 1. 生成一維數據
np.random.seed(42)
X, y = make_classification(
    n_samples=100, 
    n_features=1, 
    n_informative=1, 
    n_redundant=0, 
    n_clusters_per_class=1,
    flip_y=0.1,  # 添加一些噪聲
    class_sep=1.5,  # 類別分離程度
)

# 切分數據集為訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. 訓練模型
# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Support Vector Machine
svm_clf = SVC(kernel="linear", C=1.0)
svm_clf.fit(X_train, y_train)

# 3. 計算正確率
log_train_acc = accuracy_score(y_train, log_reg.predict(X_train))
log_test_acc = accuracy_score(y_test, log_reg.predict(X_test))

svm_train_acc = accuracy_score(y_train, svm_clf.predict(X_train))
svm_test_acc = accuracy_score(y_test, svm_clf.predict(X_test))

print("Logistic Regression:")
print(f"  Training Accuracy: {log_train_acc:.2f}")
print(f"  Testing Accuracy: {log_test_acc:.2f}")

print("\nSVM:")
print(f"  Training Accuracy: {svm_train_acc:.2f}")
print(f"  Testing Accuracy: {svm_test_acc:.2f}")

# 4. 決策邊界可視化
# 創建一個密集的 X 範圍
x_range = np.linspace(X.min() - 1, X.max() + 1, 500).reshape(-1, 1)

# Logistic Regression 預測概率
log_proba = log_reg.predict_proba(x_range)[:, 1]

# SVM 決策函數
svm_decision = svm_clf.decision_function(x_range)
svm_proba = 1 / (1 + np.exp(-svm_decision))  # 將 SVM 的決策函數轉換為概率

# 5. 畫圖
plt.figure(figsize=(10, 6))

# 繪製數據點
plt.scatter(X_train, y_train, c=y_train, cmap="bwr", edgecolor="k", label="Training data")
plt.scatter(X_test, y_test, c=y_test, cmap="coolwarm", edgecolor="k", marker="x", label="Test data")

# 繪製 Logistic Regression 決策邊界
plt.plot(x_range, log_proba, label="Logistic Regression (Prob)", color="blue", linestyle="--")

# 繪製 SVM 決策邊界
plt.plot(x_range, svm_proba, label="SVM (Prob)", color="red", linestyle="-.")

# 添加標籤與圖例
plt.axhline(0.5, color="gray", linestyle=":")  # 概率閾值
plt.title("Logistic Regression vs SVM (1D Data)")
plt.xlabel("Feature Value")
plt.ylabel("Probability")
plt.legend()
plt.grid()
plt.show()
