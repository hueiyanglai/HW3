# HW3-2: SVM Decision Boundary with Labeled Data Points and Boundary Name
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. 生成二維數據集
np.random.seed(42)
X, y = make_circles(n_samples=300, factor=0.5, noise=0.1)

# 2. 分割數據集 (訓練集和測試集)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 訓練 SVM 模型
C = 1.0  # 正則化參數
gamma = 0.5  # RBF 核函數的參數
svm_clf = SVC(kernel='rbf', C=C, gamma=gamma)
svm_clf.fit(X_train, y_train)

# 4. 模型預測
y_pred = svm_clf.predict(X_test)

# 5. 計算正確率
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of SVM Model: {accuracy * 100:.2f}%")

# 6. 繪製決策邊界
def plot_decision_boundary(model, X, y, title="SVM Decision Boundary"):
    # 創建網格
    x0, x1 = np.meshgrid(
        np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200),
        np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200)
    )
    Z = model.decision_function(np.c_[x0.ravel(), x1.ravel()])
    Z = Z.reshape(x0.shape)

    # 繪製決策邊界
    plt.figure(figsize=(10, 8))
    plt.contourf(x0, x1, Z, levels=20, cmap="coolwarm", alpha=0.8)
    decision_line = plt.contour(x0, x1, Z, levels=[0], colors="black", linewidths=2)
    plt.clabel(decision_line, fmt="Decision Boundary", fontsize=12)

    # 繪製數據點，區分兩類
    class_0 = y == 0
    class_1 = y == 1
    plt.scatter(X[class_0, 0], X[class_0, 1], c="blue", label="Class 0 (Blue)", edgecolors="k", alpha=0.8)
    plt.scatter(X[class_1, 0], X[class_1, 1], c="red", label="Class 1 (Red)", edgecolors="k", alpha=0.8)

    # 標記數據點類別名稱
    for i, label in enumerate(["Class 0", "Class 1"]):
        xc, yc = np.mean(X[y == i], axis=0)
        plt.text(xc, yc, label, color="black", fontsize=12, ha="center", bbox=dict(facecolor='white', alpha=0.6))

    # 設置圖例和標題
    plt.title(title, fontsize=16)
    plt.xlabel("Feature 1", fontsize=12)
    plt.ylabel("Feature 2", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.4)
    plt.show()

# 7. 繪製結果
plot_decision_boundary(svm_clf, X, y, title=f"SVM Decision Boundary (Accuracy: {accuracy * 100:.2f}%)")
