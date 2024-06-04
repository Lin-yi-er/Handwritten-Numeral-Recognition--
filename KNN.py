from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from DataPreprocess import x_train,x_test,y_train,y_test

# 数据转换为适合KNN格式
x_train_flat = x_train.reshape(-1, 28*28)
x_test_flat = x_test.reshape(-1, 28*28)

# 定义KNN参数搜索范围
param_grid = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance'],
}

# 创建KNN模型
knn_model = KNeighborsClassifier()

# 使用GridSearchCV进行参数搜索
grid_search = GridSearchCV(knn_model, param_grid, cv=3, verbose=2, n_jobs=-1)
grid_search.fit(x_train_flat, y_train)

# 打印最佳参数
print("Best parameters found: ", grid_search.best_params_)

# 打印每种参数组合的结果
results = grid_search.cv_results_
for mean_score, params in zip(results['mean_test_score'], results['params']):
    print(f"Accuracy: {mean_score:.4f} with params: {params}")

# 使用最佳参数评估测试集
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(x_test_flat)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {test_accuracy:.4f}")