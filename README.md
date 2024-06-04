# Handwritten-Numeral-Recognition--
By comparing SVM, CNN, DNN and KNN, we tried to find the optimal model. Finally, the uploaded image was passed into the trained model through pre-processing methods such as cutting important areas, and the result was obtained


# 版本



~~~~python
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout
import time
import numpy as np
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

print("TensorFlow version:", tf.__version__)
~~~~

$本次实验基于$
$$
tensorflow 2.10.0 \\
1.4.1.post1 \\
numpy 1.26.4 \\
matplotlib3.5.1
$$






# 数据处理

$ps：注释中带*表示可以省略的步骤$



~~~python
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# *查看数据集中第一副图和数组
first_image = x_train[0]
plt.imshow(first_image, cmap='Blues')
plt.title(f'Label: {y_train[0]}')
plt.show()
print(x_train[0])

# 归一化
x_train, x_test = x_train / 255.0, x_test / 255.0

# 增加一个维度，通道维度
x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")

# *打印形状 ----(60000, 28, 28, 1)
print(x_train.shape)

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices(
    (x_test, y_test)).batch(32)
~~~







# 模型构建及训练



为了便于寻找模型的最优参数，我们使用网格搜索法遍历每种组合。网格搜索针对超参数组合列表中的每一个组合，实例化给定的模型，做cv次交叉验证，将平均得分最高的超参数组合作为最佳的选择，返回模型对象。[sklearn网格搜索][https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV]





## SVM模型

~~~python
# 数据转换为适合SVM格式
x_train_flat = x_train.reshape(-1, 28*28)
x_test_flat = x_test.reshape(-1, 28*28)

# 定义SVM参数搜索范围
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
    'kernel': ['linear', 'rbf']
}

# 创建SVM模型
svm_model = svm.SVC()

# 使用GridSearchCV进行参数搜索
grid_search = GridSearchCV(svm_model, param_grid, cv=3, verbose=2, n_jobs=-1)
grid_search.fit(x_train_flat, y_train)

# 打印最佳参数
print("Best parameters found: ", grid_search.best_params_)

# 打印每种参数组合的结果
results = grid_search.cv_results_
for mean_score, params in zip(results['mean_test_score'], results['params']):
    print(f"Accuracy: {mean_score:.4f} with params: {params}")

# 使用最佳参数评估测试集
best_svm = grid_search.best_estimator_
y_pred = best_svm.predict(x_test_flat)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {test_accuracy:.4f}")
~~~



最优参数组合为`'C': 10, 'gamma': 'scale', 'kernel': 'rbf'`，此时测试结果为98.37%。

~~~python
Best parameters found:  {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}
Accuracy: 0.9399 with params: {'C': 0.1, 'gamma': 'scale', 'kernel': 'linear'}
Accuracy: 0.9518 with params: {'C': 0.1, 'gamma': 'scale', 'kernel': 'rbf'}
Accuracy: 0.9399 with params: {'C': 0.1, 'gamma': 'auto', 'kernel': 'linear'}
Accuracy: 0.9022 with params: {'C': 0.1, 'gamma': 'auto', 'kernel': 'rbf'}
Accuracy: 0.9319 with params: {'C': 1, 'gamma': 'scale', 'kernel': 'linear'}
Accuracy: 0.9749 with params: {'C': 1, 'gamma': 'scale', 'kernel': 'rbf'}
Accuracy: 0.9319 with params: {'C': 1, 'gamma': 'auto', 'kernel': 'linear'}
Accuracy: 0.9343 with params: {'C': 1, 'gamma': 'auto', 'kernel': 'rbf'}
Accuracy: 0.9203 with params: {'C': 10, 'gamma': 'scale', 'kernel': 'linear'}
Accuracy: 0.9802 with params: {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}
Accuracy: 0.9203 with params: {'C': 10, 'gamma': 'auto', 'kernel': 'linear'}
Accuracy: 0.9554 with params: {'C': 10, 'gamma': 'auto', 'kernel': 'rbf'}
Test accuracy: 0.9837
~~~











## KNN



~~~python
# 数据转换为适合KNN格式
x_train_flat = x_train.reshape(-1, 28*28)
x_test_flat = x_test.reshape(-1, 28*28)

# 定义KNN参数搜索范围
param_grid = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
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
~~~

~~~python
Fitting 3 folds for each of 6 candidates, totalling 18 fits
Best parameters found:  {'n_neighbors': 3, 'weights': 'distance'}
Accuracy: 0.9682 with params: {'n_neighbors': 3, 'weights': 'uniform'}
Accuracy: 0.9693 with params: {'n_neighbors': 3, 'weights': 'distance'}
Accuracy: 0.9674 with params: {'n_neighbors': 5, 'weights': 'uniform'}
Accuracy: 0.9686 with params: {'n_neighbors': 5, 'weights': 'distance'}
Accuracy: 0.9652 with params: {'n_neighbors': 7, 'weights': 'uniform'}
Accuracy: 0.9667 with params: {'n_neighbors': 7, 'weights': 'distance'}
Test accuracy: 0.9717
~~~







## DNN模型

~~~python
class DNN(Model):
    def __init__(self):
        super(DNN,self).__init__()
        self.d1 = Dense(128,input_shape=(28,28,1),activation='relu')
        self.d4 = Dense(128,activation='relu')
        self.d5 = Dense(10)
    
    def call(self, x, training=False):
        x = self.d1(x)
        x = self.d2(x)
        if training:
            x = self.d3(x, training=training)
        x = self.d4(x)
        x = self.d5(x)
        return x

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True) # dropout层存在时设置为True,else False
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss) # 其显示作用，主要时跟踪loss的平均值
    train_accuracy(labels, predictions)
    
@tf.function
def test_step(images, labels):
    predictions = model(images, training=False) # dropout层存在时设置为True,else False
    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)
    
    
model = DNN()
EPOCHS = 10

for epoch in range(EPOCHS):
  # 梯度清空
    start_time = time.time()  # 开始计时
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for images, labels in train_ds:
        train_step(images, labels)


    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    end_time = time.time()  # 结束计时
    epoch_time = end_time - start_time
    
    print(
    f'Epoch {epoch + 1}, '
    f'Time: {epoch_time:.2f} sec, '
    f'Loss: {train_loss.result()}, '
    f'Accuracy: {train_accuracy.result() * 100}, '
    f'Test Loss: {test_loss.result()}, '
    f'Test Accuracy: {test_accuracy.result() * 100}'
    )
~~~



结果大概在第四轮达到最大，约为98.69%

~~~python
Epoch 1, Time: 17.30 sec, Loss: 0.042795561254024506, Accuracy: 98.70999908447266, Test Loss: 0.05229710042476654, Test Accuracy: 98.1199951171875
Epoch 2, Time: 17.22 sec, Loss: 0.020882850512862206, Accuracy: 99.29833221435547, Test Loss: 0.05419255793094635, Test Accuracy: 98.3499984741211
Epoch 3, Time: 17.26 sec, Loss: 0.014223978854715824, Accuracy: 99.50666809082031, Test Loss: 0.06330963969230652, Test Accuracy: 98.22999572753906
Epoch 4, Time: 17.09 sec, Loss: 0.009975634515285492, Accuracy: 99.66666412353516, Test Loss: 0.04991135746240616, Test Accuracy: 98.68999481201172
Epoch 5, Time: 16.99 sec, Loss: 0.007010235916823149, Accuracy: 99.74832916259766, Test Loss: 0.05905098840594292, Test Accuracy: 98.50999450683594
Epoch 6, Time: 17.28 sec, Loss: 0.005892242304980755, Accuracy: 99.7933349609375, Test Loss: 0.05921382084488869, Test Accuracy: 98.68000030517578
Epoch 7, Time: 17.81 sec, Loss: 0.004830869380384684, Accuracy: 99.85166931152344, Test Loss: 0.07633807510137558, Test Accuracy: 98.40999603271484
Epoch 8, Time: 17.16 sec, Loss: 0.0051274956203997135, Accuracy: 99.82833099365234, Test Loss: 0.0749225839972496, Test Accuracy: 98.40999603271484
Epoch 9, Time: 17.29 sec, Loss: 0.002916652010753751, Accuracy: 99.89167022705078, Test Loss: 0.09321436285972595, Test Accuracy: 98.19999694824219
Epoch 10, Time: 17.65 sec, Loss: 0.004324355162680149, Accuracy: 99.8499984741211, Test Loss: 0.08502297848463058, Test Accuracy: 98.38999938964844
~~~





## CNN模型

~~~python
class MyModel(Model):
    def __init__(self,kernel_number,kernel_size,D1_size,D2_size):
        super(MyModel, self).__init__()
        self.input_layer = tf.keras.layers.Input(shape=(18,18,1),name="input_l")
        self.conv1 = Conv2D(kernel_number, kernel_size, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(D1_size, activation='relu')
        self.d2 = Dense(D2_size)

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

    def get_config(self):  
        config = {
            "kernel_number": self.kernel_number,
            "kernel_size": self.kernel_size,
            "D1_size": self.D1_size,
            "D2_size": self.D2_size
        }
        base_config = super(MyModel, self).get_config()
       	return dict(list(base_config.items()) + list(config.items()))

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True) # dropout层存在时设置为True,else False
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss) # 其显示作用，主要时跟踪loss的平均值
    train_accuracy(labels, predictions)
    
@tf.function
def test_step(images, labels):
    predictions = model(images, training=False) # dropout层存在时设置为True,else False
    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)
  
# Create an instance of the model
model = MyModel(32,3,128,10)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


EPOCHS = 1

for epoch in range(EPOCHS):
  # 梯度清空
    start_time = time.time()  # 开始计时
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for images, labels in train_ds:
        train_step(images, labels)


    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    end_time = time.time()  # 结束计时
    epoch_time = end_time - start_time
    
    print(
    f'Epoch {epoch + 1}, '
    f'Time: {epoch_time:.2f} sec, '
    f'Loss: {train_loss.result()}, '
    f'Accuracy: {train_accuracy.result() * 100}, '
    f'Test Loss: {test_loss.result()}, '
    f'Test Accuracy: {test_accuracy.result() * 100}'
    )

    
# *保存模型权重
model.save_weights("./checkpoints/my_checkpoints")

# *模型导入
load_model = MyModel(32,3,128,10)
load_model.load_weights("./checkpoints/my_checkpoints")
load_model.compile(optimizer=tf.keras.optimizers.Adam(),
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])

loss,acc = load_model.evaluate(test_ds)
~~~

结果，我们可以看到在第9轮时，测试结果达到最高，约为98.57%。

~~~python
Epoch 1, Time: 25.74 sec, Loss: 0.13960197567939758, Accuracy: 95.8933334350586, Test Loss: 0.056066036224365234, Test Accuracy: 98.23999786376953
Epoch 2, Time: 25.02 sec, Loss: 0.0415259450674057, Accuracy: 98.69499969482422, Test Loss: 0.051432039588689804, Test Accuracy: 98.37999725341797
Epoch 3, Time: 27.59 sec, Loss: 0.02032860927283764, Accuracy: 99.36500549316406, Test Loss: 0.06192609667778015, Test Accuracy: 98.2699966430664
Epoch 4, Time: 27.31 sec, Loss: 0.01316783670336008, Accuracy: 99.55000305175781, Test Loss: 0.06380966305732727, Test Accuracy: 98.27999877929688
Epoch 5, Time: 26.23 sec, Loss: 0.008673087693750858, Accuracy: 99.7300033569336, Test Loss: 0.07679145038127899, Test Accuracy: 97.95999908447266
Epoch 6, Time: 25.00 sec, Loss: 0.00748553266748786, Accuracy: 99.74666595458984, Test Loss: 0.06446573883295059, Test Accuracy: 98.47999572753906
Epoch 7, Time: 25.63 sec, Loss: 0.0043510738760232925, Accuracy: 99.84833526611328, Test Loss: 0.07793684303760529, Test Accuracy: 98.2699966430664
Epoch 8, Time: 25.93 sec, Loss: 0.004396883770823479, Accuracy: 99.86499786376953, Test Loss: 0.07229559868574142, Test Accuracy: 98.37999725341797
Epoch 9, Time: 26.29 sec, Loss: 0.004857382737100124, Accuracy: 99.82167053222656, Test Loss: 0.07290682196617126, Test Accuracy: 98.56999969482422
Epoch 10, Time: 25.93 sec, Loss: 0.002418240299448371, Accuracy: 99.92166900634766, Test Loss: 0.08962585031986237, Test Accuracy: 98.38999938964844
~~~

~~~python
Fitting 3 folds for each of 6 candidates, totalling 18 fits
Best parameters found:  {'n_neighbors': 3, 'weights': 'distance'}
Accuracy: 0.9682 with params: {'n_neighbors': 3, 'weights': 'uniform'}
Accuracy: 0.9693 with params: {'n_neighbors': 3, 'weights': 'distance'}
Accuracy: 0.9674 with params: {'n_neighbors': 5, 'weights': 'uniform'}
Accuracy: 0.9686 with params: {'n_neighbors': 5, 'weights': 'distance'}
Accuracy: 0.9652 with params: {'n_neighbors': 7, 'weights': 'uniform'}
Accuracy: 0.9667 with params: {'n_neighbors': 7, 'weights': 'distance'}
Test accuracy: 0.9717
~~~






## 实战环节





~~~python

from PIL import Image
def img_box(img):
    index=np.where(img)
    weight=img[index]
    index_x_mean=np.average(index[1],weights=weight)
    index_y_mean=np.average(index[0],weights=weight)
    index_xy_std=np.sqrt(np.average(((index[1]-index_x_mean)**2+(index[0]-index_y_mean)**2)/2,
                    weights=weight))
    box=(index_x_mean-3*index_xy_std,index_y_mean-3*index_xy_std,index_x_mean+3*index_xy_std,
            index_y_mean+3*index_xy_std)
    return box


def normalize_figure(img_path):
    image0=Image.open(img_path).convert('L')
    
    img=np.array(image0)
    # print(img)
    img=np.where(img==255,0,255)
    image=Image.fromarray(img)
    box=img_box(img)
    crop_img=image.crop(box)
    norm_img=crop_img.resize((28,28))
    # plt.imshow(norm_img,cmap='Greys',interpolation='nearest')
    # plt.show()
    return norm_img

import random
img_paths = ["./手写体/"+str(i)+".png" for i in range(0,10)]
random.shuffle(img_paths) # 打乱顺序
for img_path in img_paths:
    img = normalize_figure(img_path)
    img_array = np.array(img).reshape(1,28,28,1)/255
    predictions = load_model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    plt.title(f'Label: {predicted_class}')
    plt.imshow(img,cmap='Greys',interpolation='nearest')
    plt.show()
~~~









