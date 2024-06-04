# Handwritten-Numeral-Recognition--
$\text{By comparing SVM, CNN, DNN and KNN, we tried to find the optimal model.}$
$\text{Finally, the uploaded image was passed into the trained model through pre-processing methods such as cutting important areas, and the result was obtained}$

$我们基于minst数据集训练出适用于手写体识别的模型$


# instructions

$\text{just simply run the relevant model file} $

$\text{if you want cnn,just run CNN.py}$


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
$tensorflow 2.10.0 $

$sklearn 1.4.1.post1 $

$numpy 1.26.4 $

$matplotlib3.5.1 $







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



$\text{为了便于寻找模型的最优参数，我们使用网格搜索法遍历每种组合。}$
$\text{网格搜索针对超参数组合列表中的每一个组合，实例化给定的模型，做n次cv(交叉验证)。}$
$\text{将平均得分最高的超参数组合作为最佳的选择，返回模型对象。}$





## SVM模型

~~~python

param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
    'kernel': ['linear', 'rbf']
}

svm_model = svm.SVC()

grid_search = GridSearchCV(svm_model, param_grid, cv=3, verbose=2, n_jobs=-1)

~~~



$最优参数组合为$ `'C': 10, 'gamma': 'scale', 'kernel': 'rbf'` $，此时测试结果为98.37%。$

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
param_grid = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}

knn_model = KNeighborsClassifier()

grid_search = GridSearchCV(knn_model, param_grid, cv=3, verbose=2, n_jobs=-1)
grid_search.fit(x_train_flat, y_train)
~~~

$最优参数组合为$ `'n_neighbors': 3, 'weights': 'distance'` $，此时测试结果为97.17%。$
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
~~~


$结果大概在第四轮达到最大，约为97.90%，可以通过设置早停使其在最高值时停止。$

~~~python
Epoch 1, Time: 2.11 sec, Loss: 0.22810868918895721, Accuracy: 93.07666778564453, Test Loss: 0.10587550699710846, Test Accuracy: 96.76000213623047
Epoch 2, Time: 1.72 sec, Loss: 0.09427641332149506, Accuracy: 97.11499786376953, Test Loss: 0.08010327816009521, Test Accuracy: 97.39999389648438
Epoch 3, Time: 1.59 sec, Loss: 0.06614474952220917, Accuracy: 97.96666717529297, Test Loss: 0.09041441977024078, Test Accuracy: 97.25999450683594
Epoch 4, Time: 1.60 sec, Loss: 0.05139654874801636, Accuracy: 98.29499816894531, Test Loss: 0.08708411455154419, Test Accuracy: 97.27999877929688
Epoch 5, Time: 1.59 sec, Loss: 0.0400259904563427, Accuracy: 98.70500183105469, Test Loss: 0.08054660260677338, Test Accuracy: 97.72999572753906
Epoch 6, Time: 1.60 sec, Loss: 0.03196844458580017, Accuracy: 98.90833282470703, Test Loss: 0.07868018001317978, Test Accuracy: 97.86000061035156
Epoch 7, Time: 1.60 sec, Loss: 0.02776424214243889, Accuracy: 99.04833221435547, Test Loss: 0.07842708379030228, Test Accuracy: 97.73999786376953
Epoch 8, Time: 1.59 sec, Loss: 0.02377120405435562, Accuracy: 99.18000030517578, Test Loss: 0.09592799097299576, Test Accuracy: 97.47000122070312
Epoch 9, Time: 1.60 sec, Loss: 0.019003167748451233, Accuracy: 99.35832977294922, Test Loss: 0.0835547149181366, Test Accuracy: 97.89999389648438
Epoch 10, Time: 1.59 sec, Loss: 0.017659997567534447, Accuracy: 99.3883285522461, Test Loss: 0.11550948768854141, Test Accuracy: 97.58999633789062
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
~~~

$结果，我们可以看到在第9轮时，测试结果达到最高，约为98.54%。$

~~~python
Epoch 1, Time: 18.38 sec, Loss: 0.1359022557735443, Accuracy: 95.82666778564453, Test Loss: 0.05962755158543587, Test Accuracy: 97.98999786376953
Epoch 2, Time: 18.14 sec, Loss: 0.04215861111879349, Accuracy: 98.72000122070312, Test Loss: 0.04919729381799698, Test Accuracy: 98.31999969482422
Epoch 3, Time: 18.09 sec, Loss: 0.022303974255919456, Accuracy: 99.26666259765625, Test Loss: 0.05520520731806755, Test Accuracy: 98.23999786376953
Epoch 4, Time: 18.47 sec, Loss: 0.012478834018111229, Accuracy: 99.5999984741211, Test Loss: 0.05839865654706955, Test Accuracy: 98.2699966430664
Epoch 5, Time: 19.21 sec, Loss: 0.00852228607982397, Accuracy: 99.70500183105469, Test Loss: 0.06275804340839386, Test Accuracy: 98.37999725341797
Epoch 6, Time: 19.40 sec, Loss: 0.0068765003234148026, Accuracy: 99.7683334350586, Test Loss: 0.060312796384096146, Test Accuracy: 98.54000091552734
Epoch 7, Time: 19.52 sec, Loss: 0.00420769676566124, Accuracy: 99.87000274658203, Test Loss: 0.0685533732175827, Test Accuracy: 98.43000030517578
Epoch 8, Time: 21.30 sec, Loss: 0.005770161747932434, Accuracy: 99.80333709716797, Test Loss: 0.08168778568506241, Test Accuracy: 98.38999938964844
Epoch 9, Time: 18.60 sec, Loss: 0.004067363683134317, Accuracy: 99.85832977294922, Test Loss: 0.06709392368793488, Test Accuracy: 98.54000091552734
Epoch 10, Time: 21.32 sec, Loss: 0.002288063056766987, Accuracy: 99.93000030517578, Test Loss: 0.10075352340936661, Test Accuracy: 98.12999725341797
313/313 [==============================] - 1s 2ms/step - loss: 0.1009 - accuracy: 0.9813
~~~







## 实战环节


$\text{创建了一个名为“手写体”的文件夹，里面放置了上传的图片，名字即为他们的类别。如一张“9”的图片，它的名字为“9.png”。}$


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









