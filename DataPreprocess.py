import tensorflow as tf
import matplotlib.pyplot as plt

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