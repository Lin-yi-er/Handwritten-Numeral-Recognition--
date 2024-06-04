import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D
import time
from DataPreprocess import test_ds,train_ds


class MyModel(Model):
    def __init__(self, kernel_number, kernel_size, D1_size, D2_size):
        super(MyModel, self).__init__()
        self.input_layer = tf.keras.layers.Input(shape=(18, 18, 1), name="input_l")
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


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)  # dropout层存在时设置为True,else False
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)  # 其显示作用，主要时跟踪loss的平均值
    train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
    predictions = model(images, training=False)  # dropout层存在时设置为True,else False
    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)


# Create an instance of the model
model = MyModel(32, 3, 128, 10)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

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

# *保存模型权重
model.save_weights("./checkpoints/my_checkpoints")

# *模型导入
load_model = MyModel(32, 3, 128, 10)
load_model.load_weights("./checkpoints/my_checkpoints")
load_model.compile(optimizer=tf.keras.optimizers.Adam(),
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])

loss, acc = load_model.evaluate(test_ds)