import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense,Flatten
import time
from DataPreprocess import test_ds,train_ds


class DNN(Model):
    def __init__(self):
        super(DNN, self).__init__()
        self.flatten = Flatten(input_shape=(28, 28, 1))
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(128, activation='relu')
        self.d3 = Dense(10)

    def call(self, x, training=False):
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        return x


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