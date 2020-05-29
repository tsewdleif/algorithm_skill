#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/5/29 0029 下午 18:55
# @Author : West Field
# @File : model_save_in_custom_training.py

import tensorflow as tf
import os


def model_save_in_custom_training():
    '''
    在自定义训练中保存模型检查点
    :return:
    '''
    # 数据处理
    (train_image, train_label), (test_image, test_label) = tf.keras.datasets.fashion_mnist.load_data()
    print(train_image.shape, train_label.shape, test_image.shape, test_label.shape)
    # 归一化
    train_image = train_image / 255
    test_image = test_image / 255
    # 建立模型
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))
    print(model.summary())
    # 优化器
    optimizer = tf.keras.optimizers.Adam()
    # 损失函数
    loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # from_logits=True表示最后一层有激活函数

    # 定义计算loss的函数
    def loss(model, x, y):
        y_ = model(x)
        return loss_func(y, y_)

    # 定义train_step
    def train_step(model, images, labels):
        with tf.GradientTape() as t:
            pred = model(images)
            loss_step = loss_func(labels, pred)
        grads = t.gradient(loss_step, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        # 计算训练过程中loss、acc变化情况
        train_loss(loss_step)
        train_accuracy(labels, pred)

    train_loss = tf.keras.metrics.Mean("train_loss", dtype=tf.float32)
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy("train_accuracy")
    test_loss = tf.keras.metrics.Mean("test_loss", dtype=tf.float32)
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy("test_accuracy")

    # 保存检查点的路径
    cp_dir = "./customtrain_cp"
    # 保存检查点的前缀
    cp_prefix = os.path.join(cp_dir, "ckpt")
    # 初始化检查点的一个类，在训练过程中将weights和optimizer保存到checkpoint中
    checkpoint = tf.train.Checkpoint(  # 定义要保存的东西。在训练过程中，模型改变的是weights和优化器(这里的Adam自适应优化器)，所以这里需要保存优化器和model。
        optimizer=optimizer,
        model=model
    )

    # 建立dataset
    dataset = tf.data.Dataset.from_tensor_slices((train_image, train_label))
    dataset = dataset.shuffle(10000).batch(32)

    # train函数
    def train():
        for epoch in range(5):
            for (batch, (images, labels)) in enumerate(dataset):
                train_step(model, images, labels)
            print("Epoch{} loss is {}".format(epoch, train_loss.result()))
            print("Epoch{} accuracy is {}".format(epoch, train_accuracy.result()))
            train_loss.reset_states()
            train_accuracy.reset_states()
            # 每隔2个epoch，保存一次检查点
            if (epoch + 1) % 2 == 0:
                checkpoint.save(file_prefix=cp_prefix)

    # 开始模型训练
    train()


def model_restore():
    '''
    重新恢复模型
    :return:
    '''
    # 数据处理
    (train_image, train_label), (test_image, test_label) = tf.keras.datasets.fashion_mnist.load_data()
    print(train_image.shape, train_label.shape, test_image.shape, test_label.shape)
    # 归一化
    train_image = train_image / 255
    test_image = test_image / 255
    # 建立模型
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))
    print(model.summary())
    # 优化器
    optimizer = tf.keras.optimizers.Adam()

    # 保存检查点的路径
    cp_dir = "./customtrain_cp"
    # 初始化检查点的一个类
    checkpoint = tf.train.Checkpoint(
        optimizer=optimizer,
        model=model
    )

    # 取出最新的检查点
    latest_ckpt = tf.train.latest_checkpoint(cp_dir)
    print(latest_ckpt)
    # model在训练集上的正确率，由于是自定义训练，无法使用model.evaluate()，直接调用模型
    pred = model(train_image, training=False)  # 此时由于模型没有训练，所以预测效果很差
    print(pred)
    print(tf.argmax(pred, axis=-1).numpy(), train_label)
    # 把最新的检查点恢复出来
    checkpoint.restore(latest_ckpt)  # 恢复是根据模型中的权重或优化器的名称恢复所有的参数值
    pred = model(train_image, training=False)  # 此时模型准确率才可以
    print(tf.argmax(pred, axis=-1).numpy(), train_label)
    # 计算准确率
    print((tf.argmax(pred, axis=-1).numpy() == train_label).sum() / len(train_label))


if __name__ == "__main__":
    # 在自定义训练中保存模型检查点
    # model_save_in_custom_training()
    # 重新恢复模型
    model_restore()
