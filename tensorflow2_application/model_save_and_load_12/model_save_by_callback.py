#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/5/29 0029 下午 16:56
# @Author : West Field
# @File : model_save_by_callback.py

import tensorflow as tf


def use_callback_save():
    '''
    使用回调函数保存模型
    在训练期间或训练结束自动保存检查点。这样一来，您便可以使用经过训练的模型，而无需重新训练该模型，或从暂停的地方继续训练，以防训练过程中中断。
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
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"])

    # 保存检查点
    checkpoint_path = "training_cp/cp.ckpt"
    # 如果save_best_only为True时，保存最好的monitor指标对应weights的模型。save_weights=True表示只保存weights。period=1是每一个epoch保存一次检查点。
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=False,
                                                     save_weights_only=True, period=1)
    # 使用回调函数保存模型
    model.fit(train_image, train_label, epochs=3, callbacks=[cp_callback])  # 这样在训练过程中每个epoch之后会去自动保存模型权重


def load_model():
    '''
    加载保存好的检查点
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
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"])
    # 检测
    print(model.evaluate(test_image, test_label))  # 此时模型未经训练，weights是随机初始化的，准确率很差

    # 加载检查点
    checkpoint_path = "training_cp/cp.ckpt"
    model.load_weights(checkpoint_path)
    # 检测
    print(model.evaluate(test_image, test_label))  # 会发现准确率上来了


if __name__ == "__main__":
    # 使用回调函数保存模型
    # use_callback_save()
    # 加载保存好的检查点
    load_model()
