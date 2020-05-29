#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/5/29 0029 下午 12:45
# @Author : West Field
# @File : model_save.py

import tensorflow as tf
import matplotlib.pyplot as plt


def overall_save():
    '''
    保存整体模型
    保存整个模型是指将整个模型保存到一个文件中，其中包括权重值、模型配置(架构)、优化器配置。
    恢复模型之后，可以用这个模型进行预测，也可以在此模型基础之上继续训练，即为模型设置检查点，并稍后从完全相同的状态继续训练而无需访问源代码。
    keras中使用HDF5标准提供基本的保存格式。
    :return:
    '''
    # 数据处理
    (train_image, train_label), (test_image, test_label) = tf.keras.datasets.fashion_mnist.load_data()
    print(train_image.shape, train_label.shape, test_image.shape, test_label.shape)
    plt.imshow(train_image[0])
    plt.show()
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
    model.fit(train_image, train_label, epochs=3)
    # 测试
    print(model.evaluate(test_image, test_label, verbose=0))

    # 保存模型
    model.save("lesson_model.h5")
    # 使用保存好的模型
    new_model = tf.keras.models.load_model("lesson_model.h5")
    print(new_model.summary())  # 说明保存了模型架构。
    print(new_model.evaluate(test_image, test_label, verbose=0))  # 可以evaluate()，说明保存了优化器。预测结构和之前一样说明保存了权重。


def only_structure_save():
    '''
    仅仅保持模型架构
    有时我们只对模型的j架构感兴趣，而无需保存权重值和优化器。在这种情况下，可以仅保存模型的“配置”。
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
    model.fit(train_image, train_label, epochs=3)
    # 测试
    print(model.evaluate(test_image, test_label, verbose=0))

    # 仅保存模型架构，没有保存权重和优化器
    model_config = model.to_json()  # 也可以保存到磁盘上
    print(model_config)
    # 重建模型
    reinitialized_model = tf.keras.models.model_from_json(model_config)
    print(reinitialized_model.summary())
    # 测试
    # print(reinitialized_model.evaluate(test_image, test_label, verbose=0)) # 会报错提示没有编译模型，说明没有保存优化器
    # 如果需要用这个模型，就需要重新编译配置
    reinitialized_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"])
    print(reinitialized_model.evaluate(test_image, test_label, verbose=0))  # 可以预测，但是结果很差，因为是随机初始化的权重


def only_weights_save():
    '''
    仅仅保存模型的权重
    有时我们只需保存模型的状态(其权重值)，而对模型架构不感兴趣。在这种情况下，可以通过get_weights()获取权重值，并通过set_weights()设置权重值。
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
    model.fit(train_image, train_label, epochs=3)
    # 测试
    print(model.evaluate(test_image, test_label, verbose=0))

    # 仅保存权重
    weights = model.get_weights()
    model_config = model.to_json()  # 读取模型架构
    reinitialized_model = tf.keras.models.model_from_json(model_config)  # 仅有架构的模型
    reinitialized_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"])  # 配置优化器
    reinitialized_model.set_weights(weights)  # 设置权重
    print(reinitialized_model.evaluate(test_image, test_label, verbose=0))

    # 将权重保存到磁盘上
    model.save_weights("lesson_weights.h5")
    # 加载保存的权重
    model_config = model.to_json()  # 读取模型架构
    reinitialized_model = tf.keras.models.model_from_json(model_config)  # 仅有架构的模型
    reinitialized_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"])  # 配置优化器
    reinitialized_model.load_weights("lesson_weights.h5")
    print(reinitialized_model.evaluate(test_image, test_label, verbose=0))


if __name__ == "__main__":
    # 保存模型整体
    # overall_save()
    # 仅仅保持模型架构
    # only_structure_save()
    # 仅仅保存模型的权重
    only_weights_save()
