#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/5/27 0027 下午 15:52
# @Author : West Field
# @File : color_item_multi_output_model.py

import tensorflow as tf
import pathlib
import random
import matplotlib.pyplot as plt
import numpy as np


def color_item_multi_output():
    '''
    就是有多个输出，比如对衣服的颜色和类型同时进行预测，就是输出两个预测，即多输出模型
    :return:
    '''
    # 读取数据
    data_root = pathlib.Path("./dataset/")
    for d in data_root.iterdir():
        print(d)
    all_image_paths = list(data_root.glob("*/*"))
    image_count = len(all_image_paths)
    all_image_paths = [str(path) for path in all_image_paths]
    random.shuffle(all_image_paths)
    print(all_image_paths[:5])
    label_names = sorted(item.name for item in data_root.glob("*/") if item.is_dir())
    print(label_names)
    # 颜色名称、衣服名称
    color_label_names = set(name.split("_")[0] for name in label_names)
    item_label_names = set(name.split("_")[1] for name in label_names)
    color_label_to_index = dict((name, index) for index, name in enumerate(color_label_names))
    item_label_to_index = dict((name, index) for index, name in enumerate(item_label_names))
    print(color_label_names, item_label_names)
    print(color_label_to_index, item_label_to_index)
    # 将标签编码为0,1,2,...
    all_image_labels = [pathlib.Path(path).parent.name for path in all_image_paths]
    print(all_image_labels[:5])
    color_labels = [color_label_to_index[label.split("_")[0]] for label in all_image_labels]
    item_labels = [item_label_to_index[label.split("_")[1]] for label in all_image_labels]
    print(color_labels[:5], item_labels[:5])
    # 绘制图像
    image_path = all_image_paths[0]
    label = all_image_labels[0]
    plt.imshow((load_and_preprocess_image(image_path) + 1) / 2)  # 归一化还原
    plt.grid(False)
    plt.xlabel(label)
    plt.show()
    # 创建dataset
    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices((color_labels, item_labels))
    for ele in label_ds.take(3):
        print(ele[0].numpy(), ele[1].numpy())
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
    print(image_label_ds)
    # 数据集划分
    test_count = int(image_count * 0.2)
    train_count = image_count - test_count
    train_data = image_label_ds.skip(test_count)
    test_data = image_label_ds.take(test_count)
    BATCH_SIZE = 32
    train_data = train_data.shuffle(train_count).repeat()
    train_data = train_data.batch(BATCH_SIZE)
    train_data = train_data.prefetch(buffer_size=BATCH_SIZE)
    test_data = test_data.batch(BATCH_SIZE)
    print(train_data, test_data)

    # 使用函数式API建立模型
    # 小型预训练网络，用于部署在移动设备上。这里没有没有写weights参数，则不使用该网络的权重，仅仅使用它的架构
    mobile_net = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False)
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = mobile_net(inputs)
    print(x.get_shape())
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # 第一个分支
    x1 = tf.keras.layers.Dense(1024, activation="relu")(x)
    out_color = tf.keras.layers.Dense(len(color_label_names), activation="softmax", name="out_color")(x1)
    # 第二个分支
    x2 = tf.keras.layers.Dense(1024, activation="relu")(x)
    out_item = tf.keras.layers.Dense(len(item_label_names), activation="softmax", name="out_item")(x2)
    # 函数式API的模型创建
    model = tf.keras.Model(inputs=inputs, outputs=[out_color, out_item])
    print(model.summary())
    # 编译配置模型
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss={"out_color": "sparse_categorical_crossentropy", "out_item": "sparse_categorical_crossentropy"},
                  metrics=["accuracy"])
    # 训练模型
    train_steps = train_count // BATCH_SIZE
    test_steps = test_count // BATCH_SIZE
    model.fit(train_data, epochs=16, steps_per_epoch=train_steps, validation_data=test_data,
              validation_steps=test_steps)
    # 经过了3个epoch，两个正确率分别达到了0.97、0.98，可以调小learning_rate来寻找更好的正确率

    # 评价模型
    # model.evaluate(test_data) # 同sklearn中的model.score()
    # 对单张图像进行预测
    my_image = load_and_preprocess_image("./dataset/red_dress/00000002.jpg")
    print(my_image.shape)
    np.expand_dims(my_image, 0)  # 在第0个维度进行扩张一维，也可以使用tf.expand_dims(my_image, 0)
    print(my_image.shape)
    pre = model.predict(my_image)
    print(pre)
    # 调用model的方式进行预测，training=False时，调用model和使用predict的预测等价
    pre = model(my_image, training=False)  # training=False表示预测模式
    print(pre)


def load_and_preprocess_image(path):
    '''
    加载和格式化图像
    :param path:
    :return:
    '''
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    image = 2 * image - 1  # 归一化到-1到1之间
    return image


if __name__ == "__main__":
    # 颜色和服装多输出模型
    color_item_multi_output()
