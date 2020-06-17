#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/5/26 0026 上午 9:01
# @Author : West Field
# @File : cat_dog_custom_training.py

import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import numpy as np


def cat_dog_category():
    '''
    猫狗图像分类自定义训练
    :return:
    '''
    train_image_path = glob.glob('./dc/train/*/*.jpg')
    # print(len((train_image_path)))
    # print(train_image_path[:5])
    train_image_label = [int(p.split('\\')[1] == 'cat') for p in train_image_path]
    print(train_image_label[:5], train_image_label[-5:])
    # 建立dataset
    train_image_ds = tf.data.Dataset.from_tensor_slices((train_image_path, train_image_label))
    # 根据CPU个数自动设置是否并行运算
    AUTOTUNE = tf.data.experimental.AUTOTUNE  # experimental，后边可能会废弃
    train_image_ds = train_image_ds.map(load_preprocess_image, num_parallel_calls=AUTOTUNE)
    print(train_image_ds)
    # shuffle batch
    BATCH_SIZE = 16  # 如果超出分配内存则须减小BATCH_SIZE
    train_count = len(train_image_path)
    train_image_ds = train_image_ds.shuffle(train_count).batch(BATCH_SIZE)
    # 预取，在前边一部分数据训练的过程中，会在后台预取另一部分数据，可以加速训练过程
    train_image_ds = train_image_ds.prefetch(BATCH_SIZE)
    # 构建验证集dataset
    test_image_path = glob.glob('./dc/test/*/*.jpg')
    test_image_label = [int(p.split('\\')[1] == 'cat') for p in test_image_path]
    test_image_ds = tf.data.Dataset.from_tensor_slices((test_image_path, test_image_label))
    test_image_ds = test_image_ds.map(load_preprocess_image, num_parallel_calls=AUTOTUNE)
    test_image_ds = test_image_ds.batch(BATCH_SIZE)  # 验证集无需shuffle()
    test_image_ds = test_image_ds.prefetch(BATCH_SIZE)
    # 在2.0的eager模式下，dataset默认是可迭代的，用iter方法把他转化成生成器，用next方法取出下一个批次的数据
    imgs, labels = next(iter(train_image_ds))
    print(imgs.shape)
    plt.imshow(imgs[0])
    plt.show()
    print(labels[0])
    # 创建模型
    model = tf.keras.Sequential([
        # 第一层一般都是卷积层，因为卷积层提取特征的能力比较强
        tf.keras.layers.Conv2D(64, (3, 3), input_shape=(256, 256, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(1024, (3, 3), activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        # 二分类问题，输出一个值，就是一个逻辑回归问题。
        # 这里不对它进行激活，sigmoid就是一个把x映射到y，y范围是0到1，x<0时，y为0到0.5，x>0时，y为0.5到1，所以不需要激活，只需判断输出是否大于0或小于0，就能判断出猫狗。
        # sigmoid、softmax，它们的本质就是一个归一化，使得我们的输出映射到0到1之间，sigmoid是小于0和大于0作为分类，softmax可以认为是输出中哪个分量大就是哪个分类，激活就是把概率值做了一个归一化，所以这里不做激活，是完全没有问题的。
        tf.keras.layers.Dense(1)
    ])
    print(model.summary())
    # 调用模型，来做预测
    pred = model(imgs)
    print(pred.shape)
    # 可以认为预测数据大于0，预测结果为1，小于0，预测结果为0，数据绝对值越大表示预测越有信心
    print(pred)
    print(np.array([p[0].numpy() for p in tf.cast(pred > 0, tf.int32)]))  # 由于pred是二维的，所以要是p[0]
    print(np.array([l[0].numpy() for l in labels]))  # 实际值
    # 自定义训练模型
    # 自定义损失函数
    ls = tf.keras.losses.BinaryCrossentropy(
        from_logits=True)  # from_logits=True表示最后一层没有使用激活函数，它会在内部对它进行一个sigmoid运算，然后再计算二元的交叉熵损失
    # 自定义优化器
    optimizer = tf.keras.optimizers.Adam()  # 默认学习率0.001
    # metric模块，记录每个epoch的metric变化
    epoch_loss_avg = tf.keras.metrics.Mean('train_loss')
    train_accuracy = tf.keras.metrics.Accuracy()
    # 验证集的metric模块，记录每个epoch的metric变化
    epoch_loss_avg_test = tf.keras.metrics.Mean('test_loss')
    test_accuracy = tf.keras.metrics.Accuracy()

    # 定义一个批次的训练
    def train_step(model, images, labels):
        # 创建上下文管理器，记录运算过程
        with tf.GradientTape() as t:
            pred = model(images)
            # 计算一步训练的损失
            loss_step = ls(labels, pred)
            # 计算梯度
        grads = t.gradient(loss_step, model.trainable_variables)
        # 对梯度进行优化
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        # metrics
        epoch_loss_avg(loss_step)
        train_accuracy(labels, tf.cast(pred > 0, tf.int32))

    # 定义一个批次的验证集的metrics
    def test_step(model, images, labels):
        pred = model(images, training=False)  # 不训练，或者使用model.predict(images)
        loss_step = ls(labels, pred)
        # metrics
        epoch_loss_avg(loss_step)
        train_accuracy(labels, tf.cast(pred > 0, tf.int32))

    # 记录每个epoch的metrics
    train_loss_results = []
    train_acc_results = []
    # 记录验证集的每个epoch的metrics
    test_loss_results = []
    test_acc_results = []
    num_epochs = 30
    # 训练每个epoch
    for epoch in range(num_epochs):
        # 训练集
        for imgs_, labels_ in train_image_ds:
            train_step(model, imgs_, labels_)
            print(".", end="", flush=True)  # 这里用end=""，须加上flush=True才会在控制台输出
        print()  # 换行
        # 记录每个epoch的metrics
        train_loss_results.append(epoch_loss_avg.result())
        train_acc_results.append(train_accuracy.result())
        # 验证集
        for imgs_, labels_ in test_image_ds:
            test_step(model, imgs_, labels_)
        # 记录验证集每个epoch的metrics
        test_loss_results.append(epoch_loss_avg_test.result())
        test_acc_results.append(test_accuracy.result())
        # 打印每个epoch的metrics信息
        print("Epoch: {}: loss: {:.3f}, accuracy: {:.3f}, test_loss: {:.3f}, tesst_accuracy: {:.3f}".format(epoch + 1,
                                                                                                            epoch_loss_avg.result(),
                                                                                                            train_accuracy.result(),
                                                                                                            epoch_loss_avg_test.result(),
                                                                                                            test_accuracy.result()))
        # 每个epoch重置metrics
        epoch_loss_avg.reset_states()
        train_accuracy.reset_states()
        epoch_loss_avg_test.reset_states()
        test_accuracy.reset_states()


def load_preprocess_image(path, label):
    '''
    读取图片内容
    :param path:
    :param label:
    :return:
    '''
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [256, 256])
    image = tf.cast(image, tf.float32)
    image = image / 255
    label = tf.reshape(label, [1])
    return image, label


def cat_dog_optimization():
    '''
    猫狗分类模型优化：
    由于上边的模型的精确度不高，需要优化模型，增加层数来增加网络深度。
    增加网络容量即增加网络的可训练参数，有两个方案，即增加网络的深度，和增加每层的宽度。
    增加宽度不如增加深度，增加宽度容易产生过拟合。
    模型过深会导致梯度消失，使得模型不可训练。
    每层的filter个数不能太少，太少会使图像的特征信息不能很好的穿过每一层，造成信息传递的瓶颈，所以第一层设置成了64个，...
    :return:
    '''
    train_image_path = glob.glob('./dc/train/*/*.jpg')
    train_image_label = [int(p.split('\\')[1] == 'cat') for p in train_image_path]
    # 建立dataset
    train_image_ds = tf.data.Dataset.from_tensor_slices((train_image_path, train_image_label))
    # 根据CPU个数自动设置是否并行运算
    AUTOTUNE = tf.data.experimental.AUTOTUNE  # experimental，后边可能会废弃
    train_image_ds = train_image_ds.map(load_preprocess_image, num_parallel_calls=AUTOTUNE)
    # shuffle batch
    BATCH_SIZE = 16  # 如果超出分配内存则须减小BATCH_SIZE
    train_count = len(train_image_path)
    train_image_ds = train_image_ds.shuffle(train_count).batch(BATCH_SIZE)
    # 预取，在前边一部分数据训练的过程中，会在后台预取另一部分数据，可以加速训练过程
    train_image_ds = train_image_ds.prefetch(BATCH_SIZE)
    # 构建验证集dataset
    test_image_path = glob.glob('./dc/test/*/*.jpg')
    test_image_label = [int(p.split('\\')[1] == 'cat') for p in test_image_path]
    test_image_ds = tf.data.Dataset.from_tensor_slices((test_image_path, test_image_label))
    test_image_ds = test_image_ds.map(load_preprocess_image, num_parallel_calls=AUTOTUNE)
    test_image_ds = test_image_ds.batch(BATCH_SIZE)  # 验证集无需shuffle()
    test_image_ds = test_image_ds.prefetch(BATCH_SIZE)
    # 创建模型
    model = tf.keras.Sequential([
        # 第一层一般都是卷积层，因为卷积层提取特征的能力比较强
        tf.keras.layers.Conv2D(64, (3, 3), input_shape=(256, 256, 3), activation='relu'),
        # 每一个卷积层后边使用批标准化BatchNormalization，可以抑制过拟合，使得可以构建更深的网络模型
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), input_shape=(256, 256, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(1024, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(1024, (3, 3), activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        # 二分类问题，输出一个值，就是一个逻辑回归问题。
        # 这里不对它进行激活，sigmoid就是一个把x映射到y，y范围是0到1，x<0时，y为0到0.5，x>0时，y为0.5到1，所以不需要激活，只需判断输出是否大于0或小于0，就能判断出猫狗。
        # sigmoid、softmax，它们的本质就是一个归一化，使得我们的输出映射到0到1之间，sigmoid是小于0和大于0作为分类，softmax可以认为是输出中哪个分量大就是哪个分类，激活就是把概率值做了一个归一化，所以这里不做激活，是完全没有问题的。
        tf.keras.layers.Dense(1)
    ])
    # 自定义训练模型
    # 自定义损失函数
    ls = tf.keras.losses.BinaryCrossentropy(
        from_logits=True)  # from_logits=True表示最后一层没有使用激活函数，它会在内部对它进行一个sigmoid运算，然后再计算二元的交叉熵损失
    # 自定义优化器
    optimizer = tf.keras.optimizers.Adam()  # 默认学习率0.001
    # metric模块，记录每个epoch的metric变化
    epoch_loss_avg = tf.keras.metrics.Mean('train_loss')
    train_accuracy = tf.keras.metrics.Accuracy()
    # 验证集的metric模块，记录每个epoch的metric变化
    epoch_loss_avg_test = tf.keras.metrics.Mean('test_loss')
    test_accuracy = tf.keras.metrics.Accuracy()

    # 定义一个批次的训练
    def train_step(model, images, labels):
        # 创建上下文管理器，记录运算过程
        with tf.GradientTape() as t:
            pred = model(images)
            # 计算一步训练的损失
            loss_step = ls(labels, pred)
            # 计算梯度
        grads = t.gradient(loss_step, model.trainable_variables)
        # 对梯度进行优化
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        # metrics
        epoch_loss_avg(loss_step)
        train_accuracy(labels, tf.cast(pred > 0, tf.int32))

    # 定义一个批次的验证集的metrics
    def test_step(model, images, labels):
        pred = model(images, training=False)  # 不训练，或者使用model.predict(images)
        loss_step = ls(labels, pred)
        # metrics
        epoch_loss_avg(loss_step)
        train_accuracy(labels, tf.cast(pred > 0, tf.int32))

    # 记录每个epoch的metrics
    train_loss_results = []
    train_acc_results = []
    # 记录验证集的每个epoch的metrics
    test_loss_results = []
    test_acc_results = []
    num_epochs = 30
    # 训练每个epoch
    for epoch in range(num_epochs):
        # 训练集
        for imgs_, labels_ in train_image_ds:
            train_step(model, imgs_, labels_)
            print(".", end="", flush=True)  # 这里用end=""，须加上flush=True才会在控制台输出
        print()  # 换行
        # 记录每个epoch的metrics
        train_loss_results.append(epoch_loss_avg.result())
        train_acc_results.append(train_accuracy.result())
        # 验证集
        for imgs_, labels_ in test_image_ds:
            test_step(model, imgs_, labels_)
        # 记录验证集每个epoch的metrics
        test_loss_results.append(epoch_loss_avg_test.result())
        test_acc_results.append(test_accuracy.result())
        # 打印每个epoch的metrics信息
        print("Epoch: {}: loss: {:.3f}, accuracy: {:.3f}, test_loss: {:.3f}, tesst_accuracy: {:.3f}".format(epoch + 1,
                                                                                                            epoch_loss_avg.result(),
                                                                                                            train_accuracy.result(),
                                                                                                            epoch_loss_avg_test.result(),
                                                                                                            test_accuracy.result()))
        # 每个epoch重置metrics
        epoch_loss_avg.reset_states()
        train_accuracy.reset_states()
        epoch_loss_avg_test.reset_states()
        test_accuracy.reset_states()


def cat_dog_image_enhancement():
    '''
    抑制过拟合：
    最好的方法就是增加样本，其次才是其他的方法，如dropout机制，如果数据实在是少，可以使用图像增强技术来机制过拟合。
    图像增强，相当于产生了更多的样本，可以抑制过拟合，使得模型的可迁移性更高
    :return:
    '''
    train_image_path = glob.glob('./dc/train/*/*.jpg')
    train_image_label = [int(p.split('\\')[1] == 'cat') for p in train_image_path]
    # 建立dataset
    train_image_ds = tf.data.Dataset.from_tensor_slices((train_image_path, train_image_label))
    # 根据CPU个数自动设置是否并行运算
    AUTOTUNE = tf.data.experimental.AUTOTUNE  # experimental，后边可能会废弃
    train_image_ds = train_image_ds.map(load_preprocess_image2, num_parallel_calls=AUTOTUNE)
    # 查看图像
    for img, label in train_image_ds.take(1):
        plt.imshow(img)
        plt.show()
    # shuffle batch
    BATCH_SIZE = 16  # 如果超出分配内存则须减小BATCH_SIZE
    train_count = len(train_image_path)
    train_image_ds = train_image_ds.shuffle(train_count).batch(BATCH_SIZE)
    # 预取，在前边一部分数据训练的过程中，会在后台预取另一部分数据，可以加速训练过程
    train_image_ds = train_image_ds.prefetch(BATCH_SIZE)
    # 构建验证集dataset
    test_image_path = glob.glob('./dc/test/*/*.jpg')
    test_image_label = [int(p.split('\\')[1] == 'cat') for p in test_image_path]
    test_image_ds = tf.data.Dataset.from_tensor_slices((test_image_path, test_image_label))
    # 对于验证集数据没必要做数据增强
    test_image_ds = test_image_ds.map(load_preprocess_image, num_parallel_calls=AUTOTUNE)
    test_image_ds = test_image_ds.batch(BATCH_SIZE)  # 验证集无需shuffle()
    test_image_ds = test_image_ds.prefetch(BATCH_SIZE)
    # 创建模型
    model = tf.keras.Sequential([
        # 第一层一般都是卷积层，因为卷积层提取特征的能力比较强
        tf.keras.layers.Conv2D(64, (3, 3), input_shape=(256, 256, 3), activation='relu'),
        # 每一个卷积层后边使用批标准化BatchNormalization，可以抑制过拟合，使得可以构建更深的网络模型
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(1024, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(1024, (3, 3), activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        # 二分类问题，输出一个值，就是一个逻辑回归问题。
        # 这里不对它进行激活，sigmoid就是一个把x映射到y，y范围是0到1，x<0时，y为0到0.5，x>0时，y为0.5到1，所以不需要激活，只需判断输出是否大于0或小于0，就能判断出猫狗。
        # sigmoid、softmax，它们的本质就是一个归一化，使得我们的输出映射到0到1之间，sigmoid是小于0和大于0作为分类，softmax可以认为是输出中哪个分量大就是哪个分类，激活就是把概率值做了一个归一化，所以这里不做激活，是完全没有问题的。
        tf.keras.layers.Dense(1)
    ])
    # 自定义训练模型
    # 自定义损失函数
    ls = tf.keras.losses.BinaryCrossentropy(
        from_logits=True)  # from_logits=True表示最后一层没有使用激活函数，它会在内部对它进行一个sigmoid运算，然后再计算二元的交叉熵损失
    # 自定义优化器
    optimizer = tf.keras.optimizers.Adam()  # 默认学习率0.001
    # metric模块，记录每个epoch的metric变化
    epoch_loss_avg = tf.keras.metrics.Mean('train_loss')
    train_accuracy = tf.keras.metrics.Accuracy()
    # 验证集的metric模块，记录每个epoch的metric变化
    epoch_loss_avg_test = tf.keras.metrics.Mean('test_loss')
    test_accuracy = tf.keras.metrics.Accuracy()

    # 定义一个批次的训练
    def train_step(model, images, labels):
        # 创建上下文管理器，记录运算过程
        with tf.GradientTape() as t:
            pred = model(images)
            # 计算一步训练的损失
            loss_step = ls(labels, pred)
            # 计算梯度
        grads = t.gradient(loss_step, model.trainable_variables)
        # 对梯度进行优化
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        # metrics
        epoch_loss_avg(loss_step)
        train_accuracy(labels, tf.cast(pred > 0, tf.int32))

    # 定义一个批次的验证集的metrics
    def test_step(model, images, labels):
        pred = model(images, training=False)  # 不训练，或者使用model.predict(images)
        loss_step = ls(labels, pred)
        # metrics
        epoch_loss_avg(loss_step)
        train_accuracy(labels, tf.cast(pred > 0, tf.int32))

    # 记录每个epoch的metrics
    train_loss_results = []
    train_acc_results = []
    # 记录验证集的每个epoch的metrics
    test_loss_results = []
    test_acc_results = []
    num_epochs = 30
    # 训练每个epoch
    for epoch in range(num_epochs):
        # 训练集
        for imgs_, labels_ in train_image_ds:
            train_step(model, imgs_, labels_)
            print(".", end="", flush=True)  # 这里用end=""，须加上flush=True才会在控制台输出
        print()  # 换行
        # 记录每个epoch的metrics
        train_loss_results.append(epoch_loss_avg.result())
        train_acc_results.append(train_accuracy.result())
        # 验证集
        for imgs_, labels_ in test_image_ds:
            test_step(model, imgs_, labels_)
        # 记录验证集每个epoch的metrics
        test_loss_results.append(epoch_loss_avg_test.result())
        test_acc_results.append(test_accuracy.result())
        # 打印每个epoch的metrics信息
        print("Epoch: {}: loss: {:.3f}, accuracy: {:.3f}, test_loss: {:.3f}, tesst_accuracy: {:.3f}".format(epoch + 1,
                                                                                                            epoch_loss_avg.result(),
                                                                                                            train_accuracy.result(),
                                                                                                            epoch_loss_avg_test.result(),
                                                                                                            test_accuracy.result()))
        # 每个epoch重置metrics
        epoch_loss_avg.reset_states()
        train_accuracy.reset_states()
        epoch_loss_avg_test.reset_states()
        test_accuracy.reset_states()


def load_preprocess_image2(path, label):
    '''
    图像增强
    :param path:
    :param label:
    :return:
    '''
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [360, 360])
    # 随机裁剪指定大小的图像
    image = tf.image.random_crop(image, [256, 256, 3])
    # 随机上下左右翻转
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_flip_left_right(image)
    # 随机改变亮度
    # image = tf.image.random_brightness(image, 0.5)
    # # 随机改变对比度
    # image = tf.image.random_contrast(image, 0, 1)
    # # 随机改变颜色
    # image = tf.image.random_hue(image, max_delta=0.3)
    image = tf.cast(image, tf.float32)
    image = image / 255
    label = tf.reshape(label, [1])
    return image, label


def cat_dog_vgg():
    '''
    使用VGG网络优化模型（这里和VGG相似）
    :return:
    '''
    train_image_path = glob.glob('./dc/train/*/*.jpg')
    train_image_label = [int(p.split('\\')[1] == 'cat') for p in train_image_path]
    # 建立dataset
    train_image_ds = tf.data.Dataset.from_tensor_slices((train_image_path, train_image_label))
    # 根据CPU个数自动设置是否并行运算
    AUTOTUNE = tf.data.experimental.AUTOTUNE  # experimental，后边可能会废弃
    train_image_ds = train_image_ds.map(load_preprocess_image2, num_parallel_calls=AUTOTUNE)
    # 查看图像
    for img, label in train_image_ds.take(1):
        plt.imshow(img)
        plt.show()
    # shuffle batch
    BATCH_SIZE = 16  # 如果超出分配内存则须减小BATCH_SIZE
    train_count = len(train_image_path)
    train_image_ds = train_image_ds.shuffle(train_count).batch(BATCH_SIZE)
    # 预取，在前边一部分数据训练的过程中，会在后台预取另一部分数据，可以加速训练过程
    train_image_ds = train_image_ds.prefetch(BATCH_SIZE)
    # 构建验证集dataset
    test_image_path = glob.glob('./dc/test/*/*.jpg')
    test_image_label = [int(p.split('\\')[1] == 'cat') for p in test_image_path]
    test_image_ds = tf.data.Dataset.from_tensor_slices((test_image_path, test_image_label))
    # 对于验证集数据没必要做数据增强
    test_image_ds = test_image_ds.map(load_preprocess_image, num_parallel_calls=AUTOTUNE)
    test_image_ds = test_image_ds.batch(BATCH_SIZE)  # 验证集无需shuffle()
    test_image_ds = test_image_ds.prefetch(BATCH_SIZE)
    # 创建模型
    model = tf.keras.Sequential([
        # 第一层一般都是卷积层，因为卷积层提取特征的能力比较强
        tf.keras.layers.Conv2D(64, (3, 3), input_shape=(256, 256, 3), padding="same", activation='relu'),
        # 每一个卷积层后边使用批标准化BatchNormalization，可以抑制过拟合，使得可以构建更深的网络模型
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        # 二分类问题，输出一个值，就是一个逻辑回归问题。
        # 这里不对它进行激活，sigmoid就是一个把x映射到y，y范围是0到1，x<0时，y为0到0.5，x>0时，y为0.5到1，所以不需要激活，只需判断输出是否大于0或小于0，就能判断出猫狗。
        # sigmoid、softmax，它们的本质就是一个归一化，使得我们的输出映射到0到1之间，sigmoid是小于0和大于0作为分类，softmax可以认为是输出中哪个分量大就是哪个分类，激活就是把概率值做了一个归一化，所以这里不做激活，是完全没有问题的。
        tf.keras.layers.Dense(1)
    ])
    # 自定义训练模型
    # 自定义损失函数
    ls = tf.keras.losses.BinaryCrossentropy(
        from_logits=True)  # from_logits=True表示最后一层没有使用激活函数，它会在内部对它进行一个sigmoid运算，然后再计算二元的交叉熵损失
    # 自定义优化器
    optimizer = tf.keras.optimizers.Adam()  # 默认学习率0.001
    # metric模块，记录每个epoch的metric变化
    epoch_loss_avg = tf.keras.metrics.Mean('train_loss')
    train_accuracy = tf.keras.metrics.Accuracy()
    # 验证集的metric模块，记录每个epoch的metric变化
    epoch_loss_avg_test = tf.keras.metrics.Mean('test_loss')
    test_accuracy = tf.keras.metrics.Accuracy()

    # 定义一个批次的训练
    def train_step(model, images, labels):
        # 创建上下文管理器，记录运算过程
        with tf.GradientTape() as t:
            pred = model(images)
            # 计算一步训练的损失
            loss_step = ls(labels, pred)
            # 计算梯度
        grads = t.gradient(loss_step, model.trainable_variables)
        # 对梯度进行优化
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        # metrics
        epoch_loss_avg(loss_step)
        train_accuracy(labels, tf.cast(pred > 0, tf.int32))

    # 定义一个批次的验证集的metrics
    def test_step(model, images, labels):
        pred = model(images, training=False)  # 不训练，或者使用model.predict(images)
        loss_step = ls(labels, pred)
        # metrics
        epoch_loss_avg(loss_step)
        train_accuracy(labels, tf.cast(pred > 0, tf.int32))

    # 记录每个epoch的metrics
    train_loss_results = []
    train_acc_results = []
    # 记录验证集的每个epoch的metrics
    test_loss_results = []
    test_acc_results = []
    num_epochs = 30
    # 训练每个epoch
    for epoch in range(num_epochs):
        # 训练集
        for imgs_, labels_ in train_image_ds:
            train_step(model, imgs_, labels_)
            print(".", end="", flush=True)  # 这里用end=""，须加上flush=True才会在控制台输出
        print()  # 换行
        # 记录每个epoch的metrics
        train_loss_results.append(epoch_loss_avg.result())
        train_acc_results.append(train_accuracy.result())
        # 验证集
        for imgs_, labels_ in test_image_ds:
            test_step(model, imgs_, labels_)
        # 记录验证集每个epoch的metrics
        test_loss_results.append(epoch_loss_avg_test.result())
        test_acc_results.append(test_accuracy.result())
        # 打印每个epoch的metrics信息
        print("Epoch: {}: loss: {:.3f}, accuracy: {:.3f}, test_loss: {:.3f}, tesst_accuracy: {:.3f}".format(epoch + 1,
                                                                                                            epoch_loss_avg.result(),
                                                                                                            train_accuracy.result(),
                                                                                                            epoch_loss_avg_test.result(),
                                                                                                            test_accuracy.result()))
        # 每个epoch重置metrics
        epoch_loss_avg.reset_states()
        train_accuracy.reset_states()
        epoch_loss_avg_test.reset_states()
        test_accuracy.reset_states()


if __name__ == "__main__":
    # 猫狗自定义训练
    # cat_dog_category()
    # 模型优化
    # cat_dog_optimization()
    # 图像增强抑制过拟合
    # cat_dog_image_enhancement()
    # VGG网络进一步优化模型
    cat_dog_vgg()
