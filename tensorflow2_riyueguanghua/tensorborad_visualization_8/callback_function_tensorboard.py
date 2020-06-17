#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/4/13 0013 下午 18:34
# @Author : West Field
# @File : callback_function_tensorboard.py

import tensorflow as tf
# 记录每一次运行的时间
import datetime
import os

def callback_function_tensorboard():
    '''
    通过tf.keras回调函数使用tensorboard
    :return:
    '''
    (train_image, train_labels), (test_image, test_labels) = tf.keras.datasets.mnist.load_data()
    #     使用tf.data作为dataset加载进来
    train_image = tf.expand_dims(train_image, -1)  # 扩增channel维度，-1表示扩增最后一个维度
    test_image = tf.expand_dims(test_image, -1)
    train_image = tf.cast(train_image / 255, tf.float32) # 归一化
    test_image = tf.cast(test_image / 255, tf.float32)
    train_labels = tf.cast(train_labels, tf.int64)
    test_labels = tf.cast(test_labels, tf.int64)
    dataset = tf.data.Dataset.from_tensor_slices((train_image, train_labels))  # 建立dataset
    test_dataset = tf.data.Dataset.from_tensor_slices((test_image, test_labels))
    print(dataset)
#     使用tf.keras.fit()需要让数据一直repeat,自定义训练不用repeat,因为每一步调用一次dataset
    dataset = dataset.repeat().shuffle(60000).batch(128)
    test_dataset = test_dataset.repeat().batch(128)
    #     建立模型
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, [3, 3], activation='relu', input_shape=(None, None, 1)),
        # kernal_size为16，None表示任意尺寸图像都可以输入进来
        tf.keras.layers.Conv2D(32, [3, 3], activation='relu'),
        tf.keras.layers.GlobalMaxPooling2D(),
        tf.keras.layers.Dense(10, activation='softmax')  # 这里无需softmax激活，因为softmax只是对这10个输出进行归一化，最大的输出即对应哪个分类
    ])
#     编译模型
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
#     使用tensorborad对网络进行可视化,使用model.fit中使用callback方法去调用一个方法,使用调用的函数来改变keras在运行中的方式
#     建立一个调用,log_dir是放置事件的位置,histogram_freq是记录直方图的频率
    log_dir = os.path.join('logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) # 保存事件的文件,strftime是将时间转成字符串格式
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1)
#     训练模型
    model.fit(dataset, epochs=5, steps_per_epoch=60000//128, validation_data=test_dataset, validation_steps=10000//128,
              callbacks=[tensorboard_callback]) # 训练过程中,它会把事件记录到logdir中,然后就可以打开tensorboard去查看它.

def custom_scaler():
    '''
    记录自定义的标量:学习率
    :return:
    '''
    (train_image, train_labels), (test_image, test_labels) = tf.keras.datasets.mnist.load_data()
    #     使用tf.data作为dataset加载进来
    train_image = tf.expand_dims(train_image, -1)  # 扩增channel维度，-1表示扩增最后一个维度
    test_image = tf.expand_dims(test_image, -1)
    train_image = tf.cast(train_image / 255, tf.float32)  # 归一化
    test_image = tf.cast(test_image / 255, tf.float32)
    train_labels = tf.cast(train_labels, tf.int64)
    test_labels = tf.cast(test_labels, tf.int64)
    dataset = tf.data.Dataset.from_tensor_slices((train_image, train_labels))  # 建立dataset
    test_dataset = tf.data.Dataset.from_tensor_slices((test_image, test_labels))
    print(dataset)
    #     使用tf.keras.fit()需要让数据一直repeat,自定义训练不用repeat,因为每一步调用一次dataset
    dataset = dataset.repeat().shuffle(60000).batch(128)
    test_dataset = test_dataset.repeat().batch(128)
    #     建立模型
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, [3, 3], activation='relu', input_shape=(None, None, 1)),
        # kernal_size为16，None表示任意尺寸图像都可以输入进来
        tf.keras.layers.Conv2D(32, [3, 3], activation='relu'),
        tf.keras.layers.GlobalMaxPooling2D(),
        tf.keras.layers.Dense(10, activation='softmax')  # 这里无需softmax激活，因为softmax只是对这10个输出进行归一化，最大的输出即对应哪个分类
    ])
    #     编译模型
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    #     使用tensorborad对网络进行可视化,使用model.fit中使用callback方法去调用一个方法,使用调用的函数来改变keras在运行中的方式
    #     建立一个调用,log_dir是放置事件的位置,histogram_freq是记录直方图的频率
    log_dir = os.path.join('logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))  # 保存事件的文件,strftime是将时间转成字符串格式
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1)
    # 把learning_rate的变化写到磁盘上
    # 创建一个文件编写器
    file_writer = tf.summary.create_file_writer(log_dir + '/lr')
    file_writer.set_as_default() # 将file_writer设置为默认的文件编写器,我们使用tf.summary.scaler写入数据的时候,他都会默认调用这个file_writer
    # 定义学习率变化
    def lr_sche(epoch):
        learning_rate = 0.2
        if epoch > 5:
            learning_rate = 0.02
        if epoch > 10:
            learning_rate = 0.01
        if epoch > 20:
            learning_rate = 0.005
        # 记录learning_rate标量值的变化
        tf.summary.scalar('learning_rate', data=learning_rate, step=epoch)
        return learning_rate
    # 建立学习率的回调函数
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_sche)
    #     训练模型
    model.fit(dataset, epochs=25, steps_per_epoch=60000 // 128, validation_data=test_dataset,
              validation_steps=10000 // 128,
              callbacks=[tensorboard_callback, lr_callback])

def custom_training_tensorboard():
    '''
    自定义训练当中使用tensorboard
    :return:
    '''
    (train_image, train_labels), (test_image, test_labels) = tf.keras.datasets.mnist.load_data()
    print(train_image.shape, train_labels)
    #     使用tf.data作为dataset加载进来
    train_image = tf.expand_dims(train_image, -1)  # 扩增channel维度，-1表示扩增最后一个维度
    test_image = tf.expand_dims(test_image, -1)
    print(train_image.shape)
    train_image = tf.cast(train_image / 255, tf.float32)
    test_image = tf.cast(test_image / 255, tf.float32)
    train_labels = tf.cast(train_labels, tf.int64)
    test_labels = tf.cast(test_labels, tf.int64)
    dataset = tf.data.Dataset.from_tensor_slices((train_image, train_labels))  # 建立dataset
    test_dataset = tf.data.Dataset.from_tensor_slices((test_image, test_labels))
    print(dataset)
    #     数据混洗，设置batch
    dataset = dataset.shuffle(10000).batch(32)
    test_dataset = test_dataset.batch(32)
    print(dataset)
    #     建立模型
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, [3, 3], activation='relu', input_shape=(None, None, 1)),
        # kernal_size为16，None表示任意尺寸图像都可以输入进来
        tf.keras.layers.Conv2D(32, [3, 3], activation='relu'),
        tf.keras.layers.GlobalMaxPooling2D(),
        tf.keras.layers.Dense(10)  # 这里无需softmax激活，因为softmax只是对这10个输出进行归一化，最大的输出即对应哪个分类
    ])
    #     因为这里是自定义循环，不需要编译这个模型
    #     优化器
    optimizer = tf.keras.optimizers.Adam()
    #     loss函数
    loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # 由于最后一层没有激活，所以要设置from_logits=True

    #     开始训练
    # 初始化汇总计算对象,每个batch之后记录一下它的loss和accuracy
    train_loss = tf.keras.metrics.Mean('train_loss')  # 汇总对象起名为train_loss
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
    test_loss = tf.keras.metrics.Mean('test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')

    #     建立一步的train，完成对一个批次的训练
    def train_step(model, images, labels):
        with tf.GradientTape() as t:
            pred = model(images)  # 预测值
            loss_step = loss_func(labels, pred)  # 每个batch的损失
        grads = t.gradient(loss_step, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        # 每个batch之后,汇总计算方法记录损失和准确率
        train_loss(loss_step)
        train_accuracy(labels, pred)

    # 建立test_step,记录在test数据集上每一步的情况,验证集上无需求梯度优化
    def test_step(model, images, labels):
        pred = model(images)  # 预测值
        loss_step = loss_func(labels, pred)  # 每个batch的损失
        # 每个batch之后,汇总计算方法记录损失和准确率
        test_loss(loss_step)
        test_accuracy(labels, pred)

    # 创建文件编写器,把训练过程中的loss和accuracy数据写到磁盘上
    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    train_log_dir = 'logs/gradient_tape' + current_time + '/train'
    test_log_dir = 'logs/gradient_tape' + current_time + '/test'
    train_writer = tf.summary.create_file_writer(train_log_dir)
    test_writer = tf.summary.create_file_writer(test_log_dir)

    #     对数据进行多次优化训练
    def train():
        for epoch in range(10):
            for (batch, (images, labels)) in enumerate(dataset):
                train_step(model, images, labels)
            # 收集训练集标量值
            with train_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=epoch)
                tf.summary.scalar('acc', train_accuracy.result(), step=epoch)
            # 计算验证集
            for (batch, (images, labels)) in enumerate(test_dataset):
                test_step(model, images, labels)
                # print('*', end='')
            # 收集验证集的标量值
            with test_writer.as_default():
                tf.summary.scalar('loss', test_loss.result(), step=epoch)
                tf.summary.scalar('acc', test_accuracy.result(), step=epoch)
            # 每个epoch结束之后,调用result()方法计算损失和准确率的均值
            print("Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}.".format(
                epoch, train_loss.result(), train_accuracy.result(), test_loss.result(), test_accuracy.result()))
            # 在每个epoch结束之后,重置汇总计算对象
            train_loss.reset_states()
            train_accuracy.reset_states()
            test_loss.reset_states()
            test_accuracy.reset_states()

    train()  # 开始训练


if __name__ == '__main__':
    # 通过回调函数使用tensorboard
    # callback_function_tensorboard()
#     记录自定义的标量:学习率
#     custom_scaler()
#     自定义训练当中使用tensorboard
    custom_training_tensorboard()
