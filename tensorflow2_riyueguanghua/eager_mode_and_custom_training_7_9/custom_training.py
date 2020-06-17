#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/4/13 0013 下午 14:37
# @Author : West Field
# @File : custom_training.py

import tensorflow as tf

def custom_training():
    '''
    自定义训练
    tensorflow2.0的设计就是一种高低搭配的设计，它既给我们tf.keras封装好的API，使用fit()训练一个模型，
    另外当你需要研究去训练的时候，你也可以去自己做一些定义，控制这个训练的整个过程，添加一些新的东西进去。
    :return:
    '''
    (train_image, train_labels), _ = tf.keras.datasets.mnist.load_data()
    print(train_image.shape, train_labels)
    (a, b), (c, d) = tf.keras.datasets.mnist.load_data()
    print(a.shape, c.shape)
#     使用tf.data作为dataset加载进来
    train_image = tf.expand_dims(train_image, -1) # 扩增channel维度，-1表示扩增最后一个维度
    print(train_image.shape)
    train_image = tf.cast(train_image/255, tf.float32) # 改变数据类型，数据归一化
    train_labels = tf.cast(train_labels, tf.int64)
    dataset = tf.data.Dataset.from_tensor_slices((train_image, train_labels)) # 建立dataset
    print(dataset) # TensorSliceDataset
#     数据混洗，设置batch
    dataset = dataset.shuffle(10000).batch(32)
    print(dataset) # BatchDataset
#     建立模型
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, [3, 3], activation='relu', input_shape=(None, None, 1)), # kernal_size数量为16，None表示任意尺寸图像都可以输入进来
        tf.keras.layers.Conv2D(32, [3, 3], activation='relu'),
        tf.keras.layers.GlobalMaxPooling2D(),
        tf.keras.layers.Dense(10) # 这里无需softmax激活，因为softmax只是对这10个输出进行归一化，最大的输出即对应哪个分类
    ])
#     因为这里是自定义循环，不需要编译这个模型
#     优化器
    optimizer = tf.keras.optimizers.Adam()
#     loss函数
    loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) # 由于最后一层没有激活，所以要设置from_logits=True
#     写循环开始训练
    features, labels = next(iter(dataset)) # 把dataset变成可迭代对象
    print(features.shape, labels.shape)
    predictions = model(features) # model是可调用的，得到预测数据。没有训练就进行预测，效果比较差。
    print(predictions.shape)
    print(predictions)
    print("-1-" * 100)
    print(tf.argmax(predictions, axis=1)) # 取最大值的位置，即为预测结果
    print(labels) # 实际的label
    print(loss_func(labels, predictions))
    # print(loss_func(labels, tf.argmax(predictions, axis=1))) # 会报错：类型不搭配错误
    print("-2-"*100)
#     开始训练
#     定义一个损失函数，计算每个批次的损失值
    def loss(model, x, y):
        y_ = model(x)
        lossss = loss_func(y, y_)
        print('='*100)
        print(lossss)
        print('=' * 100)
        return lossss
#     建立一步的train，完成对一个批次的训练
    def train_step(model, images, labels):
        with tf.GradientTape() as t:
            loss_step = loss(model, images, labels)
        # 计算损失值对model里边的变量可训练参数的梯度，寻找下降最快的方向，来使得损失下降
        grads = t.gradient(loss_step, model.trainable_variables)
        # 优化，来使得变量梯度下降最快
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
#     对数据进行多次优化训练
    def train():
        for epoch in range(10):
            for (batch, (images, labels)) in enumerate(dataset):
                # print(type(batch),batch) # int
                # print(type(images), images) # tensorflow.python.framework.ops.EagerTensor   shape=(32, 28, 28, 1)
                # print(type(labels), labels) # tensorflow.python.framework.ops.EagerTensor   shape=(32,)
                train_step(model, images, labels)
            print("Epoch{} is finished.".format(epoch))
    train() # 开始训练

def metrics_module():
    '''
    上边已经完成了自动微分计算和自定义循环，这里看一下面向对象的汇总/指标计算模型tf.keras.metrics
    :return:
    '''
#     计算准确率均值
    m = tf.keras.metrics.Mean('acc')
    print(type(m), m) # tensorflow.python.keras.metrics.Mean
    m(10)
    m(20)
    print(type(m), m) # tensorflow.python.keras.metrics.Mean
    print(m.result().numpy())
#     在每个epoch之后，重置对象
    m.reset_states()
    m(1)
    m(2)
    print(m.result().numpy())
#     计算一个batch的准确率
    (train_image, train_labels), _ = tf.keras.datasets.mnist.load_data()
    print(train_image.shape, train_labels)
    #     使用tf.data作为dataset加载进来
    train_image = tf.expand_dims(train_image, -1)  # 扩增channel维度，-1表示扩增最后一个维度
    print(train_image.shape)
    train_image = tf.cast(train_image / 255, tf.float32)  # 改变数据类型，数据归一化
    train_labels = tf.cast(train_labels, tf.int64)
    dataset = tf.data.Dataset.from_tensor_slices((train_image, train_labels))  # 建立dataset
    print(dataset)
    #     数据混洗，设置batch
    dataset = dataset.shuffle(10000).batch(32)
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
    #     写循环开始训练
    features, labels = next(iter(dataset))  # 把dataset变成可迭代对象
    print(type(features), type(labels), type(next(iter(dataset))), type(dataset.take(1)))
    print(labels) # 真实值
#     定义准确率
    a = tf.keras.metrics.SparseCategoricalAccuracy('acc')
    print(a(labels, model(features))) # 由预测正确的个数除以32得到该值

def metrics_in_custom_training():
    '''
    汇总/指标计算模块在自定义训练中的应用
    :return:
    '''
    (train_image, train_labels), _ = tf.keras.datasets.mnist.load_data()
    print(train_image.shape, train_labels)
    #     使用tf.data作为dataset加载进来
    train_image = tf.expand_dims(train_image, -1)  # 扩增channel维度，-1表示扩增最后一个维度
    print(train_image.shape)
    train_image = tf.cast(train_image / 255, tf.float32)  # 改变数据类型，数据归一化
    train_labels = tf.cast(train_labels, tf.int64)
    dataset = tf.data.Dataset.from_tensor_slices((train_image, train_labels))  # 建立dataset
    print(dataset)
    #     数据混洗，设置batch
    dataset = dataset.shuffle(10000).batch(32)
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
    train_loss = tf.keras.metrics.Mean('train_loss') # 汇总对象起名为train_loss
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
    #     建立一步的train，完成对一个批次的训练
    def train_step(model, images, labels):
        with tf.GradientTape() as t:
            pred = model(images) # 预测值
            loss_step = loss_func(labels, pred) # 每个batch的损失
        grads = t.gradient(loss_step, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        # 每个batch之后,汇总计算方法记录损失和准确率
        train_loss(loss_step)
        train_accuracy(labels, pred)

    #     对数据进行多次优化训练
    def train():
        for epoch in range(10):
            for (batch, (images, labels)) in enumerate(dataset):
                train_step(model, images, labels)
            # 每个epoch结束之后,调用result()方法计算损失和准确率的均值
            print("Epoch{} loss is {}, accuracy is {}.".format(epoch, train_loss.result(), train_accuracy.result()))
            # 在每个epoch结束之后,重置汇总计算对象
            train_loss.reset_states()
            train_accuracy.reset_states()

    train()  # 开始训练

def add_test_dataset_metrics():
    '''
    在训练过程中同时打印验证集上的汇总计算数据
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

    #     对数据进行多次优化训练
    def train():
        for epoch in range(10):
            for (batch, (images, labels)) in enumerate(dataset):
                train_step(model, images, labels)
            # 每个epoch结束之后,调用result()方法计算损失和准确率的均值
            print("Epoch{} loss is {}, accuracy is {}.".format(epoch, train_loss.result(), train_accuracy.result()))
            # 计算验证集
            for (batch, (images, labels)) in enumerate(test_dataset):
                test_step(model, images, labels)
            # 每个epoch结束之后,调用result()方法计算损失和准确率的均值
            print("Epoch{} test_loss is {}, test_accuracy is {}.".format(epoch, test_loss.result(), test_accuracy.result()))
            # 在每个epoch结束之后,重置汇总计算对象
            train_loss.reset_states()
            train_accuracy.reset_states()
            test_loss.reset_states()
            test_accuracy.reset_states()

    train()  # 开始训练
#     到此为止,自定义的训练,基本上和tf.keras高级API封装好的的fit()的输出结果基本相同了


if __name__ == '__main__':
#     自定义训练
#     custom_training()
#     汇总/指标计算模块
#     metrics_module()
#     汇总/指标计算模块在自定义训练中的应用
#     metrics_in_custom_training()
#     在验证集中添加汇总计算
    add_test_dataset_metrics()
