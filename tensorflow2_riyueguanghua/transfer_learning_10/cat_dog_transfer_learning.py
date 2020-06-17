#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/5/26 0026 下午 23:01
# @Author : West Field
# @File : cat_dog_transfer_learning.py

import tensorflow as tf
import glob
import matplotlib.pyplot as plt


def cat_dog_vgg():
    '''
    使用预训练网络VGG16来进行猫狗分类，这里训练样本即使只有两千个，使用预训练模型仍可达到很好的训练效果。
    预训练网络一个最大的好处就是让我们实现了，对于小型数据集的分类。
    使用预训练网络帮助我们提取图像的特征，卷积层主要的作用就在于特征的提取，然后使用分类层进行分类，输出层把它的结果输出出来。
    深度学习需要大量的数据进行运算，如果训练数据比较少，很容易就会造成过拟合，因为深度学习它的可训练的参数非常多，使用预训练网络就可以帮助我们摆脱这一点，使得我们可以对小型数据进行训练和识别。
    预训练网络是一个保存好的之前在大型数据集上训练好的卷积神经网络。
    即使新任务与原始任务不同，学习到的特征在不同任务之间是可以移植的。
    它使得深度学习对于小数据集上的任务也是非常有效的。
    通过使用预训练网络，使得解决问题的范围大大的扩大了，可以在通用数据集上训练好一个网络，用它来提取特征，对一个小型数据集实现分类。比如说要实现一个小公司的员工人脸识别，那么就要使用预训练网络了。
    :return:
    '''
    train_image_path = glob.glob("../eager_mode_and_custom_training_7_9/dc/train/*/*.jpg")
    print(len(train_image_path))
    print(train_image_path[:5])
    train_image_label = [int(p.split("\\")[1] == 'cat') for p in train_image_path]
    print(train_image_label[:5])
    # 建立dataset
    train_image_ds = tf.data.Dataset.from_tensor_slices((train_image_path, train_image_label))
    # 根据CPU个数自动设置是否并行运算
    AUTOTUNE = tf.data.experimental.AUTOTUNE  # experimental，后边可能会废弃
    train_image_ds = train_image_ds.map(load_preprocess_image, num_parallel_calls=AUTOTUNE)
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
    test_image_path = glob.glob('../eager_mode_and_custom_training_7_9/dc/test/*/*.jpg')
    test_count = len(train_image_path)
    test_image_label = [int(p.split('\\')[1] == 'cat') for p in test_image_path]
    test_image_ds = tf.data.Dataset.from_tensor_slices((test_image_path, test_image_label))
    # 对于验证集数据没必要做数据增强
    test_image_ds = test_image_ds.map(load_preprocess_image, num_parallel_calls=AUTOTUNE)
    test_image_ds = test_image_ds.batch(BATCH_SIZE)  # 验证集无需shuffle()
    test_image_ds = test_image_ds.prefetch(BATCH_SIZE)

    # keras内置网络实现
    ## 预训练好的网络由卷积基加分类器组成。
    ## tf.keras.applications中包含VGG16、VGG19、ResNet50、Inception v3、Xception等预训练好的经典网络模型。
    # weights='imagenet'表示使用在imagenet上预训练好的权重，如果是weights=None表示只使用VGGNet而不使用它预训练好的权重。
    # include_top=False表示是否包含最后的几个全连接层和输出层，False表示只引入预训练好的卷积基不引入分类器
    # 第一次使用VGG16的weights，会从网络上下载，如果C:\Users\用户名\.keras\models下存在就直接从本地加载
    conv_base = tf.keras.applications.VGG16(weights='imagenet', include_top=False)
    # vgg架构
    print(conv_base.summary())
    # 添加全连接层
    model = tf.keras.Sequential()
    model.add(conv_base)  # 添加卷积基
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(512, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    print(model.summary())
    # 设置卷积基的参数为不可训练
    conv_base.trainable = False
    print(model.summary())  # 会发现可训练参数是剩下最后两层的参数
    # 编译网络
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005),
                  loss='binary_crossentropy',
                  metrics=['acc'])
    # 开始训练。会发现训练三个epoch，以及达到91%的准确率，再往后准确率下降，则可以停止训练，取第三个epoch训练后的模型
    history = model.fit(train_image_ds, steps_per_epoch=train_count // BATCH_SIZE, epochs=15,
                        validation_data=test_image_ds, validation_steps=test_count // BATCH_SIZE)


def load_preprocess_image(path, label):
    '''
    图像增强
    :param path:
    :param label:
    :return:
    '''
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [256, 256])
    image = tf.cast(image, tf.float32)
    image = image / 255
    return image, label


def cat_dog_vgg_finetuning():
    '''
    微调：冻结模型库底部卷积层，共同训练新添加的分类层和顶部部分卷积层，使得正确率进一步的提高。
    这允许我们微调基础模型中的高阶特征，以使它们与特定任务更相关。
    为什么要进行微调呢？
    底部的一些卷积层它会提取一些通用的特征，比如纹理。然后顶部的卷积层，随着视野的扩大会形成一些抽象的图像，比如一只猫一只狗。
    所以底部是一些通用的特征，顶部的卷积层会更加的与特定的任务相关。
    而基于imagenet的VGG顶部卷积层是与imagenet任务(一千种分类)相关的，所以微调顶部的卷积层来使得这些卷积层与当前特定的任务猫狗分类更加相关，使得数据集正确率会进一步提高。
    微调：只有分类器已经训练好了，才能微调卷积基的顶部卷积层。如果没有这样的话，刚开始的训练误差很大，微调之前这些卷积层学到的表示会被破坏掉。
    微调步骤：
    1.在预训练卷积基上添加自定义层
    2.冻结所有的卷积基
    3.训练添加的自定义分类层
    4.解冻卷积基的一部分层
    5.联合训练解冻的卷积层和添加的自定义层
    :return:
    '''
    train_image_path = glob.glob("../eager_mode_and_custom_training_7_9/dc/train/*/*.jpg")
    train_image_label = [int(p.split("\\")[1] == 'cat') for p in train_image_path]
    # 建立dataset
    train_image_ds = tf.data.Dataset.from_tensor_slices((train_image_path, train_image_label))
    # 根据CPU个数自动设置是否并行运算
    AUTOTUNE = tf.data.experimental.AUTOTUNE  # experimental，后边可能会废弃
    train_image_ds = train_image_ds.map(load_preprocess_image, num_parallel_calls=AUTOTUNE)
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
    test_image_path = glob.glob('../eager_mode_and_custom_training_7_9/dc/test/*/*.jpg')
    test_count = len(train_image_path)
    test_image_label = [int(p.split('\\')[1] == 'cat') for p in test_image_path]
    test_image_ds = tf.data.Dataset.from_tensor_slices((test_image_path, test_image_label))
    # 对于验证集数据没必要做数据增强
    test_image_ds = test_image_ds.map(load_preprocess_image, num_parallel_calls=AUTOTUNE)
    test_image_ds = test_image_ds.batch(BATCH_SIZE)  # 验证集无需shuffle()
    test_image_ds = test_image_ds.prefetch(BATCH_SIZE)

    # keras内置网络实现
    ## 预训练好的网络由卷积基加分类器组成。
    ## tf.keras.applications中包含VGG16、VGG19、ResNet50、Inception v3、Xception、DenseNet等预训练好的经典网络模型。
    # weights='imagenet'表示使用在imagenet上预训练好的权重，如果是weights=None表示只使用VGGNet而不使用它预训练好的权重。
    # include_top=False表示是否包含最后的几个全连接层和输出层，False表示只引入预训练好的卷积基不引入分类器
    # 第一次使用VGG16的weights，会从网络上下载，如果C:\Users\用户名\.keras\models下存在就直接从本地加载
    conv_base = tf.keras.applications.VGG16(weights='imagenet', include_top=False)
    # vgg架构
    print(conv_base.summary())
    # 添加全连接层
    model = tf.keras.Sequential()
    model.add(conv_base)  # 添加卷积基
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(512, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    print(model.summary())
    # 设置卷积基的参数为不可训练
    conv_base.trainable = False
    print(model.summary())  # 会发现可训练参数是剩下最后两层的参数
    # 编译网络
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss='binary_crossentropy',
                  metrics=['acc'])
    # 开始训练。在12个epoch达到最好的正确率。
    history = model.fit(train_image_ds, steps_per_epoch=train_count // BATCH_SIZE, epochs=12,
                        validation_data=test_image_ds, validation_steps=test_count // BATCH_SIZE)

    ## 微调
    conv_base.trainable = True
    print(conv_base.layers)  # 查看conv_base的层数
    # 设置卷积基的最后三层可训练
    fine_tune_at = -3
    for layer in conv_base.layers[:fine_tune_at]:
        layer.trainable = False
    # 重新编译模型
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005 / 10),  # 更小的学习率下探寻找一个更小的loss的极值点
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    # 开始训练
    initial_epochs = 12  # 前边已经训练的次数
    fine_tune_epochs = 10  # 准备微调再训练10次
    total_epochs = initial_epochs + fine_tune_epochs  # 总的次数
    history = model.fit(train_image_ds, steps_per_epoch=train_count // BATCH_SIZE, epochs=total_epochs,
                        initial_epochs=initial_epochs,
                        validation_data=test_image_ds, validation_steps=test_count // BATCH_SIZE)
    # 结果达到95%的正确率
    # 通过使用预训练网络加微调，仅仅使用一个很小的数据集就能够达到一个非常高的正确率


def cat_dog_xception():
    '''
    使用Xception的预训练网络进行猫狗分类
    :return:
    '''
    train_image_path = glob.glob("../eager_mode_and_custom_training_7_9/dc/train/*/*.jpg")
    train_image_label = [int(p.split("\\")[1] == 'cat') for p in train_image_path]
    # 建立dataset
    train_image_ds = tf.data.Dataset.from_tensor_slices((train_image_path, train_image_label))
    # 根据CPU个数自动设置是否并行运算
    AUTOTUNE = tf.data.experimental.AUTOTUNE  # experimental，后边可能会废弃
    train_image_ds = train_image_ds.map(load_preprocess_image, num_parallel_calls=AUTOTUNE)
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
    test_image_path = glob.glob('../eager_mode_and_custom_training_7_9/dc/test/*/*.jpg')
    test_count = len(train_image_path)
    test_image_label = [int(p.split('\\')[1] == 'cat') for p in test_image_path]
    test_image_ds = tf.data.Dataset.from_tensor_slices((test_image_path, test_image_label))
    # 对于验证集数据没必要做数据增强
    test_image_ds = test_image_ds.map(load_preprocess_image, num_parallel_calls=AUTOTUNE)
    test_image_ds = test_image_ds.batch(BATCH_SIZE)  # 验证集无需shuffle()
    test_image_ds = test_image_ds.prefetch(BATCH_SIZE)

    # keras内置网络实现
    conv_base = tf.keras.applications.xception.Xception(weights='imagenet', include_top=False,
                                                        input_shape=(256, 256, 3),
                                                        pooling='avg')  # pooling='avg'表示包含全局池化层
    # vgg架构
    print(conv_base.summary())
    # 添加全连接层
    model = tf.keras.Sequential()
    model.add(conv_base)  # 添加卷积基
    model.add(tf.keras.layers.Dense(512, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    print(model.summary())
    # 设置卷积基的参数为不可训练
    conv_base.trainable = False
    print(model.summary())  # 会发现可训练参数是剩下最后两层的参数
    # 编译网络
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005),
                  loss='binary_crossentropy',
                  metrics=['acc'])
    # 开始训练。正确率达到97%，超过了VGG16的正确率。
    history = model.fit(train_image_ds, steps_per_epoch=train_count // BATCH_SIZE, epochs=5,
                        validation_data=test_image_ds, validation_steps=test_count // BATCH_SIZE)


def cat_dog_xception_finetuning():
    '''
    Xception微调
    :return:
    '''
    train_image_path = glob.glob("../eager_mode_and_custom_training_7_9/dc/train/*/*.jpg")
    train_image_label = [int(p.split("\\")[1] == 'cat') for p in train_image_path]
    # 建立dataset
    train_image_ds = tf.data.Dataset.from_tensor_slices((train_image_path, train_image_label))
    # 根据CPU个数自动设置是否并行运算
    AUTOTUNE = tf.data.experimental.AUTOTUNE  # experimental，后边可能会废弃
    train_image_ds = train_image_ds.map(load_preprocess_image, num_parallel_calls=AUTOTUNE)
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
    test_image_path = glob.glob('../eager_mode_and_custom_training_7_9/dc/test/*/*.jpg')
    test_count = len(train_image_path)
    test_image_label = [int(p.split('\\')[1] == 'cat') for p in test_image_path]
    test_image_ds = tf.data.Dataset.from_tensor_slices((test_image_path, test_image_label))
    # 对于验证集数据没必要做数据增强
    test_image_ds = test_image_ds.map(load_preprocess_image, num_parallel_calls=AUTOTUNE)
    test_image_ds = test_image_ds.batch(BATCH_SIZE)  # 验证集无需shuffle()
    test_image_ds = test_image_ds.prefetch(BATCH_SIZE)

    # keras内置网络实现
    conv_base = tf.keras.applications.xception.Xception(weights='imagenet', include_top=False,
                                                        input_shape=(256, 256, 3),
                                                        pooling='avg')  # pooling='avg'表示包含全局池化层
    # vgg架构
    print(conv_base.summary())
    # 添加全连接层
    model = tf.keras.Sequential()
    model.add(conv_base)  # 添加卷积基
    model.add(tf.keras.layers.Dense(512, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    print(model.summary())
    # 设置卷积基的参数为不可训练
    conv_base.trainable = False
    print(model.summary())  # 会发现可训练参数是剩下最后两层的参数
    # 编译网络
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005),
                  loss='binary_crossentropy',
                  metrics=['acc'])
    # 开始训练。正确率达到97%，超过了VGG16的正确率。
    history = model.fit(train_image_ds, steps_per_epoch=train_count // BATCH_SIZE, epochs=5,
                        validation_data=test_image_ds, validation_steps=test_count // BATCH_SIZE)

    ## 微调
    conv_base.trainable = True
    print(conv_base.layers)  # 查看conv_base的层数
    # 设置卷积基的最后三层可训练
    fine_tune_at = -33
    for layer in conv_base.layers[:fine_tune_at]:
        layer.trainable = False
    # 重新编译模型
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005 / 10),  # 更小的学习率下探寻找一个更小的loss的极值点
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    # 开始训练
    initial_epochs = 5  # 前边已经训练的次数
    fine_tune_epochs = 5  # 准备微调再训练10次
    total_epochs = initial_epochs + fine_tune_epochs  # 总的次数
    history = model.fit(train_image_ds, steps_per_epoch=train_count // BATCH_SIZE, epochs=total_epochs,
                        initial_epochs=initial_epochs,
                        validation_data=test_image_ds, validation_steps=test_count // BATCH_SIZE)
    # 结果达到99%的正确率，非常高了。


if __name__ == "__main__":
    # VGG16预训练网络猫狗分类
    # cat_dog_vgg()
    # VGG16微调
    # cat_dog_vgg_finetuning()
    # Xception预训练网络猫狗分类
    # cat_dog_xception()
    # Xception微调
    cat_dog_xception_finetuning()
