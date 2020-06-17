#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time : 2020/4/16 0016 下午 17:50
# @Author : West Field
# @File : foo.py

import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family']='SimHei'
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
import matplotlib
matplotlib.rcParams['font.family']='SimHei'
matplotlib.rcParams['font.sans-serif']=['SimHei']

x=np.linspace(-10.0,10.0,100)
y=1/(1+np.exp(-x))
plt.xlabel("x")
plt.ylabel("y")
plt.title("sigmoid function and its derivative image")
plt.plot(x,y,color='r',label="sigmoid")
y=np.exp(-x)/pow((1+np.exp(-x)),2)
plt.plot(x,y,color='b',label="derivative")
plt.legend()#将plot标签里面的图注印上去
plt.show()

x=np.linspace(-10,10,100)
y=(1-np.exp(-2*x))/(1+np.exp(-2*x))
plt.xlabel('x')
plt.ylabel('y')
plt.title("Tanh function and its derivative image")
plt.plot(x,y,color='r',label='Tanh')
y=1-pow((1-np.exp(-2*x))/(1+np.exp(-2*x)),2)
plt.plot(x,y,color='b',label='derivative')
plt.legend()
plt.show()

x=np.linspace(-2,2,100)
y=x*(x>0)
plt.xlabel('x')
plt.ylabel('y')
plt.title("Relu function and its derivative image")
plt.plot(x,y,color='r',label="Relu")

x=np.linspace(-2,0)
y=np.linspace(0,0)
plt.plot(x,y,color='b')

x=np.linspace(0,2)
y=np.linspace(1,1)
plt.plot(x,y,color='b',label="derivative")

plt.legend()
plt.show()


x=np.linspace(-2,2,100)
def prelu(x):
    return np.where(x<0,0.2*x,x)
y = prelu(x)
plt.xlabel('x')
plt.ylabel('y')
plt.title("Leaky ReLU/PRelu function and its derivative image")
plt.plot(x,y,color='r',label="Relu")

x=np.linspace(-2,0)
y=np.linspace(0.2,0.2)
plt.plot(x,y,color='b')

x=np.linspace(0,2)
y=np.linspace(1,1)
plt.plot(x,y,color='b',label="derivative")

plt.legend()
plt.show()
