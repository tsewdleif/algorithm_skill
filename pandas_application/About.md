numpy能够帮我们处理处理数值型数据，但是这还不够，很多时候，我们的数据除了数值之外，还有字符串，还有时间序列等。
所以，numpy能够帮助我们处理数值，但是pandas除了处理数值之外(基于numpy)，还能够帮助我们处理其他类型的数据。

pandas的常用数据类型：
1.Series 一维，带标签数组。
本质上由两个数组构成：对象的键和对象的值。
numpy中ndarray的很多方法都可以运用于Series，如argmax、clip。
2.DataFrame 二维，Series容器。
DataFrame对象既有行索引，又有列索引。
行索引，表明不同行，横向索引，叫index，0轴，axis=0。
列索引，表名不同列，纵向索引，叫columns，1轴，axis=1。


