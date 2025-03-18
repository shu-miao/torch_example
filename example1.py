'''
torch的矩阵乘法
1.矩阵乘法是神经网络的核心运算，由于GPU的硬件特性，矩阵乘法在对大量数据进行计算时仍然可以保持高效，
2.从神经元的计算来看，输入、权重、输出都可以使用矩阵来表示，矩阵乘法可以表示多个输入的线性组合，所以矩阵乘法可以将多个神经元的计算简化成一个矩阵运算
3.矩阵乘法在各元素计算间具有独立性，所以矩阵乘法天然具有并行处理能力
4.神经元的计算是进行线性变化，而矩阵乘法本质上是对输入进行线性变换，这与神经网络中的权重更新过程相符，通过矩阵乘法可以方便的实现网络中层与层之间的连接关系
'''
import torch
import numpy as np

print(torch.__version__) # 版本

x = torch.tensor([[1,2,3,4],[5,6,7,8]]) # 二维张量（两行四列）
y = torch.tensor([2,3,1,0]) # 行向量
print(torch.matmul(x,y)) # 矩阵乘法，此时的输入形状为（2，4）（1，4），但torch会将张量y当成一个一维列向量，与x矩阵进行矩阵乘法，此时的结果是一个一维向量，长度为（2，）
print(x@y) # 另一种矩阵乘法写法，这与上一行写法等价，两者都会调用底层优化后的矩阵乘法函数
# 对于[[1,2,3,4],[5,6,7,8]]和[2,3,1,0]的矩阵乘法
# 第一行计算：1*2+2*3+3*1+4*0=11
# 第二行计算：5*2+6*3+7*1+8*0=35
# 输出为tensor([11, 35]) # tensor是一种数据结构，类似于numpy的ndarray，但是比ndarray更强大，因为它可以在GPU上运行，并且可以自动求导

y = y.view(4,1) # 把y转变4行1列的张量
print(y)
# tensor([[2],
#         [3],
#         [1],
#         [0]])
print(torch.matmul(x,y)) # 矩阵乘法，此时的输入形状为（2，4）（4，1），结果形为（2，1）
print(x@y)
# tensor([[11],
#         [35]])

'''
总结，torch的矩阵乘法分两种情况
1.y矩阵是1维张量：此时相对于将y当作列向量，与矩阵x做矩阵乘法，torch.matmul(x,y)返回一个一维张量，长度为(2,)
2.y矩阵是2维张量：此时矩阵乘法严格遵守二维矩阵的维度规则，torch.matmul(x,y)返回一个二维张量，形状为(2,1)
因为输入张量的维度数不同，导致输出的维度也发生了变化
'''