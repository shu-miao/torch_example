'''
torch中关于下三角矩阵和上三角矩阵的计算
在深度学习中，下三角矩阵和上三角矩阵是非常常见的矩阵操作，pytorch提供了torch.tril()和torch.triu()这连个函数，分别用于计算下三角矩阵和上三角矩阵
torch.tril(input,diagonal=0)返回输入张量的下三角部分，保留主对角线及其以下的元素，主对角线以上的元素全部变为0
torch.triu(input,diagonal=0)返回输入张量的上三角部分，保留主对角线及其以上的元素，主对角线以下的元素全部变为0
'''

import torch

# torch.tril
A = torch.tensor([
    [1,2,3,4],
    [5,6,7,8],
    [9,10,11,12],
    [13,14,15,16]
])
print('原始矩阵 A：\n',A)

L = torch.tril(A) # diagonal默认为0
print('\nA 的下三角矩阵（diagonal=0）：\n',L)

L = torch.tril(A,diagonal=1) # 保留主对角线以上一行
print('\nA 的下三角矩阵（diagonal=1）：\n',L)

L = torch.tril(A,diagonal=-1) # 移除主对角线
print('\nA 的下三角矩阵（diagonal=-1）：\n',L)

# torch.tril用于Masking（掩码）
seq_length = 5
mask = torch.tril(torch.ones(seq_length,seq_length)) # 创建一个下三角掩码
print("Mask\n",mask)


# torch.triu
U = torch.triu(A)
print('\nA 的上三角矩阵（diagonal=0）：\n',U)

U = torch.triu(A,diagonal=1) # 移除主对角线
print('\nA 的上三角矩阵（diagonal=1）：\n',U)

U = torch.triu(A,diagonal=-1) # 保留主对角线一下一行
print('\nA 的上三角矩阵（diagonal=-1）：\n',U)

'''
torch.tril()取下三角矩阵，可以用于Cholesky分解，Transformer Masking
torch.triu()取上三角矩阵，常用于线性代数计算和矩阵变换
'''