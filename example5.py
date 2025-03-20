'''
torch的求导操作
torch使用tensor.backward()来进行张量的求导操作,torch中的tensor具有自动微分机制，只需要在创建tensor时指定requires_grad=True即可自动计算反向传播
'''

import torch

x = torch.tensor([[2.,-1.],[1.,1.]],requires_grad=True) # 初始化张量，指定requires_grad=True，启用自动求导功能
print(x) # 输出张量
# tensor([[ 2., -1.],
#         [ 1.,  1.]], requires_grad=True)

out = x.pow(2).sum() # 对x中的每个元素求平方并加和，模拟前向传播
print(out) # 输出out
# tensor(7., grad_fn=<SumBackward0>)，grad_fn=<SumBackward0>表示out是通过对x进行求和操作得到的，反向传播时会回溯到这个操作

out.backward() # 对out进行反向传播，backward()方法会自动计算out对x的梯度，并将结果存储在x.grad中
print(x.grad) # 输出x的梯度
# tensor([[ 4., -2.], 
#         [ 2.,  2.]])

'''
requires_grad=True表示启用自动求导功能，参与计算的操作会记录到计算图中，
backward()执行反向传播，沿着计算图回溯到每个操作，计算每个操作的梯度，梯度会累加到x.grad中
在torch中，backward()计算张量的梯度，但是它只能对标量进行计算，因此在使用backward()时，需要先将张量转换为标量（即只有一个数值），
将张量转换为标量通常使用求和sum()或求平均值mean()等操作
'''
