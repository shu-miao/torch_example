'''
torch的克隆和共享操作
torch中的张量可以通过tensor.clone()方法来执行克隆操作，克隆操作会创建一个新的张量，修改clone的返回值不会影响原始数据，
tensor.detach()方法执行共享操作，共享操作会创建一个新的张量，但是新的张量和原始张量共享数据，修改detach的返回值会影响原始数据
'''

import torch

# x.clone()对张量x进行深拷贝，生成一个新的张量，
# 新的张量和原始张量具有相同的数据，但存储在不同的内存空间
# 修改clone()的返回值不会影响原始张量
x = torch.tensor([1.0,2.0,3.0],requires_grad=True) # 初始化张量
y = x.clone() # 克隆操作

y[0] = 99.0 # 修改克隆的返回值
print(x) # 输出原始数据
# tensor([1., 2., 3.], requires_grad=True)
print(y) # 输出克隆的返回值
# tensor([99.,  2.,  3.], grad_fn=<CopySlices>)

# x.detach()返回一个与x共享相同数据但与计算图断开联系的张量
# detach()方法还可以用于将一个需要梯度计算的张量转换为不需要梯度计算的张量，通常用于阻止梯度计算
# 在神经网络中，如果不希望某些操作影响反向传播时，就可以使用detach()方法
# 修改detach()的返回值会影响原始张量
x = torch.tensor([1.0,2.0,3.0],requires_grad=True) # 初始化张量
y = x.detach() # 共享操作

y[0] = 99.0
print(x) # 输出原始数据
# tensor([99.,  2.,  3.], requires_grad=True)
print(y) # 输出共享的返回值
# tensor([99.,  2.,  3.])
'''
detach在阻止梯度传播和保存模型状态时非常有用
比如：
# 阻止梯度传播
z = x.clone().detach() # z不会参与反向传播，x的梯度也不会受z的影响

# 保存模型状态
with torch.no_grad():
    output = model(x) # 临时禁用梯度计算
'''

# 另外，对于torch.tensor(x).float()操作
# torch.tensor(x).float()将输入x转换为pytorch张量，并将其中元素数据类型强制为torch.float32（默认浮点类型）
# 输入x可以是一个python列表或numpy数组，
# 某些模型可能对数据类型有严格要求，必须使用torch.tensor类型，此时可以使用torch.tensor(x).float()操作进行转换
x = [[1,2,3],[4,5,6]] # python列表
y = torch.tensor(x).float() # 转换为torch.tensor类型
print(y)
# tensor([[1., 2., 3.],
#         [4., 5., 6.]])

'''
对于x.clone().detach()和torch.tensor(x).float()的使用场景，
有如下理解：
当x是一个非torch.tensor张量对象（如python列表或者numpy数组），需要转换为torch.tensor并且确保数据类型为float时可以使用torch.tensor(x).float()；
当x是一个torch.tensor张量对象，且需要复制数据的同时与原始计算图断开时可以使用x.clone().detach()
'''