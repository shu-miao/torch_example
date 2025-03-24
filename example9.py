'''
torch中创建一个指定大小的张量
torch使用torch.empty创建一个指定形状的张量，但不会对张量中的值进行赋值初始化，这意味着这个张量中的数据可能是内存中之前残留的随机值，因此其值是不可预测的
当需要一个新的张量用来存储值是，使用torch.empty可以节省初始化的时间开销，并且需要马上使用新值将该张量中的值覆盖
torch.empty(*size,*,out=None,dtype=None,layout=torch.strided,device=None,requires_grad=False)
size：张量的形状，可以是一个整数序列。例如（3，4）表示创建一个三行四列的张量
dtype：数据类型，例如：torch.float32、torch.int64等
device：设备，可以是”cuda“或”cpu“
requires_grad：是否对张量计算梯度，默认为False
'''

import torch

x = torch.empty(5) # 创建一个长度为5的张量
print(x)
# tensor([0., 0., 0., 0., 0.]) 每次输出的值可能不相同

x = torch.empty((2,3))
print(x)
# tensor([[7.6087e+30, 1.3116e-42, 0.0000e+00],
#         [0.0000e+00, 0.0000e+00, 0.0000e+00]]) # 两行三列

x = torch.empty((2,3),dtype=torch.float32,device='cuda',requires_grad=True) # 指定数据类型和设备
print(x)
# tensor([[0., 0., 0.],
#         [0., 0., 0.]], device='cuda:0', requires_grad=True)

'''
由于torch.empty返回的张量是未初始化的，所以在使用前必须显式赋值或覆盖这些数据，否则可能导致错误的计算结果。

另外，torch中存在其他创建张量的方法，
创建一个全为0的张量，可以使用torch.zeros；创建一个全为1的张量，可以使用torch.ones
torch.empty的速度是最快的，因为它不执行初始化，产生的张量的值也是不确定的
'''