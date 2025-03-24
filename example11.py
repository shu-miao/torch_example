'''
torch中的dropout操作
在深度学习中，dropout是一种正则化方法，用于防止过拟合，
python中的，torch.nn.Dropout提供了Dropout机制，用于在训练过程中随机丢弃部分神经元的输出，以提高模型的泛化能力
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

dropout = torch.nn.Dropout(p=0.5) # 创建Dropout层，p=0.5意味着50%的神经元会被随机抛弃
example = torch.ones(6,6) # 初始化一个全为1的（6，6）的张量
output = dropout(example) # 应用Dropout，
print(output) # 由于dropout是随机的，每次结果可能不同
# tensor([[0., 0., 2., 2., 2., 2.],
#         [0., 2., 0., 2., 0., 0.],
#         [0., 0., 2., 0., 2., 0.],
#         [2., 0., 0., 2., 2., 2.],
#         [0., 0., 2., 0., 2., 2.],
#         [2., 0., 0., 0., 2., 0.]]) # 大约50%的元素变成了0，其余元素被放大两倍以保持总期望不变

# 设置不同的Dropout率
dropout_75 = torch.nn.Dropout(0.75) # 丢弃75%
dropout_25 = torch.nn.Dropout(0.25) # 丢弃25%

output_75 = dropout_75(example)
print('Dropout p=0.75:\n',output_75)

output_25 = dropout_25(example)
print('Dropout p=0.25:\n',output_25)


'''
torch中的Dropout只在训练模式train()中生效，在评估模式eval()中关闭
'''
dropout.train() # 设置为训练模式
print('Train Mode Output:\n',dropout(example)) # 输出经过dropout的值

dropout.eval() # 设置为评估模式
print('Eval Mode Output:\n',dropout(example)) # 由于设置了评估模式，所有元素保持不变


'''
在网络中添加dropout层
在神经网络中，Dropout主要用于全连接层和卷积层之间，防止模型过拟合
'''
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(10,5)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(5,2)
    def forward(self,input):
        x = torch.relu(self.fc1(input))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = Net()
print(model)
# Net(
#   (fc1): Linear(in_features=10, out_features=5, bias=True)
#   (dropout): Dropout(p=0.5, inplace=False)
#   (fc2): Linear(in_features=5, out_features=2, bias=True)
# )


'''
除了torch.nn.Dropout外，torch还提供了torch.nn.functional.dropout()，可以直接对张量进行操作
'''
x = torch.ones(6,6)
output = F.dropout(x,p=0.5,training=True)
print(output)
