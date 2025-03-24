'''
torch的模型保存和加载
在pytorch中，模型的保存和加载分为两种情况
1.保存或加载整个模型，包括模型结构和当时的状态，模型的可训练参数
2.仅保存模型的可训练参数
二者各有优劣，通常情况下仅保存模型的参数即可，但模型具有较复杂且固定的结构时，推荐保存整个模型
'''

import torch
import torch.nn as nn


# 构建简单的网络示例
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(1,5)
        self.fc2 = nn.Linear(5,25)
        self.fc3 = nn.Linear(25,5)
        self.fc4 = nn.Linear(5,1)
    def forward(self,input):
        f1 = self.fc1(input)
        f2 = self.fc2(f1)
        f3 = self.fc3(f2)
        f4 = self.fc4(f3)
        return f4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 设置计算设备
net = Net().to(device) # 实例化模型

# 创建随机输入数据
input = torch.randn(1,1).to(device)
# 模型的前向传播
output = net(input)

torch.save(net.state_dict(),'model.pth') # 只保存模型的参数
load_net = Net() # 载入模型参数文件之前需要重新定义模型结构，且需与原结构相同
state_dict = torch.load('model.pth',weights_only=False) # 载入参数文件，实际上这是一个字典

print(state_dict.keys()) # 打印加载的参数
# odict_keys(['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'fc3.weight', 'fc3.bias', 'fc4.weight', 'fc4.bias'])

load_net.load_state_dict(state_dict) # 将参数字典加载到模型
load_net.to(device)

# 可用的其他选项
state_dict = torch.load('model.pth',map_location='cpu') # 如果保存模型时模型在gpu上，而加载模型时在cpu上，可以使用map_location
load_net.load_state_dict(state_dict,strict=False) # strict=False非严格加载，当保存的参数与模型结构不完全匹配时（比如额外的层或者不同的顺序），可以使用此选项


torch.save(net,'model_full.pth') # 保存整个模型
# 加载整个模型时不需要重新定义模型结构，直接加载模型文件即可
load_net = torch.load('model_full.pth',weights_only=False) # 加载整个模型
load_net.to('cuda')
print(load_net)
# Net(
#   (fc1): Linear(in_features=1, out_features=5, bias=True)
#   (fc2): Linear(in_features=5, out_features=25, bias=True)
#   (fc3): Linear(in_features=25, out_features=5, bias=True)
#   (fc4): Linear(in_features=5, out_features=1, bias=True)
# )

'''
保存参数：大多数情况都可使用，优点是文件小、灵活性高，缺点是载入前需要手动定义模型架构
保存整个模型：在模型复杂且固定的情况下使用，优点是不需要重新定义模型，直接加载，缺点是文件大、依赖保存时的代码版本

另外：在较新版本的torch中，使用torch.load()载入文件时，需要指定weights_only=False以安全载入
'''