'''
torch中的nn.ModuleList容器
在构建深度学习模型时，经常需要管理多个网络层（例如多个nn.Linear、nn.Conv2d等）。
torch提供了nn.ModuleList容器，这是一个类似于python列表的容器，可以用来存储多个子模块（也就是继承自nn.Model的对象），并自动注册它们的参数
主要优点：
自动注册子模块：将nn.Model存储在ModuleList中后，这些模块的参数会自动被添加到父模块的参数列表中，这意味着当调用model.parameters()时，这些子模块的参数也会被包含进去，从而参与梯度计算和优化
灵活管理：它可以像普通列表一样进行索引、迭代和切片操作，方便构建动态网络结构

注意：nn.ModuleList不会像nn.Sequential那样自动定义前向传播流程，使用中需要在模型的forward()方法中手动遍历ModuleList并调用各个子模块
'''

import torch
import torch.nn as nn

# 构建一个包含两层全连接网络的模型，并借助nn.ModuleList来管理子模块
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel,self).__init__()
        # 构建一个ModuleList来存储各层
        self.layers = nn.ModuleList([
            nn.Linear(10,20),
            nn.ReLU(),
            nn.Linear(20,5)
        ])
    def forward(self,x):
        # 遍历ModuleList中的各个子模块，并依次调用forward
        for layer in self.layers:
            x = layer(x)
        return x

model = MyModel()
print('模型结构：\n',model)
# MyModel(
#   (layers): ModuleList(
#     (0): Linear(in_features=10, out_features=20, bias=True)
#     (1): ReLU()
#     (2): Linear(in_features=20, out_features=5, bias=True)
#   )
# )

input_tensor = torch.randn(3,10) # 生成一组输入
output = model(input_tensor) # 得到模型输出
print('模型输出：\n',output)
# tensor([[ 0.2198,  0.2931,  0.0489,  0.4137, -0.3391],
#         [-0.0905,  0.0536,  0.0584,  0.1338, -0.2520],
#         [-0.2319,  0.0514, -0.0330, -0.0329, -0.4490]],
#        grad_fn=<AddmmBackward0>)


'''
当网络结构比较复杂或者层数不固定时，可以利用列表生成器动态构建ModuleList
'''
# 构建动态MLP模型
class DynamicMLP(nn.Module):
    def __init__(self,layer_sizes):
        super(DynamicMLP,self).__init__()

        # 使用循环构造每一层，存储在ModuleList中
        layers = [] # 先用普通列表保存层
        for i in range(len(layer_sizes) - 1):
            linear_layer = nn.Linear(layer_sizes[i],layer_sizes[i+1])
            layers.append(linear_layer)

        # 将普通列表转换为ModuleList
        self.layers = nn.ModuleList(layers)

    def forward(self,x):
        # 遍历每一层
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x

dynamic_model = DynamicMLP([10,20,30,5]) # 动态MLP：输入10，隐藏层20，30，输出5
print('动态MLP模型：\n',dynamic_model)
#  DynamicMLP(
#   (layers): ModuleList(
#     (0): Linear(in_features=10, out_features=20, bias=True)
#     (1): Linear(in_features=20, out_features=30, bias=True)
#     (2): Linear(in_features=30, out_features=5, bias=True)
#   )
# )

# 测试模型
input_tensor = torch.randn(4,10)
output = dynamic_model(input_tensor)
print('动态MLP模型输出：\n',output)
#  tensor([[0.1295, 0.1294, 0.0000, 0.0000, 0.0000],
#         [0.0799, 0.2175, 0.1305, 0.0633, 0.0000],
#         [0.1175, 0.1866, 0.1517, 0.0000, 0.0260],
#         [0.1458, 0.1298, 0.1130, 0.0000, 0.0000]], grad_fn=<ReluBackward0>)

# Transformers中的多头注意力机制

class SingleHeadAttention(nn.Module): # 单个注意力机制
    def __init__(self,embed_dim,head_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim,head_dim) # 查询
        self.key = nn.Linear(embed_dim,head_dim) # 键
        self.value = nn.Linear(embed_dim,head_dim) # 值

    def forward(self,x):
        # 注意力计算逻辑的实现
        return 0

class MultiHeadAttention(nn.Module): # 多头注意力机制
    def __init__(self,embed_dim,num_heads):
        super().__init__()
        self.head_dim = embed_dim // num_heads

        # 显式创建每个注意力头
        self.head1 = SingleHeadAttention(embed_dim,self.head_dim)
        self.head2 = SingleHeadAttention(embed_dim,self.head_dim)
        self.head3 = SingleHeadAttention(embed_dim,self.head_dim)

        # 使用ModuleList管理多个头
        self.heads = nn.ModuleList([
            self.head1,
            self.head2,
            self.head3
        ])

        self.output_proj = nn.Linear(embed_dim,embed_dim)

    def forward(self,x):
        # 在前向传播中分别处理每个头
        head1_out = self.head1(x)
        head2_out = self.head2(x)
        head3_out = self.head3(x)

        # 拼接结果
        combined = torch.cat([head1_out,head2_out,head3_out],dim=
                             1)
        return self.output_proj(combined)

'''
总结:
nn.ModuleList是专门用于存储多个子模块的容器，它会自动注册子模块，确保所有参数能参与训练
与普通python列表相比，ModuleList可以直接通过model.parameters()获取其中所有参数，从而方便的进行优化
使用ModuleList时，前向传播需要手动遍历其中的模块，这提供了灵活性，同时也需要理解循环的过程
'''