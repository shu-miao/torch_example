'''
torch的连接操作
连接操作利用了张量的特性，达到了合并多个张量的目的。
torch的张量可以有多维结构，这使得连接操作可以在不同的维度上进行，
torch在执行连接操作时，会尽可能保持内存的连续性，
且在某些情况下，连接操作可以利用广播机制使得不同形状的张量也能进行连接

连接的目的主要有三个：
1.数据整合，将多个样本的数据整合成一个批次，便于进行批量处理
2.格式统一，在构建网络时，可通过连接将不同来源的数据连接在一起，以形成统一的输入格式
3.特征融合，连接操作可以将不同特征的张量连接，形成更丰富的特征表示
'''
import torch
import torch.nn.functional as F

# 通常使用torch.cat(tensors,dim=0)进行连接操作，tensors：一个包含多个待拼接张量的列表或元组；dim：指定在哪个维度上进行拼接操作
# torch.cat()的使用规则：在指定维度上，张量的形状可以不同（因为会拼接），在其他维度上，张量的形状必须相同

x = torch.tensor([[1,2],
                 [3,4]]) # （2，2）的张量
y = torch.tensor([[5,6],
                  [7,8]]) # （2，2）的张量
result = torch.cat((x,y),dim=0) # 在第0维拼接，也就是在行方向上拼接
print(result) # 输出为（4，2）

result = torch.cat((x,y),dim=1) # 在第1维拼接，也就是在列方向上拼接
print(result) # 输出为（2，4）

# 高维张量的拼接
x = torch.tensor([
    [[1,2,3],[4,5,6]],
    [[7,8,9],[10,11,12]]
]) # 形状（2，2，3）
y = torch.tensor([
    [[13,14,15],[16,17,18]],
    [[19,20,21],[22,23,24]]
]) # 形状（2，2，3）

result_dim0 = torch.cat((x,y),dim=0) # 在第0维拼接，也就是在最外层上拼接
print(result_dim0.shape) # torch.Size([4,2,3])
print(result_dim0) # 结果张量包含4个块，每个块的形状仍然是（2，3）

result_dim1 = torch.cat((x,y),dim=1) # 在第1维拼接，也就是在每个块中的行上拼接
print(result_dim1.shape) # torch.Size([2,4,3])
print(result_dim1) # 结果张量包含两个块，每个块增加了两行，形状是（4，3）

result_dim2 = torch.cat((x,y),dim=2) # 在第2维拼接，也就是在每个块中的列上拼接
print(result_dim2.shape) # torch.Size(2,2,6)
print(result_dim2) # 结果张量包含两个块，每个块的列数增加了一倍，形状是（2，6）

'''
dim = 0:增加块数；dim = 1:增加每块中的行数；dim = 2:增加每块中的列数
'''

# 不同形状张量的拼接
# 如果在非拼接维度上的形状不同，会报错
try:
    x = torch.tensor([[1,2],[3,4]]) # 形状（2，2）
    y = torch.tensor([[5,6,7]]) # 形状（1，3）
    result = torch.cat((x,y),dim=0) # 在行方向上拼接，此时列方向上维度不一致，将抛出错误
    print(result)
except Exception as e:
    print(e)

# 如果希望在行方向上拼接，可以通过补零或裁剪等方式使列数一致，torch使用torch.nn.functional.pad进行补零
x_padded = F.pad(x,(0,1)) # 在列方向右侧补一列零，补零后x_padded形状为（2，3）
result = torch.cat((x_padded,y),dim=0) # 在行方向拼接，此时x_padded和y的形状分别为（2，3）（1，3）
print(result) # 拼接后形状为（3，3）

'''
torch.cat()用于拼接张量，参数dim决定在那个维度进行拼接，
若指定维度之外的其他维度形状不相同，则无法拼接，
如果需要将不同形状的张量进行拼接，可以使用补零、裁剪等方式将指定维度外的其他维度形状改为一致，再进行拼接
'''