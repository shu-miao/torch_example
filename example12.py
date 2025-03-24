'''
torch中根据掩码填充张量中的特定元素，
torch使用tensor.masked_fill()和tensor.masked_fill_()来进行根据掩码填充张量中特定元素操作
区别在于：tensor.masked_fill()返回一个新的张量，不修改原张量，tensor.masked_fill_()修改原张量，返回修改后的张量
'''

import torch

# 创建原张量
tensor = torch.tensor([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])

# 创建掩码：True表示要被替换的元素
mask = torch.tensor([
    [False,True,False],
    [True,False,False],
    [False,False,True]
])

print('原张量 tensor：\n',tensor)

# 使用tensor.masked_fill()
new_tensor = tensor.masked_fill(mask,-1)
print('\n新张量 new_tensor（使用tensor.masked_fill()）：\n',new_tensor)
print('\n原张量 tensor（未修改）：\n',tensor)

# 使用tensor.masked_fill_()
tensor.masked_fill_(mask,-1)
print('\n原张量 tensor（被修改）：\n',tensor)


'''
tensor.masked_fill()在transformer自注意力中的应用
'''
# 创建一个4x4的注意力得分矩阵
attn_scores = torch.tensor([
    [0.5,0.7,0.8,0.9],
    [0.6,0.5,0.4,0.8],
    [0.2,0.4,0.5,0.7],
    [0.3,0.5,0.6,0.8]
])

# 创建一个掩码（模拟未来时间步的屏蔽）
mask = torch.tensor([
    [False,False,False,True],
    [False,False,True,True],
    [False,True,True,True],
    [True,True,True,True]
])

mask_scores = attn_scores.masked_fill(mask,float('-inf'))
print('注意力得分（mask_fill()）:\n',mask_scores)
#  tensor([[0.5000, 0.7000, 0.8000,   -inf],
#         [0.6000, 0.5000,   -inf,   -inf],
#         [0.2000,   -inf,   -inf,   -inf],
#         [  -inf,   -inf,   -inf,   -inf]])


'''
tensor.masked_fill_()在梯度计算中的应用
'''
# 创建一个需要计算梯度的张量
x = torch.tensor([0.1, 0.2, 0.3, 0.4], requires_grad=True)
x_clone = x.clone() # 克隆这个张量，断开计算图
# 创建掩码
mask = torch.tensor([False, True, False, True])
# 直接修改 x
x_clone.masked_fill_(mask, 0.0)
print("\n被修改后的 x_clone（masked_fill_()）：\n", x)
#  tensor([0.1000, 0.2000, 0.3000, 0.4000], requires_grad=True)
