'''
torch中的嵌入操作，nn.Embedding
在处理离散输入的任务中（比如自然语言处理），常常需要将离散的标识符（比如单词、字符等）转换为连续的、低维的向量表示。
torch提供了nn.Embedding模块来实现这种嵌入操作

nn.Embedding实际上是一个查找表，它内部维护一个矩阵，每一行对应一个离散标识符的向量表示
假设有一个词汇表，大小为num_embeddings，每个词映射到一个embedding_dim维的向量上
nn.Embedding会创建一个形状为[num_embeddings,embedding_dim]的矩阵
当输入一个包含单词索引的张量时，模块会直接从这个矩阵中查找相应行的向量，作为单词的嵌入表示

它的好处是直接查找而不是进行繁琐的矩阵乘法计算，更加高效和直观
'''

import torch
import torch.nn as nn

# 定义一个嵌入层
embedding = nn.Embedding(num_embeddings=10,embedding_dim=5) # 假设词汇表大小为10，每个单词用5维向量表示
print('嵌入矩阵形状：',embedding.weight.shape)
# torch.Size([10, 5])

# 定义一个包含单词索引的张量，索引表中[3],[7],[1]行
indices = torch.tensor([3,7,1])

embedded_vectors = embedding(indices)
print('查找到的嵌入向量：\n',embedded_vectors)
#  tensor([[ 0.7418,  1.0491,  1.7181, -0.7029, -1.6706],
#         [ 0.9481, -1.1572, -0.7473,  0.5966,  0.4390],
#         [-0.5021,  0.4753, -0.3223, -0.0532,  0.0341]],
#        grad_fn=<EmbeddingBackward0>)

'''
处理批次输入
'''

batch_indices = torch.tensor([
    [1,2,3,4],
    [5,6,7,8]
])
batch_embeddings = embedding(batch_indices) # 使用嵌入层查找嵌入向量
print('批次嵌入向量：\n',batch_embeddings)
# tensor([[[-0.9936, -0.3513, -0.7920, -1.8446, -0.9659],
#          [ 0.9202,  0.5230,  2.2381, -0.7954, -0.0508],
#          [ 0.3666,  0.6366, -0.1946, -1.4518, -0.0060],
#          [ 1.3253, -1.2482,  0.5468,  0.2180,  0.6459]],
#
#         [[ 0.4189,  0.3562,  0.8600, -0.1767, -0.1110],
#          [-0.3078, -1.8781,  0.0648,  0.6710,  1.7552],
#          [-2.3765, -1.1803,  0.4910,  0.6199, -2.8528],
#          [ 1.0757, -1.3970,  1.0612,  0.7177, -0.9579]]],
#        grad_fn=<EmbeddingBackward0>)