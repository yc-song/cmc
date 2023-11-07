import torch
from scipy.stats import rankdata
import numpy as np  
import random
# all_scores = torch.zeros(256, 11264)
# previous_round_idxs = torch.randperm(3).unsqueeze(0).expand(2, -1)
# scores = torch.randn(2, 174)
# num_columns_all = all_scores.size(1)
# new_scores = torch.randn(2,3)

# # Reshape tensor B to (256, 3, 1) for broadcasting
# # Create a broadcasted index tensor of shape (256, 3, 174)
# # broadcasted_indices = previous_round_idxs.unsqueeze(-1).expand(-1, -1, num_columns_all)
# # scores = scores.unsqueeze(-1).expand(-1, -1, num_columns_all)
# # index_tensor = torch.arange(num_columns_all).unsqueeze(0).expand(all_scores.size())
# expanded_score = torch.zeros_like(scores)
# print(expanded_score.shape, previous_round_idxs.shape, new_scores.shape)
# expanded_score.scatter_(1, previous_round_idxs, new_scores)

# result = scores+expanded_score
# print(scores, expanded_score)
# print(result)
# Use torch.scatter_add_ to score specific elements of tensor A using tensor C

# a= [[0,1,1,1,1,0,1,1, 1, 0],[1,1,0,1,0],[0,1,0],[1,1]]
# def find_address(a, i, j):
#     if i == 0:
#         return 2*j+a[i][j]
#     return find_address(a, i-1, 2*j+a[i][j])
# for i in range(1,3):
#     for j in range(5):
#         print('**')
#         print(i, j, find_address(a,i, j))
# students output is random float tensor of shape (batch_size, num_elements))
seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

student_outputs = torch.randn(2, 5)
teacher_outputs = torch.randn(2, 5)
labels = torch.randint(0, 5, (2,))
zero_tensors = torch.zeros_like(labels)
zero_tensors = np.expand_dims(zero_tensors.cpu().detach().numpy(), axis = 1)
idx_array_reranker = rankdata(-student_outputs.cpu().detach().numpy(), axis = 1, method = 'min')
rank_reranker = np.take_along_axis(idx_array_reranker, labels[:, None].cpu().detach().numpy(), axis=1)
# Retriever ranking
idx_array_retriever = rankdata(-teacher_outputs.cpu().detach().numpy(), axis = 1, method = 'min')
rank_retriever = np.take_along_axis(idx_array_retriever, labels[:, None].cpu().detach().numpy(), axis=1)
# penalty is defined as (max(0, 1/rank_retriever-1/rank_reranker))
# print(rank_retriever, rank_reranker)
penalty = np.maximum(zero_tensors, 1/rank_retriever-1/rank_reranker)
penalty = torch.from_numpy(penalty).squeeze(1)
# print(penalty)
loss = torch.nn.CrossEntropyLoss(reduction='none')(student_outputs, labels)
# print(loss)
loss *= penalty
loss = torch.mean(loss)
print(loss)

loss_2 = torch.nn.CrossEntropyLoss()(student_outputs, labels)
print(loss_2)