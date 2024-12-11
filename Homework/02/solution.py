import torch
import torch.nn.functional as F
import numpy as np

def compute_attention(queries, keys, values) -> torch.Tensor:
    """
    queries- (BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM)
    keys- (BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM)
    values- (BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM)
    """
    hidden_dim = keys.shape[2]
    attention_scores = F.softmax((queries @ keys.mT) / np.sqrt(hidden_dim) , dim=2)
    return attention_scores @ values


def compute_multihead_attention(queries, keys, values, projection_matrix) -> torch.Tensor:
    """
    queries- (BATCH_SIZE, N_HEADS, SEQ_LENGTH, DIM_PER_HEAD)
    keys- (BATCH_SIZE, N_HEADS, SEQ_LENGTH, DIM_PER_HEAD)
    values- (BATCH_SIZE, N_HEADS, SEQ_LENGTH, DIM_PER_HEAD)
    projection_matrix- (N_HEADS*DIM_PER_HEAD, N_HEADS*DIM_PER_HEAD)
    """
    n_heads = keys.shape[1]
    attention_scores = [compute_attention(queries[:, i, :, :],
                                           keys[:, i, :, :],
                                           values[:, i, :, :]) for i in range(n_heads)]
    attention = torch.concat(attention_scores, dim=-1)
    return attention @ projection_matrix.T


def generate_rotation_matrices(frequencies, i):
    matrices = []
    for freq in frequencies:
        cos_val, sin_val = np.cos([freq * i])[0], np.sin([freq * i])[0]
        rotation = torch.tensor([[cos_val, -sin_val], [sin_val, cos_val]])
        matrices.append(rotation)
    return matrices

def compute_rotary_embeddings(x)-> torch.Tensor:
    """
    x- (BATCH_SIZE, SEQ_LENGTH, N_HEADS, DIM_PER_HEAD)
    """
    

    batch_size, seq_length, n_heads, dim_per_head = x.shape
    frequencies = torch.tensor([10 ** (-8 * idx / dim_per_head) for idx in range(dim_per_head // 2)])
    rotation_matrices = [generate_rotation_matrices(frequencies, i) for i in range(seq_length)]
    
    stacked_matrices = torch.stack([torch.block_diag(*rotation_matrices[i]) for i in range(seq_length)])

    reshaped_x = x.permute(0, 2, 1, 3).reshape(batch_size * n_heads, seq_length, dim_per_head)
    transformed_x = torch.zeros_like(reshaped_x)
    
    for i in range(seq_length):
        transformed_x[:, i, :] = reshaped_x[:, i, :] @ stacked_matrices[i].mT
    
    result = transformed_x.reshape(batch_size, n_heads, seq_length, dim_per_head).permute(0, 2, 1, 3)
    return result
