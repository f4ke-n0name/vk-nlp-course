import torch

def scaled_dot_product_gqa(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, is_causal: bool = True, need_weights: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product attention in grouped manner.

    Args:
        query (torch.Tensor): Query tensor of shape [batch size; seq len; num heads; hidden dim]
        key (torch.Tensor): Key tensor of shape [batch size; kv seq len; num kv heads; hidden dim]
        value (torch.Tensor): Value tensor of shape [batch size; kv seq len; num kv heads; hidden dim]
        is_causal (bool): Whether causal mask of attention should be used
        need_weights (bool): Whether attention weights should be returned

    Returns:
        2-tuple of torch.Tensor:
            - Attention output with shape [batch size; seq len; num heads; hidden dim]
            - (Optional) Attention weights with shape [batch size; num heads; seq len; kv seq len].
                Only returned if 'need_weights' is True.
    """
    batch_size, seq_len, num_heads, hidden_dim = query.shape
    kv_seq_len, kv_heads = key.shape[1:3]

    if num_heads % kv_heads:
        raise ValueError('Error')

    query = query.permute(0, 2, 1, 3)
    value = value.repeat_interleave(num_heads // kv_heads, dim=2).permute(0, 2, 1, 3)
    key = key.repeat_interleave(num_heads // kv_heads, dim=2).permute(0, 2, 1, 3)

    scores = (query @ key.transpose(-2, -1)) / (hidden_dim ** 0.5)

    if is_causal:
        mask = torch.triu(torch.ones(seq_len, kv_seq_len), diagonal=1)
        scores.masked_fill_(mask == 1, float('-inf'))

    weights = F.softmax(scores, dim=-1)

    output = (weights @ value).permute(0, 2, 1, 3)

    if need_weights:
        return output, weights
    else:
        return output
