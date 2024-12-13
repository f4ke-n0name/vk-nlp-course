import torch


def compute_alibi(num_heads: int, seq_len: int) -> torch.Tensor:
    """
    Compute ALiBi for a sequence.

    ALiBi can be used not only with causal models.
    In this case, the biases will be symmetrical about the diagonal up to the sign.

    Args:
        num_heads (int): Number of attention heads.
        seq_len (int): Sequence length.

    Returns:
        torch.Tensor: A tensor containing ALiBi to be added to attention scores.
    """
    alibi = torch.zeros(num_heads, seq_len, seq_len)
    factors = 2 ** (torch.arange(1, num_heads + 1).view(-1, 1) * (8 / num_heads))
    
    for head in range(num_heads):
        alibi[head] = -(torch.arange(seq_len).view(-1, 1) - torch.arange(seq_len).view(1, -1)) / factors[head]

    return alibi


if __name__ == "__main__":
    bias = compute_alibi(4, 4)
    print(bias)
