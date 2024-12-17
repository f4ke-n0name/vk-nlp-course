import random

from fsm import FSM, build_odd_zeros_fsm


def get_valid_tokens(vocab: dict[int, str], eos_token_id: int, fsm: FSM, state: int) -> list[int]:
    """Filter tokens from the vocabulary based on the given state in the FSM.  
    1. Retain only tokens that can be achieved from the given state.  
    2. If the current state is terminal, then add the EOS token.

    Args:
        vocab (dict): vocabulary, id to token
        eos_token_id (int): index of EOS token
        fsm (FSM): Finite-State Machine
        state (int): start state
    Returns:
        valid tokens (list): list of possible tokens
    """
    result = []
    for id_token, token in vocab.items():
        if fsm.validate_continuation(state, token):
            result.append(id_token)
    if fsm.states[state].is_terminal:
        result.append(eos_token_id)
    return result


def random_generation() -> str:
    """Structured generation based on Odd-Zeros FSM with random sampling from possible tokens.

    Args:
    Returns:
        generation (str): A binary string with an odd number of zeros.
    """
    vocab = {0: "[EOS]", 1: "0", 2: "1"}
    eos_token_id = 0
    fsm, state = build_odd_zeros_fsm()
    tokens: list[int] = []
    while True:
        valid_tokens = get_valid_tokens(vocab, eos_token_id, fsm, state)
        next_token = random.choice(valid_tokens)
        if next_token == eos_token_id:
            break
        tokens.append(next_token)
        state = fsm.move(vocab[next_token], state)

    return "".join([vocab[it] for it in tokens])


if __name__ == "__main__":
    print(random_generation())
