import numpy as np
import torch

def generate_with_reward_guidance(
        main_model, main_tokenizer,
        reward_model, reward_tokenizer,
        N=16,
        device='cpu',
    ):
    """
    Generate text samples using a main model and select the best sample based on a reward model's guidance.

    This function generates multiple text samples from a main model, evaluates each sample using a reward model,
    and returns the sample with the highest reward score. The process is guided by the reward model to select
    the most desirable output.

    Parameters:
    main_model: The language model used to generate text samples.
    main_tokenizer: The tokenizer for main_model
    reward_model: The model used to compute reward scores for the generated samples.
    reward_tokenizer: The tokenizer for reward_model
    N (int, optional): The number of text samples to generate. Default is 16.
    device (str, optional): The device on which the computation should be performed. Default is 'cpu'.

    Returns:
    str: The generated text sample with the highest reward score.
    """
    samples = []
    for i in range(N):
        inputs = main_tokenizer(" ", return_tensors='pt')["input_ids"]
        generated_candidate = main_model.generate(inputs, max_new_tokens=50)
        generated = main_tokenizer.decode(generated_candidate[0])
        samples.append(generated)
    results = []
    with torch.no_grad():
        for sample in samples:
            result = compute_reward(reward_model, reward_tokenizer, [sample]).item()
            results.append(result)
    best_result_idx = torch.argmax(torch.tensor(results))
    best_reward = results[best_result_idx]
    return best_sample, best_reward
