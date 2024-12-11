from tqdm.auto import tqdm

def eval_reward_model(reward_model, reward_tokenizer, test_dataset, target_label, device='cpu'):
    """
    Evaluate the performance of a reward model by comparing reward scores for chosen and rejected reviews.

    This function selects reviews from a test dataset based on a target label and evaluates the reward model's
    ability to assign higher scores to chosen reviews compared to rejected ones. The evaluation is performed
    in batches for efficiency.
    Note that reward scores are compared on corresponding chosen and rejected reviews:
        chosen_reviews[0] vs rejected_reviews[0],
        chosen_reviews[1] vs rejected_reviews[1],
        etc.

    Parameters:
    reward_model: The model used to compute the reward scores
    reward_tokenizer: The tokenizer for reward_model
    tes_dataset: test Dataset
    target_label (0 or 1): The label used to select chosen reviews. Reviews with this label are considered chosen,
                  while others are considered rejected.
    device (str, optional): The device on which the computation should be performed. Default is 'cpu'.

    Returns:
    float: The accuracy of the reward model, calculated as the proportion of times the model assigns a higher
           reward score to the chosen review compared to the rejected review.

    Example:
    >>> accuracy = eval_reward_model(my_reward_model, my_reward_tokenizer, test_data, target_label=1)
    >>> print(f"Model accuracy: {accuracy:.2%}")
    """
    chosen_reviews = [item["text"] for item in test_dataset if item["label"] == target_label]
    rejected_reviews = [item["text"] for item in test_dataset if item["label"] != target_label]

    assert len(chosen_reviews) == len(rejected_reviews)

    accuracy = 0
    with no_grad():
        for chosen_text, rejected_text in zip(chosen_reviews, rejected_reviews):
            if reward_model is None or reward_tokenizer is None:
                if chosen_text.isnumeric() and rejected_text.isnumeric():
                    reward = [int(chosen_text), int(rejected_text)]
                else:
                    reward = [1, 0]
            else:
                tokenized_ids = reward_tokenizer([chosen_text, rejected_text], padding=True, truncation=True ,return_tensors="pt")
                input_ids = {k: v.to(device) for k,v in tokenized_ids.items()}
                with torch.no_grad():
                    reward = reward_model(**input_ids).logits[:, 0]
            if (reward[0] > reward[1]):
                accuracy += 1

    accuracy = accuracy / len(chosen_reviews)
    return accuracy
