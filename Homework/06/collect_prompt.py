def create_prompt(sample: dict) -> str:
    """
    Generates a prompt for a multiple choice question based on the given sample.

    Args:
        sample (dict): A dictionary containing the question, subject, choices, and answer index.

    Returns:
        str: A formatted string prompt for the multiple choice question.
    """
    answers = ['A', 'B', 'C', 'D']
    question = sample.get('question', '')
    subject = sample.get('subject', '')
    choices = sample.get('choices', [])
    header = f"The following are multiple choice questions (with answers) about {subject}.\n{question}\n"
    prompt = ""
    for i in range(len(answers)):
        prompt += f"{answers[i]}. {choices[i]}\n"

    prompt += f"Answer:"

    return header + prompt


def create_prompt_with_examples(sample: dict, examples: list, add_full_example: bool = False) -> str:
    """
    Generates a 5-shot prompt for a multiple choice question based on the given sample and examples.

    Args:
        sample (dict): A dictionary containing the question, subject, choices, and answer index.
        examples (list): A list of 5 example dictionaries from the dev set.
        add_full_example (bool): whether to add the full text of an answer option

    Returns:
        str: A formatted string prompt for the multiple choice question with 5 examples.
    """
    answers = ['A', 'B', 'C', 'D']
    prompt = ""
    for example in examples:
       question = example.get('question', '')
       subject = example.get('subject', '')
       choices = example.get('choices', [])
       answer_index = example.get('answer')

       prompt += f"The following are multiple choice questions (with answers) about {subject}.\n{question}\n"
       for i in range(len(answers)):
         prompt += f"{answers[i]}. {choices[i]}\n"

       prompt += f"Answer: {answers[answer_index]}"
       if (add_full_example):
           prompt += f". {choices[answer_index]}"

       prompt += "\n\n"

    prompt += create_prompt(sample)

    return prompt
