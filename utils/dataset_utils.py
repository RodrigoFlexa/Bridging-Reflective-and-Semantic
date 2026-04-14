from typing import TypedDict


def format_sample(sample: TypedDict) -> str:
    question = sample["question"]
    choices = sample["choices"]["text"]
    answer_key = sample["answerKey"]
    correct_answer = choices[ord(answer_key) - ord("A")]

    formatted = f"Question: {question}\nChoices:\n"
    for idx, choice in enumerate(choices):
        formatted += f"{chr(ord('A') + idx)}. {choice}\n"
    formatted += f"Answer: {correct_answer}"

    return formatted


def get_question(sample: TypedDict) -> str:
    question = sample["question"]
    return question


def get_alternatives_label(sample: TypedDict) -> str:
    label = sample["choices"]["label"]
    return label


def get_alternatives_text(sample: TypedDict) -> str:
    text = sample["choices"]["text"]
    return text


def get_correct_answer(sample: TypedDict) -> str:
    answer_key = sample["answerKey"]
    return answer_key


def get_correct_answer_text(sample: TypedDict) -> str:
    correct_answer = get_correct_answer(sample)
    labels = get_alternatives_label(sample)
    index = labels.index(correct_answer)
    text = sample["choices"]["text"][index]
    return text


def try_to_answer(sample: TypedDict, guess: str) -> bool:
    correct_answer = get_correct_answer(sample).strip().upper()
    treated_guess = guess.strip().upper()
    if correct_answer == treated_guess:
        return True
    else:
        return False


def format_question(sample: TypedDict) -> str:
    question = get_question(sample)
    texts = get_alternatives_text(sample)
    labels = get_alternatives_label(sample)

    formatted_choices = "\n".join(
        [f"({label}) {choice}" for label, choice in zip(labels, texts)]
    )
    formatted_question = f"{question}\n{formatted_choices}"

    return formatted_question
