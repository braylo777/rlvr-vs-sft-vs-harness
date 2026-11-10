"""Prompts. Every trained model and the base model are evaluated with COT_SYSTEM only."""

COT_SYSTEM = ("You are a careful math solver. Solve the problem step by step. "
              "Finish with a final line of the form 'The answer is N.' where N is a number.")

POT_SYSTEM = ("Solve the math problem by writing a short Python program. Use only numbers, "
              "arithmetic and variable assignments, one per line. Store the final result in a "
              "variable named answer. Output only the code.")

# Hand-written few-shot examples (not from the GSM8K test split).
POT_SHOTS = [
    ("A shop sells pens at $3 each. Mia buys 4 pens and pays with a $20 bill. How much change does she get?",
     "price = 3\ncount = 4\ncost = price * count\nanswer = 20 - cost"),
    ("A train travels 60 miles per hour for 2.5 hours, then 40 miles per hour for 1 hour. How far does it travel?",
     "first = 60 * 2.5\nsecond = 40 * 1\nanswer = first + second"),
]

COT_SHOTS = [
    ("A shop sells pens at $3 each. Mia buys 4 pens and pays with a $20 bill. How much change does she get?",
     "The pens cost 4 * 3 = 12 dollars.\nHer change is 20 - 12 = 8 dollars.\nThe answer is 8."),
    ("A train travels 60 miles per hour for 2.5 hours, then 40 miles per hour for 1 hour. How far does it travel?",
     "First part: 60 * 2.5 = 150 miles.\nSecond part: 40 * 1 = 40 miles.\nTotal: 150 + 40 = 190 miles.\nThe answer is 190."),
]


def cot_messages(question: str) -> list[dict]:
    return [{"role": "system", "content": COT_SYSTEM}, {"role": "user", "content": question}]


def fewshot_cot_messages(question: str) -> list[dict]:
    msgs = [{"role": "system", "content": COT_SYSTEM}]
    for q, a in COT_SHOTS:
        msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    return msgs + [{"role": "user", "content": question}]


def pot_messages(question: str) -> list[dict]:
    msgs = [{"role": "system", "content": POT_SYSTEM}]
    for q, a in POT_SHOTS:
        msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    return msgs + [{"role": "user", "content": question}]
