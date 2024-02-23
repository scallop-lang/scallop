import openai
import json
from tqdm import tqdm
import random
import os


ZERO_SHOT_HEADER = "Respond only with the answer choice."
FEW_SHOT_HEADER = """Respond only with the answer choice.

Examples:
Q: Alice, Bob, and Claire are playing a game. At the start of the game, they are each holding a ball: Alice has a orange ball, Bob has a white ball, and Claire has a blue ball. \n\nAs the game progresses, pairs of players trade balls. First, Alice and Bob swap balls. Then, Bob and Claire swap balls. Finally, Alice and Bob swap balls. At the end of the game, Alice has the \nAnswer choices: (A) orange ball. (B) white ball. (C) blue ball.
A: (C) blue ball.

Q: Alice, Bob, Claire, Dave, and Eve are playing a game. At the start of the game, they are each holding a ball: Alice has a pink ball, Bob has a white ball, Claire has a red ball, Dave has a purple ball, and Eve has a yellow ball. \n\nAs the game progresses, pairs of players trade balls. First, Alice and Dave swap balls. Then, Claire and Eve swap balls. Then, Alice and Bob swap balls. Then, Dave and Claire swap balls. Finally, Alice and Claire swap balls. At the end of the game, Alice has the \nAnswer choices: (A) pink ball. (B) white ball. (C) red ball. (D) purple ball. (E) yellow ball.
A: (A) pink ball.

Q: Alice, Bob, Claire, Dave, Eve, Fred, and Gertrude are playing a game. At the start of the game, they are each holding a ball: Alice has a green ball, Bob has a white ball, Claire has a yellow ball, Dave has a pink ball, Eve has a orange ball, Fred has a black ball, and Gertrude has a brown ball. \n\nAs the game progresses, pairs of players trade balls. First, Bob and Gertrude swap balls. Then, Fred and Claire swap balls. Then, Dave and Gertrude swap balls. Then, Bob and Gertrude swap balls. Then, Alice and Claire swap balls. Then, Gertrude and Claire swap balls. Finally, Eve and Claire swap balls. At the end of the game, Alice has the \nAnswer choices: (A) green ball. (B) white ball. (C) yellow ball. (D) pink ball. (E) orange ball. (F) black ball. (G) brown ball.
A: (F) black ball.

Answer the following:
"""

TASK_3 = json.load(os.path.abspath(os.path.join(__file__, "../dataset/task-3-objects.json")))
TASK_5 = json.load(os.path.abspath(os.path.join(__file__, "../dataset/task-5-objects.json")))
TASK_7 = json.load(os.path.abspath(os.path.join(__file__, "../dataset/task-7-objects.json")))
DATASET = TASK_3["examples"] + TASK_5["examples"] + TASK_7["examples"]
N = len(DATASET)
SEED = 10
SAMPLE_SIZE = 150


def run_gpt(question):
    messages = [{"role": "user", "content": question}]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0,
    )
    return response["choices"][0]["message"]["content"]


def test_tracking(range):
    out = {"score": 0, "data": []}

    pbar = tqdm(range)
    for i in pbar:
        example = DATASET[i]
        choices = []
        letter = "A"
        target = ""
        for choice in example["target_scores"]:
            choices.append(f"({letter}) {choice}")
            if example["target_scores"][choice]:
                target = choice
            letter = chr(ord(letter) + 1)
        question = f"{ZERO_SHOT_HEADER}\n{example['input']}\nAnswer choices: {' '.join(choices)}"
        try:
            answer = run_gpt(question)
            score = int(target in answer)
            out["score"] += score
            out["data"] += [
                {
                    "id": i,
                    "question": question,
                    "answer": answer,
                    "score": score,
                }
            ]
        except Exception as e:
            out["data"] += [
                {"id": i, "question": question, "exception": str(e), "score": 0}
            ]

        pbar.set_postfix({"score": out["score"]})

    json_object = json.dumps(out.copy(), indent=4)
    with open("data.json", "w") as outfile:
        outfile.write(json_object)


random.seed(SEED)
sample = random.sample(range(N), SAMPLE_SIZE)
test_tracking(sample)
