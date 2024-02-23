import openai
import json
import re
from tqdm import tqdm
import random
import os


FEW_SHOT = False
SHOTS = """
Examples:
Q: Alice, Bob, and Claire are playing a game. At the start of the game, they are each holding a ball: Alice has a orange ball, Bob has a white ball, and Claire has a blue ball. \n\nAs the game progresses, pairs of players trade balls. First, Alice and Bob swap balls. Then, Bob and Claire swap balls. Finally, Alice and Bob swap balls. At the end of the game, Alice has the
A: Let's think step by step.
1. Alice and Bob swap balls. Now Alice has the white ball, and Bob has the orange ball.
2. Bob and Claire swap balls. Now Bob has the blue ball, and Claire has the orange ball.
3. Alice and Bob swap balls. Now Alice has the blue ball, and Bob has the white ball.
At the end of the game, Alice has the blue ball.

Q: Alice, Bob, and Claire are on the same team in a soccer match. At the start of the match, they are each assigned to a position: Alice is playing striker, Bob is playing benchwarmer, and Claire is playing center midfielder. \n\nAs the game progresses, pairs of players occasionally swap positions. First, Alice and Bob trade positions. Then, Alice and Claire trade positions. Finally, Claire and Bob trade positions. At the end of the match, Bob is playing 
A: Let's think step by step.
1. Alice and Bob trade positions. Now Alice is benchwarmer, and Bob is striker.
2. Alice and Claire trade positions. Now Alice is center midfielder, and Claire is benchwarmer.
3. Claire and Bob trade positions. Now Bob is benchwarmer, and Claire is striker.
At the end of the match, Bob is playing benchwarmer.

Q: Alice, Bob, and Claire are dancers at a square dance. At the start of a song, they each have a partner: Alice is dancing with Patrick, Bob is dancing with Karl, and Claire is dancing with Izzi. \n\nThroughout the song, the dancers often trade partners. First, Bob and Claire switch partners. Then, Claire and Alice switch partners. Finally, Claire and Bob switch partners. At the end of the dance, Claire is dancing with
A: Let's think step by step.
1. Bob and Claire switch partners. Now Bob's partner is Izzi, and Claire's partner is Karl.
2. Claire and Alice switch partners. Now Alice's partner is Karl, and Claire's partner is Patrick.
3. Claire and Bob switch partners. Now Bob's partner is Patrick, and Claire's partner is Izzi.
At the end of the dance, Claire is dancing with Izzi.


Now here is the question:
"""
COT_PROMPT = "Let's think step by step."
COT_EXTRACTION = "Therefore, among A through G, the answer is"
REGEX = r"A|B|C|D|E|F|G"

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
                target = letter
            letter = chr(ord(letter) + 1)
        question = f"Q: {example['input']} Answer choices: {' '.join(choices)}\nA: {COT_PROMPT}"
        try:
            if FEW_SHOT:
                response = run_gpt(SHOTS + question)
            else:
                response = run_gpt(question)
            question2 = f"{question} {response}\n{COT_EXTRACTION}"
            response2 = run_gpt(question2)
            pred = re.findall(REGEX, response2)
            pred = pred[0] if len(pred) > 0 else ""
            score = int(pred == target)
            out["score"] += score
            out["data"] += [
                {
                    "id": i,
                    "question": question2,
                    "response": response2,
                    "answer": pred,
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
