import openai
import os
import json
import re
from tqdm import tqdm


FEW_SHOT = True
SHOTS = """
Examples:
Q: Yesterday is February 14, 2019. What is the date 1 month ago?
A: Let's think step by step. Since yesterday is February 14, 2019, that means today is February 15, 2019. 1 month ago from today is January 15, 2019. Therefore, the answer is 01/15/2019.

Q: Yesterday is February 14, 2019. What is the date 1 year later?
A: Let's think step by step. Since yesterday is February 14, 2019, that means today is February 15, 2019. 1 year later from today is February 15, 2020. Therefore, the answer is 02/15/2020.

Q: The deadline is August 15, 2023, which is today. What is the date today?
A: Let's think step by step. Since the deadline is August 15, 2023, and since today is the deadline, that means today is August 15, 2023. Therefore, the answer is 08/15/2023.

Q: Jenny began her current job on the Christmas Eve of 2016. What is the 5th anniversary?
A: Let's think step by step. Since Jenny began her current job on Christmas Eve of 2016, that means she began on her job on December 24, 2016. The 5th anniversary is 5 years later from that date, so that date is December 24, 2021. Therefore, the answer is 12/24/2021.

Q: Today is March 5th, 2010. Mark earns $1000 per day starting from now. When can Mark earn $10000?
A: Let's think step by step. Since Mark earns $1000 per day, it takes $10000 / $1000 = 10 days for Mark to earn $10000. Because today is March 5th, 2010, Mark will earn $10000 after 10 days, which is March 15th, 2010. Therefore, the answer is 03/15/2010.


Now here is the question:
"""
COT_PROMPT = "Let's think step by step."
COT_EXTRACTION = "Therefore, in MM/DD/YYYY form, the answer is"
REGEX = r"\d\d\/\d\d\/\d\d\d\d"

DATASET = os.path.abspath(os.path.join(__file__, "../dataset/task-corrected.json"))
TASK = json.load(open(DATASET))
N = len(TASK["examples"])


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
        example = TASK["examples"][i]
        question = f"Q: {example['input']}\nA: {COT_PROMPT}"
        try:
            if FEW_SHOT:
                response = run_gpt(SHOTS + question)
            else:
                response = run_gpt(question)
            question2 = f"{question} {response}\n{COT_EXTRACTION}"
            response2 = run_gpt(question2)
            pred = re.findall(REGEX, response2)
            pred = pred[0] if len(pred) > 0 else ""
            score = int(
                pred in example["target_scores"] and example["target_scores"][pred]
            )
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


test_tracking(range(N))
