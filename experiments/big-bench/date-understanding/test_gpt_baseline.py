import openai
import os
import json
from tqdm import tqdm

DATASET = os.path.abspath(os.path.join(__file__, "../dataset/task-corrected.json"))
TASK = json.load(open(DATASET))
ZERO_SHOT_HEADER = """Answer only in the format of MM/DD/YYYY, where MM is 2 digits for the month, DD is 2 digits for the day, and YYYY is 4 digits for the year. Do not include anything else with your answer.

Examples:
April 20, 2021 is 04/20/2021 in MM/DD/YYYY form.

Answer the following:
"""
FEW_SHOT_HEADER = """Answer only in the format of MM/DD/YYYY, where MM is 2 digits for the month, DD is 2 digits for the day, and YYYY is 4 digits for the year. Do not include anything else with your answer.

Examples:
Question: The deadline is Jun 1, 2021, which is 2 days away from now. What is the date 10 days ago in MM/DD/YYYY?
05/20/2021

Question: Jenny began her current job on the Christmas Eve of 2016. What is the 5th anniversary in MM/DD/YYYY?
12/24/2021

Question: Today is March 5th, 2010. Mark earns $1000 per day starting from now. When can Mark earh $10000 in MM/DD/YYYY?
03/15/2010

Answer the following:
"""


def run_gpt(question):
    messages = [{"role": "user", "content": ZERO_SHOT_HEADER + question}]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0,
    )
    return response["choices"][0]["message"]["content"]


def test_date_understanding():
    out = {"score": 0, "data": []}

    pbar = tqdm(TASK["examples"])
    for example in pbar:
        try:
            answer = run_gpt(example["input"])
            score = int(
                answer in example["target_scores"] and example["target_scores"][answer]
            )
            out["score"] += score
            out["data"] += [
                {
                    "question": example["input"],
                    "answer": answer,
                    "score": score,
                }
            ]
        except Exception as e:
            out["data"] += [
                {"question": example["input"], "exception": str(e), "score": 0}
            ]

        pbar.set_postfix({"score": out["score"]})

    json_object = json.dumps(out.copy(), indent=4)
    with open("data.json", "w") as outfile:
        outfile.write(json_object)


test_date_understanding()
