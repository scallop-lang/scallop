import openai
import json
from tqdm import tqdm

from dataset_loader import CLUTRRDataset


FEW_SHOT = True
SHOTS = """
Examples:
Q: Dorothy's brother Michael and her went to get ice cream. Michael is the proud father of the lovely Donald. Who is Dorothy to Donald?
A: aunt

Q: Michael and his daughter Jennifer like to read poems together. Jason is the proud father of the lovely Michael. Who is Jason to Jennifer?
A: grandfather

Q: Kevin loves going to plays with his wife Aida. Aida's dad James, however, does not like them at all. Who is James to Kevin?
A: father-in-law


Now here is the question:
"""
COT_PROMPT = "Let's think step by step."
COT_EXTRACTION = "Therefore, in one word, the answer is"

TASK = CLUTRRDataset()
N = len(TASK)


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
        (ctx, query), ans = TASK[i]
        input = ctx + " Who is " + query[1] + " to " + query[0] + "?"


        question = f"Q: {input}\nA: {COT_PROMPT}"
        try:
            if FEW_SHOT:
                response = run_gpt(SHOTS + question)
            else:
                response = run_gpt(question)
            question2 = f"{question} {response}\n{COT_EXTRACTION}"
            response2 = run_gpt(question2)
            final_ans = response2.split()[-1]
            score = int(final_ans.strip().lower().replace(r'[^a-zA-Z]', '') == ans.strip().lower().replace(r'[^a-zA-Z]', ''))
            out["score"] += score
            out["data"] += [
                {
                    "question": input,
                    "reasoning": response,
                    "answer": final_ans,
                    "correct_answer": ans,
                    "score": score,
                }
            ]
        except Exception as e:
            out["data"] += [
                {"question": input, "exception": str(e), "score": 0}
            ]

        pbar.set_postfix({"score": out["score"]})

        json_object = json.dumps(out.copy(), indent=4)
        with open("data_cot.json", "w") as outfile:
            outfile.write(json_object)


test_tracking(range(N))