import openai
import json
from tqdm import tqdm
from io import StringIO
import sys

from dataset_loader import GSM8KDataset

TASK = GSM8KDataset()
N = len(TASK)
MARGIN = 0.001

HEADER = '''
In this task, you will be given a math word problem to solve.
Please output your reasoning as a chain of thought and then output your answer on a new line at the end. For the final answer, do not include any non-numerical digits except for a decimal point when applicable.

Here are some examples:

Question: Lisa, Jack, and Tommy earned $60 from washing cars all week. However, half of the $60 was earned by Lisa. Tommy earned half of what Lisa earned. How much more money did Lisa earn than Tommy?
Answer: Let's think step by step. First, we know that Lisa earned half of the $60, which is $60 / 2 = $30. Then, we know that Tommy earned half of what Lisa earned, which is $30 / 2 = $15. So, Lisa earned $30 - $15 = $15 more than Tommy.

15

Now, answer the following question:

\n
Question:\s
'''

def run_gpt(question):
    messages = [{"role": "user", "content": HEADER + question + "\nAnswer: Let's think step by step. "}]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0,
    )
    return response["choices"][0]["message"]["content"]

def test_gsm8k(range):
    out = {"exact_score": 0, "margin_score": 0, "data": []}

    for i in tqdm(range):
        input, ans = TASK[i]

        try:
            output_ans = run_gpt(input)
            final_ans = output_ans.split('\n')[-1].replace(',', '')
            exact_score = int(float(final_ans) == float(ans))
            margin_score = int(abs(float(final_ans) - float(ans)) <= MARGIN)
            out["exact_score"] += exact_score
            out["margin_score"] += margin_score
            out["data"] += [
                {
                    "id": i,
                    "question": input,
                    "reasoning": output_ans,
                    "answer": final_ans,
                    "correct_answer": ans,
                    "exact_score": exact_score,
                    "margin_score": margin_score,
                }
            ]
        except Exception as e:
            out["data"] += [
                {"id": i, "question": input, "exception": str(e), "exact_score": 0, "margin_score": 0}
            ]
        
        json_object = json.dumps(out.copy(), indent=4)
        with open("data_baseline.json", "w") as outfile:
            outfile.write(json_object)

test_gsm8k(range(N))