import openai
import json
from tqdm import tqdm
from io import StringIO
import sys

from dataset_loader import CLUTRRDataset

TASK = CLUTRRDataset()
N = len(TASK)

HEADER = '''
In this task, you will be given a question regarding kinships between characters in a story.
Please output your reasoning as a chain of thought and then output your answer on a new line at the end.

Here are some examples:

Question: Dorothy's brother Michael and her went to get ice cream. Michael is the proud father of the lovely Donald. Who is Dorothy to Donald?
Answer: Dorothy is Donald's aunt. In the given scenario, Michael is Dorothy's brother, and Michael is the father of Donald. This makes Dorothy the sister of Donald's father, which means she is Donald's aunt.

aunt

Question: Michael and his daughter Jennifer like to read poems together. Jason is the proud father of the lovely Michael. Who is Jason to Jennifer?
Answer: Jason is Jennifer's grandfather. In the given scenario, Michael is Jennifer's father, and he enjoys reading poems with her. It is also mentioned that Jason is the proud father of Michael. Therefore, Jason is Jennifer's grandfather, as he is the father of her father, Michael.

grandfather

Question: Kevin loves going to plays with his wife Aida. Aida's dad James, however, does not like them at all. Who is James to Kevin?
Answer: James is Aida's father, and Aida is Kevin's wife. Therefore, James is Kevin's father-in-law.

father-in-law

Now, answer the following question:

\n
Question:\s
'''

def run_gpt(question):
    messages = [{"role": "user", "content": HEADER + question + "\nAnswer:"}]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0,
    )
    return response["choices"][0]["message"]["content"]

def test_kinship(range):
    out = {"score": 0, "data": []}

    for i in tqdm(range):
        (ctx, query), ans = TASK[i]
        input = ctx + " Who is " + query[1] + " to " + query[0] + "?"

        try:
            output_ans = run_gpt(input)
            final_ans = output_ans.split('\n')[-1]
            score = int(final_ans.strip().lower().replace(r'[^a-zA-Z]', '') == ans.strip().lower().replace(r'[^a-zA-Z]', ''))
            out["score"] += score
            out["data"] += [
                {
                    "question": input,
                    "reasoning": output_ans,
                    "answer": final_ans,
                    "score": score,
                }
            ]
        except Exception as e:
            out["data"] += [
                {"question": input, "exception": str(e), "score": 0}
            ]
        
        json_object = json.dumps(out.copy(), indent=4)
        with open("data_baseline.json", "w") as outfile:
            outfile.write(json_object)

test_kinship(range(N))