import json
from tqdm import tqdm
from io import StringIO
import sys

import scallopy
import scallopy_ext

import openai

from dataset_loader import GSM8KDataset

TASK = GSM8KDataset()
N = len(TASK)
SCALLOP_FILE = "semantic_parser_cot.scl"
MARGIN = 0.001

HEADER = """
Suppose we have the following symbolic expression language:

Expr ::= Const(float) | Add(Expr, Expr) | Sub(Expr, Expr) | Mult(Expr, Expr) | Div(Expr, Expr)

In the following task, you will be given a math word problem. Please semantically parse it into a symbolic expression.
First, output your reasoning, then put the symbolic expression on a new line.

Here are a few examples:

Question: Lisa, Jack, and Tommy earned $60 from washing cars all week. However, half of the $60 was earned by Lisa. Tommy earned half of what Lisa earned. How much more money did Lisa earn than Tommy?
Answer: Let's think step by step. First, we know that Lisa earned half of the $60, which corresponds to Mult(Const(0.5), Const(60)). Then, we know that Tommy earned half of what Lisa earned (which was Mult(Const(0.5), Const(60))), which corresponds to Mult(Const(0.5), Mult(Const(0.5), Const(60))). \
Finally, to see how much more money Lisa earned than Tommy, we must subtract the amount Tommy earned, Mult(Const(0.5), Mult(Const(0.5), Const(60))), from the amount that Lisa earned, Mult(Const(0.5), Const(60)). This gives us a final answer of \
Sub(Mult(Const(0.5), Const(60)), Mult(Const(0.5), Mult(Const(0.5), Const(60)))).

Sub(Mult(Const(0.5), Const(60)), Mult(Const(0.5), Mult(Const(0.5), Const(60))))

Question: Colton had 72 dolphin stickers. He gave 4 stickers each to 3 friends.  He also gave his friend Mandy 2 more than he gave his three friends total.   And he gave Justin 10 less than Mandy.  How many stickers does Colton have left?
Answer: Let's think step by step. To find the number of stickers Colton has left, we must subtract the number of stickers he gave away to all of his friends from 72 or Const(72), which is the number of stickers he started with. We will first add up \
the total number of stickers he gave away.

First, he gave 4 stickers each to the first 3 friends, which gives a total of Mult(Const(4), Const(3)) stickers. Then, he gave Mandy 2 more than he gave his three friends total. Since he gave his three friends \
Mult(Const(4), Const(3)) total, this means that he gave Mandy Add(Mult(Const(4), Const(3)), Const(2)) \
stickers. So far, Colton has given away Add(Mult(Const(4), Const(3)), Add(Mult(Const(4), Const(3)), Const(2))) \
stickers. Next, Colton gave Justin 10 fewer stickers than Mandy. Since he gave Mandy Add(Mult(Const(4), Const(3)), Const(2)) \
stickers, this means that he gave Justin Sub(Add(Mult(Const(4), Const(3)), Const(2)), Const(10)) stickers. \
Since we must also add the stickers Colton gave Justin to the total number of stickers Colton gave away, \
Colton has given away Add(Add(Mult(Const(4), Const(3)), Add(Mult(Const(4), Const(3)), Const(2))), Sub(Add(Mult(Const(4), Const(3)), Const(2)), Const(10))) \
stickers in total.

Finally, since Colton started out with Const(72) stickers but gave away Add(Add(Mult(Const(4), Const(3)), Add(Mult(Const(4), Const(3)), Const(2))), Sub(Add(Mult(Const(4), Const(3)), Const(2)), Const(10))) \
of them, Colton has Sub(Const(72), Add(Add(Mult(Const(4), Const(3)), Add(Mult(Const(4), Const(3)), Const(2))), Sub(Add(Mult(Const(4), Const(3)), Const(2)), Const(10)))) \
stickers remaining.

Sub(Const(72), Add(Add(Mult(Const(4), Const(3)), Add(Mult(Const(4), Const(3)), Const(2))), Sub(Add(Mult(Const(4), Const(3)), Const(2)), Const(10))))

Now, please semantically parse the following question:

Question: 
"""

class Args:
    def __init__(self):
        self.cuda = False
        self.gpu = None
        self.num_allowed_openai_request = 100
        self.openai_gpt_model = "gpt-4"
        self.openai_gpt_temperature = 0

def run_gpt(question):
    messages = [{"role": "user", "content": HEADER + question + "\nAnswer: Let's think step by step. "}]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0,
    )
    return response["choices"][0]["message"]["content"]

def test_semantic_parser_cot(range=range(N)):
    out = {"exact_score": 0, "margin_score": 0, "data": [], "logs": []}

    for i in tqdm(range):
        question, answer = TASK[i]

        buffer = StringIO()
        sys.stdout = buffer
        try:
            # GPT should put the parsed expression on the last line
            output = run_gpt(question)
            parsed_expr = output.split('\n')[-1]

            ctx = scallopy.ScallopContext(provenance="unit")
            scallopy_ext.config.configure(Args(), [])
            scallopy_ext.extlib.load_extlib(ctx)
            ctx.import_file(SCALLOP_FILE)
            ctx.add_facts("parsed_expr", [(parsed_expr,)])
            ctx.run()

            final_ans, = list(ctx.relation("result"))[0]
            exact_score = int(final_ans == answer)
            margin_score = int(abs(final_ans - answer) <= MARGIN)

            out["data"] += [
                {
                    "id": i,
                    "question": question,
                    "correct_answer": answer,
                    "final_answer": final_ans,
                    "exact_score": exact_score,
                    "margin_score": margin_score,
                    "reasoning": output,
                    "parsed_expr": parsed_expr,
                    "outputted_answers": list(ctx.relation("result")),
                }
            ]
            out["exact_score"] += exact_score
            out["margin_score"] += margin_score

        except Exception as e:
            out["data"] += [
                {
                    "id": i,
                    "question": question,
                    "exception": str(e),
                    "exact_score": 0,
                    "margin_score": 0,
                }
            ]
        
        out["logs"] += [
            {
                "id": i,
                "log": buffer.getvalue().encode("utf-8").decode("unicode_escape"),
            }
        ]

        json_object = json.dumps(out.copy(), indent=2)
        with open("data_scallop_cot.json", "w") as outfile:
            outfile.write(json_object)

    sys.stdout = sys.__stdout__
    print(out["exact_score"])

test_semantic_parser_cot(range(N))