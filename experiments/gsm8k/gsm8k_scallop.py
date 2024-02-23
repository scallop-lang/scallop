import json
from tqdm import tqdm
from io import StringIO
import sys

import scallopy
import scallopy_ext

from dataset_loader import GSM8KDataset

TASK = GSM8KDataset()
N = len(TASK)
SCALLOP_FILE = "semantic_parser.scl"
MARGIN = 0.001

class Args:
    def __init__(self):
        self.cuda = False
        self.gpu = None
        self.num_allowed_openai_request = 100
        self.openai_gpt_model = "gpt-4"
        self.openai_gpt_temperature = 0

def test_semantic_parser(range=range(N)):
    out = {"exact_score": 0, "margin_score": 0, "data": [], "logs": []}

    for i in tqdm(range):
        question, answer = TASK[i]

        buffer = StringIO()
        sys.stdout = buffer
        try:
            ctx = scallopy.ScallopContext(provenance="unit")
            scallopy_ext.config.configure(Args(), [])
            scallopy_ext.extlib.load_extlib(ctx)
            ctx.import_file(SCALLOP_FILE)
            ctx.add_facts("question", [(question,)])
            ctx.run()
            res = list(ctx.relation("result"))
            exact_score = 0
            margin_score = 0
            final_answer = None
            for output_ans, in res:
                if float(output_ans) == answer:
                    exact_score = 1
                    margin_score = 1
                    final_answer = output_ans
                if exact_score == 0 and abs(float(output_ans) - answer) <= MARGIN:
                    margin_score = 1
                    final_answer = output_ans
            out["data"] += [
                {
                    "id": i,
                    "question": question,
                    "correct_answer": answer,
                    "final_answer": final_answer,
                    "exact_score": exact_score,
                    "margin_score": margin_score,
                    "parsed_expr": list(ctx.relation("parsed_expr")),
                    "outputted_answers": list(ctx.relation("result")),
                    "num_answers": len(res),
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
        with open("data_scallop.json", "w") as outfile:
            outfile.write(json_object)

    sys.stdout = sys.__stdout__
    print(out["exact_score"])
          
test_semantic_parser()