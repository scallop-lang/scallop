import json
from tqdm import tqdm
from io import StringIO
import sys
import argparse

import scallopy
import scallopy_ext

from dataset_loader import CLUTRRDataset

TASK = CLUTRRDataset()
N = len(TASK)
SCALLOP_FILE = "kinship.scl"

class Args:
    def __init__(self):
        self.cuda = False
        self.gpu = None
        self.num_allowed_openai_request = 100
        self.openai_gpt_model = "gpt-4"
        self.openai_gpt_temperature = 0

def test_kinship(range=range(N)):
    out = {"score": 0, "data": [], "logs": []}
    
    plugins = scallopy_ext.PluginRegistry()
    parser = argparse.ArgumentParser()
    plugins.setup_argument_parser(parser)
    known_args, unknown_args = parser.parse_known_args()
    plugins.configure(known_args, unknown_args)

    for i in tqdm(range):
        (ctx, query), ans = TASK[i]
        input = ctx + " Who is " + query[1] + " to " + query[0] + "?"

        buffer = StringIO()
        sys.stdout = buffer
        try:
            ctx = scallopy.ScallopContext(provenance="unit")
            plugins.load_into_ctx(ctx)
            ctx.import_file(SCALLOP_FILE)
            ctx.add_facts("context", [(input,)])
            ctx.run()
            res = list(ctx.relation("result"))
            score = 0
            final_answer = ""
            for output_ans, in res:
                if output_ans.strip().lower().replace(r'[^a-zA-Z]', '') == ans.strip().lower().replace(r'[^a-zA-Z]', ''):
                    score = 1
                    final_answer = output_ans
            out["data"] += [
                {
                    "id": i,
                    "question": input,
                    "final_answer": final_answer,
                    "score": score,
                    "mentioned_kinship": list(ctx.relation("kinship")),
                    "derived_kinship": list(ctx.relation("derived_kinship")),
                    "query": list(ctx.relation("question")),
                    "answer": list(ctx.relation("result")),
                    "num_answers": len(res),
                }
            ]
            out["score"] += score
        except Exception as e:
            out["data"] += [
                {
                    "id": i,
                    "question": input,
                    "exception": str(e),
                    "score": 0,
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
    print(out["score"])
          
test_kinship()