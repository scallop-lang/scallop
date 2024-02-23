import json
from tqdm import tqdm
from datetime import datetime
import scallopy
import scallopy_ext
import os
from io import StringIO
import sys

DATASET = os.path.abspath(os.path.join(__file__, "../dataset/task-corrected.json"))
TASK = json.load(open(DATASET))
N = len(TASK["examples"])
SCALLOP_FILE = os.path.abspath(os.path.join(__file__, "../date-compute.scl"))


class Args:
    def __init__(self):
        self.cuda = False
        self.gpu = None
        self.num_allowed_openai_request = N
        self.openai_gpt_model = "gpt-4"
        self.openai_gpt_temperature = 0


def test_date_understanding(range):
    out = {"score": 0, "data": [], "logs": []}

    # Configure scallop extension library
    scallopy_ext.config.configure(Args())

    # Setup scallop context
    ctx = scallopy.ScallopContext(provenance="unit")
    scallopy_ext.extlib.load_extlib(ctx)
    ctx.import_file(SCALLOP_FILE)

    # Iterate through al the datapoints
    pbar = tqdm(range)
    for i in pbar:
        example = TASK["examples"][i]
        buffer = StringIO()
        sys.stdout = buffer

        try:
            temp_ctx = ctx.clone()
            temp_ctx.add_facts("question", [(example["input"],)])
            temp_ctx.run()
            res = [
                datetime.strptime(x[0].split(" ")[0], "%Y-%m-%d").strftime("%m/%d/%Y")
                for x in list(temp_ctx.relation("answer"))
            ]
            score = 0
            final_answer = ""
            for answer in example["target_scores"]:
                if answer in res:
                    final_answer = answer
                    score = example["target_scores"][answer]
                    break
            out["score"] += score
            out["data"] += [
                {
                    "id": i,
                    "question": example["input"],
                    "final_answer": final_answer,
                    "score": score,
                    "mentioned_date": list(temp_ctx.relation("mentioned_date")),
                    "relationship": list(temp_ctx.relation("relationship")),
                    "goal": list(temp_ctx.relation("goal")),
                    "answer": list(temp_ctx.relation("answer")),
                }
            ]
        except Exception as e:
            out["data"] += [
                {
                    "id": i,
                    "question": example["input"],
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

        pbar.set_postfix({"score": out["score"]})

    json_object = json.dumps(out.copy(), indent=2)
    with open("data.json", "w") as outfile:
        outfile.write(json_object)


test_date_understanding(range(N))
