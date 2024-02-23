import json
import jsbeautifier
from tqdm import tqdm
import scallopy
import scallopy_ext
from io import StringIO
import sys
import random
import os

TASK_3 = json.load(os.path.abspath(os.path.join(__file__, "../dataset/task-3-objects.json")))
TASK_5 = json.load(os.path.abspath(os.path.join(__file__, "../dataset/task-5-objects.json")))
TASK_7 = json.load(os.path.abspath(os.path.join(__file__, "../dataset/task-7-objects.json")))
DATASET = TASK_3["examples"] + TASK_5["examples"] + TASK_7["examples"]
N = len(DATASET)
SEED = 10
SAMPLE_SIZE = 150
SCALLOP_FILE = "tracking.scl"


class Args:
    def __init__(self):
        self.cuda = False
        self.gpu = None
        self.num_allowed_openai_request = SAMPLE_SIZE
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
        example = DATASET[i]
        buffer = StringIO()
        sys.stdout = buffer

        try:
            temp_ctx = ctx.clone()
            temp_ctx.add_facts("question", [(example["input"],)])
            temp_ctx.run()
            res = list(temp_ctx.relation("answer"))[0][0]
            score = 0
            final_answer = ""
            for answer in example["target_scores"]:
                if res in answer:
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
                    "possessions": list(temp_ctx.relation("possessions")),
                    "swaps": list(temp_ctx.relation("swaps")),
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

    options = jsbeautifier.default_options()
    options.indent_size = 2
    json_object = jsbeautifier.beautify(json.dumps(out.copy()), options)
    with open("data.json", "w") as outfile:
        outfile.write(json_object)


random.seed(SEED)
sample = random.sample(range(N), SAMPLE_SIZE)
test_date_understanding(sample[100:])
