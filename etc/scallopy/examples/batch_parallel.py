# This script tests the batch-level parallelism of Scallop.
#
# To run this script in non-parallel mode (and time it)
#
# ```
# $ time python batch_parallel.py
# ```
#
# To run this script in parallel mode (and time it)
#
# ```
# $ time python batch_parallel.py --parallel
# ```

import scallopy
import argparse

# Setup command line argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--amount", type=int, default=1000000)
parser.add_argument("--parallel", action="store_true")
args = parser.parse_args()

# Setup a context
ctx = scallopy.ScallopContext()

# It has two relations `digit_1` and `digit_2`
ctx.add_relation("digit_1", int)
ctx.add_relation("digit_2", int)

# It performs digit addition
ctx.add_rule("sum_2(a + b) = digit_1(a), digit_2(b)")

# Compile the context so that it can be ran in batch
ctx.compile()

# Setup the batched input. There will be `args.amount` number of tasks, each one of them will perform one simple addition
inputs = {
  "digit_1": [[(i,)] for i in range(args.amount)],
  "digit_2": [[(i,)] for i in range(args.amount)],
}

# Run the program in batch mode
batched_result = ctx.run_batch(inputs, "sum_2", parallel=args.parallel)

# Check if there are in total `args.amount` result collections being computed
assert len(batched_result) == args.amount
