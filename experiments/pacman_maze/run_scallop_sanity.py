from argparse import ArgumentParser
from tqdm import tqdm
import numpy
import random
import scallopy
import os
import cv2

from arena import AvoidingArena

FILE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../"))

class ScallopPolicy:
  def __init__(self, grid_x, grid_y):
    self.grid_x, self.grid_y = grid_x, grid_y
    self.cells = [(x, y) for x in range(self.grid_x) for y in range(self.grid_y)]
    self.ctx = scallopy.ScallopContext(provenance="topkproofs", k=1)
    self.ctx.import_file(os.path.join(FILE_DIR, "scl", "arena.scl"))
    self.ctx.add_facts("grid_node", [(args.attenuation, c) for c in self.cells])

  def __call__(self, hidden_state):
    curr_pos, goal_pos, enemies = hidden_state
    temp_ctx = self.ctx.clone()
    temp_ctx.add_facts("curr_position", [(None, curr_pos)])
    temp_ctx.add_facts("goal_position", [(None, goal_pos)])
    temp_ctx.add_facts("is_enemy", [(None, p) for p in enemies])
    temp_ctx.run()
    output = list(temp_ctx.relation("action_score"))
    if len(output) > 0:
      action_id = numpy.argmax(numpy.array([p for (p, _) in output]))
      action = output[action_id][1][0]
      return action
    else:
      return random.randrange(0, 4)


def show_image(raw_image):
  cv2.imshow("Current Arena", raw_image)
  cv2.waitKey(int(args.show_run_interval * 1000))


def test_scallop_model():
  # Initialize
  arena = AvoidingArena((args.grid_x, args.grid_y), args.cell_size, args.dpi, args.num_enemies)
  model = ScallopPolicy(args.grid_x, args.grid_y)
  success, failure = 0, 0
  iterator = tqdm(range(args.num_episodes))
  for episode_i in iterator:
    _ = arena.reset()
    if args.show_run:
      show_image(arena.render())
    for _ in range(args.num_steps):
      action = model(arena.hidden_state())
      _, done, reward, _ = arena.step(action)
      if args.show_run:
        show_image(arena.render())
      if done:
        if reward > 0: success += 1
        else: failure += 1
        break

    # Print
    success_rate = (success / (episode_i + 1)) * 100.0
    iterator.set_description(f"[Test] {success}/{episode_i + 1} ({success_rate:.2f}%)")


if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--grid-x", type=int, default=5)
  parser.add_argument("--grid-y", type=int, default=5)
  parser.add_argument("--cell-size", type=float, default=0.5)
  parser.add_argument("--dpi", type=int, default=80)
  parser.add_argument("--num-enemies", type=int, default=5)
  parser.add_argument("--num-episodes", type=int, default=1000)
  parser.add_argument("--num-steps", type=int, default=30)
  parser.add_argument("--attenuation", type=float, default=0.9)
  parser.add_argument("--seed", type=int, default=12345)
  parser.add_argument("--show-run", action="store_true")
  parser.add_argument("--show-run-interval", type=float, default=0.001)
  args = parser.parse_args()

  # Other arguments
  args.show_run = True
  random.seed(args.seed)

  # Test scallop model
  test_scallop_model()
