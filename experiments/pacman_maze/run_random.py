from argparse import ArgumentParser
from tqdm import tqdm
import random

from arena import AvoidingArena

class RandomPolicy:
  def __init__(self): pass

  def __call__(self, _):
    return random.randint(0, 3)

def test_random_model():
  # Initialize
  arena = AvoidingArena((args.grid_x, args.grid_y), args.cell_size, args.dpi, args.num_enemies)
  model = RandomPolicy()
  success, failure = 0, 0
  iterator = tqdm(range(args.num_episodes))
  for episode_i in iterator:
    state = arena.reset()
    for _ in range(args.num_steps):
      action = model(state)
      _, done, reward, _ = arena.step(action)
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
  args = parser.parse_args()

  test_random_model()
