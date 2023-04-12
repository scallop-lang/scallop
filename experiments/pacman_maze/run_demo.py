import random
import argparse
import cv2

from arena import AvoidingArena

ESC_KEY = 27
LEFT_KEY = 63234
UP_KEY = 63232
DOWN_KEY = 63233
RIGHT_KEY = 63235

def show_image(env: AvoidingArena):
  # Render an image
  image = env.render()

  # Show the image
  cv2.imshow("Avoid Enemy", image)

  # Show the cells
  if args.show_cells:
    cells = [(i, j) for i in range(args.grid_x) for j in range(args.grid_y)]
    for cell_pos in cells:
      cell_image = env.get_cell_image(image, cell_pos)
      cv2.imshow(f"Cell {cell_pos}", cell_image)

  # Wait for next frame
  if args.auto:
    cv2.waitKey(int(args.interval * 1000))
  elif args.manual:
    key = cv2.waitKeyEx()
    if key == ESC_KEY:
      cv2.destroyAllWindows()
      exit()
    elif key == UP_KEY:
      return env.step(0)
    elif key == RIGHT_KEY:
      return env.step(1)
    elif key == DOWN_KEY:
      return env.step(2)
    elif key == LEFT_KEY:
      return env.step(3)
  else:
    key = cv2.waitKey()
    if key == ESC_KEY:
      cv2.destroyAllWindows()
      exit()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--show-image", action="store_true")
  parser.add_argument("--show-cells", action="store_true")
  parser.add_argument("--print-arena", action="store_true")
  parser.add_argument("--auto", action="store_true")
  parser.add_argument("--manual", action="store_true")
  parser.add_argument("--interval", type=float, default=0.3)
  parser.add_argument("--grid-x", type=int, default=5)
  parser.add_argument("--grid-y", type=int, default=5)
  parser.add_argument("--cell-size", type=float, default=0.5)
  parser.add_argument("--num-enemies", type=int, default=5)
  parser.add_argument("--dpi", type=int, default=80)
  parser.add_argument("--seed", type=int, default=1358)
  args = parser.parse_args()

  # Manual
  if args.manual:
    args.show_image = True
  random.seed(args.seed)

  # Start environment
  env = AvoidingArena((args.grid_x, args.grid_y), args.cell_size, args.dpi, args.num_enemies)
  done, reward = False, 0
  env.reset()

  # Print or show image
  if args.print_arena: env.print_state()
  if args.show_image: result = show_image(env)

  # Enter loop
  while not done:
    if not args.manual:
      action = random.randint(0, 3)
      print(f"Taking action: {env.string_of_action(action)}")
      _, done, reward, _ = env.step(action)
    else:
      _, done, reward, _ = result

    # If finished
    if done: print("Success!" if reward > 0 else "Failed!")

    # Print or show image
    if args.print_arena: env.print_state()
    if args.show_image: result = show_image(env)
