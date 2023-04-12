from typing import *
import gym
import numpy
import cv2
import random
import torch
import os

RES_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../res"))

class AvoidingArena(gym.Env):
  def __init__(
    self,
    grid_dim: Tuple[int, int] =(5, 5),
    cell_size: float = 0.5,
    dpi: int = 80,
    num_enemies: int = 5,
    easy: bool = False,
    default_reward: float = 0.00,
    on_success_reward: float = 1.0,
    on_failure_reward: float = 0.0,
    remain_unchanged_reward: float = 0.0,
  ):
    """
    :param grid_dim, (int, int), a tuple of two integers for (grid_x, grid_y)
    :param cell_size, float, the side length of each cell, in inches
    :param dpi, int, dimension (number of pixels) per inch
    :param num_enemies, int, maximum number of enemies on the arena
    """
    self.grid_x, self.grid_y = grid_dim
    self.cell_size = cell_size
    self.dpi = dpi
    self.num_enemies = num_enemies
    self.image_w, self.image_h = self.grid_x * self.cell_size, self.grid_y * self.cell_size
    self.easy = easy
    self.default_reward = default_reward
    self.on_success_reward = on_success_reward
    self.on_failure_reward = on_failure_reward
    self.remain_unchanged_reward = remain_unchanged_reward

    # Initialize environment states
    self.curr_pos = None
    self.start_pos = None
    self.goal_pos = None
    self.enemies = None

    # Load background and enemy images
    self.background_image = cv2.imread(os.path.join(RES_DIR, "back.webp"))
    enemy_image_1 = cv2.imread(os.path.join(RES_DIR, "enemy1.webp"), cv2.IMREAD_UNCHANGED)
    enemy_image_2 = cv2.imread(os.path.join(RES_DIR, "enemy2.webp"), cv2.IMREAD_UNCHANGED)
    self.enemy_images = [enemy_image_1, enemy_image_2]
    self.goal_image = cv2.imread(os.path.join(RES_DIR, "flag.png"), cv2.IMREAD_UNCHANGED)
    self.agent_image = cv2.imread(os.path.join(RES_DIR, "agent.png"), cv2.IMREAD_UNCHANGED)

  def reset(self):
    # Generate start position
    self.start_pos = self.sample_point()
    self.curr_pos = self.start_pos

    if self.easy:
      possible_positions = []
      for (off_x, off_y) in [(0, 1), (1, 0), (0, -1), (1, 0)]:
        goal_x = self.start_pos[0] + off_x
        goal_y = self.start_pos[1] + off_y
        if 0 <= goal_x < self.grid_x and 0 <= goal_y < self.grid_y:
          possible_positions.append((goal_x, goal_y))
      self.goal_pos = possible_positions[random.randrange(0, len(possible_positions))]
    else:
      # Generate end position
      self.goal_pos = self.sample_point()
      while self.goal_pos == self.start_pos:
        self.goal_pos = self.sample_point()

    # Generate enemy positions
    self.enemies = []
    self.enemy_types = []
    num_tries = 0
    while len(self.enemies) < self.num_enemies and num_tries < 100:
      num_tries += 1
      try_pos = self.sample_point()
      if self.ok_enemy_position(try_pos):
        self.enemies.append(try_pos)
        self.enemy_types.append(random.randint(0, 1))

    # Return
    return ()

  def step(self, action):
    prev_pos = self.curr_pos

    # Step action
    if action == 0 and self.curr_pos[1] < self.grid_y - 1: # If can move up
      self.curr_pos = (self.curr_pos[0], self.curr_pos[1] + 1)
    elif action == 1 and self.curr_pos[0] < self.grid_x - 1: # If can move right
      self.curr_pos = (self.curr_pos[0] + 1, self.curr_pos[1])
    elif action == 2 and self.curr_pos[1] > 0: # If can move down
      self.curr_pos = (self.curr_pos[0], self.curr_pos[1] - 1)
    elif action == 3 and self.curr_pos[0] > 0: # If can move left
      self.curr_pos = (self.curr_pos[0] - 1, self.curr_pos[1])

    # Check if reached goal position
    done, reward = False, self.default_reward
    if self.curr_pos in self.enemies: done, reward = True, self.on_failure_reward # Hitting enemy
    elif self.curr_pos == self.goal_pos: done, reward = True, self.on_success_reward # Reaching goal
    elif self.curr_pos == prev_pos: done, reward = False, self.remain_unchanged_reward # Stay in same position

    # Return
    return ((), done, reward, ())

  def hidden_state(self):
    """
    Return a tuple (current_position, goal_position, enemy_positions)
    where enemy positions is a list of enemy positions

    This hidden_state should not be used by model that desires to solve the game
    """
    return (self.curr_pos, self.goal_pos, self.enemies)

  def render(self):
    w, h = int(self.image_w * self.dpi), int(self.image_h * self.dpi)
    # image = numpy.zeros((w, h, 3), dtype=numpy.uint8)
    image = numpy.zeros((h, w, 3), dtype=numpy.uint8)

    # Setup the background
    image[0:h, 0:w] = self.background_image[0:h, 0:w]

    # Draw the current position
    self._paint_spirit(image, self.agent_image, self.curr_pos)

    # Draw the goal position
    self._paint_spirit(image, self.goal_image, self.goal_pos)

    # Draw the enemy position
    for (i, enemy_pos) in enumerate(self.enemies):
      self._paint_spirit(image, self.enemy_images[self.enemy_types[i]], enemy_pos)

    return image

  def render_torch_tensor(self, image=None):
    image = self.render() if image is None else image
    image = numpy.ascontiguousarray(image, dtype=numpy.float32) / 255
    torch_image = torch.tensor(image).permute(2, 0, 1).float()
    return torch.stack([torch_image])

  def _paint_spirit(self, background, spirit, orig_cell_pos):
    cell_pos = (orig_cell_pos[0], self.grid_y - orig_cell_pos[1] - 1)
    cell_w, cell_h = self.cell_pixel_size()
    agent_image = cv2.resize(spirit, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
    agent_offset_x, agent_offset_y = cell_pos[0] * cell_w, cell_pos[1] * cell_h
    agent_end_x, agent_end_y = agent_offset_x + cell_w, agent_offset_y + cell_h
    agent_img_gray = agent_image[:, :, 3]
    _, mask = cv2.threshold(agent_img_gray, 120, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(agent_img_gray)
    source = background[agent_offset_y:agent_end_y, agent_offset_x:agent_end_x]
    bg = cv2.bitwise_or(source, source, mask=mask_inv)
    fg = cv2.bitwise_and(agent_image, agent_image, mask=mask)
    background[agent_offset_y:agent_end_y, agent_offset_x:agent_end_x] = cv2.add(bg, fg[:, :, 0:3])

  def paint_color(self, background, colors, cell_pos):
    size_x, size_y = 10, 10
    cell_pos = (cell_pos[0], self.grid_y - cell_pos[1] - 1)
    cell_w, cell_h = self.cell_pixel_size()
    agent_offset_x, agent_offset_y = cell_pos[0] * cell_w, cell_pos[1] * cell_h
    agent_end_x, agent_end_y = agent_offset_x + size_x, agent_offset_y + size_y
    get_channel = lambda c: numpy.ones((size_y, size_x), dtype=numpy.uint8) * int(255 * colors[c])
    color = numpy.transpose(numpy.stack([get_channel(i) for i in range(3)]), (1, 2, 0))
    background[agent_offset_y:agent_end_y, agent_offset_x:agent_end_x] = color

  def print_state(self):
    print("┌" + ("─" * ((self.grid_x + 2) * 2 - 3)) + "┐")
    for j in range(self.grid_y - 1, -1, -1):
      print("│", end=" ")
      for i in range(self.grid_x):
        print(self.pos_char((i, j)), end=" ")
      print("│")
    print("└" + ("─" * ((self.grid_x + 2) * 2 - 3)) + "┘")

  def pos_char(self, pos):
    if pos == self.curr_pos: return 'C'
    elif pos == self.start_pos: return 'S'
    elif pos == self.goal_pos: return 'G'
    elif pos in self.enemies: return '▒'
    else: return ' '

  def string_of_action(self, action):
    if action == 0: return "up"
    elif action == 1: return "right"
    elif action == 2: return "down"
    elif action == 3: return "left"
    else: raise Exception(f"Unknown action `{action}`")

  def sample_point(self):
    return (random.randint(0, self.grid_x - 1), random.randint(0, self.grid_y - 1))

  def ok_enemy_position(self, pos):
    return self.manhatten_distance(pos, self.start_pos) > 1 and self.manhatten_distance(pos, self.goal_pos) > 1

  def cell_pixel_size(self):
    return (int(self.cell_size * self.dpi), int(self.cell_size * self.dpi))

  def manhatten_distance(self, p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def crop_cell_image(image, grid_dim, cell_pixel_size, orig_cell_pos):
  cell_pos = (orig_cell_pos[0], grid_dim[1] - orig_cell_pos[1] - 1)
  cell_w, cell_h = cell_pixel_size
  agent_offset_x, agent_offset_y = cell_pos[0] * cell_w, cell_pos[1] * cell_h
  agent_end_x, agent_end_y = agent_offset_x + cell_w, agent_offset_y + cell_h
  return image[agent_offset_y:agent_end_y, agent_offset_x:agent_end_x]

def crop_cell_image_torch(image, grid_dim, cell_pixel_size, orig_cell_pos):
  cell_pos = (orig_cell_pos[0], grid_dim[1] - orig_cell_pos[1] - 1)
  cell_w, cell_h = cell_pixel_size
  agent_offset_x, agent_offset_y = cell_pos[0] * cell_w, cell_pos[1] * cell_h
  agent_end_x, agent_end_y = agent_offset_x + cell_w, agent_offset_y + cell_h
  return image[:, agent_offset_y:agent_end_y, agent_offset_x:agent_end_x]
