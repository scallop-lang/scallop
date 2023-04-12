from argparse import ArgumentParser

import random
import torch
from torch import nn
from torch import optim
import cv2
from tqdm import tqdm
from collections import namedtuple, deque

from arena import AvoidingArena

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

class PolicyNet(nn.Module):
  def __init__(self):
    super(PolicyNet, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=8, stride=4, padding=2)
    self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=4, padding=1)
    self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=4, padding=1)
    self.fc1 = nn.Linear(in_features=576, out_features=256)
    self.fc2 = nn.Linear(in_features=256, out_features=4)
    self.relu = nn.ReLU()

  def forward(self, x):
    batch_size, _, _, _ = x.shape
    x = self.relu(self.conv1(x)) # In: (80, 80, 4) Out: (20, 20, 16)
    x = self.relu(self.conv2(x)) # In: (20, 20, 16) Out: (10, 10, 32)
    x = self.relu(self.conv3(x)) # In: (20, 20, 32) Out: (10, 10, 64)
    x = x.view(batch_size, -1) # In: (10, 10, 32) Out: (3200,)
    x = self.relu(self.fc1(x)) # In: (3200,) Out: (256,)
    x = self.fc2(x) # In: (256,) Out: (4,)
    return x


class ReplayMemory:
  def __init__(self, capacity):
    self.memory = deque([], maxlen=capacity)

  def push(self, transition):
    self.memory.append(transition)

  def sample(self, batch_size):
    return random.sample(self.memory, batch_size)

  def __len__(self):
    return len(self.memory)


class DQN:
  def __init__(self):
    self.policy_net = PolicyNet()

    # Create another network
    self.target_net = PolicyNet()
    self.target_net.load_state_dict(self.policy_net.state_dict())
    self.target_net.eval()

    # Store replay memory
    self.memory = ReplayMemory(args.replay_memory_capacity)

    # Loss function and optimizer
    self.criterion = nn.SmoothL1Loss()
    self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=args.learning_rate)

  def predict_action(self, state_image):
    action_scores = self.policy_net(state_image)
    action = torch.argmax(action_scores, dim=1)
    return action

  def observe_transition(self, transition):
    self.memory.push(transition)

  def optimize_model(self):
    if len(self.memory) < args.batch_size: return 0.0

    # Pull out a batch and its relevant features
    batch = self.memory.sample(args.batch_size)
    non_final_mask = torch.tensor([transition.next_state != None for transition in batch], dtype=torch.bool)
    non_final_next_states = torch.stack([transition.next_state[0] for transition in batch if transition.next_state is not None])
    action_batch = torch.stack([transition.action for transition in batch])
    state_batch = torch.stack([transition.state[0] for transition in batch])
    reward_batch = torch.stack([torch.tensor(transition.reward) for transition in batch])

    # Prepare the loss function
    state_action_values = self.policy_net(state_batch).gather(1, action_batch)[:, 0]
    next_state_values = torch.zeros(args.batch_size)
    next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * args.gamma) + reward_batch

    # Compute the loss
    loss = self.criterion(state_action_values, expected_state_action_values)
    self.optimizer.zero_grad()
    loss.backward()
    for param in self.policy_net.parameters():
      param.grad.data.clamp_(-1, 1)
    self.optimizer.step()

    # Return loss
    return loss.detach()

  def update_target(self):
    self.target_net.load_state_dict(self.policy_net.state_dict())


class Trainer:
  def __init__(self, grid_x, grid_y, cell_size, dpi, num_enemies, epsilon):
    self.dqn = DQN()
    self.epsilon = epsilon
    self.arena = AvoidingArena(
      (grid_x, grid_y),
      cell_size,
      dpi,
      num_enemies,
      default_reward=0.0,
      on_success_reward=1.0,
      on_failure_reward=0.0,
      remain_unchanged_reward=0.0,
    )

  def show_image(self, image):
    cv2.imshow("Current Arena", image)
    cv2.waitKey(int(args.show_run_interval * 1000))

  def train_epoch(self, epoch):
    self.epsilon = self.epsilon * args.epsilon_falloff
    success, failure, optimize_count, sum_loss = 0, 0, 0, 0.0
    iterator = tqdm(range(args.num_train_episodes))
    for episode_i in iterator:
      _ = self.arena.reset()
      curr_raw_image = self.arena.render()
      curr_state_image = self.arena.render_torch_tensor(image=curr_raw_image)
      sum_loss = 0.0
      for _ in range(args.num_steps):
        # Render
        if args.show_train_run:
          self.show_image(curr_raw_image)

        # Pick an action
        if random.random() < self.epsilon: action = torch.tensor([random.randint(0, 3)])
        else: action = self.dqn.predict_action(curr_state_image)

        # Step the environment
        _, done, reward, _ = self.arena.step(action[0])

        # Record the transition in memory buffer
        if done:
          next_raw_image = None
          next_state_image = None
        else:
          next_raw_image = self.arena.render()
          next_state_image = self.arena.render_torch_tensor(image=next_raw_image)
        transition = Transition(curr_state_image, action, next_state_image, reward)
        self.dqn.observe_transition(transition)

        # Update the model
        loss = self.dqn.optimize_model()
        sum_loss += loss
        optimize_count += 1

        # Update the next state
        if done:
          if reward > 0: success += 1
          else: failure += 1
          break
        else:
          curr_raw_image = next_raw_image
          curr_state_image = next_state_image

      # Update the target net
      if episode_i % args.target_update == 0:
        self.dqn.update_target()

      # Print information
      success_rate = (success / (episode_i + 1)) * 100.0
      avg_loss = sum_loss / optimize_count
      iterator.set_description(f"[Train Epoch {epoch}] Avg Loss: {avg_loss}, Success: {success}/{episode_i + 1} ({success_rate:.2f}%)")

  def test_epoch(self, epoch):
    success, failure = 0, 0
    iterator = tqdm(range(args.num_test_episodes))
    for episode_i in iterator:
      _ = self.arena.reset()
      raw_image = self.arena.render()
      state_image = self.arena.render_torch_tensor(image=raw_image)
      for _ in range(args.num_steps):
        # Show image
        if args.show_test_run:
          self.show_image(raw_image)

        # Pick an action
        action = self.dqn.predict_action(state_image)
        _, done, reward, _ = self.arena.step(action[0])
        raw_image = self.arena.render()
        state_image = self.arena.render_torch_tensor(image=raw_image)

        # Update the next state
        if done:
          if reward > 0: success += 1
          else: failure += 1
          break

      # Print information
      success_rate = (success / (episode_i + 1)) * 100.0
      iterator.set_description(f"[Test Epoch {epoch}] Success {success}/{episode_i + 1} ({success_rate:.2f}%)")

  def run(self):
    # self.test_epoch(0)
    for i in range(1, args.num_epochs + 1):
      self.train_epoch(i)
      self.test_epoch(i)


if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--grid-x", type=int, default=5)
  parser.add_argument("--grid-y", type=int, default=5)
  parser.add_argument("--cell-size", type=float, default=0.5)
  parser.add_argument("--dpi", type=int, default=80)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--panelize-staying", action="store_true")
  parser.add_argument("--num-enemies", type=int, default=5)
  parser.add_argument("--num-epochs", type=int, default=100)
  parser.add_argument("--num-train-episodes", type=int, default=500)
  parser.add_argument("--num-test-episodes", type=int, default=50)
  parser.add_argument("--num-steps", type=int, default=30)
  parser.add_argument("--learning-rate", type=float, default=0.0001)
  parser.add_argument("--target-update", type=int, default=10)
  parser.add_argument("--epsilon", type=float, default=0.9)
  parser.add_argument("--epsilon-falloff", type=float, default=0.98)
  parser.add_argument("--gamma", type=float, default=0.999)
  parser.add_argument("--replay-memory-capacity", type=int, default=1000)
  parser.add_argument("--seed", type=int, default=1357)
  parser.add_argument("--cuda", action="store_true")
  parser.add_argument("--gpu", type=int, default=0)
  parser.add_argument("--show-run", action="store_true")
  parser.add_argument("--show-train-run", action="store_true")
  parser.add_argument("--show-test-run", action="store_true")
  parser.add_argument("--show-run-interval", type=int, default=0.001)
  args = parser.parse_args()

  # Set parameters
  torch.manual_seed(args.seed)
  random.seed(args.seed)
  if args.show_run:
    args.show_train_run = True
    args.show_test_run = True
  if args.cuda:
    if torch.cuda.is_available(): device = torch.device(f"cuda:{args.gpu}")
    else: raise Exception("No cuda available")
  else: device = torch.device("cpu")

  # Train
  trainer = Trainer(args.grid_x, args.grid_y, args.cell_size, args.dpi, args.num_enemies, args.epsilon)
  trainer.run()
