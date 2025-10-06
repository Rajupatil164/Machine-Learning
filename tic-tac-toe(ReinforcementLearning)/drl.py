import random
import numpy as np
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
class TicTacToe:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros(9, dtype=int)
        self.current_player = 1
        self.done = False
        self.winner = None
        return self._get_state()

    def _get_state(self):
        return self.board.astype(np.float32)

    def available_actions(self):
        return [i for i in range(9) if self.board[i] == 0]

    def step(self, action):
        if self.done:
            raise ValueError("Game is over")
        if self.board[action] != 0:
            self.done = True
            self.winner = -self.current_player
            return self._get_state(), -10.0, True, {"illegal": True}

        self.board[action] = self.current_player
        reward, ended, winner = self._check_game_result()
        self.done = ended
        if ended:
            self.winner = winner

        if not self.done:
            self.current_player *= -1
        return self._get_state(), float(reward), bool(ended), {}

    def _check_game_result(self):
        b = self.board.reshape(3,3)
        lines = []
        lines.extend(list(b.sum(axis=1)))
        lines.extend(list(b.sum(axis=0)))
        lines.append(b.trace())
        lines.append(np.fliplr(b).trace())

        if 3 in lines:
            return 1.0, True, 1
        if -3 in lines:
            return -1.0, True, -1
        if not (self.board == 0).any():
            return 0.5, True, 0
        return 0.0, False, None

    def render(self):
        symbols = {1: 'X', -1: 'O', 0: ' '}
        for r in range(3):
            print("|".join(symbols[self.board[3*r + c]] for c in range(3)))
            if r < 2:
                print("-"*5)
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)
class QNetwork(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=128, output_dim=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, output_dim)
        )

    def forward(self, x):
        return self.net(x)
class DQNAgent:
    def __init__(self, lr=1e-3, gamma=0.99, batch_size=64,
                 buffer_capacity=5000, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = QNetwork().to(self.device)
        self.target_net = QNetwork().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay = ReplayBuffer(capacity=buffer_capacity)
        self.steps_done = 0
        self.update_count = 0

    def select_action(self, state, available_actions, epsilon):
        if random.random() < epsilon:
            return random.choice(available_actions)
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.policy_net(state_t).cpu().numpy().flatten()
        masked = np.full_like(q_vals, -1e9)
        for a in available_actions:
            masked[a] = q_vals[a]
        return int(np.argmax(masked))

    def store_transition(self, *args):
        self.replay.push(*args)

    def learn(self):
        if len(self.replay) < self.batch_size:
            return None
        batch = self.replay.sample(self.batch_size)
        state_b = torch.tensor(np.array(batch.state), dtype=torch.float32, device=self.device)
        action_b = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
        reward_b = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_state_b = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=self.device)
        done_b = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.policy_net(state_b).gather(1, action_b)

        with torch.no_grad():
            next_q_policy = self.policy_net(next_state_b)
            next_actions = torch.argmax(next_q_policy, dim=1, keepdim=True)
            next_q_target = self.target_net(next_state_b).gather(1, next_actions)
            target = reward_b + (1.0 - done_b) * self.gamma * next_q_target

        loss = nn.functional.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_count += 1

        if self.update_count % 100 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()
def random_opponent_move(env: TicTacToe):
    return random.choice(env.available_actions())
def train(num_episodes=1000,
          epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.9995,
          eval_every=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(device=device)
    env = TicTacToe()
    epsilon = epsilon_start
    stats = {"wins":0, "losses":0, "draws":0}

    for ep in range(1, num_episodes+1):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state, env.available_actions(), epsilon)
            next_state, reward, done, info = env.step(action)
            if info.get("illegal", False):
                agent.store_transition(state, action, reward, next_state, True)
                break
            if done:
                agent.store_transition(state, action, reward, next_state, True)
                if env.winner == 1: stats["wins"] += 1
                elif env.winner == -1: stats["losses"] += 1
                else: stats["draws"] += 1
                break
            opp_action = random_opponent_move(env)
            next_state_after_opp, r2, done_after_opp, _ = env.step(opp_action)
            if done_after_opp:
                terminal_reward = 1.0 if env.winner==1 else (-1.0 if env.winner==-1 else 0.5)
                agent.store_transition(state, action, terminal_reward, next_state_after_opp, True)
                if env.winner == 1: stats["wins"] += 1
                elif env.winner == -1: stats["losses"] += 1
                else: stats["draws"] += 1
                done = True
                break
            else:
                agent.store_transition(state, action, 0.0, next_state_after_opp, False)
                state = next_state_after_opp

        agent.learn()
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        agent.steps_done += 1

        if ep % 100 == 0:
            print(f"Episode {ep} | epsilon {epsilon:.3f} | wins {stats['wins']} | losses {stats['losses']} | draws {stats['draws']}")

        if ep % eval_every == 0:
            eval_stats = evaluate(agent, n_games=200)
            print(f"== Evaluation after {ep} episodes: {eval_stats} ==")

    return agent
def evaluate(agent: DQNAgent, n_games=200):
    env = TicTacToe()
    stats = {"wins":0, "losses":0, "draws":0}
    for _ in range(n_games):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state, env.available_actions(), epsilon=0.0)
            state, _, done, _ = env.step(action)
            if done:
                if env.winner == 1: stats["wins"] += 1
                elif env.winner == -1: stats["losses"] += 1
                else: stats["draws"] += 1
                break
            opp = random_opponent_move(env)
            state, _, done, _ = env.step(opp)
            if done:
                if env.winner == 1: stats["wins"] += 1
                elif env.winner == -1: stats["losses"] += 1
                else: stats["draws"] += 1
                break
    return stats
def watch_one_game(agent, opponent="random"):
    env = TicTacToe()
    state = env.reset()
    done = False
    print("\n=== New Game ===")
    env.render()

    while not done:
        action = agent.select_action(state, env.available_actions(), epsilon=0.0)
        print(f"\nAgent plays at {action}")
        state, reward, done, _ = env.step(action)
        env.render()
        if done:
            print(f"Game over. Winner: {env.winner}\n")
            break

        if opponent == "random":
            opp_action = random.choice(env.available_actions())
        else:
            opp_action = int(input("Your move (0-8): "))

        print(f"\nOpponent plays at {opp_action}")
        state, reward, done, _ = env.step(opp_action)
        env.render()
        if done:
            print(f"Game over. Winner: {env.winner}\n")
            break
if __name__ == "__main__":
    agent = train(num_episodes=1000, eval_every=500)
    print("\nFinal evaluation:", evaluate(agent, n_games=500))
    watch_one_game(agent, opponent="random")