import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
class TicTacToe:
    def __init__(self):
        self.reset()
    def reset(self):
        self.board = np.zeros(9, dtype=int)
        self.current_player = 1
        return self.board.copy()
    def available_actions(self):
        return [i for i in range(9) if self.board[i] == 0]
    def step(self, action):
        if self.board[action] != 0:
            return self.board.copy(), -10, True 
        self.board[action] = self.current_player
        if self.check_winner(self.current_player):
            return self.board.copy(), 1, True
        if len(self.available_actions()) == 0:
            return self.board.copy(), 0.5, True 
        self.current_player *= -1
        return self.board.copy(), 0, False
    def check_winner(self, player):
        b = self.board.reshape(3,3)
        return any([
            np.all(b[i,:] == player) for i in range(3)
        ]) or any([
            np.all(b[:,j] == player) for j in range(3)
        ]) or np.all(np.diag(b) == player) or np.all(np.diag(np.fliplr(b)) == player)
    def render(self):
        symbols = {1:"X", -1:"O", 0:" "}
        for r in range(3):
            print("|".join(symbols[self.board[3*r+c]] for c in range(3)))
            if r < 2: print("-"*5)
class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(9, 64), nn.ReLU(),
            nn.Linear(64, 9)
        )
    def forward(self, x):
        return self.fc(x)
def train(episodes=500):
    env = TicTacToe()
    model = QNet()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    for ep in range(episodes):
        state = env.reset()
        done = False
        while not done:
            s_tensor = torch.tensor(state, dtype=torch.float32)
            q_values = model(s_tensor)
            available = env.available_actions()
            if random.random() < max(0.1, 1-ep/episodes):
                action = random.choice(available)
            else:
                mask = torch.full((9,), -1e9)
                mask[available] = q_values[available]
                action = torch.argmax(mask).item()
            next_state, reward, done = env.step(action)
            target = q_values.clone().detach()
            target[action] = reward
            loss = loss_fn(q_values, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            state = next_state
    return model
def play(model):
    env = TicTacToe()
    state = env.reset()
    done = False
    env.render()
    while not done:
        s_tensor = torch.tensor(state, dtype=torch.float32)
        q_values = model(s_tensor)
        available = env.available_actions()
        mask = torch.full((9,), -1e9)
        mask[available] = q_values[available]
        action = torch.argmax(mask).item()
        state, _, done = env.step(action)
        print("\nAgent move:")
        env.render()
        if done: break
        human_action = int(input("Your move (0-8): "))
        state, _, done = env.step(human_action)
        print("\nYour move:")
        env.render()
if __name__ == "__main__":
    model = train(episodes=500)
    play(model)