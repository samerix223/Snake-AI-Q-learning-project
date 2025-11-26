# Snake AI – Q-learning in Pygame

Simple Snake game written in Python with **Pygame**, where the snake learns to play using **Q-learning** (reinforcement learning).  
Designed to run in **MU Editor**, but works in any Python 3 environment.

## Features

- Classic Snake on a grid (Pygame).
- AI agent using Q-learning:
  - state: simple vision (danger ahead/left/right, food direction, snake direction),
  - actions: turn left / go straight / turn right,
  - rewards for eating food, penalties for dying and moving away from food.
- Training mode and play mode:
  - `train` – fast training over many episodes,
  - `play` – watch the trained AI.
- Q-table is saved/loaded from `qtable_snake.pkl`
