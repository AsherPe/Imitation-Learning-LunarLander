# Imitation Learning on LunarLander-v3

This project explores Imitation Learning in the LunarLander-v3 environment using Behavioral Cloning.

Instead of learning only through trial and error, the agent learns to imitate an expert policy by training on state-action demonstrations generated from a PPO expert agent.

## Project Overview

The project follows a complete imitation learning pipeline:

1. Analyze a trained PPO expert agent
2. Collect expert demonstrations
3. Train Behavioral Cloning models
4. Evaluate imitation policies
5. Analyze how dataset size affects performance

The main goal is to understand how many expert demonstrations are needed for an imitation policy to approach expert-level performance.

## Environment

The project uses the LunarLander-v3 environment from Gymnasium.

The agent observes an 8-dimensional state vector that includes:

- Lander position
- Linear velocity
- Angle
- Angular velocity
- Left and right leg contact sensors

The action space contains 4 discrete actions:

- Do nothing
- Fire left engine
- Fire main engine
- Fire right engine

An episode is considered successful when the total reward is at least 200.

## Methodology

### Stage A: PPO Expert Agent

A PPO expert agent was provided as part of the assignment.

The expert was trained using:

| Parameter | Value |
|---|---|
| Algorithm | PPO |
| Policy | MLP |
| Parallel Environments | 8 |
| Learning Rate | 3e-4 |
| Discount Factor | 0.99 |
| Total Timesteps | 1.2M |

The expert achieved an average reward of approximately 269-271 over 50 evaluation episodes, which is above the success threshold.

### Stage B: Demonstration Collection

Expert demonstrations were collected by running the expert policy and recording state-action pairs at every environment step.

Datasets of different sizes were created:

| Expert Episodes | Environment Steps |
|---|---|
| 3 | 726 |
| 5 | 1,262 |
| 10 | 3,209 |
| 50 | 13,350 |
| 100 | 26,860 |
| 250 | 68,607 |
| 500 | 136,934 |

### Stage C: Behavioral Cloning

Behavioral Cloning was used to train a neural network policy using supervised learning.

The model learns the mapping:

```text
state -> action
```

Model architecture:

```text
Input: 8-dimensional state vector
Dense(64, ReLU)
Dense(64, ReLU)
Dense(4, Softmax)
```

Training setup:

| Parameter | Value |
|---|---|
| Loss Function | Categorical Crossentropy |
| Epochs | 30 |
| Batch Size | 64 |
| Train/Validation Split | 80/20 |
| Output | 4-way action classification |

## Results

The imitation policy was evaluated using:

- Average episode reward
- Success rate
- Comparison against the PPO expert

### Expert vs Imitation Performance

| Dataset Size | Success Rate | Average Reward |
|---|---|---|
| 3 episodes | 0% | -182 |
| 5 episodes | 75% | ~189 |
| 10 episodes | 94% | ~258 |
| 50+ episodes | 96-98% | ~264-267 |

## Key Insight

The results show that a relatively small number of high-quality demonstrations can be enough to reach near-expert performance.

The largest improvement happened between 3 and 10 expert episodes. After around 50 episodes, performance started to saturate, meaning that adding more data gave only small improvements.

This shows that both data quality and state coverage are important in imitation learning.

## Few-Shot Imitation Learning

The project also tested few-shot imitation learning using very small demonstration datasets.

The 3-episode model failed completely, while the 5-episode model achieved partial success. This shows a common limitation of Behavioral Cloning: when the dataset is too small, the model may not see enough states to recover from mistakes during inference.

## Main Takeaways

- Behavioral Cloning can successfully imitate a strong PPO expert.
- Very small datasets are not reliable.
- Around 10-50 expert episodes were enough to approach expert-level performance.
- More data improves stability, but after a certain point the benefit becomes smaller.
- Imitation Learning can simplify reinforcement learning problems by turning them into supervised learning tasks.
- A key limitation is distribution shift: during inference, the agent may enter states that were not well represented in the expert demonstrations.

## Repository Structure

```text
Imitation-Learning-LunarLander/
│
├── README.md
├── .gitignore
├── Assignment Report.pdf
│
├── notebooks/
│   ├── RL_Trained_Agent.ipynb
│   ├── Lmitation Learning.ipynb
│   └── Inference.ipynb
│
├── data/
│   └── demonstration datasets
│
└── videos/
    └── inference videos
```

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/AsherPe/Imitation-Learning-LunarLander.git
```

2. Open the notebooks in Google Colab or Jupyter Notebook.

3. Run the notebooks in this order:

```text
1. RL_Trained_Agent.ipynb
2. Lmitation Learning.ipynb
3. Inference.ipynb
```

## Technologies Used

- Python
- TensorFlow / Keras
- Gymnasium
- Stable-Baselines3
- NumPy
- Matplotlib
- Reinforcement Learning
- Imitation Learning
- Behavioral Cloning

## Future Improvements

Possible future improvements include:

- Implementing DAgger to reduce distribution shift
- Comparing Behavioral Cloning with DQN or PPO
- Adding more robust evaluation under unseen initial states
- Improving the expert demonstration selection process
- Adding more visual analysis of failed landing cases

## Authors

- Asher Pe'er
- Angel Sasin
- Shahar Solema

## Academic Context

This project was submitted as part of:

```text
Advanced Topics in Deep Learning
The College of Management Academic Studies
2026
```

## License

This project was developed for educational and research purposes.
