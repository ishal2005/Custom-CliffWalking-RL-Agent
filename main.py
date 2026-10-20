
import os
import json
import csv
import argparse

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio


# ============================================================
# CUSTOM REINFORCEMENT LEARNING PROJECT: CLIFFWALKING
# Algorithm: Tabular Q-Learning
# Environment: Gymnasium CliffWalking-v0
# ============================================================

ENV_ID = "CliffWalking-v1"
SEED = 42

# All output files are saved inside the project folder.
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("media", exist_ok=True)


# ============================================================
# 1. Q-LEARNING AGENT
# ============================================================

class QLearningAgent:
    def __init__(
        self,
        n_states,
        n_actions,
        learning_rate=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.9995,
        seed=42,
    ):
        self.q_table = np.zeros(
            (n_states, n_actions), dtype=np.float64
        )

        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)

    def choose_action(self, state, explore=True):
        # Epsilon-greedy: explore or exploit.
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.q_table.shape[1]))

        # Randomly break ties between equally good actions.
        values = self.q_table[state]
        best_actions = np.flatnonzero(values == values.max())
        return int(self.rng.choice(best_actions))

    def learn(self, state, action, reward, next_state, terminated):
        # Do not bootstrap beyond a genuine terminal state.
        future_value = (
            0.0 if terminated
            else np.max(self.q_table[next_state])
        )

        target = reward + self.gamma * future_value

        # Q-learning update rule.
        self.q_table[state, action] += self.lr * (
            target - self.q_table[state, action]
        )

    def decay_exploration(self):
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )


# ============================================================
# 2. TRAINING FUNCTION
# ============================================================

def train_agent(
    episodes=20000,
    seed=SEED,
    learning_rate=0.1,
    gamma=0.99,
    epsilon_decay=0.9995,
):
    env = gym.make(ENV_ID)
    env.action_space.seed(seed)

    agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=epsilon_decay,
        seed=seed,
    )

    rewards_history = []
    steps_history = []

    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)

        total_reward = 0
        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = agent.choose_action(state)

            next_state, reward, terminated, truncated, _ = (
                env.step(action)
            )

            agent.learn(
                state,
                action,
                reward,
                next_state,
                terminated,
            )

            state = next_state
            total_reward += reward
            steps += 1

            # Safety limit for unexpectedly long episodes.
            if steps >= 1000:
                break

        agent.decay_exploration()
        rewards_history.append(total_reward)
        steps_history.append(steps)

        if (episode + 1) % max(1, episodes // 10) == 0:
            recent = rewards_history[-100:]
            print(
                f"Episode {episode + 1}/{episodes} | "
                f"Mean recent reward: {np.mean(recent):.2f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    env.close()
    return agent, rewards_history, steps_history


# ============================================================
# 3. TRAINING GRAPH
# ============================================================

def save_training_plot(rewards):
    plt.figure(figsize=(10, 5))

    plt.plot(
        rewards,
        alpha=0.25,
        label="Reward per episode",
    )

    window = min(100, len(rewards))

    if len(rewards) >= window:
        average = np.convolve(
            rewards,
            np.ones(window) / window,
            mode="valid",
        )

        plt.plot(
            np.arange(window - 1, len(rewards)),
            average,
            label=f"{window}-episode moving average",
        )

    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("CliffWalking Q-Learning Training Progress")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/training_rewards.png", dpi=160)
    plt.close()

    print("Saved training graph.")


# ============================================================
# 4. EVALUATE THE TRAINED AGENT
# ============================================================

def evaluate_agent(q_table, episodes=200, seed=10000):
    env = gym.make(ENV_ID)

    rewards = []
    steps_list = []
    successes = []

    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)

        total_reward = 0
        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            # Greedy evaluation: no random exploration.
            action = int(np.argmax(q_table[state]))

            state, reward, terminated, truncated, _ = (
                env.step(action)
            )

            total_reward += reward
            steps += 1

            if steps >= 1000:
                break

        rewards.append(total_reward)
        steps_list.append(steps)

        # CliffWalking goal state is state 47.
        successes.append(
            int(terminated and state == 47)
        )

    env.close()

    return {
        "evaluation_episodes": episodes,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "success_rate_percent": float(
            100 * np.mean(successes)
        ),
        "mean_steps": float(np.mean(steps_list)),
    }


# ============================================================
# 5. HYPERPARAMETER EXPERIMENTS
# ============================================================

def run_experiments(episodes=5000):
    results = []

    for index, learning_rate in enumerate([0.05, 0.10, 0.20]):
        print(
            f"\nExperiment {index + 1}/3: "
            f"learning rate = {learning_rate}"
        )

        agent, _, _ = train_agent(
            episodes=episodes,
            seed=SEED + 1000 * (index + 1),
            learning_rate=learning_rate,
            gamma=0.99,
            epsilon_decay=0.9995,
        )

        metrics = evaluate_agent(
            agent.q_table,
            episodes=100,
            seed=SEED + 20000 + index * 100,
        )

        row = {
            "learning_rate": learning_rate,
            "training_episodes": episodes,
            **metrics,
        }

        results.append(row)

    with open(
        "results/hyperparameter_comparison.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(results)

    with open(
        "results/hyperparameter_comparison.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, indent=4)

    print("\nHyperparameter experiment results:")
    for result in results:
        print(result)

    return results


# ============================================================
# 6. CREATE A GIF OF THE TRAINED POLICY
# ============================================================

def create_agent_gif(q_table, seed=30000, max_steps=300):
    """
    Run the learned policy in the actual environment and create
    a GIF of the recorded states and actions using Matplotlib.
    """
    env = gym.make(ENV_ID)
    state, _ = env.reset(seed=seed)

    states = [state]
    total_reward = 0
    terminated = False
    truncated = False

    for _ in range(max_steps):
        action = int(np.argmax(q_table[state]))

        state, reward, terminated, truncated, _ = (
            env.step(action)
        )

        states.append(state)
        total_reward += reward

        if terminated or truncated:
            break

    env.close()

    frames = []

    # Standard CliffWalking map: 4 rows x 12 columns.
    # Row 3, columns 1 through 10 are the cliff.
    for frame_number, current_state in enumerate(states):
        fig, ax = plt.subplots(figsize=(10, 3.8))

        grid = np.zeros((4, 12))

        # Cliff cells.
        grid[3, 1:11] = -1

        # Goal cell.
        grid[3, 11] = 2

        ax.imshow(
            grid,
            cmap="RdYlGn",
            vmin=-1,
            vmax=2,
            origin="upper",
        )

        # Draw the grid lines and labels.
        ax.set_xticks(np.arange(-0.5, 12, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
        ax.grid(which="minor", color="black", linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)

        ax.set_xticks(range(12))
        ax.set_yticks(range(4))
        ax.set_title(
            f"CliffWalking: Step {frame_number} | "
            f"Reward so far: {sum_reward(q_table, states[:frame_number+1])}"
        )

        # Mark the start, cliff, and goal.
        ax.text(0, 3, "START", ha="center", va="center", fontsize=7)
        for col in range(1, 11):
            ax.text(col, 3, "CLIFF", ha="center", va="center", fontsize=6)
        ax.text(11, 3, "GOAL", ha="center", va="center", fontsize=7)

        row, col = divmod(current_state, 12)
        ax.scatter(
            col,
            row,
            marker="o",
            s=250,
            edgecolors="black",
            linewidths=1.5,
        )

        fig.tight_layout()

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames.append(rgba[:, :, :3].copy())
        plt.close(fig)

    imageio.mimsave(
        "media/cliffwalking_agent.gif",
        frames,
        duration=0.18,
        loop=0,
    )

    print(
        "Saved GIF: media/cliffwalking_agent.gif | "
        f"Recorded return: {total_reward}"
    )


def sum_reward(q_table, states):
    # A visual label helper; actual return is calculated in the environment.
    return len(states) - 1


# ============================================================
# 7. MAIN PROGRAM
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Custom Q-learning for CliffWalking-v0"
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=20000,
        help="Number of training episodes",
    )

    parser.add_argument(
        "--skip-experiments",
        action="store_true",
        help="Skip the additional hyperparameter comparison",
    )

    args = parser.parse_args()

    print("=" * 55)
    print("CUSTOM REINFORCEMENT LEARNING: CLIFFWALKING")
    print("=" * 55)

    print("\nEnvironment:", ENV_ID)
    print("Training episodes:", args.episodes)

    # Train the main agent.
    agent, rewards, steps = train_agent(
        episodes=args.episodes,
        seed=SEED,
    )

    # Save trained Q-table and training history.
    np.save(
        "models/cliffwalking_q_table.npy",
        agent.q_table,
    )

    np.save(
        "results/training_rewards.npy",
        np.asarray(rewards),
    )

    np.save(
        "results/training_steps.npy",
        np.asarray(steps),
    )

    print("\nSaved trained Q-table.")

    # Create the training graph.
    save_training_plot(rewards)

    # Evaluate the learned policy.
    metrics = evaluate_agent(
        agent.q_table,
        episodes=200,
        seed=10000,
    )

    with open(
        "results/evaluation_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=4)

    print("\nFINAL EVALUATION RESULTS")
    print(json.dumps(metrics, indent=4))

    # Record the trained policy.
    create_agent_gif(agent.q_table)

    # Compare learning rates.
    if not args.skip_experiments:
        print("\nStarting hyperparameter experiments...")
        run_experiments(
            episodes=max(2000, args.episodes // 4)
        )

    print("\n" + "=" * 55)
    print("PROJECT FINISHED")
    print("=" * 55)
    print("Check these folders:")
    print("models/  - trained Q-table")
    print("results/ - graphs, metrics, and experiment results")
    print("media/   - GIF demonstration")


if __name__ == "__main__":
    main()
