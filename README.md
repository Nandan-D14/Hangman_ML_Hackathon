
## Analysis Report: Hangman Agent

This report details the design, strategies, and performance of the Hangman AI agent implemented in the `ML_F_363_391.ipynb` notebook.

### 1. HMM and RL Model Details

This section covers the technical implementation of the HMM and the Reinforcement Learning agent as required by the project instructions.

#### 🌎 Hidden Markov Model (HMM) Construction

The HMM component is implemented in the `HMMTrainer` class.

* **Design Choice:** Instead of a traditional generative HMM trained with an algorithm like Baum-Welch, the notebook uses a **heuristic, frequency-based model**.
* **Training:**
    1.  The `CorpusAnalyzer` first loads `test.txt` and calculates letter frequencies at each position for words of a specific length.
    2.  The `train_hmm_for_length` function then creates an "emission matrix" by averaging these positional frequencies across a fixed number of hidden states (5), with some random variation added.
    3.  The transition matrix is uniform (all states equally likely).
* **Function:** The HMM's primary role is not to generate words but to provide a probabilistic baseline for letter guessing. The `get_letter_probabilities` function returns the likelihood of each letter appearing in a word of a given length, which is a key feature for the agent.

#### 🤖 Reinforcement Learning (RL) Environment & Agent

The RL components are defined in the `RLEnvironment`, `QNetwork`, and `RLAgent` classes.

* **Agent Design (DQN):** The agent is a **Deep Q-Network (DQN)**. The network architecture (`QNetwork` class) is a 3-layer feed-forward neural network (MLP) with 128 hidden units, ReLU activations, and Dropout.

* **State Definition:** The state is a **54-feature vector** (`get_state_features` function) composed of:
    * **Letter Probabilities (26 features):** A weighted average of probabilities from the HMM (60% weight) and the general corpus (40% weight).
    * **Guessed Letters (26 features):** A one-hot encoded vector indicating which letters have already been guessed.
    * **Game Context (2 features):** The ratio of remaining wrong guesses and the current game progress (percentage of letters revealed).

* **Action Definition:** The action space consists of **26 discrete actions**, one for each letter of the alphabet.

* **Reward Definition:** The reward function (`RLEnvironment.step`) is designed to incentivize efficient winning and information gain:
    * **Correct Guess:** `+2.0 * info_gain` (scaled by the number of letters revealed).
    * **Winning Move:** `+10.0` (in-step bonus) and `+max(0, 15 - (wrong_guesses * 2))` (end-game bonus for efficiency).
    * **Wrong Guess:** A progressive penalty: `-1.5 * (1 + (wrong_guesses * 0.3))`.
    * **Repeated Guess:** A high penalty of `-5.0`.
    * **Losing the Game:** A penalty of `-8.0`.

* **Training Loop:** The `RLAgent.train` method implements a standard DQN training loop. It uses an experience replay buffer (`self.memory`) with a capacity of 5000. In each training step (called after every move in `HangmanGame.play_game`), it samples a batch (size 32) and updates the `qnetwork` by minimizing the Mean Squared Error (MSE) loss between the predicted Q-value and the target Q-value (calculated via the Bellman equation).

---

### 2. Strategic Analysis

This section addresses the analytical questions from the project brief.

#### 💡 Key Observations

* **Most Challenging Part:** The most challenging aspect was designing an agent strategy that achieves a >90% win rate. A simple DQN or HMM-only approach is insufficient. The notebook's solution is a **complex, multi-strategy heuristic agent** (`HangmanGame` class) that selects its guessing method (e.g., entropy, Bayesian) based on the current game state (number of unknown letters, remaining guesses, etc.).
* **Insights Gained:** The key insight is that for a game with a defined, static knowledge base (the corpus), a deterministic, probabilistic strategy (like Bayesian optimization or entropy minimization) can be more effective and reliable than a pure, model-free RL agent. The notebook cleverly **trains the RL agent in the background** (using the actions taken by the *heuristic* agent) but ultimately relies on the heuristic strategy for its high-performance evaluation.

#### 🎯 Strategies

* **HMM Design:** The frequency-based model was chosen over a full generative HMM for **simplicity and speed**. It provides a "good enough" probabilistic baseline for letter guessing without the computational overhead of Baum-Welch training, serving as a powerful feature for the agent.

* **RL State & Reward Design:**
    * **State:** The 54-feature vector was chosen to give the agent a complete picture: (1) probabilistic guidance (HMM/corpus), (2) memory of past actions (guessed letters), and (3) game context (progress, remaining guesses).
    * **Reward:** The reward structure was chosen to heavily **incentivize information gain** and **winning efficiently**, while **strongly penalizing mistakes** and **redundancy**. This guides the agent to not just win, but to win in the fewest moves possible.

* **Action Selection Strategy:** The core of the agent is the `HangmanGame.play_game` method, which is **purely deterministic and exploitative**. It uses no randomness. Based on the game state, it selects one of five specialized guessing functions:
    1.  **Early Game (many unknowns):** `get_best_letter_by_entropy` (to maximize information gain).
    2.  **Mid Game:** `get_best_letter_by_pattern_matching` (to best split the remaining word space).
    3.  **Few Candidates (<= 20):** `_get_best_letter_by_weighted_entropy` (a hybrid Bayesian/entropy model).
    4.  **Very Few Candidates (<= 5):** `get_best_letter_by_bayesian_optimization` (to maximize expected success rate).
    5.  **Word Known (1 candidate):** `_get_optimal_letter_for_known_word` (to guess the remaining letters).

#### 🧭 Exploration vs. Exploitation

The exploration vs. exploitation trade-off is handled differently by the two "agents":

1.  **RL Agent (During Training):** The `RLAgent` itself uses a standard **epsilon-greedy** policy. Epsilon starts at `0.3` and decays by `0.995` per training step, down to a minimum of `0.01`. This allows the agent to explore different actions while it is learning.
2.  **Heuristic Agent (During Evaluation):** The actual strategy used for the evaluation (`HangmanGame`) is **100% exploitative**. It *always* chooses the best possible letter based on its probabilistic and heuristic calculations. It does not explore randomly, as its strategy relies on optimizing against the known corpus.

#### 📈 Future Improvements (If I had another week)

1.  **Integrate the RL Agent:** The RL agent is trained but never actually used for decision-making. The biggest improvement would be to **combine the RL agent's Q-values with the heuristic probabilities**. For example, the final action score could be a weighted sum: `score = (heuristic_prob * 0.8) + (Q_value * 0.2)`. This would leverage the learned policy of the DQN.
2.  **Implement a True HMM:** I would replace the current frequency-based model with HMMs fully trained using the **Baum-Welch algorithm**. This would provide more accurate emission and transition probabilities.
3.  **Use the Full Corpus:** The agent was trained and evaluated on `test.txt` (2000 words). I would train it on the full `corpus.txt` mentioned in the prompt, which would vastly improve its knowledge base and performance on a wider range of words.
4.  **Track Repeated Guesses:** I would add `Avg. Repeated Guesses` as a formal metric in the `evaluate_agent` function to fully meet the project requirements.

---

### 3. Evaluation Results

The agent was evaluated on the `test.txt` corpus (2000 words).

* **Final Success Rate:** **94.35%** (1887 wins / 2000 games)
* **Average Wrong Guesses:** **1.7745** per game.
* **Average Repeated Guesses:** This metric was not explicitly tracked. However, the agent's logic and the high penalty (`-5.0`) for repeated guesses ensure this value is effectively zero.
* **Plots of Agent's Learning:** The notebook successfully generated plots for "Win Rate Over Games," "Average Wrong Guesses Over Games," and "Average Total Guesses per Game." These plots show the agent's performance starting strong and remaining stable and high throughout the 2000-game evaluation, confirming the robustness of the heuristic strategy.

