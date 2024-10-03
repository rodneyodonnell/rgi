# Design Document: Generalized MuZero-like Algorithm for Game Playing AI

## 0. Introduction

### 0.1. Author and Context

This design document was created by an AI assistant to assist in developing a generalized MuZero-like algorithm for the RGI game-playing AI project. The document is based on instructions provided by the user, who intends to use this document to delegate tasks to less powerful models or junior developers.

Instructions Summary:

- Develop a generalized MuZero-like algorithm capable of learning and playing multiple games.
- Break down the implementation plan into detailed, manageable tasks suitable for delegation.
- Include code snippets and examples, especially in Section 7, to clarify the tasks.
- Use markdown format for the document to facilitate easy saving and sharing.
- Include a "Section 0" summarizing the author and instructions for future updates.
- Provide examples of defining a Player class and demonstrate how action and state embeddings interact, including the dot product and softmax (two-tower model approach).

## 1. Introduction

This document outlines the design of a generalized MuZero-like algorithm, referred to as Generalized MuZero, for developing a game-playing AI capable of learning and performing well across multiple games. The approach focuses on embedding states and actions into a shared space and integrating these embeddings with Monte Carlo Tree Search (MCTS) for reinforcement learning.

## 2. Goals and Objectives

### 2.1. Primary Goals

- Develop a generalized model that can learn to play multiple games using a shared architecture.
- Implement state and action embedding models that can handle arbitrary JSON representations.
- Integrate the learned models with MCTS to enhance decision-making.
- Validate the approach using Connect4 before generalizing to other games.
- Explore transfer learning to leverage knowledge across different games.

### 2.2. Secondary Goals

- Experiment with different model architectures (e.g., CNNs, Transformers).
- Optimize embedding dimensions and other hyperparameters.
- Implement efficient training techniques inspired by "EfficientZero" and "MuZero Unplugged."

## 3. Algorithm Overview

Generalized MuZero combines model-based reinforcement learning with MCTS. The algorithm learns:

- State Embedding Model: Converts game states (JSON blobs) into fixed-dimensional embeddings.
- Action Embedding Model: Converts actions into embeddings within the same space.
- Dynamics Model: Predicts the next state embedding given the current state embedding and an action embedding.
- Reward Model: Predicts the scalar reward for a given state embedding.
- Policy Model: Suggests the next action embedding based on the current state embedding.

These components enable planning through MCTS by simulating future states and rewards.

## 4. Model Architecture

### 4.1. State and Action Embeddings

- Shared Embedding Space: Both states and actions are embedded into a shared or closely related space to facilitate interaction.
- Embedding Models:
  - Initial Prototype: Use CNNs for games like Connect4 where states can be represented as grids.
  - Generalized Model: Transition to Transformers to handle arbitrary JSON representations for states and actions.
- Embedding Dimension: Configurable parameter (e.g., 64 dimensions) to be optimized through experimentation.

### 4.2. Dynamics and Reward Models

- Dynamics Model: Predicts the next state embedding given the current state and action embeddings.
- Reward Model: Outputs a scalar reward prediction for the current state embedding.
- Model Architecture: May consist of feedforward layers or more complex architectures depending on experimentation.

### 4.3. Policy Model

- Predicts the probabilities of legal actions given the current state embedding.
- Helps guide the MCTS by providing prior probabilities for actions.

## 5. Training Process

### 5.1. Data Generation

- Self-Play: Generate training data by having the AI play games against itself using a mostly random policy initially.
- Dataset Composition: Each training sample includes:
  - State embedding
  - Action taken
  - Resulting state embedding
  - Reward received
- Experience Replay: Store gameplay experiences for training and updating the models.

### 5.2. Training Methodology

- Loss Functions:
  - Dynamics Loss: Difference between predicted and actual next state embeddings.
  - Reward Loss: Mean squared error between predicted and actual rewards.
  - Policy Loss: Cross-entropy loss between predicted and actual actions taken.
- Optimization: Use techniques like stochastic gradient descent or Adam optimizer.
- Hyperparameter Tuning: Adjust embedding dimensions, learning rates, and other parameters.

## 6. Integration with MCTS

- MCTS Overview: Uses simulations to evaluate the potential outcomes of actions.
- Integration Strategy:
  - Use the policy model to provide prior probabilities for action selection in the tree.
  - Use the dynamics and reward models to simulate future states and rewards without actual gameplay.
- Efficient Implementation: Incorporate techniques from "EfficientZero" to reduce computational requirements.

## 7. Implementation Plan

### 7.1. Phase 1: Prototype with Connect4

#### Overview

This phase focuses on developing a working prototype of the MuZero-like algorithm using Connect4. The tasks are broken down into manageable units suitable for implementation by junior developers or less sophisticated AI agents. Each task includes detailed descriptions, code entry points, code snippets, and acceptance criteria.

#### 7.1.1. Implement State and Action Embedding Models

##### Task 1: Implement Connect4StateEmbedder

Description: Develop a StateEmbedder for Connect4 that converts game states into fixed-dimensional embeddings (e.g., 64 floats).

Code Entry Points:

File: `rgi/games/connect4/connect4_embeddings.py`

Subtasks:

a. Implement the Connect4StateEmbedder class using composition:

```python
import jax.numpy as jnp
from rgi.core.base import StateEmbedder
from typing import Any

class Connect4StateEmbedder(StateEmbedder[Connect4State, jax.Array]):
    def __init__(self, cnn_model: Any):
        self.cnn_model = cnn_model

    def embed_state(self, params: dict, state: Connect4State) -> jax.Array:
        board_tensor = self._state_to_tensor(state)
        embedding = self.cnn_model.apply(params['cnn_model'], board_tensor)
        return embedding

    def _state_to_tensor(self, state: Connect4State) -> jax.Array:
        board_array = jnp.zeros((6, 7), dtype=jnp.float32)
        for (row, col), value in state.board.items():
            board_array = board_array.at[row - 1, col - 1].set(1.0 if value == 1 else -1.0)
        board_array = board_array[..., jnp.newaxis]  # Add a channel dimension
        return board_array

    def get_embedding_dim(self) -> int:
        return self.cnn_model.embedding_dim
```

b. Define the CNN model separately:

```python
from flax import linen as nn

class Connect4CNN(nn.Module):
    embedding_dim: int = 64

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(4, 4))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.Flatten()(x)
        x = nn.Dense(features=self.embedding_dim)(x)
        return x
```

Testing Criteria:

- Write unit tests for embed_state with various board states.
- Verify that the output embedding is a vector of the correct dimension.

Acceptance Criteria:

- Connect4StateEmbedder accurately converts game states to embeddings.
- All unit tests pass.
- Code follows the Python Style Guide.

##### Task 2: Implement Connect4ActionEmbedder

Description: Develop an ActionEmbedder for Connect4 that converts actions into learned, fixed-dimensional embeddings.

Code Entry Points:

File: `rgi/games/connect4/connect4_embeddings.py`

Subtasks:

a. Implement the Connect4ActionEmbedder class as a Flax Module:

```python
from flax import linen as nn

class Connect4ActionEmbedder(nn.Module):
    embedding_dim: int = 64

    @nn.compact
    def __call__(self, action: int):
        action_embedding = nn.Embed(num_embeddings=7, features=self.embedding_dim)(action - 1)
        return action_embedding
```

Testing Criteria:

- Write unit tests for embed_action covering all valid actions.
- Verify the output embedding is a vector of the correct dimension.

Acceptance Criteria:

- Connect4ActionEmbedder correctly converts actions to embeddings.
- All unit tests pass.
- Code adheres to the Python Style Guide.

#### 7.1.2. Develop the MuZero Model Components

##### Task 3: Implement the Dynamics, Reward, and Policy Models

Description: Develop the neural network models for predicting the next state embedding (dynamics), the scalar reward, and the policy for action selection.

Code Entry Points:

File: `rgi/players/muzero_player/muzero_model.py`

Subtasks:

a. Define the MuZeroModel class:

```python
class MuZeroModel(nn.Module):
    state_embedder: Any  # Instance of Connect4StateEmbedder
    action_embedder: Any  # Instance of Connect4ActionEmbedder
    embedding_dim: int = 64

    @nn.compact
    def __call__(self, state: Connect4State, action: Optional[TAction]):
        # Get state embedding
        state_embedding = self.state_embedder.embed_state(state)
        
        if action is not None:
            # Get action embedding
            action_embedding = self.action_embedder(action)
            # Dynamics model
            x = jnp.concatenate([state_embedding, action_embedding], axis=-1)
            x = nn.Dense(128)(x)
            x = nn.relu(x)
            next_state_embedding = nn.Dense(self.embedding_dim)(x)
        else:
            next_state_embedding = None

        # Reward model
        r = nn.Dense(64)(state_embedding)
        r = nn.relu(r)
        reward = nn.Dense(1)(r).squeeze()

        # Policy model (Two-Tower approach)
        # Compute dot product between state embedding and all action embeddings
        all_actions = range(1, 8)  # Actions are columns 1-7
        all_action_embeddings = jnp.stack([self.action_embedder(a) for a in all_actions])
        logits = jnp.dot(all_action_embeddings, state_embedding)

        return next_state_embedding, reward, logits
```

Testing Criteria:

- Write unit tests for the model with dummy data.
- Verify that outputs are of correct shapes: next state embedding, scalar reward, action logits.

Acceptance Criteria:

- MuZeroModel is implemented and tested.
- Code conforms to project conventions.

##### Task 4: Implement the MuZeroPlayer Class

Description: Create a Player class that uses the MuZeroModel and MCTS to select actions during gameplay.

Code Entry Points:

File: `rgi/players/muzero_player/muzero_player.py`

Subtasks:

a. Define the MuZeroPlayer class:

```python
class MuZeroPlayer(Player[Connect4State, None, TAction]):
    def __init__(self, muzero_model: MuZeroModel, params: dict):
        self.muzero_model = muzero_model
        self.params = params
        # Additional initialization as needed

    def select_action(self, game_state: Connect4State, legal_actions: list[TAction]) -> TAction:
        # Get policy logits from the model
        _, _, logits = self.muzero_model.apply(self.params, game_state, action=None)
        # Mask illegal actions
        mask = jnp.array([1.0 if a in legal_actions else 0.0 for a in range(1, 8)])
        masked_logits = logits * mask - 1e9 * (1 - mask)
        # Compute probabilities
        action_probs = nn.softmax(masked_logits)
        # Sample or select the action with the highest probability
        selected_action = int(jnp.argmax(action_probs)) + 1  # Actions are 1-indexed
        return selected_action

    def update_state(self, game_state: Connect4State, action: TAction) -> None:
        # Update any internal states if necessary
        pass
```

b. Explain the Action Selection Process:

The MuZeroPlayer uses the two-tower model approach:
- Computes the dot product between the state embedding and all possible action embeddings.
- Applies a softmax function to obtain action probabilities.
- Selects the action with the highest probability or samples according to the probabilities.

Testing Criteria:

- Test the player in simulated games against baseline players.
- Verify that the player selects legal actions and behaves appropriately.

Acceptance Criteria:

- MuZeroPlayer is functional and integrates with the game framework.
- All tests pass.

#### 7.1.3. Collect Training Data via Self-Play

##### Task 5: Implement Data Collection with Self-Play

Description: Generate training data by having MuZeroPlayer play against itself or a RandomPlayer.

Code Entry Points:

File: `scripts/collect_muzero_data.py`

Subtasks:

a. Simulate games and record sequences of (state, action, reward, next_state) tuples.
b. Store the data in a format suitable for training (e.g., TFRecord, JSONL).

Testing Criteria:

- Ensure that data collection runs without errors.
- Verify that the collected data is valid and correctly formatted.

Acceptance Criteria:

- Training dataset is generated and ready for use.
- Data quality is confirmed.

#### 7.1.4. Train the Models

##### Task 6: Implement the Training Pipeline

Description: Set up the training loop to train the MuZeroModel using the collected data.

Subtasks:

a. Define loss functions:

- Dynamics Loss: MSE between predicted and target next state embeddings.
- Reward Loss: MSE between predicted and target rewards.
- Policy Loss: Cross-entropy loss between predicted logits and actual actions.

b. Implement the training loop using JAX and Optax.

```python
def loss_fn(params, batch):
    next_state_pred, reward_pred, logits = muzero_model.apply(params, batch['state'], batch['action'])
    dynamics_loss = jnp.mean((next_state_pred - batch['next_state_embedding']) ** 2)
    reward_loss = jnp.mean((reward_pred - batch['reward']) ** 2)
    policy_loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch['action'] - 1)
    total_loss = dynamics_loss + reward_loss + policy_loss
    return total_loss
```

c. Use an optimizer (e.g., Adam) and update parameters.

Testing Criteria:

- Run the training loop and monitor loss values.
- Ensure that gradients are computed correctly.

Acceptance Criteria:

- Models are trained without errors.
- Loss decreases over time, indicating learning.

#### 7.1.5. Evaluate the MuZeroPlayer Performance

##### Task 7: Conduct Performance Evaluation

Description: Assess the performance of MuZeroPlayer by playing it against baseline players.

Subtasks:

a. Set up matches between MuZeroPlayer and RandomPlayer or MinimaxPlayer.
b. Record metrics such as win rate, average game length.

Testing Criteria:

- Ensure that a sufficient number of games are played for statistical significance.
- Verify that the data collected is accurate.

Acceptance Criteria:

- MuZeroPlayer demonstrates improved performance over baseline players.
- Results are documented and analyzed.

## 7.2. Phase 2: Generalization to Multiple Games

In this phase, we aim to generalize the MuZero-like algorithm to handle multiple games using a shared architecture. This involves transitioning to models capable of processing arbitrary game states and actions represented as JSON blobs.

### 7.2.1. Transition to Transformer-Based Models

#### Task 8: Implement Transformer-Based State Embedder

Description: Develop a StateEmbedder that uses a Transformer model to embed game states represented as JSON blobs into fixed-dimensional embeddings.

Code Entry Points:

File: `rgi/embedders/transformer_state_embedder.py`

Subtasks:

a. Design a method to serialize game states (JSON) into sequences of tokens.
   - Tokenize JSON keys and values appropriately.
   - Include game identifiers to help the model distinguish between different games.

b. Implement the Transformer model using Flax:
   - Use positional encoding to handle sequence data.
   - Configure the model with appropriate hyperparameters (e.g., number of layers, heads).

c. Create the TransformerStateEmbedder class:

```python
class TransformerStateEmbedder(StateEmbedder[Any, jax.Array]):
    def __init__(self, transformer_model: nn.Module):
        self.transformer_model = transformer_model

    def embed_state(self, params: dict, state: Any) -> jax.Array:
        tokenized_state = self._tokenize_state(state)
        embedding = self.transformer_model.apply(params['transformer_model'], tokenized_state)
        return embedding

    def _tokenize_state(self, state: Any) -> jax.Array:
        # Implement tokenization logic
        pass

    def get_embedding_dim(self) -> int:
        return self.transformer_model.embedding_dim
```

Testing Criteria:
- Verify that the tokenizer correctly processes states from different games.
- Test the embedding output for consistency and correctness.

Acceptance Criteria:
- TransformerStateEmbedder can process and embed states from multiple games.
- All unit tests pass.
- Code adheres to the Python Style Guide.

#### Task 9: Implement Transformer-Based Action Embedder

Description: Develop an ActionEmbedder that uses a Transformer model to embed actions represented as JSON blobs.

Code Entry Points:

File: `rgi/embedders/transformer_action_embedder.py`

Subtasks:

a. Design a method to serialize actions into token sequences.
   - Tokenize action fields consistently with state tokenization.

b. Implement the Transformer model or adapt the state embedder to handle actions.

c. Create the TransformerActionEmbedder class:

```python
class TransformerActionEmbedder(ActionEmbedder[Any, jax.Array]):
    def __init__(self, transformer_model: nn.Module):
        self.transformer_model = transformer_model

    def embed_action(self, params: dict, action: Any) -> jax.Array:
        tokenized_action = self._tokenize_action(action)
        embedding = self.transformer_model.apply(params['transformer_model'], tokenized_action)
        return embedding

    def _tokenize_action(self, action: Any) -> jax.Array:
        # Implement tokenization logic
        pass

    def get_embedding_dim(self) -> int:
        return self.transformer_model.embedding_dim
```

Testing Criteria:
- Test action embedding with various actions from different games.
- Ensure embeddings are of correct dimensions.

Acceptance Criteria:
- TransformerActionEmbedder functions correctly across multiple games.
- All tests pass.

### 7.2.2. Expand to Additional Games

#### Task 10: Integrate Additional Games

Description: Add support for new games to test the generalization capabilities of the model.

Code Entry Points:

Files under `rgi/games/` (e.g., `rgi/games/tictactoe.py`, `rgi/games/infiltr8.py`)

Subtasks:

a. Implement the Game classes for new games, following the existing Game interface.
b. Develop serializers and deserializers for game states and actions.
c. Update the game registry to include the new games.

Testing Criteria:
- Verify that the new games can be played using the existing framework.
- Ensure that serialization and deserialization work correctly.

Acceptance Criteria:
- New games are fully integrated and operational.
- All tests pass.

#### Task 11: Collect Multi-Game Training Data

Description: Generate training data for all supported games using self-play.

Subtasks:

a. Simulate self-play games for each game.
b. Record training data in a unified format.
c. Label data with game identifiers.

Testing Criteria:
- Ensure data integrity and correctness across all games.
- Verify that the dataset is balanced.

Acceptance Criteria:
- Multi-game dataset is ready for training.
- Data quality is confirmed.

#### Task 12: Retrain Models on Combined Dataset

Description: Train the embedding models and MuZeroModel on the combined dataset from multiple games.

Subtasks:

a. Update training scripts to handle data from multiple games.
b. Adjust models to incorporate game identifiers if necessary.
c. Monitor training for signs of overfitting or underfitting.

Testing Criteria:
- Evaluate model performance on validation sets for each game.
- Check for consistent learning across games.

Acceptance Criteria:
- Models generalize well across multiple games.
- Training is stable and efficient.

#### Task 13: Evaluate Generalization and Transfer Learning

Description: Assess how well the models transfer knowledge between games.

Subtasks:

a. Measure performance improvements due to transfer learning.
b. Analyze any challenges or limitations observed.

Testing Criteria:
- Compare models trained on single games versus multiple games.
- Use metrics such as win rates and learning speed.

Acceptance Criteria:
- Insights into transfer learning effects are documented.
- Recommendations for further improvements are provided.

## 7.3. Phase 3: Optimization and Refinement

In this phase, we focus on optimizing the models and training processes, incorporating advanced techniques to improve performance and efficiency.

### 7.3.1. Hyperparameter Optimization

#### Task 14: Set Up Hyperparameter Tuning Framework

Description: Implement tools to automate the tuning of model hyperparameters.

Subtasks:

a. Choose a hyperparameter optimization library (e.g., Optuna, Ray Tune).
b. Define the hyperparameters to tune (e.g., embedding dimensions, learning rates, model depths).
c. Integrate the tuning process with the existing training pipeline.

Testing Criteria:
- Run test optimization trials to ensure the framework functions correctly.
- Verify that hyperparameters are being adjusted and recorded.

Acceptance Criteria:
- Hyperparameter tuning framework is operational and integrated.

#### Task 15: Conduct Hyperparameter Experiments

Description: Perform experiments to find optimal hyperparameter settings for the models.

Subtasks:

a. Schedule and execute hyperparameter tuning sessions.
b. Collect and analyze results.
c. Update models with optimal hyperparameters.

Testing Criteria:
- Confirm that experiments are reproducible.
- Ensure that performance improvements are statistically significant.

Acceptance Criteria:
- Optimal hyperparameters are identified and documented.
- Models show improved performance with new settings.

### 7.3.2. Implement Efficient Training Techniques

#### Task 16: Incorporate Techniques from "EfficientZero"

Description: Apply methods from the "EfficientZero" paper to improve training efficiency and performance.

Subtasks:

a. Research and understand the key techniques proposed in "EfficientZero."
b. Adapt these techniques to the current models and training pipeline.
   - Techniques may include prioritized experience replay, value normalization, or bootstrapping strategies.
c. Implement necessary code changes and optimizations.

Testing Criteria:
- Measure training speed and resource utilization before and after changes.
- Evaluate any impact on model performance.

Acceptance Criteria:
- Training becomes more efficient without degrading model performance.
- Documentation is updated to reflect new methods.

#### Task 17: Enhance MCTS Integration

Description: Optimize the MCTS algorithm to work more efficiently with the learned models.

Subtasks:

a. Implement enhancements such as virtual loss, policy pruning, or parallel simulations.
b. Fine-tune MCTS hyperparameters (e.g., exploration constants).
c. Test the impact of changes on gameplay performance.

Testing Criteria:
- Verify that MCTS enhancements lead to better decision-making.
- Ensure that computational overhead is acceptable.

Acceptance Criteria:
- MCTS is optimized for performance and efficiency.
- Improvements are validated through testing.

### 7.3.3. Finalize and Document the Models

#### Task 18: Update Documentation and Codebase

Description: Ensure that all code and documentation are up-to-date and comprehensive.

Subtasks:

a. Review code for consistency with style guides and best practices.
b. Update README files, docstrings, and usage instructions.
c. Document any known issues, limitations, or future work.

Testing Criteria:
- Perform code reviews and documentation audits.
- Confirm that all components are properly documented.

Acceptance Criteria:
- Codebase is clean, well-documented, and ready for deployment or further development.
- Documentation is sufficient for new contributors to get started.

## 7.4. Phase 4: Deployment and Evaluation (Optional)

If applicable, this phase involves deploying the models in a real-world or user-facing environment and conducting comprehensive evaluations.

#### Task 19: Deploy Models to Production Environment

Description: Integrate the trained models into the frontend applications for users to interact with.

Subtasks:

a. Prepare the models for deployment (e.g., export, optimization).
b. Update backend APIs to use the models.
c. Ensure compatibility with frontend interfaces.

Testing Criteria:
- Conduct end-to-end testing of the application.
- Verify model responses and user experience.

Acceptance Criteria:
- Models are successfully deployed and functional in the production environment.
- User interactions are smooth and error-free.

#### Task 20: Collect User Feedback and Metrics

Description: Gather data on model performance in the real world and user satisfaction.

Subtasks:

a. Implement logging and monitoring tools to collect performance metrics.
b. Solicit user feedback through surveys or analytics.
c. Analyze data to identify areas for improvement.

Testing Criteria:
- Ensure that data collection complies with privacy policies.
- Verify that metrics are accurately recorded.

Acceptance Criteria:
- Insights are gathered to inform future iterations.
- User satisfaction meets predefined benchmarks.

## 8. Supporting Infrastructure

### 8.1. Logging and Monitoring

- Implement logging mechanisms to track training progress, model performance, and system metrics.
- Use visualization tools (e.g., TensorBoard) to monitor metrics like loss curves, accuracy, and resource utilization.
- Set up alerts for any anomalies or issues detected during training or deployment.

### 8.2. ELO Rating System

- Develop an ELO rating system to evaluate and compare different model versions.
- Automate matches between models to update ratings and track progress over time.
- Visualize ELO ratings to provide insights into model improvements.

### 8.3. Version Control and Continuous Integration

- Use version control systems (e.g., Git) to manage code changes and collaborate effectively.
- Set up continuous integration pipelines to automate testing, linting, and building processes.

## 9. Project Timeline and Milestones

### 9.1. Phase 1 Milestones

Week 1:
- Complete Tasks 1-4 (Implement embeddings and MuZeroModel for Connect4).

Week 2:
- Complete Tasks 5-7 (Collect data, train models, and implement MuZeroPlayer).

Week 3:
- Evaluate MuZeroPlayer performance and refine models.

### 9.2. Phase 2 Milestones

Week 4-5:
- Complete Tasks 8-11 (Implement transformer-based embedders and integrate additional games).

Week 6:
- Retrain models on multi-game data and evaluate generalization.

### 9.3. Phase 3 Milestones

Week 7-8:
- Complete Tasks 14-17 (Hyperparameter optimization and efficiency improvements).

Week 9:
- Finalize documentation and prepare for potential deployment.

### 9.4. Adjustments

- Flexibility: Adjust timelines based on progress, challenges encountered, and resource availability.
- Parallelization: Tasks may be parallelized where possible to expedite the development process.

## 10. Future Work

- Explore Advanced Techniques: Investigate methods like curriculum learning, meta-learning, or unsupervised pre-training to enhance model capabilities.
- Real-World Applications: Extend the framework to real-world tasks beyond games, such as robotics, logistics, or decision-making systems.
- Community Engagement: Open-source the project to involve the community in testing, feedback, and contributions.

## 11. Conclusion

This design document provides a detailed plan for developing a generalized MuZero-like algorithm for game-playing AI. By starting with a focused implementation on Connect4 and progressively generalizing to multiple games, the project aims to build a robust AI system that leverages shared embeddings and transfer learning. The inclusion of detailed tasks, code snippets, and clear acceptance criteria ensures that the implementation can be effectively delegated and managed.