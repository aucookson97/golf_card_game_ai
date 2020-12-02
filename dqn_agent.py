from training_environment import GolfGame

import numpy as np
import random
from collections import deque
from game import Game
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Input
from tensorflow.keras.optimizers import Adam

DISCOUNT = 0.95
EPSILON = .1

MODEL_LOCATION = 'models'

class DqnAgent:

    def __init__(self, train=True):
        self.env = GolfGame()
        self.buffer = ReplayBuffer()
        self.model = self.build_model()
        self.target_model = self.build_model()

        if train:
            # Save checkpoints in case of crash
            self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0), net=self.model)
            self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, 'checkpoints', max_to_keep=10)
            # self.load_checkpoint()
        else:
            self.load_model()

    def train_model(self, num_episodes=1000):
        """ Train Agent 
        """
        best_performance = -10000

        for episode in range(num_episodes):
            self.collect_experience()
            batch = self.buffer.sample_batch()
            loss = self.train(batch)

            if episode % 20 == 0:
                self.update_target_network()
                # performance = self.evaluate_model()
                performance = self.evaluate_model_game(10)
                print ('Model Performance: {}'.format(performance))
                self.save_checkpoint()

                # Save best model
                if performance > best_performance:
                    best_performance = performance
                    self.save_model()

    def evaluate_model_game(self, num_rounds):
        game = Game(None, num_rounds)
        avg_score = game.run()

        return avg_score

    def evaluate_model(self, num_episodes=10):
        """ Check model's performance 
        """
        total_reward = 0.0
        for _ in range(num_episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0.0
            while not done:
                action = self.policy(state)
                next_state, reward, done = self.env.step(action)
                episode_reward += reward
                state = next_state
            total_reward += episode_reward
        average_reward = total_reward / num_episodes

        return average_reward

    def update_target_network(self):
        """ Updates target network to match training network
        """
        self.target_model.set_weights(self.model.get_weights())

    def collect_experience(self):
        """ Step through environment and store experiences in the buffer
        """
        state = self.env.reset()
        done = False

        while not done:
            action = self.policy(state)
            next_state, reward, done = self.env.step(action)
            self.buffer.store_experience(state, next_state, reward, action, done)
            state = next_state

    def policy(self, state):
        """ Takes a state and returns an action
        """
        if random.random() < EPSILON: # e-greedy Random exploration
            return random.randint(0, 6)
        else: # Exploitation
            state_input = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
            action_q = self.model(state_input)
            action = np.argmax(action_q.numpy()[0], axis=0)
            return action

    def train(self, batch):
        """ Takes a batch of experiences from replay buffer and trains model
        """
        state_batch, next_state_batch, action_batch, reward_batch, done_batch = batch
        current_q = self.model(state_batch)
        target_q = np.copy(current_q)
        next_q = self.target_model(next_state_batch)
        max_next_q = np.amax(next_q, axis=1)
        for i in range(state_batch.shape[0]):
            target_q[i][action_batch[i]] = reward_batch[i] if done_batch[i] else reward_batch[i] + DISCOUNT * max_next_q[i]
        result = self.model.fit(x=state_batch, y=target_q)
        return result.history['loss']

    def build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=7, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(7, activation='linear', kernel_initializer='he_uniform'))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

        return model

    def save_checkpoint(self):
        self.checkpoint_manager.save()
    
    def load_checkpoint(self):
        self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
    
    def save_model(self):
        tf.saved_model.save(self.model, MODEL_LOCATION)

    def load_model(self):
        self.model = tf.saved_model.load(MODEL_LOCATION)

class ReplayBuffer:

    def __init__(self):
        self.gameplay_experiences = deque(maxlen=100000)

    def store_experience(self, state, next_state, reward, action, done):
        """ Stores a single experience transition
        """
        self.gameplay_experiences.append((state, next_state, reward, action, done))

    def sample_batch(self):
        """ Samples a batch of gameplay experiences 
        """
        batch_size = min(128, len(self.gameplay_experiences))
        sampled_gameplay_batch = random.sample(self.gameplay_experiences, batch_size)
        state_batch, next_state_batch, action_batch, reward_batch, done_batch = [], [], [], [], []
        for gameplay_experience in sampled_gameplay_batch:
            state_batch.append(gameplay_experience[0])
            next_state_batch.append(gameplay_experience[1])
            reward_batch.append(gameplay_experience[2])
            action_batch.append(gameplay_experience[3])
            done_batch.append(gameplay_experience[4])
        return np.array(state_batch), np.array(next_state_batch), action_batch, reward_batch, done_batch




if __name__=="__main__":
    agent = DqnAgent()

    agent.train_model(num_episodes=1000)

    print ('done!')