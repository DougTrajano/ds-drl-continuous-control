import numpy as np
import random
import copy
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque

from model import Actor, Critic
from memory import ReplayMemory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed=399, lr_actor=1e-4, lr_critic=1e-4,
                 memory_size=int(1e6), batch_size=128, gamma=0.99, tau=1e-3, weight_decay=0,
                 actor_units=(256, 128), critic_units=(256, 128), action_min=-1, action_max=1,
                update_every=16):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            lr_actor (float): learning rate for actor model
            lr_critic (float): learning rate for critic model
            memory_size (int): The total amount of memory to save experiences
            prioritized_memory (bool): Use prioritized memory (takes more time in the training session)
            batch_size (int): subset size for each training step
            gamma (float): discount factor
            tau (float): interpolation parameter
            weight_decay (float): L2 weight decay
            actor_units (tuple): A tuple with numbers of nodes in 1st and 2nd hidden layer for actor network
            critic_units (tuple): A tuple with numbers of nodes in 1st and 2nd hidden layer for critic network
            action_min (int or float): The min value in the action range
            action_max (int or float): The max value in the action range
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = np.random.seed(seed)
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.weight_decay = weight_decay
        self.actor_units = actor_units
        self.critic_units = critic_units
        self.action_min = action_min
        self.action_max = action_max
        self.update_every = update_every

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, seed,
                                 fc1_units=self.actor_units[0], fc2_units=self.actor_units[1]).to(device)
        self.actor_target = Actor(state_size, action_size, seed,
                                 fc1_units=self.actor_units[0], fc2_units=self.actor_units[1]).to(device)
        
        self.actor_optimizer = optim.Adam(
            self.actor_local.parameters(), lr=self.lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, seed,
                                   fc1_units=self.critic_units[0], fc2_units=self.critic_units[1]).to(device)
        
        self.critic_target = Critic(state_size, action_size, seed,
                                    fc1_units=self.critic_units[0], fc2_units=self.critic_units[1]).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(
        ), lr=self.lr_critic, weight_decay=self.weight_decay)

        # Noise process
        self.noise = OUNoise(action_size, seed)

        # Define memory
        self.memory = ReplayMemory(self.memory_size, self.batch_size, self.seed)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:    
            # Learn, if enough samples are available in memory
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, self.action_min, self.action_max)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # UPDATE CRITIC
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # UPDATE ACTOR
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # UPDATE TARGET NETWORKS
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                tau*local_param.data + (1.0-tau)*target_param.data)

    def save_model(self, actor_path='checkpoint_actor.pth', critic_path='checkpoint_critic.pth'):
        torch.save(self.actor_local.state_dict(), actor_path)
        torch.save(self.critic_local.state_dict(), critic_path)
        
    def load_model(self, actor_path='checkpoint_actor.pth', critic_path='checkpoint_critic.pth'):
        self.actor_local.load_state_dict(torch.load(actor_path))
        self.critic_local.load_state_dict(torch.load(critic_path))
        
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * \
            np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
