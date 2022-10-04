import random
from collections import namedtuple, deque
import gym

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import time

def weight_init(layers):
    for layer in layers:
        torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, device, seed, gamma, n_step=1):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.device = device
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.gamma = gamma
        self.n_step = n_step
        self.n_step_memory = deque(maxlen=self.n_step)
    
    def store(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        self.n_step_memory.append((state, action, reward, next_state, done))
        if len(self.n_step_memory) == self.n_step:
            state, action, reward, next_state, done = self.calc_multistep_return()
            e = self.experience(state, action, reward, next_state, done)
            self.memory.append(e)
    
    def calc_multistep_return(self):
        Return = 0
        for idx in range(self.n_step):
            Return += self.gamma**idx * self.n_step_memory[idx][2]
        
        return self.n_step_memory[0][0], self.n_step_memory[0][1], Return, self.n_step_memory[-1][3], self.n_step_memory[-1][4]
        
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.stack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(
            np.stack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class Network(nn.Module):
    """Actor (Policy) Model."""
    
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size
        
        self.network = nn.Sequential(nn.Linear(state_size,fc1_units),
                                     nn.ReLU(),
                                     nn.Linear(fc1_units,fc2_units),
                                     nn.ReLU())
        self.advantage = nn.Sequential(nn.Linear(fc2_units, action_size*N_ATOMS))
        self.value = nn.Sequential(nn.Linear(fc2_units,N_ATOMS))
        self.register_buffer("supports", torch.arange(VMIN, VMAX+DZ, DZ)) # basic value vector - shape n_atoms stepsize dz
        self.softmax = nn.Softmax(dim = 1)
    
    def forward(self, state):
        batch_size = state.size()[0]
        x = self.network(state)
        value = self.value(x).view(batch_size,1,N_ATOMS)
        advantage = self.advantage(x).view(batch_size,-1, N_ATOMS)
        
        q_distr = value + advantage - advantage.mean(dim = 1, keepdim = True)
        prob = F.softmax(q_distr.view(-1, N_ATOMS)).view(-1, self.action_size, N_ATOMS)
        return prob
    
    def get_action(self,state):
        prob = self.forward(state).data.cpu()
        expected_value = prob * self.supports
        actions = expected_value.sum(2)
        return actions
    
class DQNAgent():
    """Interacts with and learns from the environment."""

    def __init__(self,
                 state_size,
                 action_size,
                 layer_size,
                 n_step,
                 BATCH_SIZE,
                 BUFFER_SIZE,
                 LR,
                 TAU,
                 GAMMA,
                 UPDATE_EVERY,
                 device,
                 seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            layer_size (int): size of the hidden layer
            BATCH_SIZE (int): size of the training batch
            BUFFER_SIZE (int): size of the replay memory
            LR (float): learning rate
            TAU (float): tau for soft updating the network weights
            GAMMA (float): discount factor
            UPDATE_EVERY (int): update frequency
            device (str): device that is used for the compute
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device = device
        self.TAU = TAU
        self.GAMMA = GAMMA
        self.UPDATE_EVERY = UPDATE_EVERY
        self.BATCH_SIZE = BATCH_SIZE
        self.Q_updates = 0
        self.n_step = n_step
        self.N_ATOMS = N_ATOMS
        
        self.action_step = 4
        self.last_action = None
        
        # Q-Network
        self.dqn = Network(state_size, action_size, seed).to(device)
        self.dqn_target = Network(state_size, action_size, seed).to(device)
        
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=LR)
        print(self.dqn)
        
        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, self.device, seed, self.GAMMA, n_step)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def projection_distribution(self, next_distr, next_state, rewards, dones, gamma):
        batch_size  = next_state.size(0)

        delta_z = float(VMAX - VMIN) / (N_ATOMS - 1)
        support = torch.linspace(VMIN, VMAX, N_ATOMS)
        rewards = rewards.expand_as(next_distr)
        dones   = dones.expand_as(next_distr)
        support = support.unsqueeze(0).expand_as(next_distr)
        ## Compute the projection of T̂ z onto the support {z_i}
        Tz = rewards + (1 - dones) * gamma * support
        Tz = Tz.clamp(min=VMIN, max=VMAX)
        b  = (Tz - VMIN) / delta_z
        l  = b.floor().long()
        u  = b.ceil().long()

        offset = torch.linspace(0, (batch_size - 1) * N_ATOMS, batch_size).long()\
                        .unsqueeze(1).expand(batch_size, N_ATOMS)
        # Distribute probability of T̂ z
        proj_dist = torch.zeros(next_distr.size())    
        proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_distr * (u.float() - b)).view(-1))
        proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_distr * (b - l.float())).view(-1))

        return proj_dist

    
    def step(self, state, action, reward, next_state, done, writer):
        # Save experience in replay memory
        self.memory.store(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.BATCH_SIZE:
                experiences = self.memory.sample()
                loss = self.learn(experiences)
                self.Q_updates += 1
                writer.add_scalar("Q_loss", loss, self.Q_updates)

    def get_action(self, state, eps=0.):
        """Returns actions for given state as per current policy. Acting only every 4 frames!
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        if self.action_step == 4:
            state = np.array(state)
            
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            self.dqn.eval()
            with torch.no_grad():
                action_values = self.dqn.get_action(state)
            self.dqn.train()

            # Epsilon-greedy action selection
            if random.random() > eps: # select greedy action if random number is higher than epsilon or noisy network is used!
                action = np.argmax(action_values.cpu().data.numpy())
                self.last_action = action
                return action
            else:
                action = random.choice(np.arange(self.action_size))
                self.last_action = action 
                return action
        
        else:
            self.action_step += 1
            return self.last_action
    
    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """

        states, actions, rewards, next_states, dones = experiences
        batch_size = len(states)

        # next_state distribution
        next_distr = self.dqn_target(next_states)
        next_actions = self.dqn_target.get_action(next_states)
        #chose max action indx
        next_actions = next_actions.max(1)[1].data.cpu().numpy()
        # gather best 
        next_best_distr = next_distr[range(batch_size), next_actions]

        proj_distr = self.projection_distribution(next_best_distr, next_states, rewards, dones, self.GAMMA)

        # Compute loss
        # calculates the prob_distribution for the actions based on the given state
        prob_distr = self.dqn(states)
        actions = actions.unsqueeze(1).expand(batch_size, 1, N_ATOMS)
        # gathers the the prob_distribution for the chosen action
        state_action_prob = prob_distr.gather(1, actions).squeeze(1)
        loss = -(state_action_prob.log() * proj_distr.detach()).sum(dim=1).mean()
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # ------------------- update target network ------------------- #
        self._target_soft_update(self.dqn, self.dqn_target)
        return loss.detach().cpu().numpy()            

    def _target_soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.TAU*local_param.data + (1.0-self.TAU)*target_param.data)

def eval_runs(eps, frame):
    """
    Makes an evaluation run with the current epsilon
    """
    env = gym.make("CartPole-v0")
    reward_batch = []
    for i in range(5):
        state = env.reset()
        rewards = 0
        while True:
            action = agent.get_action(state, eps)
            state, reward, done, _ = env.step(action)
            rewards += reward
            if done:
                break
        reward_batch.append(rewards)
        
    writer.add_scalar("Reward", np.mean(reward_batch), frame)

def run(max_episodes=1000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.99):
    """Deep Q-Learning.
    
    Params
    ======
        max_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon

    start_time = time.time()
    frame = 1

    for episode in range(max_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        # for t in range(max_t):
        while not done:
            action = agent.get_action(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done, writer)
            state = next_state
            episode_reward += reward
            
            # evaluation runs
            frame += 1
            if frame % 1000 == 0:
                eval_runs(eps, frame)

            if done:
                break 
        scores_window.append(episode_reward)       # save most recent score
        scores.append(episode_reward)              # save most recent score
        writer.add_scalar("Average100", np.mean(scores_window), frame)
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tFrame {} \tAverage Score: {:.2f} \tEpsilon: {:.2f}'.format(episode+1, frame, np.mean(scores_window), eps), end="")
        if (episode+1) % 100 == 0:
            print('\rEpisode {}\tFrame {}\tAverage Score: {:.2f} \tEpsilon: {:.2f}'.format(episode+1,frame, np.mean(scores_window), eps)) 
            elapsed_time = time.time() - start_time
            print("     Duration: {} sec".format(int(elapsed_time)))
        if np.mean(scores_window)>=199.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode+1-100, np.mean(scores_window)))
            torch.save(agent.dqn.state_dict(), 'checkpoint.pth')
            break
    elapsed_time = time.time() - start_time
    print("Training duration: {} min".format(elapsed_time/60,2))
    return scores


if __name__ == "__main__":
    
    writer = SummaryWriter("runs/"+"C51_n_step_summary")
    seed = 777
    BUFFER_SIZE = 100000    # replay buffer size
    BATCH_SIZE = 64         # minibatch size
    GAMMA = 0.99            # discount factor
    TAU = 1e-3              # for soft update of target parameters
    LR = 1e-3               # learning rate 
    UPDATE_EVERY = 1        # how often to update the network
    n_step = 3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using ", device)
    
    N_ATOMS = 51
    VMIN = -10
    VMAX = 10
    DZ = (VMAX-VMIN) / (N_ATOMS -1)
    np.random.seed(seed)
    env = gym.make("CartPole-v0")

    env.seed(seed)
    action_size = env.action_space.n
    state_size = env.observation_space.shape[0]

    agent = DQNAgent(state_size=state_size,
                        action_size=action_size,
                        layer_size=256,
                        n_step=n_step,
                        BATCH_SIZE=BATCH_SIZE, 
                        BUFFER_SIZE=BUFFER_SIZE, 
                        LR=LR, 
                        TAU=TAU, 
                        GAMMA=GAMMA, 
                        UPDATE_EVERY=UPDATE_EVERY, 
                        device=device, 
                        seed=seed)
    
    scores = run(max_episodes = 500)

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

