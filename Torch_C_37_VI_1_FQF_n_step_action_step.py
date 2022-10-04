import random
from collections import namedtuple, deque
import gym

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
from torch.nn.utils import clip_grad_norm_
import math
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
    """Quantile Value Network"""
    def __init__(self, state_size, action_size,layer_size, seed, N_ATOMS, layer_type="ff"):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size
        self.K = 48
        self.N_ATOMS = N_ATOMS
        self.n_cos = 128
        self.layer_size = layer_size
        self.pis = torch.FloatTensor([np.pi*i for i in range(1, self.n_cos+1)]).view(1,1,self.n_cos).to(device) # Starting from 0 as in the paper 
        self.head_1 = nn.Linear(self.state_size[0], layer_size) # cound be a cnn 
        self.cos_embedding = nn.Linear(self.n_cos, layer_size)
        self.ff_1 = nn.Linear(layer_size, layer_size)
        self.ff_2 = nn.Linear(layer_size, action_size)
        weight_init([self.head_1, self.ff_1])

    def calc_cos(self,taus):
        """
        Calculating the cosinus values depending on the number of tau samples
        """
        batch_size = taus.shape[0]
        n_tau = taus.shape[1]
        cos = torch.cos(taus.unsqueeze(-1)*self.pis)

        assert cos.shape == (batch_size,n_tau,self.n_cos), "cos shape is incorrect"
        return cos
    
    def forward(self, input):
        """Calculate the state embeddings"""
        return torch.relu(self.head_1(input))
    
    def get_quantiles(self, input, taus, embedding=None):
        """
        Quantile Calculation depending on the number of tau
        
        Return:
        quantiles [ shape of (batch_size, num_tau, action_size)]
        
        """
        if embedding==None:
            x = self.forward(input)
        else:
            x = embedding
        batch_size = x.shape[0]
        num_tau = taus.shape[1]
        cos = self.calc_cos(taus) # cos shape (batch, num_tau, layer_size)
        cos = cos.view(batch_size*num_tau, self.n_cos)
        cos_x = torch.relu(self.cos_embedding(cos)).view(batch_size, num_tau, self.layer_size) # (batch, n_tau, layer)
        
        # x has shape (batch, layer_size) for multiplication –> reshape to (batch, 1, layer)
        x = (x.unsqueeze(1)*cos_x).view(batch_size*num_tau, self.layer_size)
        
        x = torch.relu(self.ff_1(x))
        out = self.ff_2(x)
        return out.view(batch_size, num_tau, self.action_size)

def weight_init_xavier(layers):
    for layer in layers:
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.01)

class FPN(nn.Module):
    """Fraction proposal network"""
    def __init__(self, layer_size, seed, num_tau=8, device="cuda:0"):
        super(FPN,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.num_tau = num_tau
        self.device = device
        self.ff = nn.Linear(layer_size, num_tau)
        self.softmax = nn.LogSoftmax(dim=1)
        weight_init_xavier([self.ff])
        
    def forward(self,x):
        """
        Calculates tau, tau_ and the entropy
        
        taus [shape of (batch_size, num_tau)]
        taus_ [shape of (batch_size, num_tau)]
        entropy [shape of (batch_size, 1)]
        """
        q = self.softmax(self.ff(x)) 
        q_probs = q.exp()
        taus = torch.cumsum(q_probs, dim=1)
        taus = torch.cat((torch.zeros((q.shape[0], 1)).to(device), taus), dim=1)
        taus_ = (taus[:, :-1] + taus[:, 1:]).detach() / 2.
        
        entropy = -(q * q_probs).sum(dim=-1, keepdim=True)
        assert entropy.shape == (q.shape[0], 1), "instead shape {}".format(entropy.shape)
        
        return taus, taus_, entropy
    
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
        self.tseed = torch.manual_seed(seed)
        self.device = device
        self.TAU = TAU
        self.GAMMA = GAMMA
        self.UPDATE_EVERY = UPDATE_EVERY
        self.BATCH_SIZE = BATCH_SIZE
        self.Q_updates = 0
        self.n_step = n_step
        self.N_ATOMS = N_ATOMS
        self.entropy_coeff = 0.001
        
        self.action_step = 4
        self.last_action = None
        
        # FQF-Network
        self.dqn = Network(state_size, action_size,layer_size, seed, self.N_ATOMS).to(device)
        self.dqn_target = Network(state_size, action_size,layer_size, seed, self.N_ATOMS).to(device)
        
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=LR)
        print(self.dqn)
        
        self.FPN = FPN(layer_size, seed, N_ATOMS, device).to(device)
        self.frac_optimizer = optim.RMSprop(self.FPN.parameters(), lr=LR*0.000001, alpha=0.95, eps=0.00001)
        
        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, self.device, seed, self.GAMMA, n_step)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
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
            
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            self.dqn.eval()
            with torch.no_grad():
                embedding = self.dqn.forward(state)
                taus, taus_, entropy = self.FPN(embedding)
                F_Z = self.dqn.get_quantiles(state, taus_, embedding)
                action_values = ((taus[:, 1:].unsqueeze(-1) - taus[:, :-1].unsqueeze(-1)) * F_Z).sum(1)
                assert action_values.shape == (1, self.action_size)
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
        embedding = self.dqn.forward(states)
        taus, taus_, entropy = self.FPN(embedding.detach())
        
        # Get expected Q values from local model
        F_Z_expected = self.dqn.get_quantiles(states, taus_, embedding)
        Q_expected = F_Z_expected.gather(2, actions.unsqueeze(-1).expand(self.BATCH_SIZE, self.N_ATOMS, 1))
        assert Q_expected.shape == (BATCH_SIZE, self.N_ATOMS, 1)
        
        # calc fractional loss 
        with torch.no_grad():
            F_Z_tau = self.dqn.get_quantiles(states, taus[:, 1:-1], embedding.detach())
            FZ_tau = F_Z_tau.gather(2, actions.unsqueeze(-1).expand(self.BATCH_SIZE, self.N_ATOMS-1, 1))
            
        frac_loss = calc_fraction_loss(Q_expected.detach(), FZ_tau, taus)
        entropy_loss = self.entropy_coeff * entropy.mean() 
        frac_loss += entropy_loss

        # Get max predicted Q values (for next states) from target model
        with torch.no_grad():
            
            next_state_embedding_loc = self.dqn.forward(next_states)  
            n_taus, n_taus_, _ = self.FPN(next_state_embedding_loc)
            F_Z_next = self.dqn.get_quantiles(next_states, n_taus_, next_state_embedding_loc)  
            Q_targets_next = ((n_taus[:, 1:].unsqueeze(-1) - n_taus[:, :-1].unsqueeze(-1))*F_Z_next).sum(1)
            action_indx = torch.argmax(Q_targets_next, dim=1, keepdim=True)
            
            next_state_embedding = self.dqn_target.forward(next_states)
            F_Z_next = self.dqn_target.get_quantiles(next_states, taus_, next_state_embedding)
            Q_targets_next = F_Z_next.gather(2, action_indx.unsqueeze(-1).expand(self.BATCH_SIZE, self.N_ATOMS, 1)).transpose(1,2)
            Q_targets = rewards.unsqueeze(-1) + (self.GAMMA**self.n_step * Q_targets_next.to(self.device) * (1. - dones.unsqueeze(-1)))
        
        # Quantile Huber loss
        td_error = Q_targets - Q_expected
        assert td_error.shape == (self.BATCH_SIZE, self.N_ATOMS, self.N_ATOMS), "wrong td error shape"
        huber_l = calculate_huber_loss(td_error, 1.0)
        quantil_l = abs(taus_.unsqueeze(-1) -(td_error.detach() < 0).float()) * huber_l / 1.0
        
        loss = quantil_l.sum(dim=1).mean(dim=1) 
        loss = loss.mean()
        
        # Minimize the frac loss
        self.frac_optimizer.zero_grad()
        frac_loss.backward(retain_graph=True)
        self.frac_optimizer.step()
        
        # Minimize the huber loss
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(),1)
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

def calc_fraction_loss(FZ_,FZ, taus):
    """calculate the loss for the fraction proposal network """
    
    gradients1 = FZ - FZ_[:, :-1]
    gradients2 = FZ - FZ_[:, 1:] 
    flag_1 = FZ > torch.cat([FZ_[:, :1], FZ[:, :-1]], dim=1)
    flag_2 = FZ < torch.cat([FZ[:, 1:], FZ_[:, -1:]], dim=1)
    gradients = (torch.where(flag_1, gradients1, - gradients1) + torch.where(flag_2, gradients2, -gradients2)).view(BATCH_SIZE, N_ATOMS-1)
    assert not gradients.requires_grad
    loss = (gradients * taus[:, 1:-1]).sum(dim=1).mean()
    return loss 
    
def calculate_huber_loss(td_errors, k=1.0):
    """
    Calculate huber loss element-wisely depending on kappa k.
    """
    loss = torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))
    assert loss.shape == (td_errors.shape[0], N_ATOMS, N_ATOMS), "huber loss has wrong shape"
    return loss

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
    
    writer = SummaryWriter("runs/"+"FQF_CP_summary")
    seed = 1
    BUFFER_SIZE = 100000    # replay buffer size
    BATCH_SIZE = 64         # minibatch size
    GAMMA = 0.99            # discount factor
    TAU = 1e-3              # for soft update of target parameters
    LR = 1e-3               # learning rate 
    UPDATE_EVERY = 1        # how often to update the network
    N_ATOMS = 51
    n_step = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using ", device)
    
    np.random.seed(seed)
    env = gym.make("CartPole-v0")

    env.seed(seed)
    action_size = env.action_space.n
    state_size = env.observation_space.shape

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

