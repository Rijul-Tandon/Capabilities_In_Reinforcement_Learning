"""
learned_penalty.py - A ONE-LAYER head that replaces the fixed stuck penalty with a learned per-transition penalty
================================================================================================================
ddqn_reward_shaping.py uses a hand-written rule: "penalise the step by a fixed amount if the observation did not
change".  This module replaces that constant with a one-layer perceptron, the MiniGrid version of the event-
transition head of the VQ-VAE action-masking pipeline.  In MiniGrid every state already IS an event, so the head is
indexed directly by the enumerated state:

    s = (x, y, direction)                              Empty environments      N = W * H * 4
    s = (x, y, direction, key picked, door opened)     DoorKey environments    N = W * H * 4 * 2 * 2

The head (TransitionPenaltyHead)
--------------------------------
    one-hot(s_t)  ->  one logit per possible next state  ->  read the logit at the observed s_{t+1}

    logit(s, s') = W[s, s'] + b                 full table, W in R^{N x N}            (--penalty-rank 0, default)
    logit(s, s') = U[s] . V[s'] + b             low-rank table, U, V in R^{N x rank}  (--penalty-rank r > 0)

    p_t        = sigmoid(logit(s_t, s_{t+1}))   probability of penalising this transition  (b = 0 at start: p = 0.5)
    m_t        ~ Bernoulli(p_t)                 the decision, sampled while training
    penalty_t  = penalty_scale * m_t            >= 0, subtracted from the environment reward

The head never sees the action: the Q-network credits the shaped reward to the action it took.  In expectation the
penalty equals penalty_scale * p_t, i.e. the head's score IS the penalty; the sample is what gives a gradient
without differentiating through the agent.

Training (once per episode)
---------------------------
No reconstruction, codebook or commitment terms.  The objective is the one the whole pipeline optimises:
maximise the TRUE episode return and minimise the number of steps.

    G     = episode_return - step_cost * episode_length
    loss  = -(G - b) * sum_t log Bern(m_t | p_t)  -  entropy_coef * sum_t H(p_t)      (REINFORCE, b = running mean of G)

Size
----
    Empty 6x6: N = 144 (table 20.7k)   DoorKey 6x6: N = 576 (table 332k)   DoorKey 10x10: N = 1600 (table 2.56M)
    A full table only ever updates entries the agent has visited; --penalty-rank 32 shares weights across states
    and needs 2 * N * 32 parameters instead.

Diagnostics logged to metrics.csv (every log interval)
------------------------------------------------------
    penalty_prob_stuck   mean p_t on transitions where the observation did NOT change (s_{t+1} = s_t)
    penalty_prob_move    mean p_t on transitions where it did change
    shaper_objective     the latest G = return - step_cost * length seen by the head
If the learned rule rediscovers the manual one, penalty_prob_stuck rises towards 1 and penalty_prob_move falls to 0.
"""

from collections import deque

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Bernoulli


class DiscreteStateSpace:
    """Enumerates the MiniGrid states of one environment and maps a state tuple to its index."""

    def __init__(self, env, env_id):
        self.width = int(env.unwrapped.width)
        self.height = int(env.unwrapped.height)
        self.with_key_door = "DoorKey" in env_id
        self.size = self.width * self.height * 4 * (4 if self.with_key_door else 1)

    def index(self, x, y, direction, has_key=False, door_open=False):
        i = (int(x) * self.height + int(y)) * 4 + int(direction)
        if self.with_key_door:
            i = (i * 2 + int(bool(has_key))) * 2 + int(bool(door_open))
        return i


class TransitionPenaltyHead(nn.Module):
    """One-layer perceptron over states: one-hot(s) -> one logit per next state s'.  Read at the observed s'."""

    def __init__(self, num_states, rank=0):
        super().__init__()
        self.num_states = int(num_states)
        self.rank = int(rank)
        if self.rank <= 0:
            self.table = nn.Parameter(torch.zeros(self.num_states, self.num_states))
            self.u = self.v = None
        else:
            self.table = None
            self.u = nn.Parameter(0.1 * torch.randn(self.num_states, self.rank))
            self.v = nn.Parameter(0.1 * torch.randn(self.num_states, self.rank))
        self.bias = nn.Parameter(torch.zeros(1))

    def logit(self, s, s_next):
        if self.table is not None:
            return self.table[s, s_next] + self.bias[0]
        return (self.u[s] * self.v[s_next]).sum() + self.bias[0]

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


class LearnedPenalty:
    """Decides a penalty for every transition and learns from the episode outcome.  See the module docstring."""

    def __init__(self, num_states, device, rank=0, penalty_scale=1.0, step_cost=0.01, lr=1e-3,
                 entropy_coef=0.01, baseline_momentum=0.9):
        self.device = device
        self.net = TransitionPenaltyHead(num_states, rank).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.penalty_scale = float(penalty_scale)
        self.step_cost = float(step_cost)
        self.entropy_coef = float(entropy_coef)
        self.baseline_momentum = float(baseline_momentum)
        self.baseline = None                      # running mean of the objective G
        self._log_probs = []                      # one entry per decision of the current episode
        self._entropies = []
        self.recent_p_stuck = deque(maxlen=1000)  # p_t on "observation unchanged" transitions
        self.recent_p_move = deque(maxlen=1000)   # p_t on the others
        self.last_objective = float("nan")
        self.last_loss = float("nan")
        self.updates = 0

    def num_parameters(self):
        return self.net.num_parameters()

    def decide(self, state_index, next_state_index, no_change):
        """Returns the penalty (>= 0) for the transition s -> s'.  Keeps the graph for the episode update."""
        logit = self.net.logit(int(state_index), int(next_state_index))
        dist = Bernoulli(logits=logit)
        decision = dist.sample()
        self._log_probs.append(dist.log_prob(decision))
        self._entropies.append(dist.entropy())
        p = float(torch.sigmoid(logit).item())
        (self.recent_p_stuck if no_change else self.recent_p_move).append(p)
        return self.penalty_scale * float(decision.item())

    def end_episode(self, episode_return, episode_length):
        """One REINFORCE step on the episode's decisions: maximise return, minimise steps."""
        if not self._log_probs:
            return
        objective = float(episode_return) - self.step_cost * float(episode_length)
        if self.baseline is None:
            self.baseline = objective
        advantage = objective - self.baseline
        self.baseline = self.baseline_momentum * self.baseline + (1.0 - self.baseline_momentum) * objective
        log_probs = torch.stack(self._log_probs)
        entropies = torch.stack(self._entropies)
        loss = -advantage * log_probs.sum() - self.entropy_coef * entropies.sum()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
        self.optimizer.step()
        self.last_objective = objective
        self.last_loss = float(loss.item())
        self.updates += 1
        self._log_probs.clear()
        self._entropies.clear()

    def stats(self):
        """Extra columns for metrics.csv."""
        return {
            "penalty_prob_stuck": float(np.mean(self.recent_p_stuck)) if self.recent_p_stuck else "",
            "penalty_prob_move": float(np.mean(self.recent_p_move)) if self.recent_p_move else "",
            "shaper_objective": self.last_objective,
        }
