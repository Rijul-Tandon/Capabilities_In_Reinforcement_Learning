"""
learned_penalty.py - A ONE-LAYER head that replaces the fixed stuck penalty with a learned per-transition penalty
================================================================================================================
ddqn_reward_shaping.py uses a hand-written rule: "penalise the step by a fixed amount if the observation did not
change".  This module replaces that constant with a one-layer perceptron, the MiniGrid version of the event-
transition head of the VQ-VAE action-masking pipeline.  In MiniGrid every state is an event, so the head is
indexed directly by the enumerated state (--penalty-state):

    position_dir (default)   s = (x, y, direction)                            N = W * H * 4     (144 on 6x6, 400 on 10x10)
    position                 s = (x, y)                                       N = W * H         (36 on 6x6, 100 on 10x10)
    full                     s = (x, y, direction, key picked, door opened)   N = W * H * 4 * 4 on DoorKey

The head (TransitionPenaltyHead)
--------------------------------
    one-hot(s_t)  ->  one logit per possible next state  ->  read the logit at the observed s_{t+1}

    logit(s, s') = W[s, s'] + b                 full table, W in R^{N x N}            (--penalty-rank 0, default)
    logit(s, s') = U[s] . V[s'] + b             low-rank table, U, V in R^{N x rank}  (--penalty-rank r > 0)

    score_t    = sigmoid(logit(s_t, s_{t+1}))   in (0, 1)
    penalty_t  = penalty_scale * score_t        >= 0 and never above penalty_scale (the cap, 0.02 by default)

The head never sees the action: the Q-network credits the shaped reward r - penalty to the action it took.

Two ways to train it (--penalty-learning)
-----------------------------------------
meta (default, MetaLearnedPenalty)
    The weights start at random and are trained to MAXIMISE THE TRUE EPISODIC REWARD of the environment, even
    though the agent acts on reward + penalty.  The true reward depends on the penalty only through the agent's
    learning, so the gradient is taken THROUGH the agent's own TD step (a one-step meta-gradient):

        theta' = theta - lr * grad_theta L_TD(theta; r - rho_eta)          one simulated DQN step; rho sits in the target
        J(eta) = E_s[ sum_a softmax(Q_theta'(s, .) / tau)_a * Q_true(s, a) ] true value of the policy the update produces
        eta   <- eta + meta_lr * grad_eta J                                 back-propagated through theta'

    Q_true is a separate critic trained by Double DQN on the TRUE reward from the same replay buffer.  To first
    order dJ / d rho(s, s') is proportional to -A_true(s, a): the penalty on a transition rises when the action
    that produced it is worse than the best action under the true reward, and falls otherwise.  The replay buffer
    stores the true reward; the current penalty is re-applied at sampling time, so old transitions are shaped by
    the head as it is now, not as it was when they were collected.

hindsight (LearnedPenalty)
    Once per episode, every transition is labelled by loop erasure: wasted = 1 if it changed nothing or was part of
    a closed loop of the trajectory, 0 for the steps on the loop-erased path of a successful episode; the head is
    trained with binary cross-entropy against those labels.

Diagnostics logged to metrics.csv (every log interval)
------------------------------------------------------
    penalty_prob_stuck     mean score on transitions where the observation did NOT change (s_{t+1} = s_t)
    penalty_prob_move      mean score on transitions where it did change
    head_loss              meta: the outer objective -J (lower = higher true value); hindsight: cross-entropy
    head_wasted_fraction   hindsight only: fraction of the last episode's steps labelled wasted
    true_q_loss            meta only: TD loss of the true-reward critic
"""

import copy
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

STATE_MODES = ("position_dir", "position", "full")


class DiscreteStateSpace:
    """Enumerates the MiniGrid states of one environment and maps a state tuple to its index.

    mode = "position_dir"  s = (x, y, direction)                           N = W * H * 4          (default)
    mode = "position"      s = (x, y)                                      N = W * H
                           a turn and a wall bump are then the same self-transition: the head cannot tell them apart
    mode = "full"          s = (x, y, direction, key picked, door opened)  N = W * H * 4 * 4 on DoorKey, else as position_dir
    """

    def __init__(self, env, env_id, mode="position_dir"):
        if mode not in STATE_MODES:
            raise ValueError(f"unknown penalty state mode {mode!r}; choose one of {STATE_MODES}")
        self.mode = mode
        self.width = int(env.unwrapped.width)
        self.height = int(env.unwrapped.height)
        self.with_direction = mode != "position"
        self.with_key_door = mode == "full" and "DoorKey" in env_id
        self.size = self.width * self.height * (4 if self.with_direction else 1) * (4 if self.with_key_door else 1)

    def index(self, x, y, direction=0, has_key=False, door_open=False):
        i = int(x) * self.height + int(y)
        if self.with_direction:
            i = i * 4 + int(direction)
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

    def logits(self, s, s_next):
        """Vectorised: s and s_next are equal-length index tensors; returns one logit per pair."""
        if self.table is not None:
            return self.table[s, s_next] + self.bias[0]
        return (self.u[s] * self.v[s_next]).sum(dim=-1) + self.bias[0]

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


class _PenaltyBase:
    """What the trainer needs from every learned penalty: decide / record / end_episode / stats."""

    recomputes_penalty = False     # True: the replay buffer stores the TRUE reward and the penalty is re-applied at sampling time

    def __init__(self, num_states, device, rank, penalty_scale):
        self.device = device
        self.net = TransitionPenaltyHead(num_states, rank).to(device)
        self.penalty_scale = float(penalty_scale)
        self.recent_p_stuck = deque(maxlen=1000)          # score on "observation unchanged" transitions
        self.recent_p_move = deque(maxlen=1000)           # score on the others
        self.last_loss = float("nan")
        self.updates = 0

    def num_parameters(self):
        return self.net.num_parameters()

    def score(self, state_index, next_state_index):
        with torch.no_grad():
            return float(torch.sigmoid(self.net.logit(int(state_index), int(next_state_index))).item())

    def attach(self, **kwargs):
        pass

    def record(self, state_index, next_state_index):
        pass

    def end_episode(self, episode_return, episode_length, reached_goal):
        pass

    def stats(self):
        return {
            "penalty_prob_stuck": float(np.mean(self.recent_p_stuck)) if self.recent_p_stuck else "",
            "penalty_prob_move": float(np.mean(self.recent_p_move)) if self.recent_p_move else "",
            "head_loss": self.last_loss,
        }


# ══════════════════════════════════════════════════════════════════════════════
# meta-gradient: the head maximises the true episodic reward through the agent's TD step
# ══════════════════════════════════════════════════════════════════════════════
class MetaLearnedPenalty(_PenaltyBase):
    """See the module docstring, section "meta"."""

    recomputes_penalty = True

    def __init__(self, num_states, device, rank=0, penalty_scale=0.02, lr=5e-3, init_std=0.1, temperature=0.1):
        super().__init__(num_states, device, rank, penalty_scale)
        with torch.no_grad():                               # random start, as asked: the first penalties are ~scale/2
            if self.net.table is not None:
                self.net.table.normal_(0.0, float(init_std))
            self.net.bias.zero_()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.temperature = float(temperature)
        self.last_true_loss = float("nan")
        self.buffer_s = self.buffer_s_next = None
        self.pos = 0
        self.true_q = self.true_target = self.true_optimizer = None
        self.gamma = self.agent_lr = None
        self.target_sync_updates = 1

    def attach(self, buffer_size, obs_dim, num_actions, hidden_size, gamma, agent_lr, target_sync_updates=25):
        """Mirror of the replay buffer positions plus the true-reward critic.  Called by the trainer once."""
        from dqn_common import QNetwork        # local import: dqn_common imports this module lazily
        self.capacity = int(buffer_size)
        self.buffer_s = np.zeros(self.capacity, dtype=np.int64)
        self.buffer_s_next = np.zeros(self.capacity, dtype=np.int64)
        self.pos = 0
        self.true_q = QNetwork(obs_dim, num_actions, hidden_size).to(self.device)
        self.true_target = copy.deepcopy(self.true_q)
        self.true_optimizer = torch.optim.Adam(self.true_q.parameters(), lr=agent_lr)
        self.gamma, self.agent_lr = float(gamma), float(agent_lr)
        self.target_sync_updates = max(1, int(target_sync_updates))

    def decide(self, state_index, next_state_index, no_change, state_key=None, next_state_key=None):
        """The penalty of this transition under the current head (for logging; the buffer re-applies it at sampling)."""
        p = self.score(state_index, next_state_index)
        (self.recent_p_stuck if no_change else self.recent_p_move).append(p)
        return self.penalty_scale * p

    def record(self, state_index, next_state_index):
        """Store the state indices at the same position the replay buffer just wrote (both advance in lockstep)."""
        self.buffer_s[self.pos] = int(state_index)
        self.buffer_s_next[self.pos] = int(next_state_index)
        self.pos = (self.pos + 1) % self.capacity

    def penalties_at(self, idx):
        """Current penalties of the sampled buffer positions idx (a LongTensor); differentiable in the head."""
        idx_np = idx.detach().cpu().numpy()
        s = torch.as_tensor(self.buffer_s[idx_np], device=self.device)
        s_next = torch.as_tensor(self.buffer_s_next[idx_np], device=self.device)
        return self.penalty_scale * torch.sigmoid(self.net.logits(s, s_next))

    def meta_update(self, q_net, target_net, b_obs, b_next_obs, b_actions, b_rewards, b_dones, b_idx, double_dqn=True):
        """One step of the true-reward critic, then one meta-gradient step of the head.  b_rewards are TRUE rewards."""
        # 1. the true-reward critic (plain Double DQN on the true reward, same batch)
        with torch.no_grad():
            best_next = self.true_q(b_next_obs).argmax(dim=1, keepdim=True)
            true_next = self.true_target(b_next_obs).gather(1, best_next).squeeze(1)
            true_target = b_rewards + self.gamma * true_next * (1.0 - b_dones)
        true_loss = F.mse_loss(self.true_q(b_obs).gather(1, b_actions.unsqueeze(1)).squeeze(1), true_target)
        self.true_optimizer.zero_grad()
        true_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.true_q.parameters(), 10.0)
        self.true_optimizer.step()
        self.last_true_loss = float(true_loss.item())
        if self.updates % self.target_sync_updates == 0:
            self.true_target.load_state_dict(self.true_q.state_dict())

        # 2. the agent's TD step, simulated functionally so that it stays differentiable in the head
        params = dict(q_net.named_parameters())
        rho = self.penalties_at(b_idx)                                           # depends on the head
        with torch.no_grad():
            if double_dqn:
                best_next = q_net(b_next_obs).argmax(dim=1, keepdim=True)
                target_max = target_net(b_next_obs).gather(1, best_next).squeeze(1)
            else:
                target_max = target_net(b_next_obs).max(dim=1).values
        td_target = (b_rewards - rho) + self.gamma * target_max * (1.0 - b_dones)
        q_sa = functional_call(q_net, params, (b_obs,)).gather(1, b_actions.unsqueeze(1)).squeeze(1)
        inner_loss = F.mse_loss(q_sa, td_target)
        grads = torch.autograd.grad(inner_loss, list(params.values()), create_graph=True)
        updated = {name: p - self.agent_lr * g for (name, p), g in zip(params.items(), grads)}

        # 3. outer objective: the TRUE value of the policy the updated Q-values induce
        q_updated = functional_call(q_net, updated, (b_obs,))
        policy = torch.softmax(q_updated / self.temperature, dim=1)
        with torch.no_grad():
            q_true = self.true_q(b_obs)
        outer_loss = -(policy * q_true).sum(dim=1).mean()
        head_params = list(self.net.parameters())
        head_grads = torch.autograd.grad(outer_loss, head_params, allow_unused=True)
        self.optimizer.zero_grad()
        for p, g in zip(head_params, head_grads):
            p.grad = None if g is None else g.detach()
        torch.nn.utils.clip_grad_norm_(head_params, 10.0)
        self.optimizer.step()
        self.last_loss = float(outer_loss.item())
        self.updates += 1

    def stats(self):
        out = super().stats()
        out["true_q_loss"] = self.last_true_loss
        return out


# ══════════════════════════════════════════════════════════════════════════════
# hindsight: cross-entropy on loop-erasure labels of each episode
# ══════════════════════════════════════════════════════════════════════════════
def hindsight_labels(state_keys, next_state_keys):
    """Loop erasure: wasted[t] = 1 for every step inside a closed loop of the trajectory, 0 for the steps that
    survive on the loop-erased path from the start state to the final state.

    state_keys / next_state_keys: hashable FULL states (x, y, direction, key, door) of each transition, in order.
    A step that changes nothing (s' == s) is a loop of length one.  Whenever the trajectory re-enters a state that
    is already on the current path, every step since that state's last visit is marked wasted and the path is cut
    back to it, so A -> B -> C -> A marks all three steps.
    """
    wasted = np.zeros(len(state_keys), dtype=np.float32)
    path_states = [state_keys[0]]             # states on the current loop-erased path
    path_steps = []                           # step indices that lead from one path state to the next
    position = {state_keys[0]: 0}
    for t in range(len(state_keys)):
        s_next = next_state_keys[t]
        if s_next in position:                # the loop closes here (includes s_next == s_t)
            k = position[s_next]
            for step in path_steps[k:]:
                wasted[step] = 1.0
            wasted[t] = 1.0
            for s in path_states[k + 1:]:
                del position[s]
            path_states = path_states[:k + 1]
            path_steps = path_steps[:k]
        else:
            path_steps.append(t)
            path_states.append(s_next)
            position[s_next] = len(path_states) - 1
    return wasted


class LearnedPenalty(_PenaltyBase):
    """See the module docstring, section "hindsight"."""

    recomputes_penalty = False

    def __init__(self, num_states, device, rank=0, penalty_scale=1.0, lr=5e-3, init_logit=-2.0):
        super().__init__(num_states, device, rank, penalty_scale)
        with torch.no_grad():
            self.net.bias.fill_(float(init_logit))
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self._states, self._next_states = [], []          # head indices of the current episode's transitions
        self._keys, self._next_keys = [], []              # full states, for the loop test
        self.last_wasted_fraction = float("nan")

    def decide(self, state_index, next_state_index, no_change, state_key=None, next_state_key=None):
        """Returns the penalty (>= 0) for the transition s -> s' and remembers the transition for the episode update."""
        p = self.score(state_index, next_state_index)
        self._states.append(int(state_index))
        self._next_states.append(int(next_state_index))
        self._keys.append(state_key if state_key is not None else int(state_index))
        self._next_keys.append(next_state_key if next_state_key is not None else int(next_state_index))
        (self.recent_p_stuck if no_change else self.recent_p_move).append(p)
        return self.penalty_scale * p

    def end_episode(self, episode_return, episode_length, reached_goal):
        """One cross-entropy step on the episode's transitions against their hindsight labels."""
        if not self._states:
            return
        labels = hindsight_labels(self._keys, self._next_keys)
        # A step that is not part of a loop is certainly useful only when the episode reached the goal.
        valid = np.logical_or(labels == 1.0, bool(reached_goal))
        self.last_wasted_fraction = float(labels.mean())
        if valid.any():
            s = torch.as_tensor(np.asarray(self._states)[valid], device=self.device)
            s_next = torch.as_tensor(np.asarray(self._next_states)[valid], device=self.device)
            target = torch.as_tensor(labels[valid], device=self.device)
            loss = F.binary_cross_entropy_with_logits(self.net.logits(s, s_next), target)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
            self.optimizer.step()
            self.last_loss = float(loss.item())
            self.updates += 1
        self._states.clear()
        self._next_states.clear()
        self._keys.clear()
        self._next_keys.clear()

    def stats(self):
        out = super().stats()
        out["head_wasted_fraction"] = self.last_wasted_fraction
        return out
