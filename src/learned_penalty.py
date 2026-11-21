"""A one-layer head that replaces the fixed stuck penalty by a learned penalty for every transition s -> s', and
the two rules that train it: a meta-gradient through the agent's TD step, or hindsight loop-erasure labels."""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

MAX_GRADIENT_NORM = 10.0


class DiscreteStateSpace:
    """Enumerates the MiniGrid states of one environment: maps a state to its index in 0 .. size - 1.

    mode = "position_dir"  s = (x, y, direction)                           N = W * H * 4
    mode = "position"      s = (x, y)                                      N = W * H
                           a turn and a wall bump are then the same self-transition: the head cannot tell them apart
    mode = "full"          s = (x, y, direction, key picked, door opened)  N = W * H * 4 * 4 on DoorKey, else as position_dir
    """

    def __init__(self, env, env_id, mode):
        self.mode = mode
        self.width = int(env.unwrapped.width)
        self.height = int(env.unwrapped.height)
        self.with_direction = mode != "position"
        self.with_key_door = mode == "full" and "DoorKey" in env_id
        self.size = self.width * self.height * (4 if self.with_direction else 1) * (4 if self.with_key_door else 1)

    def index(self, x, y, direction=0, has_key=False, door_open=False):
        state_index = int(x) * self.height + int(y)
        if self.with_direction:
            state_index = state_index * 4 + int(direction)
        if self.with_key_door:
            state_index = (state_index * 2 + int(bool(has_key))) * 2 + int(bool(door_open))
        return state_index


class TransitionPenaltyHead(nn.Module):
    """One-layer perceptron over the enumerated states: one-hot(s) -> one logit per next state, read at the observed s'.

        rank = 0:   logit(s, s') = W[s, s'] + b          full table, W in R^{N x N}
        rank > 0:   logit(s, s') = U[s] . V[s'] + b      low-rank table, U, V in R^{N x rank}

    penalty(s, s') = penalty_scale * sigmoid(logit(s, s')), so it never exceeds penalty_scale.  The head never sees
    the action: the Q-network credits the shaped reward r - penalty to the action it took.
    """

    def __init__(self, num_states, rank):
        super().__init__()
        self.rank = rank
        if rank <= 0:
            self.table = nn.Parameter(torch.zeros(num_states, num_states))        # W[s, s']
        else:
            self.u = nn.Parameter(0.1 * torch.randn(num_states, rank))            # U[s]
            self.v = nn.Parameter(0.1 * torch.randn(num_states, rank))            # V[s']
        self.bias = nn.Parameter(torch.zeros(1))

    def logits(self, state_indices, next_state_indices):
        """logit(s, s') for each pair.  The arguments are two equal-length index tensors, or two single indices."""
        if self.rank <= 0:
            return self.table[state_indices, next_state_indices] + self.bias[0]
        return (self.u[state_indices] * self.v[next_state_indices]).sum(dim=-1) + self.bias[0]


# ============================================================================
# meta: the head maximises the true reward through the agent's TD step
# ============================================================================

class MetaGradientPenaltyTrainer:
    """Trains the head to maximise the TRUE reward, although the agent learns from r - penalty.

    The true reward depends on the penalty only through the agent's learning, so the gradient is taken through one
    simulated TD step of the agent (theta: agent weights, eta: head weights, rho: penalty, tau: temperature):

        theta' = theta - lr * grad_theta L_TD(theta; r - rho_eta)
        J(eta) = E_s[ sum_a softmax(Q_theta'(s, .) / tau)_a * Q_true(s, a) ]
        eta   <- eta + head_lr * grad_eta J

    Q_true is a separate critic trained by Double DQN on the true reward.  The replay buffer stores the true reward
    and the penalty is applied when a batch is sampled, so old transitions are shaped by the head as it is now.
    """

    def __init__(self, penalty_head, observation_dim, num_actions, device, args):
        from dqn_common import QNetwork      # imported here because dqn_common imports this module

        self.penalty_head = penalty_head
        self.device = device
        self.penalty_scale = args.penalty_scale
        self.temperature = args.penalty_meta_temperature
        self.gamma = args.gamma
        self.agent_learning_rate = args.learning_rate

        # Random start: with logits around 0 the first penalties are about penalty_scale / 2.
        with torch.no_grad():
            if args.penalty_rank <= 0:
                penalty_head.table.normal_(0.0, args.penalty_init_std)
            penalty_head.bias.zero_()
        self.head_optimizer = torch.optim.Adam(penalty_head.parameters(), lr=args.penalty_lr)

        # (s, s') of every transition in the replay buffer, stored at the transition's buffer position
        self.buffer_state_indices = np.zeros(args.buffer_size, dtype=np.int64)
        self.buffer_next_state_indices = np.zeros(args.buffer_size, dtype=np.int64)

        # Q_true: a critic of the agent's size, trained by Double DQN on the TRUE reward
        self.true_q_net = QNetwork(observation_dim, num_actions, args.hidden_size).to(device)
        self.true_target_net = copy.deepcopy(self.true_q_net)
        self.true_q_optimizer = torch.optim.Adam(self.true_q_net.parameters(), lr=args.learning_rate)
        # Q_true's target network is refreshed as often as the agent's, counted in updates instead of env steps
        self.target_sync_updates = max(1, args.target_network_frequency // max(1, args.train_frequency))

        self.num_updates = 0
        self.last_head_loss = float("nan")
        self.last_true_q_loss = float("nan")

    def record_transition(self, buffer_position, state_index, next_state_index):
        self.buffer_state_indices[buffer_position] = state_index
        self.buffer_next_state_indices[buffer_position] = next_state_index

    def penalties(self, buffer_indices):
        """Current penalty rho(s, s') of the transitions at these buffer positions, shape (batch,).  Differentiable in the head."""
        state_indices = torch.as_tensor(self.buffer_state_indices[buffer_indices], device=self.device)
        next_state_indices = torch.as_tensor(self.buffer_next_state_indices[buffer_indices], device=self.device)
        return self.penalty_scale * torch.sigmoid(self.penalty_head.logits(state_indices, next_state_indices))

    def update(self, q_net, target_net, observations, actions, true_rewards, next_observations, terminated, buffer_indices, double_dqn):
        """One step of the true-reward critic, then one meta-gradient step of the head, on one replay batch."""

        # 1. Q_true: plain Double DQN on the true reward
        with torch.no_grad():
            best_next_actions = self.true_q_net(next_observations).argmax(dim=1, keepdim=True)                      # (batch, 1)
            true_next_q_values = self.true_target_net(next_observations).gather(1, best_next_actions).squeeze(1)    # (batch,)
            true_td_target = true_rewards + self.gamma * true_next_q_values * (1.0 - terminated)
        true_q_values_taken = self.true_q_net(observations).gather(1, actions.unsqueeze(1)).squeeze(1)
        true_q_loss = F.mse_loss(true_q_values_taken, true_td_target)
        self.true_q_optimizer.zero_grad()
        true_q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.true_q_net.parameters(), MAX_GRADIENT_NORM)
        self.true_q_optimizer.step()
        self.last_true_q_loss = float(true_q_loss.item())
        if self.num_updates % self.target_sync_updates == 0:
            self.true_target_net.load_state_dict(self.true_q_net.state_dict())

        # 2. The agent's TD step, written functionally so that theta' stays differentiable in the head:
        #        theta' = theta - lr * grad_theta L_TD(theta; r - rho)
        agent_parameters = dict(q_net.named_parameters())
        penalties = self.penalties(buffer_indices)                                 # rho, depends on the head
        with torch.no_grad():
            if double_dqn:
                best_next_actions = q_net(next_observations).argmax(dim=1, keepdim=True)
                next_q_values = target_net(next_observations).gather(1, best_next_actions).squeeze(1)
            else:
                next_q_values = target_net(next_observations).max(dim=1).values
        # The target is NOT detached here: rho carries the gradient back to the head.
        td_target = (true_rewards - penalties) + self.gamma * next_q_values * (1.0 - terminated)
        q_values_taken = functional_call(q_net, agent_parameters, (observations,)).gather(1, actions.unsqueeze(1)).squeeze(1)
        inner_td_loss = F.mse_loss(q_values_taken, td_target)
        agent_gradients = torch.autograd.grad(inner_td_loss, list(agent_parameters.values()), create_graph=True)
        updated_agent_parameters = {}
        for (name, parameter), gradient in zip(agent_parameters.items(), agent_gradients):
            updated_agent_parameters[name] = parameter - self.agent_learning_rate * gradient

        # 3. Outer objective J: the TRUE value of the softmax policy of the updated Q-values
        updated_q_values = functional_call(q_net, updated_agent_parameters, (observations,))      # (batch, num_actions)
        policy = torch.softmax(updated_q_values / self.temperature, dim=1)                        # (batch, num_actions)
        with torch.no_grad():
            true_q_values = self.true_q_net(observations)                                         # (batch, num_actions)
        head_loss = -(policy * true_q_values).sum(dim=1).mean()                                   # -J
        head_parameters = list(self.penalty_head.parameters())
        head_gradients = torch.autograd.grad(head_loss, head_parameters, allow_unused=True)
        self.head_optimizer.zero_grad()
        for parameter, gradient in zip(head_parameters, head_gradients):
            parameter.grad = None if gradient is None else gradient.detach()
        torch.nn.utils.clip_grad_norm_(head_parameters, MAX_GRADIENT_NORM)
        self.head_optimizer.step()
        self.last_head_loss = float(head_loss.item())
        self.num_updates += 1


# ============================================================================
# hindsight: cross-entropy on loop-erasure labels of each episode
# ============================================================================

def hindsight_labels(states, next_states):
    """Loop erasure: wasted[t] = 1 for every step inside a closed loop of the trajectory, 0 for the steps that
    survive on the loop-erased path from the start state to the final state.

    states / next_states: hashable FULL states (x, y, direction, key, door) of each transition, in order.
    A step that changes nothing (s' == s) is a loop of length one.  Whenever the trajectory re-enters a state that
    is already on the current path, every step since that state's last visit is marked wasted and the path is cut
    back to it, so A -> B -> C -> A marks all three steps.
    """
    wasted = np.zeros(len(states), dtype=np.float32)
    path_states = [states[0]]               # states on the current loop-erased path
    path_steps = []                         # path_steps[i] is the step that leads from path_states[i] to path_states[i + 1]
    position_on_path = {states[0]: 0}
    for step in range(len(states)):
        next_state = next_states[step]
        if next_state in position_on_path:      # the loop closes here (includes next_state == state)
            loop_start = position_on_path[next_state]
            for loop_step in path_steps[loop_start:]:
                wasted[loop_step] = 1.0
            wasted[step] = 1.0
            for erased_state in path_states[loop_start + 1:]:
                del position_on_path[erased_state]
            path_states = path_states[:loop_start + 1]
            path_steps = path_steps[:loop_start]
        else:
            path_steps.append(step)
            path_states.append(next_state)
            position_on_path[next_state] = len(path_states) - 1
    return wasted


class HindsightPenaltyTrainer:
    """Trains the head once per episode with binary cross-entropy against loop-erasure labels (hindsight_labels):
    wasted = 1 for a step that changed nothing or was part of a closed loop, 0 for the steps on the loop-erased
    path of a successful episode."""

    def __init__(self, penalty_head, device, args):
        self.penalty_head = penalty_head
        self.device = device
        with torch.no_grad():
            penalty_head.bias.fill_(args.penalty_init_logit)
        self.head_optimizer = torch.optim.Adam(penalty_head.parameters(), lr=args.penalty_lr)

        # transitions of the current episode
        self.state_indices = []             # head index of s
        self.next_state_indices = []        # head index of s'
        self.full_states = []               # (x, y, direction, key, door) of s, for the loop test
        self.next_full_states = []

        self.last_head_loss = float("nan")
        self.last_wasted_fraction = float("nan")

    def remember_transition(self, state_index, next_state_index, full_state, next_full_state):
        self.state_indices.append(state_index)
        self.next_state_indices.append(next_state_index)
        self.full_states.append(full_state)
        self.next_full_states.append(next_full_state)

    def end_episode(self, reached_goal):
        """One cross-entropy step on this episode's transitions against their hindsight labels."""
        if not self.state_indices:
            return
        wasted = hindsight_labels(self.full_states, self.next_full_states)
        self.last_wasted_fraction = float(wasted.mean())
        # A step outside every loop is known to be useful only when the episode reached the goal;
        # in a failed episode such steps get no label.
        has_label = np.logical_or(wasted == 1.0, bool(reached_goal))
        if has_label.any():
            state_indices = torch.as_tensor(np.asarray(self.state_indices)[has_label], device=self.device)
            next_state_indices = torch.as_tensor(np.asarray(self.next_state_indices)[has_label], device=self.device)
            targets = torch.as_tensor(wasted[has_label], device=self.device)
            head_loss = F.binary_cross_entropy_with_logits(self.penalty_head.logits(state_indices, next_state_indices), targets)
            self.head_optimizer.zero_grad()
            head_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.penalty_head.parameters(), MAX_GRADIENT_NORM)
            self.head_optimizer.step()
            self.last_head_loss = float(head_loss.item())
        self.state_indices.clear()
        self.next_state_indices.clear()
        self.full_states.clear()
        self.next_full_states.clear()
