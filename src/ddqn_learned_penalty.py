"""
ddqn_learned_penalty.py - Double DQN Agent WITH a LEARNED Reward-Shaping Penalty
================================================================================
Same Double DQN as ddqn_reward_shaping.py, but the fixed stuck penalty is replaced by a ONE-LAYER head
(learned_penalty.py).  In MiniGrid every state is an event, so the head is a table indexed by the
enumerated state, (x, y, direction) by default (see --penalty-state): one-hot(s) -> one logit per next state,
read at the observed next state, sigmoid -> probability of penalising the transition.  The head is
trained once per episode to maximise the true episode return and minimise the number of steps; the
agent learns from the shaped reward r - penalty.

This is the MiniGrid test of the second layer of the VQ-VAE action-masking pipeline: can a one-layer
head that only sees transitions learn a useful penalty on its own, and how does the agent perform with
it compared with the hand-written stuck penalty?

Usage:
  python ddqn_learned_penalty.py --env-id MiniGrid-DoorKey-6x6-v0 --total-timesteps 200000

Extra arguments (defaults in dqn_common.parse_args):
  --penalty-state position_dir  state that indexes the head: position_dir = (x, y, direction) [N = W*H*4],
                               position = (x, y) [N = W*H; a turn then equals a wall bump], full = + key/door flags
  --penalty-rank 0             0 = full N x N table; r > 0 = low-rank factors U, V in R^{N x r} (2 N r parameters)
  --penalty-scale 1.0          size of the penalty when the head decides to penalise (matches the manual rule)
  --penalty-step-cost 0.01     cost per step in the head's objective  G = return - step_cost * length
  --penalty-lr 5e-3            learning rate of the head (Adam: each visited table entry moves ~lr per episode)
  --penalty-entropy-coef 0.01  entropy bonus on the penalise / do-not-penalise decision
"""

from dqn_common import parse_args, train
from learned_penalty import LearnedPenalty

if __name__ == "__main__":
    args = parse_args(default_exp_name="ddqn_learned_penalty", use_shaping=True)
    args.double_dqn = True
    args.learned_penalty = True          # recorded in config.json

    def make_penalty_learner(num_states, device):
        return LearnedPenalty(
            num_states, device,
            rank=args.penalty_rank,
            penalty_scale=args.penalty_scale,
            step_cost=args.penalty_step_cost,
            lr=args.penalty_lr,
            entropy_coef=args.penalty_entropy_coef,
        )

    train(args, use_shaping=False, penalty_learner_factory=make_penalty_learner)
