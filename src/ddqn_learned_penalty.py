"""
ddqn_learned_penalty.py - Double DQN Agent WITH a LEARNED Reward-Shaping Penalty
================================================================================
Same Double DQN as ddqn_reward_shaping.py, but the fixed stuck penalty is replaced by a ONE-LAYER head
(learned_penalty.py).  In MiniGrid every state is an event, so the head is a table indexed by the
enumerated state, (x, y, direction) by default (see --penalty-state): one-hot(s) -> one logit per next state,
read at the observed next state, sigmoid -> score in (0, 1).  The penalty is --penalty-scale times the score,
so it is capped at --penalty-scale (0.02 by default), and it is subtracted from the reward the agent learns from.

How the head is trained (--penalty-learning):
  meta (default)   The table starts at random and is trained to MAXIMISE THE TRUE EPISODIC REWARD although the
                   agent acts on reward + penalty: the gradient of the true-reward objective is taken through the
                   agent's own TD step (a one-step meta-gradient), with a separate true-reward critic.
  hindsight        Cross-entropy on loop-erasure labels of each episode (a step is wasted if it changed nothing or
                   was part of a loop; the rest of a successful episode was useful).

This is the MiniGrid test of the second layer of the VQ-VAE action-masking pipeline: can a one-layer
head that only sees transitions learn a useful penalty on its own, and how does the agent perform with
it compared with the hand-written stuck penalty?

Usage:
  python ddqn_learned_penalty.py --env-id MiniGrid-DoorKey-6x6-v0 --total-timesteps 200000

Extra arguments (defaults in dqn_common.parse_args):
  --penalty-learning meta       meta | hindsight (see above)
  --penalty-state position_dir  state that indexes the head: position_dir = (x, y, direction) [N = W*H*4],
                               position = (x, y) [N = W*H; a turn then equals a wall bump], full = + key/door flags
  --penalty-rank 0             0 = full N x N table; r > 0 = low-rank factors U, V in R^{N x r} (2 N r parameters)
  --penalty-scale 0.02         cap of the penalty (penalty = scale * score)
  --penalty-lr 5e-3            learning rate of the head (Adam)
  --penalty-init-std 0.1       meta: random initialisation of the table
  --penalty-meta-temperature 0.1
                               meta: softmax temperature of the policy whose true value is maximised
  --penalty-init-logit -2.0    hindsight: initial logit (-2 -> score 0.12)
"""

from dqn_common import parse_args, train
from learned_penalty import LearnedPenalty, MetaLearnedPenalty

if __name__ == "__main__":
    args = parse_args(default_exp_name="ddqn_learned_penalty", use_shaping=True)
    args.double_dqn = True
    args.learned_penalty = True          # recorded in config.json

    def make_penalty_learner(num_states, device):
        if args.penalty_learning == "meta":
            return MetaLearnedPenalty(
                num_states, device,
                rank=args.penalty_rank,
                penalty_scale=args.penalty_scale,
                lr=args.penalty_lr,
                init_std=args.penalty_init_std,
                temperature=args.penalty_meta_temperature,
            )
        return LearnedPenalty(
            num_states, device,
            rank=args.penalty_rank,
            penalty_scale=args.penalty_scale,
            lr=args.penalty_lr,
            init_logit=args.penalty_init_logit,
        )

    train(args, use_shaping=False, penalty_learner_factory=make_penalty_learner)
