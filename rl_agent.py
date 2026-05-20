"""
RadarFX — rl_agent.py
Double DQN agent implemented in pure NumPy.
No PyTorch / TensorFlow / stable-baselines — only numpy (already in requirements).

Architecture:  Input(15) → Dense(64, ReLU) → Dense(32, ReLU) → Output(3)
Optimizer:     Adam
Algorithm:     Double DQN with experience replay + target network
"""

import os
import random
import numpy as np
from collections import deque

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

ACTIONS = {0: "HOLD", 1: "BUY", 2: "SELL"}


# ─── Q-Network ────────────────────────────────────────────────────────────────

class QNetwork:
    """
    2-hidden-layer fully-connected network.
    All operations in NumPy; Adam optimizer for stable training.
    """

    def __init__(self, in_dim: int, out_dim: int,
                 h1: int = 64, h2: int = 32, lr: float = 5e-4):
        self.lr  = lr
        # He initialisation
        self.W1 = (np.random.randn(in_dim, h1) * np.sqrt(2.0 / in_dim)).astype(np.float32)
        self.b1 = np.zeros(h1, dtype=np.float32)
        self.W2 = (np.random.randn(h1, h2)    * np.sqrt(2.0 / h1)).astype(np.float32)
        self.b2 = np.zeros(h2, dtype=np.float32)
        self.W3 = (np.random.randn(h2, out_dim)* np.sqrt(2.0 / h2)).astype(np.float32)
        self.b3 = np.zeros(out_dim, dtype=np.float32)

        # Adam state — one m/v pair per parameter tensor
        self._t  = 0
        n_params = 6
        shapes   = [p.shape for p in self._params()]
        self._m  = [np.zeros(s, dtype=np.float32) for s in shapes]
        self._v  = [np.zeros(s, dtype=np.float32) for s in shapes]

    # ── helpers ───────────────────────────────────────────────────────────────
    def _params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    @staticmethod
    def _relu(x):
        return np.maximum(0.0, x)

    # ── forward ───────────────────────────────────────────────────────────────
    def forward(self, X: np.ndarray) -> np.ndarray:
        """X: (batch, in_dim)  →  Q: (batch, out_dim)"""
        self._X  = X
        self._z1 = X  @ self.W1 + self.b1;   self._a1 = self._relu(self._z1)
        self._z2 = self._a1 @ self.W2 + self.b2; self._a2 = self._relu(self._z2)
        return self._a2 @ self.W3 + self.b3

    # ── backward + Adam update ────────────────────────────────────────────────
    def backward(self, dOut: np.ndarray):
        """dOut: (batch, out_dim)  — gradient of loss w.r.t. network output."""
        # Layer 3
        dW3 = self._a2.T @ dOut
        db3 = dOut.sum(0)
        da2 = dOut @ self.W3.T

        # Layer 2 (ReLU gate)
        dz2 = da2 * (self._z2 > 0)
        dW2 = self._a1.T @ dz2
        db2 = dz2.sum(0)
        da1 = dz2 @ self.W2.T

        # Layer 1 (ReLU gate)
        dz1 = da1 * (self._z1 > 0)
        dW1 = self._X.T  @ dz1
        db1 = dz1.sum(0)

        self._adam([dW1, db1, dW2, db2, dW3, db3])

    def _adam(self, grads, b1=0.9, b2=0.999, eps=1e-8):
        self._t += 1
        params = self._params()
        for i, (g, p) in enumerate(zip(grads, params)):
            self._m[i] = b1 * self._m[i] + (1 - b1) * g
            self._v[i] = b2 * self._v[i] + (1 - b2) * g * g
            mh = self._m[i] / (1 - b1 ** self._t)
            vh = self._v[i] / (1 - b2 ** self._t)
            p -= self.lr * mh / (np.sqrt(vh) + eps)

    # ── weight transfer ───────────────────────────────────────────────────────
    def copy_from(self, src: "QNetwork"):
        for dst_p, src_p in zip(self._params(), src._params()):
            dst_p[:] = src_p

    # ── persistence ───────────────────────────────────────────────────────────
    def save(self, path: str):
        np.savez(path,
                 W1=self.W1, b1=self.b1,
                 W2=self.W2, b2=self.b2,
                 W3=self.W3, b3=self.b3)

    @classmethod
    def load(cls, path: str, in_dim: int, out_dim: int) -> "QNetwork":
        d   = np.load(path)
        net = cls(in_dim, out_dim)
        net.W1[:] = d["W1"]; net.b1[:] = d["b1"]
        net.W2[:] = d["W2"]; net.b2[:] = d["b2"]
        net.W3[:] = d["W3"]; net.b3[:] = d["b3"]
        return net


# ─── Experience Replay Buffer ─────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int = 10_000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append((
            s.astype(np.float32), int(a), float(r),
            s2.astype(np.float32), bool(done)
        ))

    def sample(self, batch: int):
        batch = random.sample(self.buf, batch)
        s, a, r, s2, d = zip(*batch)
        return (np.array(s), np.array(a), np.array(r, np.float32),
                np.array(s2), np.array(d, np.float32))

    def __len__(self):
        return len(self.buf)


# ─── DQN Agent ────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    Double DQN:
      - online net  : selects best next action
      - target net  : evaluates that action's value (reduces overestimation)
    """

    def __init__(self, state_dim: int = 15, n_actions: int = 3,
                 lr: float = 5e-4, gamma: float = 0.99,
                 eps_start: float = 1.0, eps_end: float = 0.05,
                 eps_decay: float = 0.994,
                 batch: int = 64, buf_cap: int = 10_000,
                 target_update: int = 20):

        self.state_dim  = state_dim
        self.n_actions  = n_actions
        self.gamma      = gamma
        self.epsilon    = eps_start
        self.eps_end    = eps_end
        self.eps_decay  = eps_decay
        self.batch      = batch
        self.tgt_freq   = target_update
        self._upd_cnt   = 0

        self.q   = QNetwork(state_dim, n_actions, lr=lr)
        self.tgt = QNetwork(state_dim, n_actions, lr=lr)
        self.tgt.copy_from(self.q)

        self.buf = ReplayBuffer(buf_cap)

    # ── action selection ──────────────────────────────────────────────────────
    def act(self, state: np.ndarray) -> int:
        """ε-greedy for training."""
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return self._greedy(state)

    def _greedy(self, state: np.ndarray) -> int:
        q = self.q.forward(state.reshape(1, -1))
        return int(np.argmax(q[0]))

    # ── training step ─────────────────────────────────────────────────────────
    def push(self, s, a, r, s2, done):
        self.buf.push(s, a, r, s2, done)

    def train_step(self) -> float:
        if len(self.buf) < self.batch:
            return 0.0

        S, A, R, S2, D = self.buf.sample(self.batch)
        B = self.batch
        idx = np.arange(B)

        # Double DQN targets
        q_online_next  = self.q.forward(S2)
        best_a         = np.argmax(q_online_next, axis=1)
        q_target_next  = self.tgt.forward(S2)
        td_targets     = R + self.gamma * q_target_next[idx, best_a] * (1 - D)

        # Current Q values
        q_vals         = self.q.forward(S)
        td_err         = q_vals[idx, A] - td_targets

        # Gradient — only on taken actions
        grad           = np.zeros_like(q_vals)
        grad[idx, A]   = td_err / B           # MSE gradient
        self.q.backward(grad)

        loss = float(np.mean(td_err ** 2))

        self._upd_cnt += 1
        if self._upd_cnt % self.tgt_freq == 0:
            self.tgt.copy_from(self.q)

        return loss

    def decay_epsilon(self):
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

    # ── persistence ───────────────────────────────────────────────────────────
    def save(self, path: str):
        """Save online network weights. path must NOT include .npz."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        self.q.save(path)

    @classmethod
    def load(cls, path: str, state_dim: int, n_actions: int = 3) -> "DQNAgent":
        """Load for inference (ε=0)."""
        agent         = cls(state_dim, n_actions)
        agent.q       = QNetwork.load(path, state_dim, n_actions)
        agent.epsilon = 0.0
        return agent

    # ── live inference ────────────────────────────────────────────────────────
    def signal(self, state: np.ndarray) -> dict:
        """
        Given current market state vector, return RL trading signal.
        state should be built from the last bar's features with position=[0,0,0]
        (i.e. assume we're flat and deciding whether to enter).
        """
        q_vals  = self.q.forward(np.array(state, dtype=np.float32).reshape(1, -1))[0]
        action  = int(np.argmax(q_vals))

        # Softmax confidence
        ev = np.exp(q_vals - q_vals.max())
        probs = ev / ev.sum()

        return {
            "action":      ACTIONS[action],
            "confidence":  round(float(probs[action]) * 100, 1),
            "q_hold":      round(float(q_vals[0]), 3),
            "q_buy":       round(float(q_vals[1]), 3),
            "q_sell":      round(float(q_vals[2]), 3),
        }


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from rl_env import ForexTradingEnv
    STATE_DIM = ForexTradingEnv.STATE_DIM

    agent = DQNAgent(STATE_DIM)
    dummy = np.random.randn(STATE_DIM).astype(np.float32)
    print("action:", agent.act(dummy))
    print("signal:", agent.signal(dummy))
    print("rl_agent.py ✓")
