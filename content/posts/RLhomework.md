---
title: 使用 DPO,DDPO,PPO 完成 LunarLander-v3
date: 2025-06-15T10:00:00+08:00
draft: false
authors:
  - 唐豆
tags:
  - RL
categories:
  - 学习
math: true
summary: 无
ShowToc: true
TocOpen: false
---

本文的源码在这里 --> https://github.com/tangdoou/LunarLander-RL-Comparison
## Part 1: 启动 —— 环境安装
我们的目标环境是 `LunarLander-v3`，一个模拟驾驶月球着陆器在无空气的月球表面安全降落的游戏。

- **状态空间 (State Space)**：一个8维向量，包含着陆器的坐标、速度、角度等信息。
- **动作空间 (Action Space)**：4个离散动作（什么都不做、喷射左侧引擎、主引擎、右侧引擎）。
- **任务目标**：在连续100个回合中，平均得分达到200分。

### 1.1 安装必备工具库

首先，安装 `gymnasium`、`pytorch` 和其他辅助库。
```bash
# 安装 gymnasium, box2d物理引擎, pytorch, numpy 和 matplotlib  
pip install "gymnasium[box2d]" torch numpy matplotlib
```

> **macOS 用户注意**
>
> 如果在 macOS（ Apple Silicon 芯片）上运行上述命令，会遇到一个关于 `box2d-py` 编译失败的错误，错误信息中包含 `error: command 'swig' failed`。
>
> *   **原因**：`box2d-py` 的安装需要一个名为 `SWIG` 的编译工具，而系统中默认没有安装。
> *   **解决方案**：使用 Homebrew（macOS 的包管理器）来安装它。
>
>     ```bash
>     # 1. (如果没装) 安装 Homebrew
>     /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
>     
>     # 2. 安装 SWIG
>     brew install swig
>     
>     # 3. 再次运行 pip 安装命令
>     pip install "gymnasium[box2d]"
>     ```

### 1.2 "Hello, LunarLander!" - 环境测试

下面这段脚本让着陆器执行随机动作，帮助我们检查环境是否就绪。

`test_env.py`
```python
import gymnasium as gym

def test_lunar_lander():
    """测试 LunarLander-v3 环境是否正常工作"""
    
    # 注意：根据对话中的错误，我们将版本从 v2 升级到了 v3
    env = gym.make("LunarLander-v3", render_mode="human")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    print(f"状态空间维度: {state_dim}") # 输出: 8
    print(f"动作空间维度: {action_dim}") # 输出: 4
    
    observation, info = env.reset(seed=42)
    
    for _ in range(1000):
        action = env.action_space.sample() # 随机选择一个动作
        observation, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print("一个回合结束！")
            observation, info = env.reset()
            
    env.close()

if __name__ == "__main__":
    test_lunar_lander()
```
运行 `python test_env.py`，如果你看到了一个游戏窗口，里面的着陆器在胡乱翻滚，那么环境就绪了。

项目结构如下
```
LunarLander/
├── agents/
│   ├── dqn_agent.py
│   ├── ppo_agent.py
│   └── ppo_gae_agent.py
├── logs/
│   └── sb3_ppo_lunarlabder/
│       └── ppo_lunarlander_1/
│           └── events.out.tfevents...
├── results/
│   ├── double_dqn/
│   ├── dqn/
│   ├── ppo/
│   └── ppo_gae/
├── saved_models/
│   ├── double_dqn/
│   ├── dqn/
│   ├── ppo/
│   ├── ppo_gae/
│   └── sb3/
├── evaluate.py
├── main.py
├── train_double_dqn.py
├── train_dqn.py
├── train_ppo.py
├── train_ppo_gae.py
└── train_sb3.py
```

## Part 2: 值迭代的基础—— 标准 DQN

现在，我们开始搭建第一个智能体：**深度Q网络 (Deep Q-Network, DQN)**。DQN 是深度强化学习的开山之作，它的核心思想是用一个神经网络 $Q(s, a; \theta)$ 来近似最优动作价值函数 $Q^*(s, a)$。

为了稳定训练，DQN 引入了两个关键技巧：**经验回放 (Experience Replay)** 和 **固定Q目标 (Fixed Q-Targets)**。DQN的更新目标基于贝尔曼方程，其损失函数旨在最小化当前Q值与目标Q值之间的均方误差（MSE）：

$$
L(\theta) = \mathbb{E}_{(s, a, r, s', d) \sim U(B)} \left[ \left( \underbrace{r + \gamma \max_{a'} Q(s', a'; \theta^-)}_{\text{TD Target}} - \underbrace{Q(s, a; \theta)}_{\text{Current Q-value}} \right)^2 \right]
$$

其中，$\theta$ 是主网络的参数，$\theta^-$ 是目标网络的参数（定期从主网络复制而来），$\gamma$ 是折扣因子。

### 2.1 DQN 的核心组件

我们将所有和DQN相关的代码都放在 `agents/dqn_agent.py` 文件中。

#### 经验回放池 (`ReplayBuffer`)
顾名思义，它是一个经验仓库，存储智能体与环境交互的记录 `(s, a, r, s', done)`，并在训练时随机采样，打破数据相关性。

```python
# agents/dqn_agent.py
import torch
import numpy as np
import random
from collections import deque

class ReplayBuffer:
    """经验回放池"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=int(capacity))

    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
        
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
```

#### Q网络 (`QNetwork`)
这是智能体的“大脑”，一个简单的全连接神经网络，输入状态，输出每个动作的Q值。为了控制变量，这里都使用了 3 层的全链接网络。

```python
# agents/dqn_agent.py (续)
import torch.nn as nn

class QNetwork(nn.Module):
    """Q值网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        return self.network(state)
```

#### DQN Agent 
我们将上述组件组装成一个完整的 `DQNAgent` 类，它负责决策、学习和网络同步。

```python
# agents/dqn_agent.py (续)
class DQNAgent:
    """DQN Agent"""
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=0.9,
                 target_update_freq=100, buffer_capacity=5000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update_freq = target_update_freq
        
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)
        self.update_count = 0

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state)
            return q_values.argmax().item()

    def update(self, batch_size):
        if len(self.buffer) < batch_size:
            return
            
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        
        # 计算当前Q值: Q(s, a)
        current_q_values = self.q_network(states).gather(1, actions)
        
        # 计算目标Q值: r + γ * max_a' Q_target(s', a')
        next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # 计算损失并更新
        loss = nn.MSELoss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 定期更新目标网络
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
```

### 2.2 训练并观察结果

`train_dqn.py`
```python
import gymnasium as gym
import torch
import matplotlib.pyplot as plt
from agents.dqn_agent import DQNAgent
import os

def train_dqn(num_episodes=600):
    env = gym.make('LunarLander-v3')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(state_dim, action_dim, lr=0.001, buffer_capacity=10000)
    
    all_rewards = []
    for i_episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            
            if len(agent.buffer) >= 64: # batch_size
                agent.update(64)

        agent.epsilon = max(0.01, agent.epsilon * 0.995)
        all_rewards.append(episode_reward)
        print(f"Episode: {i_episode+1}, Reward: {episode_reward:.2f}")

    env.close()
    
    # 绘图逻辑...
    plt.figure(figsize=(10, 5))
    plt.plot(all_rewards)
    plt.title('DQN Training Rewards on LunarLander-v3')
    plt.savefig('results/dqn/dqn_rewards.png')
    plt.show()

if __name__ == "__main__":
    train_dqn()
```

**结果分析**：当我们运行完600个回合，得到的奖励曲线非常典型：

![DQN 训练曲线](https://tc.z.wiki/autoupload/EPCrGlomy_dW_TeigVD2VjbX6Z0L9jPJG2fnSMjU_pGyl5f0KlZfm6UsKj-HyTuv/20250615/Mo4y/1000X500/image.png)

> 可以看出来，奖励蹦来蹦去的。即使在后期也频繁出现得分从+200骤降至-200的“灾难性遗忘”。
>
> 这就是标准DQN的“通病”。这条剧烈震荡的曲线告诉我们，智能体学得非常不稳定。其根源在于 **Q值过估计 (Q-value Overestimation)**。
>
> `update` 函数中的 `max` 操作会倾向于选择一个被高估的Q值作为目标，导致目标系统性偏高。智能体以为某个策略很好，实际执行却效果很差（坠毁，奖励-100），从而造成性能的剧烈波动。

为了解决这个问题，接下来引出 Double DQN。

## Part 3: 为了稳定训练 —— Double DQN

**Double DQN** 的提出就是为了解决Q值过估计问题。它的思想非常巧妙：将**“动作的选择”和“价值的评估”**两个过程解耦。

-   **标准DQN**：在下一个状态 `s'`，用同一个**目标网络**来**选择**最大Q值的动作并**评估**它的Q值。
-   **Double DQN**：在下一个状态 `s'`，我们用**主网络** (`q_network`) 来**选择**哪个动作最好，然后用**目标网络** (`target_network`) 来**评估**这个选定动作的Q值。

这样就避免了总是选择一个被目标网络自身高估的值。

损失函数的目标值计算公式从：
$$
Y_t^{\text{DQN}} = r + \gamma \max_{a'} Q(s', a'; \theta^{-})
$$
变为：
$$
Y_t^{\text{DoubleDQN}} = r + \gamma Q(s', \underset{a'}{\arg\max} \, Q(s', a'; \theta); \theta^{-})
$$
注意这个细微但是关键的区别：选择动作的 $\arg\max$ 操作是在主网络（参数 $\theta$）上完成的，而最终值的评估是在目标网络（参数 $\theta^-$）上完成的。

### 3.1 最小的改动，最大的提升
我们只需在 `agents/dqn_agent.py` 中添加一个继承自 `DQNAgent` 的新类，并重写 `update` 方法即可。

```python
# agents/dqn_agent.py (续)
class DoubleDQNAgent(DQNAgent):
    """Double DQN Agent"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("Initialized Double DQN Agent")

    def update(self, batch_size):
        if len(self.buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        
        # --- Double DQN 的核心改动在这里 ---
        # 1. 使用主网络(q_network)选择下一个状态的最佳动作
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1).unsqueeze(1)
            # 2. 使用目标网络(target_network)评估这些动作的Q值
            next_q_values = self.target_network(next_states).gather(1, next_actions)
        # --- 后续计算与标准DQN相同 ---
        
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        current_q_values = self.q_network(states).gather(1, actions)
        
        loss = nn.MSELoss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
```

**结果分析**：用修改后的脚本训练 `DoubleDQNAgent`，得到了一条稳定得多的奖励曲线。

![Double DQN 训练曲线](https://tc.z.wiki/autoupload/EPCrGlomy_dW_TeigVD2VjbX6Z0L9jPJG2fnSMjU_pGyl5f0KlZfm6UsKj-HyTuv/20250615/dwaJ/1000X500/image.png)

> 可以看到，断崖式的下降减少了（好像也没有减少很多 TwT）

## Part 4: 策略梯度与PPO

DQN系列算法属于**价值学习 (Value-based Learning)**，而接下来我们要实现的 **PPO (Proximal Policy Optimization)** 则属于**策略学习 (Policy-based Learning)**。它不学习价值函数，而是直接学习一个策略函数 $\pi(a|s)$，即在状态`s`下执行各个动作的概率。

PPO 由 OpenAI 在2017年提出，以其极高的训练稳定性和卓越性能，成为目前最流行的强化学习算法之一。它通常采用 **Actor-Critic (演员-评论家)** 架构：

*   **Actor (演员)**：策略网络，负责决策，输出动作的概率分布。
*   **Critic (评论家)**：价值网络，负责评价当前状态的好坏，以指导 Actor 学习。

PPO的核心是其**裁剪代理目标函数（Clipped Surrogate Objective）**。这个思想很简单：限制每一次策略更新的步长，不要让新旧策略的差距过大，从而避免“步子迈太大扯着蛋”的情况。

PPO的目标函数如下：
$$
L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]
$$
其中，$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ 是新旧策略的概率比，$\hat{A}_t$ 是优势函数估计，$\epsilon$ 是一个超参数（通常为0.2），用来定义裁剪范围。

另外要注意的是，PPO是 **On-Policy (在策略)** 算法，它不像DQN那样使用巨大的经验回放池。它收集一小段经验（trajectory），用它们更新一次网络，然后就丢弃这些经验。

### 4.1 PPO Agent 的实现

我们创建一个新文件 `agents/ppo_agent.py`。

```python
# agents/ppo_agent.py (伪代码)
# ...

class ActorCritic(nn.Module):
    # ... Actor和Critic网络定义 ...

class PPOAgent:
    def __init__(self, ...):
        # ...
        # policy是新策略网络，policy_old是旧策略网络
        self.policy = ActorCritic(...)
        self.policy_old = ActorCritic(...)
        self.policy_old.load_state_dict(self.policy.state_dict())
        # ...

    def update(self):
        # 1. 计算优势函数 A_t
        # ...

        # 2. 多轮（K_epochs）优化
        for _ in range(self.K_epochs):
            # 3. 计算概率比 r_t
            # 4. 计算 PPO 的 clip 损失
            # 5. 计算价值损失和熵损失
            # 6. 反向传播更新网络

        # 7. 将新策略复制到旧策略
        self.policy_old.load_state_dict(self.policy.state_dict())
```
*后边代码看 github 吧，有点多了，占地方*

### 4.2 PPO 训练与结果

PPO 的训练循环与 DQN 不同，它需要收集一整段轨迹（trajectory）的数据后，再进行一次集中的更新。运行 `train_ppo.py` 脚本，我们会得到一条相对平滑的性能曲线。
因为 PPO 的训练方式和 DQN 完全不同，需要进行收集-更新的循环，所以运行的时间大大增加，比 DQN 方法慢了好几倍。

![PPO 训练曲线](https://tc.z.wiki/autoupload/EPCrGlomy_dW_TeigVD2VjbX6Z0L9jPJG2fnSMjU_pGyl5f0KlZfm6UsKj-HyTuv/20250615/J7YP/1000X500/image.png)

> 在这张图中，**蓝色的线**代表每个回合的原始奖励，可以看到它依然存在一定的波动。而**红色的线**是50个回合的移动平均奖励（50-episode moving average），它能更清晰地揭示学习的长期趋势。
>
> 这条红色的移动平均线，完美诠释了 PPO 的最大优点：**稳定**。它通过**裁剪代理目标函数 (Clipped Surrogate Objective)**，严格限制了每次策略更新的幅度，避免了性能的剧烈波动，保证了智能体稳步地、可靠地变强。

### 4.3 升级PPO：引入GAE（广义优势估计）

我们最初的PPO实现中，优势函数 $\hat{A}_t$ 的计算方式比较朴素（通常是回报减去价值基线），这种方法的方差较大。为了提升性能，我们引入了业界标准的**GAE（Generalized Advantage Estimation）**。

GAE 的思想是，一个动作的“优势”不应该只看它后面一步的回报，而应该综合考虑后续多步的影响。它通过一个超参数 $\lambda \in [0, 1]$ 来巧妙地权衡偏差和方差，提供更稳定和准确的优势估计。

GAE的计算公式为：
$$
\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}
$$
其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ 是时序差分误差（TD Error）。
- 当 $\lambda=0$ 时，GAE退化为一步TD估计，偏差高，方差低。
- 当 $\lambda=1$ 时，GAE等价于蒙特卡洛估计，偏差低，方差高。
- 通常取 $\lambda$ 为 0.95 可以在两者之间取得很好的平衡。

我们创建一个新文件 `agents/ppo_gae_agent.py` 来实现这个改进版。

训练后，PPO-GAE的性能曲线如下：

![PPO-GAE 训练曲线](https://tc.z.wiki/autoupload/EPCrGlomy_dW_TeigVD2VjbX6Z0L9jPJG2fnSMjU_pGyl5f0KlZfm6UsKj-HyTuv/20250615/DYNe/1000X500/image.png)

> **结果分析**：红色的移动平均线展示了一条非常平滑、几乎单调递增的学习轨迹。它不像DQN那样大起大落，而是在稳固的基础上，步步为营，持续优化。

## Part 5: DQN，DDQN，PPO 对比及反思

### 5.1 性能汇总与可视化评估

我们先来总结一下手写的几个算法的特点和最终表现。

| 特性 (Feature) | DQN (标准版)             | Double DQN (改进版) | PPO-GAE (策略梯度)              |
| :----------- | :-------------------- | :--------------- | :-------------------------- |
| **Q值过估计**    | **严重**，是其不稳定的根源       | **显著缓解**，性能提升的关键 | **不存在**，从根本上绕开了此问题          |
| **训练稳定性**    | 差，奖励曲线剧烈震荡            | 中，比DQN稳定，但仍有波动   | **优**，学习曲线平滑，稳步提升           |
| **收敛速度**     | 慢，需要大量样本              | 中，比DQN快          | **快**，学习效率高，能更快掌握有效策略       |
| **数据利用率**    | 高 (Off-Policy)，使用经验回放 | 高 (Off-Policy)   | 低 (On-Policy)，样本用完即弃        |
| **实现复杂度**    | 中等                    | 中等 (在DQN上小改)     | **高**，需要Actor和Critic双网络及GAE |

我们编写一个 `evaluate.py` 脚本（详细代码见 github），加载训练好的模型，让它们在图形界面中跑几次。

```bash
# 评估DQN
python evaluate.py --model_type dqn --path saved_models/dqn/dqn_episode_500.pth

# 评估Double DQN
python evaluate.py --model_type dqn --path saved_models/double_dqn/double_dqn_episode_400.pth

# 评估PPO-GAE
python evaluate.py --model_type ppo_gae --path saved_models/ppo_gae/ppo_gae_final.pth
```
**评估结果（10回合平均奖励）：**
*   **PPO-GAE**: **+65.13**
*   **DQN**: -10.13
*   **Double DQN**: -56.43 (第400回合模型，可见其不稳定性)

> **PPO 算法一个明显的问题：局部最优**
>
> 在观察PPO-GAE的演示时，我们发现飞船总是倾向于降落在着陆区的右侧，而不是正中央。这是为什么？
>
> 这其实是强化学习中经典的**局部最优解**问题。智能体发现，冒险去挑战高难度的“中心降落”可能会失败并受到巨大惩罚（-100分），而“随便找个地方安全降落”虽然拿不到最高分，但每次都能稳定获得正分。在“最大化平均奖励”的目标下，智能体选择了这个更“保守”也更“安全”的策略。这深刻地揭示了奖励设计是如何塑造智能体行为的。
>
> **如何解决？** 我们可以通过**奖励工程 (Reward Shaping)** 来引导智能体，比如：
> 1.  增加一个与中心距离成反比的连续奖励。
> 2.  在成功降落后，根据离中心的距离再给予额外的奖励。
> 这样就能激励智能体去追求更高质量的解决方案。

### 5.2 工业级实现：Stable-Baselines3

现在，让我们见识一下工业级、研究级的强化学习库 **Stable-Baselines3 (SB3)**。它集成了大量的最佳实践，是 PyTorch 生态中强化学习的黄金标准。

**安装:**
```bash
# 安装SB3，[extra]包含了TensorBoard等额外依赖
pip install stable-baselines3[extra]
```

**使用SB3训练PPO，代码简洁：**
`train_sb3.py`:
```python
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

# 1. 创建环境 (SB3推荐使用矢量化环境以加速)
vec_env = make_vec_env("LunarLander-v3", n_envs=4)

# 2. 初始化PPO模型 (内置所有最佳实践)
# "MlpPolicy"表示使用多层感知机作为策略和价值网络
# tensorboard_log设置了日志保存目录，用于后续可视化
model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="./logs/sb3_logs/")

# 3. 开始训练 (注意，timesteps是总步数，不是回合数)
# SB3会自动处理多环境的并行数据收集和训练
model.learn(total_timesteps=200000, tb_log_name="ppo_lunarlander_run")

# 4. 保存模型
model.save("saved_models/sb3/ppo_lunarlander_final")

# 5. 评估模型
# 注意：评估时需要使用非矢量化的环境
eval_env = gym.make("LunarLander-v3")
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)
print(f"SB3 PPO 平均奖励: {mean_reward:.2f} +/- {std_reward:.2f}")

eval_env.close()
vec_env.close()
```
运行后，SB3的评估结果能轻松达到 100+ 的平均分，远超我们手写的版本。

![1244X468/image.png](https://tc.z.wiki/autoupload/EPCrGlomy_dW_TeigVD2VjbX6Z0L9jPJG2fnSMjU_pGyl5f0KlZfm6UsKj-HyTuv/20250615/kGvZ/1244X468/image.png)
### 5.3 为什么差距这么大？

SB3 在其简洁的接口背后，集成了大量我们没有手动实现的“行业秘诀” (Tricks of the Trade)，这些是提升算法性能和稳定性的关键：

*   **矢量化环境 (Vectorized Environments)**：同时运行多个环境实例，极大提高了数据收集效率和训练速度。
*   **数据标准化 (Data Normalization)**：对观测值(Observation)和优势函数(Advantage)进行标准化，这是稳定训练的**至关重要**的一步。
*   **正确的权重初始化 (Proper Weight Initialization)**：使用正交初始化等更优的方法，而不是默认的Xavier/Kaiming初始化。
*   **学习率调度 (Learning Rate Annealing)**：在训练过程中自动进行学习率的线性衰减。
*   **价值函数裁剪 (Value Function Clipping)**：与策略裁剪类似，也对价值函数的损失进行裁剪，进一步稳定训练。
*   **精细的超参数调优**：SB3为每个算法和环境都提供了经过大量实验验证的默认超参数。

### 5.4 TensorBoard：专业选手的仪表盘

SB3的另一个强大之处是与TensorBoard的无缝集成。我们只需在训练时指定日志目录，然后运行：
```bash
tensorboard --logdir ./logs/sb3_logs/
```
就能看到SB3生成的数十种详细、专业的可视化图表，帮助我们深入分析训练的每一个细节。

![TensorBoard 概览](https://tc.z.wiki/autoupload/EPCrGlomy_dW_TeigVD2VjbX6Z0L9jPJG2fnSMjU_pGyl5f0KlZfm6UsKj-HyTuv/20250615/WwuW/2166X4888/%E5%BE%AE%E4%BF%A1_2025-06-15_14.54.32.png)

**几个关键图表解读：**
- **`rollout/ep_rew_mean(平均回合奖励)`**: 
	- 曲线从负值（大约 -150）开始，这表示在训练初期，智能体完全不知道如何降落，几乎每次都坠毁，因此受到惩罚（负奖励）。
	- 随着训练步数（Step）的增加，曲线稳步上升，最终在 200 分以上达到平稳。这清晰地表明，智能体通过学习，策略不断优化，逐渐掌握了安全、精准降落的技巧，从而获得了很高的正奖励。
![平均奖励曲线](https://tc.z.wiki/autoupload/EPCrGlomy_dW_TeigVD2VjbX6Z0L9jPJG2fnSMjU_pGyl5f0KlZfm6UsKj-HyTuv/20250615/Wg3h/2130X932/image.png)

- **`rollout/ep_len_mean(平均回合长度)`**: 
	- 曲线一开始比较短，然后迅速增长并趋于平稳。在月球降落任务中，过早结束通常意味着坠毁。
	- 曲线的增长说明智能体学会了如何通过控制推进器在空中停留更长的时间，以便更好地调整姿态和速度，最终实现成功降落，而不是很快就掉下去。
![损失函数曲线](https://tc.z.wiki/autoupload/EPCrGlomy_dW_TeigVD2VjbX6Z0L9jPJG2fnSMjU_pGyl5f0KlZfm6UsKj-HyTuv/20250615/aEyX/2150X946/image.png)

- **`time/fps(每秒帧数)`**: 
	- 这个值主要反映了你的**计算性能**或**训练速度**。数值越高，训练得越快。
	- 图中的曲线在短暂的初始波动后稳定在 3300 左右，说明训练速度是相当稳定的。
![可解释方差](https://tc.z.wiki/autoupload/EPCrGlomy_dW_TeigVD2VjbX6Z0L9jPJG2fnSMjU_pGyl5f0KlZfm6UsKj-HyTuv/20250615/GDue/2116X1066/image.png)

