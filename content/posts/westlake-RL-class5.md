---
title: "西湖大学 RL 第五课：蒙特卡洛方法" 
date: 2025-05-20T10:00:00+08:00 
draft: false # 设置为 false 来发布，true 则为草稿，不会显示在最终网站上
authors: ["唐豆"] # 作者
tags: ["深度学习", "强化学习"] # 标签
categories: ["学习"] 
math: true
summary: "西湖大学强化学习第五讲蒙特卡洛方法。" 

ShowToc: true
TocOpen: false 
---


## 强化学习的数学原理 - 第五讲：基于蒙特卡洛的强化学习算法

**课程回顾与本次课内容概要**

*   **课程地图回顾**:
    *   上节课介绍了基于模型 (model-based) 的强化学习方法：价值迭代 (Value Iteration) 和策略迭代 (Policy Iteration)。这些方法在课程中也被统称为动态规划 (Dynamic Programming, DP) 方法。
    *   更准确地说，近年的模型基强化学习 (Model-Based Reinforcement Learning, MBRL) 通常指用数据估计模型，再基于模型进行强化学习。但在本课程中，价值迭代和策略迭代统一归为 MBRL。
    *   本次课将介绍第一个无模型 (model-free) 的强化学习方法：蒙特卡洛 (Monte Carlo, MC) 方法。
*   **与上节课的联系**:
    *   上节课的策略迭代是本次课的基础。本次课的算法可以看作是将策略迭代中依赖模型的部分替换为不依赖模型的部分。
*   **本次课大纲**:
    1.  通过一个启发性例子 (Motivating Example) 介绍蒙特卡洛估计 (Monte Carlo Estimation) 的基本思想。
    2.  介绍三个基于蒙特卡洛的强化学习算法：
        *   **MC Basic**: 最简单的 MC 强化学习算法，主要用于揭示如何去除模型依赖，实际效率较低。
        *   **MC Exploring Starts**: MC Basic 的改进，提高数据使用效率。
        *   **MC ε-Greedy**: 进一步改进，去除 Exploring Starts 的假设。
        这三个算法是层层递进、环环相扣的。

---

**第一部分：启发性例子 (Motivating Example) - 蒙特卡洛估计**

*   **核心问题**: 在没有模型的情况下，如何估计某些量？
*   **重要思想**: 蒙特卡洛估计 (Monte Carlo Estimation)。
*   **例子：抛硬币 (Flip a coin)**
    *   随机变量 $X$：正面朝上 $X = +1$，反面朝上 $X = -1$。
    *   目标：计算 $X$ 的期望 $E[X]$。
    *   **方法1：基于模型 (Model-Based)**
        *   假设已知概率模型：$p(X=+1) = 0.5$, $p(X=-1) = 0.5$。
        *   根据期望定义计算：
            $$
            E[X] = \sum\_x x \cdot p(x) = (+1) \times 0.5 + (-1) \times 0.5 = 0
            $$
        *   **问题**: 精确的概率分布（模型）往往是未知的。
    *   **方法2：无模型 (Model-Free) / 蒙特卡洛估计**
        *   **思想**: 进行多次实验（抛硬币），然后计算结果的平均值。
        *   假设进行了 $N$ 次实验，得到样本序列 $\{x\_1, x\_2, \dots, x\_N\}$。
        *   均值近似：
            $$
            E[X] \approx \bar{x} = \frac{1}{N} \sum\_{j=1}^{N} x\_j
            $$
        *   这就是蒙特卡洛估计的基本思想。
*   **蒙特卡洛估计的准确性**:
    *   当样本数量 $N$ 较小时，近似不精确。
    *   当样本数量 $N$ 较大时，近似会越来越精确。
    *   (参考幻灯片第7页图示：随着样本数量增加，样本均值逐渐收敛到真实的期望值0)。
*   **数学支撑：大数定律 (Law of Large Numbers, LLN)**
    *   假设 $\{x\_j\}\_{j=1}^N$ 是独立同分布 (independent and identically distributed, i.i.d.) 的样本。令 $\bar{x} = \frac{1}{N} \sum\_{j=1}^{N} x\_j$ 为样本均值。
    *   则有：
        1.  $E[\bar{x}] = E[X]$ (样本均值 $\bar{x}$ 是 $E[X]$ 的无偏估计)。
        2.  $Var[\bar{x}] = \frac{1}{N} Var[X]$ (样本均值 $\bar{x}$ 的方差)。
    *   结论：当 $N \to \infty$ 时，$Var[\bar{x}] \to 0$，这意味着 $\bar{x}$ 收敛于一个常数，即 $E[X]$。
    *   **注意**: 样本必须是 i.i.d. 的。
*   **蒙特卡洛估计总结**:
    *   指一类依赖于重复随机采样来解决近似问题的技术。
    *   **为什么关心蒙特卡洛估计？**因为它不需要模型。
    *   **为什么关心均值估计？**因为状态价值 (state value) $v_\pi(s)$ 和动作价值 (action value) $q_\pi(s,a)$ 在强化学习中被定义为随机变量的期望：
        *   $v_\pi(s) = E_\pi[G\_t | S\_t=s]$
        *   $q_\pi(s,a) = E_\pi[G\_t | S\_t=s, A\_t=a]$
        其中 $G\_t$ 是回报 (return)。

---

**第二部分：最简单的基于MC的强化学习算法 - MC Basic**

*   **核心思想**: 将策略迭代 (Policy Iteration) 算法转换为无模型的版本。
    *   理解此算法需要熟悉：
        1.  策略迭代算法。
        2.  蒙特卡洛均值估计思想。
*   **回顾策略迭代 (Policy Iteration)**: 每个迭代包含两步：
    1.  **策略评估 (Policy Evaluation)**: 对于当前策略 $\pi\_k$，计算其状态价值 $v\_{\pi\_k}$，通常通过求解贝尔曼期望方程：
        $v\_{\pi\_k} = r^{\pi\_k} + \gamma P^{\pi\_k} v\_{\pi\_k}$ (向量形式)
        $v\_{\pi\_k}(s) = \sum\_a \pi\_k(a|s) \sum\_{s',r} p(s',r|s,a) [r + \gamma v\_{\pi\_k}(s')]$ (标量形式)
    2.  **策略改进 (Policy Improvement)**: 基于 $v\_{\pi\_k}$ (或相应的 $q\_{\pi\_k}$) 贪心地更新策略得到 $\pi\_{k+1}$：
        $\pi\_{k+1} = \arg\max\_{\pi} (r^\pi + \gamma P^\pi v\_{\pi\_k})$
        针对每个状态 $s$ 的元素形式：
        $$
        \pi\_{k+1}(s) = \arg\max\_a \sum\_{s',r} p(s',r|s,a) [r + \gamma v\_{\pi\_k}(s')] = \arg\max\_a q\_{\pi\_k}(s,a)
        $$
        其中 $q\_{\pi\_k}(s,a) = \sum\_{s',r} p(s',r|s,a) [r + \gamma v\_{\pi\_k}(s')]$。
*   **关键点**: 计算动作价值 $q\_{\pi\_k}(s,a)$。
    *   **计算 $q\_{\pi\_k}(s,a)$ 的两种方式**:
        1.  **依赖模型 (Expression 1)**: (策略迭代中使用的方式)
            $$
            q\_{\pi\_k}(s,a) = \sum\_r p(r|s,a)r + \gamma \sum\_{s'} p(s'|s,a)v\_{\pi\_k}(s')
            $$
            这需要知道状态转移概率 $p(s'|s,a)$ 和奖励函数 $p(r|s,a)$ (即模型)。
        2.  **不依赖模型 (Expression 2)**: (利用动作价值的原始定义)
            $$
            q\_{\pi\_k}(s,a) = E[G\_t | S\_t=s, A\_t=a]
            $$
            其中 $G\_t = R\_{t+1} + \gamma R\_{t+2} + \gamma^2 R\_{t+3} + \dots$ 是从时间步 $t$ 开始的累积折扣回报。
*   **Model-Free RL 的思想**: 使用**表达式2**，基于数据（样本或经验）来估计 $q\_{\pi\_k}(s,a)$，而不是依赖模型的表达式1。
*   **如何基于数据获得 $q\_{\pi\_k}(s,a)$? (蒙特卡洛方法)**
    1.  从一个指定的状态-动作对 $(s,a)$ 出发。
    2.  遵循当前策略 $\pi\_k$，生成一个完整的轨迹/回合 (episode)。
    3.  计算这个 episode 的实际回报 $g(s,a)$，这个 $g(s,a)$ 就是随机变量 $G\_t$ 的一个样本。
    4.  假设我们从 $(s,a)$ 出发，生成了 $N$ 个这样的 episodes，得到回报样本集合 $\{g^{(1)}(s,a), g^{(2)}(s,a), \dots, g^{(N)}(s,a)\}$。
    5.  则 $q\_{\pi\_k}(s,a)$ 可以通过这些样本的平均值来估计：
        $$
        q\_{\pi\_k}(s,a) = E[G\_t | S\_t=s, A\_t=a] \approx \frac{1}{N} \sum\_{i=1}^{N} g^{(i)}(s,a)
        $$
    *   **基本理念**: 当模型不可用时，我们可以使用数据。在强化学习中，这些数据称为经验 (experience)。

*   **MC Basic 算法**:
    *   给定初始策略 $\pi\_0$。在第 $k$ 次迭代中：
        1.  **步骤1：策略评估 (Policy Evaluation)**
            *   目标：估计所有状态-动作对 $(s,a)$ 的 $q\_{\pi\_k}(s,a)$。
            *   方法：对于每一个 $(s,a)$，从 $(s,a)$ 出发，遵循策略 $\pi\_k$，运行足够多的 episodes。这些 episodes 的平均回报，记为 $q\_k(s,a)$，作为 $q\_{\pi\_k}(s,a)$ 的近似。
            *   **与策略迭代的区别**: 策略迭代的第一步是求解状态价值 $v\_{\pi\_k}$（需要模型），然后计算 $q\_{\pi\_k}(s,a)$。MC Basic 直接从经验样本估计 $q\_k(s,a)$（不需要模型）。
        2.  **步骤2：策略改进 (Policy Improvement)**
            *   目标：求解 $\pi\_{k+1}(s) = \arg\max\_a q\_k(s,a)$ 对所有 $s \in S$。
            *   方法：贪心策略，$\pi\_{k+1}(a|s) = 1$ 如果 $a = a\_k^\*(s) = \arg\max\_{a'} q\_k(s,a')$，否则为0。
            *   **与策略迭代的相同点**: 这一步与策略迭代的第二步完全相同。

*   **MC Basic 算法伪代码 (Pseudocode)** (参考幻灯片第16页)
    ```
    Algorithm: MC Basic (a model-free variant of policy iteration)
    Initialization: Initial guess policy π₀.
    Aim: Search for an optimal policy.
    
    For the k-th iteration (k = 0, 1, 2, ...), do:
        For every state s ∈ S, do:
            For every action a ∈ A(s), do:
                Collect sufficiently many episodes starting from (s,a) following policy πₖ.
                Policy evaluation:
                    qₖ(s,a) ≈ average return of all the episodes starting from (s,a).
        Policy improvement:
            For every state s ∈ S, do:
                a∗ₖ(s) = argmaxₐ qₖ(s,a)
                πₖ₊₁(a|s) = 1 if a = a∗ₖ(s), and πₖ₊₁(a|s) = 0 otherwise.
    ```

*   **MC Basic 算法的讨论**:
    *   MC Basic 是策略迭代算法的一个变体。
    *   理解基于模型的算法（如策略迭代）是学习无模型算法的基础。
    *   MC Basic 有助于清晰地揭示基于 MC 的无模型强化学习的核心思想，但由于效率低下，并不实用。
        *   讲者特意起了 "MC Basic" 这个名字，是为了将核心思想（如何去掉模型）与其他复杂化因素（如数据高效利用、去除假设等）剥离开。
    *   **为什么 MC Basic 估计动作价值 ($q$) 而不是状态价值 ($v$)？** 因为如果只估计状态价值 $v$，之后要改进策略（即选择动作）时，从 $v$ 到 $q$ 的转换 ($q(s,a) = \sum\_{s',r} p(s',r|s,a)[r+\gamma v(s')]$) 仍然需要模型。无模型情况下，应直接估计动作价值 $q$。
    *   **收敛性**: 由于策略迭代是收敛的，在有足够多 episodes 的情况下，MC Basic 的收敛性也能得到保证。随着后续算法变得更复杂，收敛性分析也会更复杂或收敛性可能消失。

*   **MC Basic 算法例子 (Illustrative Example 1: Step by Step)** (参考幻灯片第18-24页)
    *   **任务**: 在一个网格世界 (Grid World) 中，给定初始策略 $\pi\_0$（如图所示，在 $s\_1, s\_3$ 处策略非最优），使用 MC Basic 找到最优策略。
    *   **环境设置**: $r\_{boundary}=-1$ (撞墙奖励), $r\_{forbidden}=-1$ (进入禁止区域奖励), $r\_{target}=1$ (到达目标奖励), $\gamma=0.9$ (折扣因子)。
    *   **状态-动作对数量**: 9个状态，每个状态5个动作（上、下、左、右、保持原地），共 $9 \times 5 = 45$ 个状态-动作对。
    *   **MC Basic 步骤概览**:
        1.  **策略评估**: 计算当前策略 $\pi\_k$下所有45个 $q\_{\pi\_k}(s,a)$。
        2.  **策略改进**: 选择贪心动作 $a^\*(s) = \arg\max\_a q\_{\pi\_k}(s,a)$。
    *   **以状态 $s\_1$ 为例进行策略评估 (计算 $q\_{\pi\_0}(s\_1,a)$)**:
        *   **重要说明**: 在此特定例子中，由于当前策略 $\pi\_0$ 是确定性的 (deterministic)，并且环境也是确定性的，所以从一个特定的 $(s,a)$ 出发，无论采样多少次，得到的 episode 都是相同的。因此，只需要一个 episode 就足以得到精确的动作价值。如果策略或环境是随机的 (stochastic)，则需要多个 episodes 并取平均。
        *   从 $(s\_1, a\_1)$ (向上) 出发：episode 为 $s\_1 \xrightarrow{a\_1} s\_1 \xrightarrow{a\_1} s\_1 \dots$ (假设向上撞墙回到 $s\_1$)。
            $q\_{\pi\_0}(s\_1,a\_1) = -1 + \gamma(-1) + \gamma^2(-1) + \dots = \frac{-1}{1-\gamma} = \frac{-1}{1-0.9} = -10$.
        *   从 $(s\_1, a\_2)$ (向右) 出发：episode 为 $s\_1 \xrightarrow{a\_2} s\_2 \xrightarrow{a\_3 (\text{down})} s\_5 \xrightarrow{a\_3 (\text{down})} \dots \rightarrow \text{target}$。
            $q\_{\pi\_0}(s\_1,a\_2) = 0 + \gamma \cdot 0 + \gamma^2 \cdot 0 + \gamma^3 \cdot 1 + \gamma^4 \cdot (\text{stay in target, assume reward 0}) \dots = \gamma^3 = 0.9^3 = 0.729$ (假设到达目标后停在目标，后续奖励为0，或者根据具体设置，如果目标是终止状态，回报就是到目标为止的累积奖励)。幻灯片的计算是 $0 + \gamma 0 + \gamma^2 0 + \gamma^3(1) + \gamma^4(1) + ...$ ，这暗示目标态可能持续给1的奖励或者是一个吸收态但计算到无限。实际计算会根据具体回报定义。假设到达目标后，保持在目标状态并持续获得奖励1，则 $q\_{\pi\_0}(s\_1,a\_2) = \gamma^3 \sum\_{i=0}^{\infty} \gamma^i = \frac{0.9^3}{1-0.9} = 7.29$。如果目标是吸收态，只获得一次1的奖励，则为 $0.729$。这里根据讲者后面比较 $a\_2, a\_3$ 均为最大，其路径相似，可推断其值。
        *   从 $(s\_1, a\_3)$ (向下) 出发：episode 为 $s\_1 \xrightarrow{a\_3} s\_4 \xrightarrow{a\_2 (\text{right})} s\_5 \xrightarrow{a\_3 (\text{down})} \dots \rightarrow \text{target}$。
            $q\_{\pi\_0}(s\_1,a\_3)$ 计算方式类似 $q\_{\pi\_0}(s\_1,a\_2)$。
        *   从 $(s\_1, a\_4)$ (向左) 出发：episode 为 $s\_1 \xrightarrow{a\_4} s\_1 \xrightarrow{a\_1} s\_1 \dots$ (假设向左撞墙回到 $s\_1$，然后执行 $s\_1$ 的原定策略 $a\_1$)。
            $q\_{\pi\_0}(s\_1,a\_4) = -1 + \gamma(-1) + \gamma^2(-1) + \dots = -10$.
        *   从 $(s\_1, a\_5)$ (原地不动) 出发：episode 为 $s\_1 \xrightarrow{a\_5} s\_1 \xrightarrow{a\_1} s\_1 \dots$ (原地不动后执行 $s\_1$ 的原定策略 $a\_1$)。
            $q\_{\pi\_0}(s\_1,a\_5) = 0 + \gamma(-1) + \gamma^2(-1) + \dots = -9$.
    *   **策略改进 (针对 $s\_1$)**:
        *   比较 $q\_{\pi\_0}(s\_1,a\_1), \dots, q\_{\pi\_0}(s\_1,a\_5)$。发现 $q\_{\pi\_0}(s\_1,a\_2)$ 和 $q\_{\pi\_0}(s\_1,a\_3)$ 最大。
        *   因此，新的策略 $\pi\_1$ 在 $s\_1$ 处可以是选择 $a\_2$ (向右) 或 $a\_3$ (向下)。
        *   在这个简单例子中，一次迭代就使得 $s\_1$ 的策略达到最优。
    *   **练习**: 使用 MC Basic 更新状态 $s\_3$ 的策略。

*   **MC Basic 算法例子 (Illustrative Example 2: 影响 Episode 长度)** (这部分内容在视频 Part 3 讲解，逻辑上是 MC Basic 的例子)
    *   **背景**: MC Basic 需要数据（episodes）来计算回报。Episode 的长度理论上越长越好（最好到终止状态或无穷长以精确估计回报）。但在实际中，episode 不可能无限长，特别是在没有明确终止条件的 Grid World 中。
    *   **问题**: Episode 长度应设为多少才合适？
    *   **实验设置**: 标准 Grid World，使用 MC Basic 得到最优状态价值和最优策略的估计。
    *   **当 Episode Length = 1**:
        *   回报只考虑立即奖励 $R\_{t+1}$。
        *   结果：只有紧挨着目标（Target）的状态的价值为正数，其对应策略正确。其他所有状态的价值均为0，策略不好。
        *   原因：这些状态在1步内无法到达目标，因此其回报估计不准确。
        *   **类比**: 在沙漠中找水源，探索半径只有1公里，如果水源在5公里外，则探索不到。
    *   **当 Episode Length = 2**:
        *   除了之前正确的状态，离目标2步远的状态的策略也变正确，价值为正。
        *   某些看起来正确的策略可能是随机产生的（因为其动作价值可能都为0，随机选择了一个恰好正确的方向）。
    *   **当 Episode Length = 3**:
        *   离目标3步远的状态的策略也变正确，价值为正。
    *   **逐渐增加 Episode Length**:
        *   Episode Length = 14 (从最远非目标点到目标所需步数): 所有状态都能到达目标，策略接近最优，价值为正。
        *   某些状态的策略仍可能是随机选择（因为其状态价值为0，表示它没有通过这条路径到达目标）。
    *   **当 Episode Length = 15**:
        *   之前随机选择的状态现在也能到达目标，状态价值为正。
    *   **当 Episode Length 很大 (如30, 100)**:
        *   策略基本不再变化，已达到最优。
        *   状态价值的估计值仍在变化，逐渐接近真实的最优状态价值。
    *   **结论**:
        1.  当 episode 长度较短时，只有离目标较近的状态才能在有限步内找到目标，从而学到正确策略。
        2.  随着 episode 长度增加，离目标越来越远的状态也能慢慢到达目标，学到最优策略。
        3.  Episode 长度必须**足够长**，使得所有状态都有机会到达目标。
        4.  但也不需要无限长，只要**充分长**即可。具体长度需根据问题分析。

---

**第三部分：更有效地使用数据 - MC Exploring Starts 算法**

*   **MC Basic 算法的缺点**:
    *   优点：清晰揭示核心思想。
    *   缺点：过于简单，效率低，不实用。
*   **如何改进 MC Basic 使其更高效？**
    *   **1. 更有效地利用 episode 内的数据**
        *   **回顾 Grid World 例子中的 episode**: $s\_1 \xrightarrow{a\_2} s\_2 \xrightarrow{a\_4} s\_1 \xrightarrow{a\_2} s\_2 \xrightarrow{a\_3} s\_5 \xrightarrow{a\_1} \dots$
        *   **访问 (Visit)**: Episode 中每一次出现某个状态-动作对 $(s,a)$，都称为对该 $(s,a)$ 的一次访问。
        *   **MC Basic 的数据使用方式 (Initial-visit method)**: 只考虑 episode 的起始状态-动作对 $(s\_1, a\_2)$，用整个 episode 的回报来估计 $q_\pi(s\_1, a\_2)$。这浪费了 episode 中后续访问到的其他状态-动作对的信息。
        *   **更充分利用数据**:
            *   Episode $s\_1 \xrightarrow{a\_2} s\_2 \xrightarrow{a\_4} s\_1 \dots$ 也访问了 $(s\_2,a\_4)$, 再次访问 $(s\_1,a\_2)$, 还访问了 $(s\_2,a\_3)$, $(s\_5,a\_1)$ 等。
            *   从 $(s\_2,a\_4)$ 开始的子序列 $s\_2 \xrightarrow{a\_4} s\_1 \xrightarrow{a\_2} \dots$ 可以看作一个新的 episode，其回报可以用来估计 $q_\pi(s\_2,a\_4)$。
            *   **数据高效的方法**:
                *   **首次访问蒙特卡洛 (First-Visit MC)**: 对于一个 episode 中某个状态-动作对 $(s,a)$ 的多次访问，只使用其**第一次**出现后的回报来估计 $q_\pi(s,a)$。
                *   **每次访问蒙特卡洛 (Every-Visit MC)**: 对于一个 episode 中某个状态-动作对 $(s,a)$ 的多次访问，**每一次**出现后的回报都用来估计 $q_\pi(s,a)$ (即更新其平均值)。
    *   **2. 更高效地更新价值估计和策略**
        *   **MC Basic 的更新方式**: 在策略评估步骤，收集从某个 $(s,a)$ 出发的所有 episodes，然后计算平均回报来近似动作价值，最后才进行策略改进。这需要等待所有 episodes 完成。
        *   **更高效的更新方式**: 每当获得一个 episode 后，立即使用该 episode 的回报来更新相应的动作价值估计，并马上改进策略。即 "episode-by-episode" 更新。
        *   **疑问**: 单个 episode 的回报来估计动作价值会不准确，可行吗？
            *   **回答**: 可行。这类似于上节课讲到的**截断策略迭代 (Truncated Policy Iteration)**，其中策略评估步骤（求解贝尔曼方程）也只进行有限步迭代，得到的 $v\_{\pi\_k}$ 并不完全精确，但算法依然有效。
        *   **广义策略迭代 (Generalized Policy Iteration, GPI)**:
            *   不是一个具体的算法，而是一种思想或框架。
            *   指策略评估 (policy evaluation) 和策略改进 (policy improvement) 过程之间不断切换、相互作用的总体思路。
            *   策略评估不需要完全精确。许多强化学习算法都属于 GPI 框架。

*   **MC Exploring Starts 算法**: 结合了更有效的数据利用和更高效的更新估计。
    *   **Exploring Starts (探索性开端) 的含义**:
        *   **Exploring (探索)**: 需要确保每个状态-动作对 $(s,a)$ 都有机会被访问到并被评估。如果某个 $(s,a)$ 从未被访问，其 $q_\pi(s,a)$ 无法被估计，可能错过最优动作。
        *   **Starts (开端)**: 确保每个 $(s,a)$ 都能作为 episode 的起始点被选择。这意味着我们需要从**每一个**状态-动作对出发生成足够多的 episodes。
        *   例如，在一个有9个状态，每个状态5个动作的环境中，需要有从 $(s\_1,a\_1), (s\_1,a\_2), \dots, (s\_1,a\_5), (s\_2,a\_1), \dots, (s\_9,a\_5)$ 所有这些组合出发的 episodes。
        *   MC Basic 和 MC Exploring Starts 都需要这个**探索性开端 (Exploring Starts)** 的假设。
    *   **MC Exploring Starts 算法伪代码 (Pseudocode)** (参考幻灯片第31页)
        ```
        Algorithm: MC Exploring Starts (an efficient variant of MC Basic)
        Initialization: Initial policy π₀(a|s) and initial value q(s,a) for all (s,a).
                      Returns(s,a) = empty list (or 0) for all (s,a).
                      Num(s,a) = 0 for all (s,a). // Or Returns(s,a) stores sum, Num(s,a) stores count
        
        Goal: Search for an optimal policy.
        
        Loop for each episode:
            Episode generation:
                Select a starting state-action pair (s₀, a₀) such that all pairs (s,a)
                can be possibly selected (this is the exploring-starts condition).
                Following the current policy π, generate an episode of length T:
                s₀, a₀, r₁, s₁, a₁, r₂, ..., s_{T-1}, a_{T-1}, r_T.
        
            Initialization for each episode: G ← 0 (cumulative discounted return)
        
            For each step of the episode, t = T-1, T-2, ..., 0 (iterate backwards):
                G ← r_{t+1} + γG  // r_{t+1} is reward received after (s_t, a_t)
                // This is for First-Visit MC. If (s_t, a_t) has not appeared earlier in this episode:
                // Or for Every-Visit MC, just proceed:
                Append G to Returns(s_t, a_t) // Or: Returns(s_t,a_t) ← Returns(s_t,a_t) + G
                Num(s_t,a_t) ← Num(s_t,a_t) + 1
                Policy evaluation (update q-value for (s_t, a_t)):
                    q(s_t, a_t) ← average(Returns(s_t, a_t)) // Or: q(s_t,a_t) ← Returns(s_t,a_t) / Num(s_t,a_t)
                Policy improvement (update policy for state s_t):
                    Let a* = argmaxₐ q(s_t, a)
                    π(a|s_t) = 1 if a = a*, and π(a|s_t) = 0 otherwise.
        ```
        *   **注意伪代码中的倒序循环 (t = T-1, ..., 0) 和 G 的计算**:
            *   这是为了高效地计算一个 episode 中所有访问过的 $(s\_t, a\_t)$ 的回报 $G\_t$。
            *   从 episode 的末尾开始， $G$ 初始化为0 (如果 $T$ 是终止状态前的最后一步，则 $r\_T$ 是最后奖励，$G\_{T-1} = r\_T$)。
            *   然后向前迭代：$G \leftarrow r\_{t+1} + \gamma G$。此时的 $G$ 就是从 $(s\_t, a\_t)$ 出发到 episode 结束的回报。
            *   这是计算技巧，并非核心思想的改变，但能提高计算效率。

*   **为什么需要考虑 Exploring Starts？**
    *   **理论上**: 只有当每个状态的每个动作价值都被充分探索和评估后，才能正确地选择最优动作。否则，如果某个动作从未被探索，它恰好可能是最优的，从而被错过。
    *   **实践上**: Exploring Starts 难以实现。对于许多应用（特别是涉及与环境物理交互的应用，如机器人），很难从每个状态-动作对开始收集 episodes（例如，需要手动将机器人放置到特定状态并执行特定初始动作）。
*   **能否移除 Exploring Starts 的要求？**
    *   答案是可以的，通过使用**软性策略 (soft policies)**。这引出了下一部分内容。

---

**第四部分：没有 Exploring Starts 的蒙特卡洛方法 - MC ε-Greedy 算法**

*   **如何去掉 Exploring Starts 条件？** 引入软性策略 (Soft Policies)。
*   **什么是软性策略 (Soft Policy)？**
    *   一个策略是软性的，如果它选择任何一个动作的概率都是正的，即 $\pi(a|s) > 0$ 对所有的 $s,a$ 都成立。
    *   与确定性策略（如贪心策略，$\pi(a|s)$ 对某个 $a$ 为1，其他为0）相对。
    *   随机性策略 (Stochastic Policy) 的一种，例如软性策略。
*   **为什么引入软性策略？**
    *   使用软性策略，即使只有少数几个足够长的 episodes，也有可能访问到每一个状态-动作对 $(s,a)$。
    *   这样就不再需要大量从每个 $(s,a)$ 出发的 episodes，从而可以移除 Exploring Starts 的要求。
*   **我们将使用什么软性策略？ $\epsilon$-贪心策略 ($\epsilon$-greedy policies)**
    *   当然软性策略不止 $\epsilon$-greedy 一种，但这里重点关注它。
    *   **什么是 $\epsilon$-贪心策略？**
        *   在一个状态 $s$下，假设 $a^\*$ 是当前的贪心动作（即 $q(s,a^\*)$ 最大）。
        *   $\epsilon$-贪心策略的定义为：
            $$
            \pi(a|s) = \begin{cases}
            1 - \epsilon + \frac{\epsilon}{|A(s)|} & \text{if } a = a^\* \text{ (贪心动作)} \\
            \frac{\epsilon}{|A(s)|} & \text{if } a \neq a^\* \text{ (其他 } |A(s)|-1 \text{ 个非贪心动作)}
            \end{cases}
            $$
            其中 $\epsilon \in [0,1]$ 是一个小的正数， $|A(s)|$ 是状态 $s$ 下可用动作的数量。
            (幻灯片第36页的公式为：
            $\pi(a|s) = \begin{cases}
            1 - \epsilon \frac{|A(s)|-1}{|A(s)|} & \text{for the greedy action} \\
            \frac{\epsilon}{|A(s)|} & \text{for the other } |A(s)|-1 \text{ actions}
            \end{cases}$
            这两个公式在思想上是一致的，表示以 $1-\epsilon$ 的概率选择贪心动作，以 $\epsilon$ 的概率从所有动作中均匀随机选择一个。后一个公式更精确地描述了分配给贪心动作和非贪心动作的概率。当只有一个贪心动作时，第一个公式中的 $1-\epsilon + \frac{\epsilon}{|A(s)|}$ 对应了讲者说的“大部分概率给贪心，剩余概率均分给所有动作包括贪心本身”，而幻灯片的公式是“大部分概率 $1-\epsilon$ 给贪心，剩余$\epsilon$均分给所有动作（所以贪心动作额外获得 $\epsilon/|A(s)|$ 的部分），而其他动作只获得 $\epsilon/|A(s)|$”。更常见的实现是：以 $1-\epsilon$ 概率选贪心，$\epsilon$ 概率均匀选所有动作之一。或者：以 $1-\epsilon$ 概率选贪心，$\epsilon$ 概率均匀选 *非贪心* 动作之一。幻灯片的公式是：greedy action 概率为 $1-\epsilon' $， non-greedy actions 概率为 $\epsilon'/(|A(s)|-1)$，其中 $\epsilon'$ 是探索的总概率。这里的 $\epsilon$ 指的是分配给每个非贪心动作的基线概率，而贪心动作获得 $1 - \sum\_{\text{non-greedy}} \text{prob}$。
            以幻灯片公式为准：
            P(greedy action) $= 1 - \epsilon + \epsilon/|A(s)|$ （这是总的概率，不是讲义的公式）
            讲义公式 (slide 36):
            Prob(greedy action) $ = 1 - \epsilon \frac{|A(s)|-1}{|A(s)|} $
            Prob(any other action) $ = \frac{\epsilon}{|A(s)|} $
            例：$\epsilon=0.2, |A(s)|=5$。
            Prob(other action) $= 0.2/5 = 0.04$。
            Prob(greedy action) $= 1 - 0.2 \times (4/5) = 1 - 0.16 = 0.84$。
            选择贪心动作的概率总是大于其他动作：$1 - \epsilon \frac{|A(s)|-1}{|A(s)|} \ge \frac{\epsilon}{|A(s)|}$，因为 $1-\epsilon \ge 0$ for $\epsilon \in [0,1]$。
            *   **$\epsilon$-greedy 策略平衡了利用 (Exploitation) 和探索 (Exploration)**:
                *   **利用 (Exploitation)**: 根据当前已知的最好策略行动，即选择当前认为具有最大价值的动作。
                *   **探索 (Exploration)**: 尝试非贪心动作，以便发现可能更好的未知动作或更准确地估计现有动作的价值。
                *   当 $\epsilon \to 0$ 时，策略变为纯贪心策略。更多利用，更少探索。
                    *   P(greedy action) $\to 1$, P(other action) $\to 0$.
                *   当 $\epsilon \to 1$ 时，策略变为均匀随机策略。更多探索，更少利用。
                    *   P(greedy action) $= 1 - (5-1)/5 = 1/5$. P(other action) $= 1/5$.
*   **将 $\epsilon$-greedy 嵌入到基于 MC 的 RL 算法中**:
    *   **原始策略改进步骤** (在 MC Basic 和 MC Exploring Starts 中): 求解
        $\pi\_{k+1}(s) = \arg\max\_{\pi \in \Pi} \sum\_a \pi(a|s)q\_{\pi\_k}(s,a)$，其中 $\Pi$ 是所有可能策略的集合。最优策略是贪心策略：$\pi\_{k+1}(a|s) = 1$ 若 $a=a\_k^\* = \arg\max\_{a'} q\_{\pi\_k}(s,a')$，否则为0。
    *   **新的策略改进步骤 (使用 $\epsilon$-greedy)**: 求解
        $\pi\_{k+1}(s) = \arg\max\_{\pi \in \Pi_\epsilon} \sum\_a \pi(a|s)q\_k(s,a)$，其中 $\Pi_\epsilon$ 是所有具有固定 $\epsilon$ 值的 $\epsilon$-greedy 策略的集合。最优策略是相对于 $q\_k(s,a)$ 的 $\epsilon$-greedy 策略：
        令 $a\_k^\* = \arg\max\_a q\_k(s,a)$。则 $\pi\_{k+1}(a|s)$ 按照上述 $\epsilon$-greedy 定义（幻灯片36页公式）给出。
*   **MC $\epsilon$-Greedy 算法**:
    *   与 MC Exploring Starts 基本相同，除了：
        1.  它使用 $\epsilon$-greedy 策略进行动作选择（用于生成 episodes）。
        2.  策略改进步骤产生的是一个新的 $\epsilon$-greedy 策略。
    *   它不需要 Exploring Starts 条件，但仍需要以其他形式（例如，通过足够长的 episodes 和 $\epsilon > 0$ 的探索）访问所有状态-动作对。
    *   通常采用 **Every-Visit** 方法，因为使用软性策略时，一个长的 episode 可能会多次访问同一个状态-动作对，Every-Visit 能更充分利用这些数据。

*   **MC $\epsilon$-Greedy 算法伪代码 (Pseudocode)** (参考幻灯片第42页)
    ```
    Algorithm: MC ε-Greedy (a variant of MC Exploring Starts)
    Initialization: Initial ε-greedy policy π₀(a|s) and initial value q(s,a) for all (s,a).
                  Returns(s,a) = empty list (or 0) for all (s,a).
                  Num(s,a) = 0 for all (s,a).
                  ε ∈ (0,1] (small positive constant for exploration).
    
    Goal: Search for an optimal policy.
    
    Loop for each episode:
        Episode generation:
            Select a starting state-action pair (s₀, a₀) (exploring starts condition is NOT required).
            Following the current ε-greedy policy π, generate an episode of length T:
            s₀, a₀, r₁, s₁, a₁, r₂, ..., s_{T-1}, a_{T-1}, r_T.
    
        Initialization for each episode: G ← 0
    
        For each step of the episode, t = T-1, T-2, ..., 0 (iterate backwards, Every-Visit MC):
            G ← r_{t+1} + γG
            Append G to Returns(s_t, a_t) // Or: Returns(s_t,a_t) ← Returns(s_t,a_t) + G
            Num(s_t,a_t) ← Num(s_t,a_t) + 1
            Policy evaluation (update q-value for (s_t, a_t)):
                q(s_t, a_t) ← average(Returns(s_t, a_t)) // Or: q(s_t,a_t) ← Returns(s_t,a_t) / Num(s_t,a_t)
            Policy improvement (update policy for state s_t to be ε-greedy w.r.t new q-values):
                Let a* = argmaxₐ q(s_t, a)
                For all actions a' ∈ A(s_t):
                    if a' = a*:
                        π(a'|s_t) = 1 - ε + ε/|A(s_t)| // Or slide 36 formula
                    else:
                        π(a'|s_t) = ε/|A(s_t)|       // Or slide 36 formula
    ```
    *(这里的策略改进部分应该使用幻灯片第36页的公式，为保持一致性)*
    $\pi(a|s\_t) = \begin{cases} 1 - \epsilon \frac{|A(s\_t)|-1}{|A(s\_t)|} & \text{if } a = a^\* \\ \frac{\epsilon}{|A(s\_t)|} & \text{if } a \neq a^\* \end{cases}$

*   **例子：展示 $\epsilon$-greedy 的探索能力** (参考幻灯片第38-39页)
    *   **情况1: $\epsilon = 1$ (策略为均匀随机分布，探索能力最强)**
        *   一个 agent 根据此策略在环境中探索。
        *   **100 步**: 探索到不少状态。
        *   **1000 步**: 基本所有状态及其相应动作都被探索到。
        *   **10000 步**: 探索次数更多。
        *   **100万步**: (图d) 横轴是状态-动作对索引 (共25个状态 x 5个动作/状态 = 125个 s-a pair)，纵轴是被访问次数。即使只有一个 episode，每个 s-a pair 都被访问了成千上万次 (约7000-8000次)。
        *   **视频演示**: agent 从一个点出发，在均匀随机策略下，最终能探索到地图的各个角落和各种动作。
        *   **结论**: 当 $\epsilon$ 较大时，探索性强，不需要 Exploring Starts。一个足够长的 episode 就能覆盖所有 $(s,a)$。
    *   **情况2: $\epsilon$ 较小 (探索能力较弱)**
        *   **100步, 1000步, 10000步**: 访问的状态较少，有些动作未被访问。
        *   **100万步**: (图d) 访问次数呈现非常不均匀的分布。有的 s-a pair 被访问次数极多，有的较少。但相比纯贪心策略，仍有一定探索能力，所有 s-a pair 还是被访问到了（只要 $\epsilon > 0$ 且 episode 足够长）。
*   **例子：演示 MC $\epsilon$-Greedy 算法** (参考幻灯片第43页)
    *   **设置**: $r\_{boundary}=-1, r\_{forbidden}=-10, r\_{target}=1, \gamma=0.9$。
    *   **每次迭代**:
        1.  使用当前 $\epsilon$-greedy 策略生成一个**非常长 (100万步)**的 episode。
        2.  使用这一个 episode 更新所有状态-动作对的价值及策略。
    *   **结果**:
        *   **(a) 初始策略**: 例如均匀随机。
        *   **(b) 第一次迭代后**: 策略有所改善，但仍不理想（例如，在某些点会保持不动）。
        *   **(c) 第二次迭代后**: 策略收敛到该 $\epsilon$ 下的**最优 $\epsilon$-greedy 策略**。
            *   如果只看概率最大的动作，策略相对合理，能从任何点到达目标。
            *   但它可能不是真正的最优策略（例如，可能会穿过障碍物，因为它为了探索，会以小概率选择次优动作，如果次优动作导致穿过障碍物但整体期望仍满足 $\epsilon$-greedy 最优，就会这样）。真正的最优策略应绕过障碍物。
*   **最优性 (Optimality) vs 探索 (Exploration)** (参考幻灯片第44-47页)
    *   与贪心策略相比：
        *   **$\epsilon$-greedy 的优点**: 当 $\epsilon$ 较大时，有强探索能力。不需要 Exploring Starts 条件。
        *   **$\epsilon$-greedy 的缺点**: 通常不是全局最优的。它只在所有 $\epsilon$-greedy 策略集合 $\Pi_\epsilon$ 中是最优的。
        *   $\epsilon$ 不能太大。可以使用衰减的 $\epsilon$ (decaying $\epsilon$)：开始时 $\epsilon$ 较大以鼓励探索，随时间推移逐渐减小 $\epsilon$ 以加强利用，最终趋向于贪心策略。
    *   **例子：$\epsilon$ 对状态价值的影响 (Optimality)** (幻灯片45)
        *   **$\epsilon = 0$ (最优贪心策略)**: 得到一组状态价值 $v^\*(s)$。目标态价值最高。
        *   **$\epsilon = 0.1$ (与最优贪心策略一致的 $\epsilon$-greedy 策略)**: 即其贪心部分与 $\epsilon=0$ 的策略相同。其状态价值 $v^{\pi_\epsilon}(s)$ 比 $v^\*(s)$ 要小。因为有 $\epsilon$ 的概率会选择次优动作。
        *   **$\epsilon = 0.2, \epsilon = 0.5$**: 随着 $\epsilon$ 增加，状态价值变得更小。
        *   **目标态的价值为何为负？** 当 $\epsilon$ 较大时，在目标态有较高概率执行随机动作，可能移出目标区域进入惩罚区域，导致长期期望回报为负。
    *   **例子：最优 $\epsilon$-greedy 策略的一致性 (Consistency)** (幻灯片46)
        *   **目标**: 我们希望通过 MC $\epsilon$-Greedy 算法学到的最优 $\epsilon$-greedy 策略 $\pi^\*_\epsilon$，其对应的贪心策略（即从 $q^\*_\epsilon(s,a)$ 中选择 $\arg\max\_a q^\*_\epsilon(s,a)$）与真正的全局最优贪心策略 $\pi^\*$ 相同。
        *   **$\epsilon = 0$ (基准，最优贪心策略 $\pi^\*$及其状态价值)**
        *   **MC $\epsilon$-Greedy 算法，$\epsilon = 0.1$**:
            *   学到的最优 $\epsilon$-greedy 策略 $\pi^\*\_{0.1}$，其“贪心部分”与 $\pi^\*$ 是一致的。这意味着，如果在使用 $\pi^\*\_{0.1}$ 收集数据并学习其 $q$-values 后，我们切换到纯粹的贪心策略（即令 $\epsilon=0$），那么得到的策略就是真正的最优策略 $\pi^\*$。
        *   **MC $\epsilon$-Greedy 算法，$\epsilon = 0.2$**:
            *   学到的最优 $\epsilon$-greedy 策略 $\pi^\*\_{0.2}$，其“贪心部分”可能与 $\pi^\*$ **不一致**。例如图中 $s\_5$ 处，$\pi^\*$ 向右，$\pi^\*\_{0.2}$ 的贪心部分向上。
        *   **MC $\epsilon$-Greedy 算法，$\epsilon = 0.5$**:
            *   同样，学到的最优 $\epsilon$-greedy 策略 $\pi^\*\_{0.5}$ 的“贪心部分”也与 $\pi^\*$ 不一致。
        *   **结论**:
            *   如果想用 MC $\epsilon$-Greedy 算法找到接近最优的（贪心）策略，$\epsilon$ 的选择很重要，不能太大。
            *   实际中常用技巧：**$\epsilon$ 衰减**。开始时 $\epsilon$ 较大，保证充分探索；然后逐渐减小 $\epsilon$ (趋向于0)，使得策略逐渐收敛到最优贪心策略。这满足 GLIE (Greedy in the Limit with Infinite Exploration) 条件，可以保证收敛到最优策略。
            *   **为什么不一致？** 以目标态为例，如果 $\epsilon$ 较大，在目标态停留时，它会以一定概率执行非“停留”动作，可能进入惩罚区。为了最大化 $\epsilon$-greedy下的回报，它可能学会一种策略，其贪心部分是在目标态“逃离”（如果周围惩罚很大），这与真正的最优贪心策略（在目标态停留）不一致。

---

**本次课总结 (Summary)**

*   **关键点**:
    *   通过蒙特卡洛方法进行均值估计 (Mean estimation by Monte Carlo methods)。
    *   三个算法：
        *   **MC Basic**: 核心思想的展示。
        *   **MC Exploring Starts**: 提高数据和更新效率，但有 Exploring Starts 假设。
        *   **MC $\epsilon$-Greedy**: 去除 Exploring Starts 假设，引入 $\epsilon$-greedy 策略。
    *   三个算法之间的关系：逐步改进和完善。
    *   $\epsilon$-greedy 策略的最优性与探索性之间的权衡。
