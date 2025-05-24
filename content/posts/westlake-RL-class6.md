---
title: "西湖大学 RL 第六课：随机近似与随机梯度下降" 
date: 2025-05-20T10:00:00+16:00 
draft: false # 设置为 false 来发布，true 则为草稿，不会显示在最终网站上
authors: ["唐豆"] # 作者
tags: ["深度学习", "强化学习"] # 标签
categories: ["学习"] 
math: true
summary: "西湖大学强化学习第六讲随机近似与随机梯度下降。" 

ShowToc: true
TocOpen: false 
---
## **第六讲：随机近似与随机梯度下降 (Lecture 6: Stochastic Approximation and Stochastic Gradient Descent)**

**主讲人：赵世钰 (Shiyu Zhao)**
**机构：人工智能系, 西湖大学 (Department of Artificial Intelligence, Westlake University)**

---

### **课程大纲回顾 (Course Outline Review)**

本次课程处于强化学习知识体系中的一个承上启下的位置：
*   **已学内容**：基础概念、贝尔曼方程、贝尔曼最优方程、值迭代与策略迭代（有模型）、蒙特卡洛方法（无模型）。
*   **当前讲座 (第六章)**：**随机近似 (Stochastic Approximation, SA)**。
*   **后续内容**：时序差分方法 (TD learning)、值函数方法、策略梯度方法、行动者-评论家方法。

---

### **引言 (Introduction)**

**1. 学习本讲的动机 (Why this lecture?)**

*   **知识鸿沟**：在学习了蒙特卡洛方法之后，直接学习时序差分 (TD) 学习可能会让初学者感到困惑。TD 算法的设计思想和表达方式与之前的算法有显著不同，学生可能会不理解其设计原理和有效性。
*   **承上启下**：本讲旨在通过介绍随机近似 (SA) 算法来填补这个知识鸿沟。
*   **核心关联**：
    *   时序差分算法可以被视为一种特殊的 SA 算法。理解 SA 将有助于更容易地理解 TD 算法。
    *   随机梯度下降 (Stochastic Gradient Descent, SGD) 也是一个重要的 SA 算法，在机器学习和强化学习中有广泛应用。
*   **建议**：讲者强烈建议学习本讲内容，因为它对理解后续的 TD 学习非常有帮助，并且本讲介绍的数学工具（如 SGD）本身也具有很高的实用价值。

**2. 本讲主要内容概要**

*   通过介绍基本的随机近似 (SA) 算法来为后续学习做准备。
*   理解 SA 如何帮助理解 TD 算法。
*   学习随机梯度下降 (SGD) 算法。

---

### **本讲详细大纲 (Detailed Outline for This Lecture)**

1.  **启发性示例 (Motivating examples)**
    *   主要关注均值估计 (Mean Estimation)。
2.  **罗宾斯-蒙罗算法 (Robbins-Monro algorithm, RM 算法)**
    *   算法描述 (Algorithm description)
    *   示例说明 (Illustrative examples)
    *   收敛性分析 (Convergence analysis)
    *   在均值估计中的应用 (Application to mean estimation)
3.  **随机梯度下降 (Stochastic gradient descent, SGD)**
    *   算法描述 (Algorithm description)
    *   示例与应用 (Examples and application)
    *   收敛性分析 (Convergence analysis)
    *   收敛模式 (Convergence pattern)
    *   批量梯度下降 (BGD)、小批量梯度下降 (MBGD) 和随机梯度下降 (SGD) 的比较。
4.  **总结 (Summary)**

---

### **1. 启发性示例：再谈均值估计 (Motivating example: mean estimation, again)**

**1.1. 均值估计问题回顾**

*   **问题**：估计一个随机变量 $X$ 的期望 $E[X]$。
*   **方法**：收集一系列独立同分布 (iid) 的样本 $\{x_i\}_{i=1}^{N}$。
*   **蒙特卡洛估计**：使用样本均值 $\bar{x}$ 来近似 $E[X]$：
    $$
    E[X] \approx \bar{x} := \frac{1}{N} \sum_{i=1}^{N} x_i.
    $$
*   **收敛性**：根据大数定律，当 $N \to \infty$ 时，$\bar{x} \to E[X]$。
*   **重要性**：强化学习中的许多量（如状态值、动作值、梯度）都定义为期望值，因此均值估计非常关键。

**1.2. 计算均值 $\bar{x}$ 的方法**

*   **方法一：批处理 (Batch processing)**
    *   收集所有 $N$ 个样本后，一次性计算总和再除以 $N$。
    *   **缺点**：如果样本是随时间逐个产生的，必须等待所有样本都收集完毕才能计算。
*   **方法二：增量/迭代计算 (Incremental/Iterative processing)**
    *   可以避免上述缺点，每当新样本到达时，都可以更新当前的均值估计。
    *   **思路**：来一个样本，计算一次；再来一个，再更新一次。这样效率更高。

**1.3. 增量计算均值的推导**

*   令 $w_{k+1}$ 表示前 $k$ 个样本的均值：
    $$
    w_{k+1} = \frac{1}{k} \sum_{i=1}^{k} x_i, \quad k=1,2,...
    $$
    (注意：讲者在此处使用 $w_{k+1}$ 代表前 $k$ 个样本的均值，是为了后续迭代公式形式上的简洁。)
*   类似地，前 $k-1$ 个样本的均值为：
    $$
    w_k = \frac{1}{k-1} \sum_{i=1}^{k-1} x_i, \quad k=2,3,...
    $$
    这意味着 $\sum_{i=1}^{k-1} x_i = (k-1)w_k$。
*   $w_{k+1}$ 可以用 $w_k$ 表示：
    $$
    \begin{aligned}
    w_{k+1} &= \frac{1}{k} \sum_{i=1}^{k} x_i \\
            &= \frac{1}{k} \left( \sum_{i=1}^{k-1} x_i + x_k \right) \\
            &= \frac{1}{k} ((k-1)w_k + x_k) \\
            &= \frac{k-1}{k}w_k + \frac{1}{k}x_k \\
            &= w_k - \frac{1}{k}w_k + \frac{1}{k}x_k \\
            &= w_k - \frac{1}{k}(w_k - x_k).
    \end{aligned}
    $$
*   **迭代算法**：
    $$
    w_{k+1} = w_k - \frac{1}{k}(w_k - x_k).
    $$
    这个公式表示，新的均值估计 $w_{k+1}$ 是在旧的估计 $w_k$ 的基础上，根据新样本 $x_k$ 与旧估计的差异 $(w_k - x_k)$ 进行调整。调整的幅度是 $1/k$。

**1.4. 验证增量计算**

*   设 $w_1$ 为某种初始估计 (或者，如果我们严格按照定义，当 $k=1$ 时，即第一个样本 $x_1$ 到来时，均值估计应该是 $x_1$)。
    *   如果将迭代公式的 $k$ 理解为已处理的样本数，并且 $w_1$ 初始化为 $x_1$ (即，当收到第一个样本 $x_1$ 时，均值估计就是 $x_1$):
        *   $w_1 = x_1$
        *   当第二个样本 $x_2$ 到来时 (此时迭代步数为 $k=2$ 来更新，得到 $w_2$ 代表前两个样本的均值，公式中 $k$ 应该用 $2$):
            $w_2 = w_1 - \frac{1}{2}(w_1 - x_2) = x_1 - \frac{1}{2}(x_1 - x_2) = \frac{1}{2}(x_1 + x_2)$.
        *   当第三个样本 $x_3$ 到来时 (迭代步数 $k=3$):
            $w_3 = w_2 - \frac{1}{3}(w_2 - x_3) = \frac{1}{2}(x_1+x_2) - \frac{1}{3}\left(\frac{1}{2}(x_1+x_2) - x_3\right) = \frac{1}{3}(x_1+x_2+x_3)$.
    *   字幕中的验证过程 (将 $k$ 理解为迭代序号，从 $k=1$ 开始，使用 $x_k$ 作为第 $k$ 个样本):
        *   $w_1 = x_1$ (初始化)
        *   $w_2 = w_1 - \frac{1}{1}(w_1 - x_1) = x_1$ (使用 $x_1$ 更新，得到 $w_2$ 是 $x_1$)
        *   $w_3 = w_2 - \frac{1}{2}(w_2 - x_2) = x_1 - \frac{1}{2}(x_1 - x_2) = \frac{1}{2}(x_1+x_2)$ (使用 $x_2$ 更新，得到 $w_3$ 是前两个样本的均值)
        *   $w_4 = w_3 - \frac{1}{3}(w_3 - x_3) = \frac{1}{2}(x_1+x_2) - \frac{1}{3}(\frac{1}{2}(x_1+x_2)-x_3) = \frac{1}{3}(x_1+x_2+x_3)$ (使用 $x_3$ 更新，得到 $w_4$ 是前三个样本的均值)
    *   因此，迭代公式 $w_{k+1} = w_k - \frac{1}{k}(w_k - x_k)$ (其中 $x_k$ 是第 $k$ 个新样本，而 $w_k$ 是前 $k-1$ 个样本的均值， $w_{k+1}$ 是前 $k$ 个样本的均值)，正确地计算了样本均值。

**1.5. 增量算法的备注**

*   **优点**：
    *   **增量性**：每当收到一个新样本，就可以立即更新均值估计。
    *   **即时可用**：更新后的均值可以立即用于其他计算或决策。
    *   **渐进改进**：初期由于样本少，估计不准确 ($w_k \neq E[X]$)，但随着样本增多 ($k \to \infty$)，估计会逐渐改进并收敛到 $E[X]$ ($w_k \to E[X]$)。所谓“有总比没有强”。

**1.6. 推广的增量算法**

*   考虑更一般的形式：
    $$
    w_{k+1} = w_k - \alpha_k(w_k - x_k),
    $$
    其中 $\alpha_k > 0$ 是一个正的学习率 (learning rate) 或步长 (step size)，取代了原先的 $1/k$。
*   **问题**：这个推广的算法是否仍然收敛到 $E[X]$？
*   **答案**：是的，如果步长序列 $\{\alpha_k\}$ 满足某些温和条件 (稍后会详细介绍)。
*   **联系**：
    *   这种形式的算法是一种特殊的**随机近似 (SA)** 算法。
    *   它也是一种特殊的**随机梯度下降 (SGD)** 算法。
    *   下一讲的**时序差分 (TD)** 算法也将具有类似但更复杂的表达式。

---

### **2. 罗宾斯-蒙罗算法 (Robbins-Monro algorithm, RM 算法)**

**2.1. 随机近似 (Stochastic Approximation, SA) 简介**

*   **定义**：SA 指的是一大类用于解决**求根问题 (root-finding problems)** 或**优化问题 (optimization problems)** 的**随机迭代算法 (stochastic iterative algorithms)**。
    *   **随机**：算法中会涉及到对随机变量的采样。
    *   **迭代**：通过一系列步骤逐步逼近解。
*   **SA 的优势 (相对于梯度下降等方法)**：
    *   **无模型 (Model-free)**：不需要知道目标函数的显式表达式。
    *   **无需导数**：也不需要知道目标函数的导数或梯度的表达式。
    *   讲者类比：就像不知道函数 $g(w)$ 具体长什么样，但可以输入 $w$ 并观察到一个带噪声的输出。

**2.2. 罗宾斯-蒙罗 (RM) 算法**

*   **地位**：SA 领域的开创性工作。
*   **与 SGD 的关系**：著名的 SGD 算法是 RM 算法的一种特殊形式。
*   **与均值估计的关系**：前面介绍的增量均值估计算法也是 RM 算法的特例。

**2.3. RM 算法 – 问题陈述 (Problem Statement)**

*   **目标**：找到方程 $g(w) = 0$ 的根 (root) $w^*$，其中 $w \in \mathbb{R}$ 是待求解的变量，$g: \mathbb{R} \to \mathbb{R}$ 是一个函数。
    *   为了简化，这里先考虑 $w$ 和 $g(w)$ 都是标量的情况。
*   **广泛性**：许多问题可以转化为求根问题。
    *   **优化问题示例**：最小化目标函数 $J(w)$。其一阶最优性条件是 $\nabla_w J(w) = 0$。令 $g(w) = \nabla_w J(w)$，就转化为求 $g(w)=0$ 的根。
    *   **一般方程**：形如 $g(w) = c$ (其中 $c$ 是常数) 的方程，可以通过定义新函数 $g'(w) = g(w) - c$ 来转化为 $g'(w) = 0$ 的形式。

**2.4. 求解 $g(w)=0$ 的方法**

*   **基于模型 (Model-based)**：如果 $g(w)$ 的表达式已知，可以使用多种数值算法求解。
*   **无模型 (Model-free)**：如果 $g(w)$ 的表达式未知（例如，$g(w)$ 由一个黑箱函数或神经网络表示），RM 算法提供了一种解决方案。
    *   讲者例子：神经网络的输入是 $w$，输出是 $y=g(w)$，我们不知道 $g(w)$ 的具体表达式，但想找到使输出 $y=0$ 的输入 $w$。

**2.5. RM 算法描述 (The Algorithm)**

*   RM 算法的迭代更新规则：
    $$
    w_{k+1} = w_k - a_k \tilde{g}(w_k, \eta_k), \quad k=1,2,3,...
    $$
    其中：
    *   $w_k$：根 $w^*$ 的第 $k$ 次估计。
    *   $a_k$：一个正常数，称为步长或学习率。
    *   $\tilde{g}(w_k, \eta_k)$：在 $w_k$ 处对 $g(w_k)$ 的**带噪声的观测 (noisy observation/measurement)**。
        *   $\tilde{g}(w_k, \eta_k) = g(w_k) + \eta_k$。
        *   $\eta_k$：观测噪声或误差，在第 $k$ 次迭代时产生。
        *   **为什么有噪声？** 例如，在均值估计中，$x_k$ 是 $E[X]$ 的一个随机样本，本身就带有随机性。

*   **算法依赖于数据而非模型**：
    *   输入序列：$\{w_k\}$ (我们尝试的值)。
    *   输出序列 (带噪声)：$\{\tilde{g}(w_k, \eta_k)\}$ (我们观测到的函数值)。
    *   将 $g(w)$ 视为一个黑箱：输入 $w$，得到带噪声的输出 $\tilde{g}(w, \eta)$。
    *   **核心思想**：没有模型（函数表达式），就需要数据（观测值）来驱动算法。

**2.6. RM 算法 – 示例说明 (Illustrative Examples)**

*   **手动求解示例**：求解 $g(w) = w - 10 = 0$。真实根 $w^* = 10$。
    *   设置：初始猜测 $w_1 = 20$，步长 $a_k \equiv 0.5$ (固定步长)，无观测误差 $\eta_k = 0$ (即 $\tilde{g}(w_k, \eta_k) = g(w_k)$)。
    *   **迭代过程**：
        1.  $k=1$: $w_1 = 20 \implies g(w_1) = 20 - 10 = 10$.
            $w_2 = w_1 - a_1 g(w_1) = 20 - 0.5 \times 10 = 15$.
        2.  $k=2$: $w_2 = 15 \implies g(w_2) = 15 - 10 = 5$.
            $w_3 = w_2 - a_2 g(w_2) = 15 - 0.5 \times 5 = 12.5$.
        3.  $k=3$: $w_3 = 12.5 \implies g(w_3) = 12.5 - 10 = 2.5$.
            $w_4 = w_3 - a_3 g(w_3) = 12.5 - 0.5 \times 2.5 = 11.25$.
        *   ...以此类推，$w_k \to 10$。

**2.7. RM 算法 – 收敛性质 (Convergence Properties)**

*   **问题**：RM 算法为什么能找到 $g(w)=0$ 的根？
*   **分析方法**：
    1.  通过一个说明性示例进行直观解释。
    2.  给出严格的收敛性分析（罗宾斯-蒙罗定理）。

*   **说明性示例 (无噪声情况)**：
    *   函数：$g(w) = \tanh(w-1)$。真实根 $w^* = 1$ (因为 $\tanh(0)=0$)。
    *   参数：初始值 $w_1 = 3$，步长 $a_k = 1/k$，无噪声 $\eta_k \equiv 0$。
    *   RM 算法此时为：$w_{k+1} = w_k - a_k g(w_k)$。
    *   **仿真结果**：$w_k$ 会逐渐收敛到真实根 $w^*=1$。
        *   $w_1=3$, $g(w_1) = \tanh(2) > 0$.
        *   $w_2 = w_1 - a_1 g(w_1) < w_1$.
        *   迭代点 $w_1, w_2, w_3, \dots$ 从 $w_1=3$ 开始，在 $g(w)$ 曲线上移动，逐步逼近 $w^*=1$。
    *   **直观解释**：$w_{k+1}$ 比 $w_k$ 更接近 $w^*$。
        *   若 $w_k > w^\*$ (例如 $w_1=3 > 1$)，且 $g(w)$ 在 $w^\*$ 附近单调递增，则 $g(w_k) > 0$。由于 $a_k > 0$，所以 $a_k g(w_k) > 0$。因此 $w_{k+1} = w_k - a_k g(w_k) < w_k$，使得 $w_{k+1}$ 向 $w^*$ 移动。
        *   若 $w_k < w^\*$，则 $g(w_k) < 0$。因此 $a_k g(w_k) < 0$。所以 $w_{k+1} = w_k - a_k g(w_k) > w_k$，使得 $w_{k+1}$ 也向 $w^*$ 移动。
        *   讲者强调，关键在于每次调整的幅度，即 $-a_k g(w_k)$。

**2.8. 罗宾斯-蒙罗定理 (Robbins-Monro Theorem) – 严格收敛性分析**

*   **定理内容**：在罗宾斯-蒙罗算法 $w_{k+1} = w_k - a_k \tilde{g}(w_k, \eta_k)$ 中，如果满足以下条件，则 $w_k$ **以概率1 (with probability 1, w.p.1)** 收敛到满足 $g(w^*) = 0$ 的根 $w^*$。
    *   **概率1收敛的含义**：由于算法涉及随机变量的采样 ($\eta_k$)， $w_k$ 本身也是一个随机变量序列。其收敛不是常规意义上的确定性收敛，而是概率意义上的收敛。
    *   **条件**：
        1.  **关于函数 $g(w)$ 的条件**：对所有 $w$，存在常数 $c_1, c_2$ 使得 $0 < c_1 \le \nabla_w g(w) \le c_2$。
            *   $\nabla_w g(w) > c_1 > 0$：要求 $g(w)$ (至少在根附近) 是**单调递增**的。这保证了根 $w^*$ 的存在性和唯一性。
            *   $\nabla_w g(w) \le c_2$：要求 $g(w)$ 的梯度 (或导数) 是**有上界的**，不能增长过快。
            *   **与优化问题的联系**：如果 $g(w) = \nabla_w J(w)$ (即 RM 用于优化问题)，则 $\nabla_w g(w) = \nabla_w^2 J(w)$ (Hessian 矩阵)。条件 $\nabla_w^2 J(w) > 0$ (正定) 意味着 $J(w)$ 是（严格）**凸函数 (convex function)**，这使得梯度下降能找到全局最优。
            *   讲者补充：如果 $g(w)$ 是递减的，此定理不适用。

        2.  **关于步长序列 $\{a_k\}$ 的条件** (非常重要，TD learning 中也会遇到)：
            $$
            \sum_{k=1}^{\infty} a_k = \infty \quad \text{且} \quad \sum_{k=1}^{\infty} a_k^2 < \infty
            $$
            *   $\sum_{k=1}^{\infty} a_k^2 < \infty$ (平方和收敛)：这意味着 $a_k \to 0$ 当 $k \to \infty$。
                *   **重要性**：因为 $w_{k+1} - w_k = -a_k \tilde{g}(w_k, \eta_k)$。如果 $w_k$ 要收敛，那么相邻两项的差 $w_{k+1} - w_k$ 必须趋于0。如果 $a_k \to 0$ 且 $\tilde{g}$ 有界，则此项趋于0。如果 $w_k \to w^*$，则 $g(w_k) \to 0$，$\tilde{g}(w_k, \eta_k)$ 主要由噪声 $\eta_k$ 主导，乘以趋于0的 $a_k$ 可以抑制噪声影响。
            *   $\sum_{k=1}^{\infty} a_k = \infty$ (和发散)：这意味着 $a_k$ **不能过快地收敛到零**。
                *   **重要性**：考虑累加 $w_{k+1} - w_k$：
                    $w_N - w_1 = \sum_{k=1}^{N-1} (w_{k+1} - w_k) = -\sum_{k=1}^{N-1} a_k \tilde{g}(w_k, \eta_k)$.
                    如果 $w_N \to w^*$，则 $w^* - w_1 = -\sum_{k=1}^{\infty} a_k \tilde{g}(w_k, \eta_k)$。
                    如果 $\sum a_k < \infty$ (收敛过快)，那么右边的和可能是一个有界值。这意味着 $w^* - w_1$ 也是有界的，这与 $w_1$ 可以任意选择（可能离 $w^*$ 很远）相矛盾。因此，$\sum a_k = \infty$ 确保算法有足够的能力从任意初始点 $w_1$ 到达 $w^*$，能够克服初始误差和累积噪声。
            *   **典型的满足条件的序列**：$a_k = 1/k$。
                *   $\sum_{k=1}^{\infty} (1/k)$ 是调和级数，发散到 $\infty$。($\lim_{n\to\infty} (\sum_{k=1}^{n} \frac{1}{k} - \ln n) = \kappa \approx 0.577$，欧拉-马歇罗尼常数)。
                *   $\sum_{k=1}^{\infty} (1/k^2) = \pi^2/6 < \infty$ (巴塞尔问题)。
            *   **实际应用中的调整**：理论上 $a_k \to 0$。但在实践中，有时会选择一个很小的常数作为 $a_k$，而不是让它严格趋于0。因为如果 $a_k$ 变得太小，后期样本对参数更新的贡献会微乎其微，算法可能学习过慢或对环境变化不敏感。

        3.  **关于噪声序列 $\{\eta_k\}$ 的条件**：
            *   $E[\eta_k | \mathcal{H}\_k] = 0$：给定历史信息 $\mathcal{H}\_k = \{w_1, ..., w_k, \eta\_1, ..., \eta\_{k-1}\}$ (即到第 $k$ 步为止的所有信息)，噪声 $\eta\_k$ 的条件期望为0。这意味着噪声在平均意义上不会系统性地将估计推离真值（即噪声是**无偏的**）。
            *   $E[\eta_k^2 | \mathcal{H}_k] < \infty$ (或者 $E[\eta_k^2] < C$ for some constant $C$)：噪声的条件方差有界。这意味着噪声不会无限大。
            *   **常见特例**：如果 $\{\eta_k\}$ 是独立同分布 (iid) 的随机序列，满足 $E[\eta_k] = 0$ 和 $E[\eta_k^2] < \infty$，则此条件满足。噪声 $\eta_k$ 不需要是高斯分布。

**2.9. RM 算法 – 应用于均值估计 (Application to Mean Estimation)**

*   回顾均值估计算法：
    $$
    w_{k+1} = w_k - \alpha_k(w_k - x_k)
    $$
    (注意：这里与幻灯片中的 $w_{k+1} = w_k + \alpha_k(x_k - w_k)$ 符号上等价，因为可以写成 $w_{k+1} = w_k - \alpha_k(w_k - x_k)$。)
*   **目标**：证明这个均值估计算法是 RM 算法的一个特例，从而其收敛性可以由 RM 定理保证。

*   **步骤**：
    1.  **定义 $g(w)$**：令 $g(w) := w - E[X]$。我们的目标是求解 $g(w) = 0$，其根 $w^* = E[X]$。
        *   问题：我们通常不知道 $E[X]$，所以 $g(w)$ 的表达式实际上是未知的。
    2.  **定义带噪声的观测 $\tilde{g}(w_k, x_k)$**：在第 $k$ 步，我们有一个样本 $x_k$。我们可以构造观测：
        $$
        \tilde{g}(w_k, x_k) := w_k - x_k.
        $$
        这个观测可以写成 $g(w_k) + \eta_k$ 的形式：
        $$
        \begin{aligned}
        \tilde{g}(w_k, x_k) &= w_k - x_k \\
                           &= (w_k - E[X]) + (E[X] - x_k) \\
                           &= g(w_k) + \eta_k,
        \end{aligned}
        $$
        其中噪声 $\eta_k = E[X] - x_k$。
        *   验证噪声条件：如果 $x_k$ 是 $X$ 的 iid 样本，那么 $E[\eta_k] = E[E[X] - x_k] = E[X] - E[x_k] = E[X] - E[X] = 0$ (假设 $x_k$ 是对 $X$ 的无偏采样)。方差 $Var(\eta_k) = Var(E[X] - x_k) = Var(x_k) = Var(X)$，如果 $X$ 的方差有限，则噪声方差有界。
    3.  **应用 RM 算法**：使用上述定义的 $g(w)$ 和 $\tilde{g}(w_k, x_k)$，RM 算法的更新规则为：
        $$
        w_{k+1} = w_k - \alpha_k \tilde{g}(w_k, x_k) = w_k - \alpha_k (w_k - x_k).
        $$
        这正是我们开始时介绍的增量均值估计算法。
    4.  **结论**：均值估计算法是 RM 算法的一个特例。因此，如果步长 $\alpha_k$ 满足 RM 定理的条件 (如 $\alpha_k = 1/k$)，并且 $g(w) = w - E[X]$ 的导数 $\nabla_w g(w) = 1$ (满足 $0 < c_1 \le 1 \le c_2$)，则 $w_k$ 会以概率1收敛到 $E[X]$。

**2.10. 德沃雷茨基收敛定理 (Dvoretzky's Convergence Theorem) (可选内容)**

*   **定理内容**：考虑一个更一般的随机过程：
    $$
    w_{k+1} = (1 - \alpha_k) w_k + \beta_k \eta_k,
    $$
    其中 $\{\alpha_k\}, \{\beta_k\}, \{\eta_k\}$ 是随机序列，$\alpha_k \ge 0, \beta_k \ge 0$。
    如果满足以下条件，$w_k$ 将以概率 1 收敛到零：
    1.  $\sum_{k=1}^{\infty} \alpha_k = \infty$, $\sum_{k=1}^{\infty} \alpha_k^2 < \infty$; $\sum_{k=1}^{\infty} \beta_k^2 < \infty$ 一致地以概率1成立。
    2.  $E[\eta_k | \mathcal{H}_k] = 0$ 且 $E[\eta_k^2 | \mathcal{H}_k] \le C$ (有界) 以概率1成立。
*   **意义**：
    *   比 RM 定理更一般。
    *   可用于证明 RM 定理。
    *   可用于直接分析均值估计问题。
    *   其扩展版本可用于分析 Q-learning 和 TD 学习算法的收敛性。

---

### **3. 随机梯度下降 (Stochastic Gradient Descent, SGD)**

**3.1. SGD 简介**

*   SGD 是机器学习和强化学习中广泛使用的优化算法。
*   **与 RM 的关系**：SGD 是 RM 算法的一种特殊情况。
*   **与均值估计的关系**：均值估计算法也是 SGD 算法的一种特殊情况。
*   三者关系密切：均值估计 $\subset$ SGD $\subset$ RM。

**3.2. SGD – 问题设置 (Problem Setting)**

*   **目标**：解决以下优化问题：
    $$
    \min_w J(w) = E[f(w,X)]
    $$
    其中：
    *   $w$：待优化的参数 (可以是标量或向量)。
    *   $X$：一个随机变量 (其概率分布通常未知，但可以从中采样)。
    *   $f(w,X)$：一个标量函数。
    *   $E[\cdot]$：期望是关于随机变量 $X$ 的。

**3.3. 求解 $\min_w E[f(w,X)]$ 的方法**

*   **方法1：梯度下降 (Gradient Descent, GD)**
    *   更新规则：
        $$
        w_{k+1} = w_k - \alpha_k \nabla_w J(w_k) = w_k - \alpha_k \nabla_w E[f(w_k, X)]
        $$
    *   假设梯度和期望可以交换顺序 (通常成立)：
        $$
        w_{k+1} = w_k - \alpha_k E[\nabla_w f(w_k, X)]
        $$
        这里的 $E[\nabla_w f(w_k, X)]$ 称为**真实梯度 (true gradient)**。
    *   **缺点**：计算期望 $E[\nabla_w f(w_k, X)]$ 需要知道 $X$ 的完整概率分布，或者需要对所有可能的 $X$ 值进行积分/求和，这在实际中往往不可行。

*   **方法2：批量梯度下降 (Batch Gradient Descent, BGD)**
    *   用蒙特卡洛方法近似真实梯度：使用一批 $n$ 个从 $X$ 中独立同分布采样的样本 $\{x\_i\}_{i=1}^n$：
        $$
        E[\nabla\_w f(w\_k, X)] \approx \frac{1}{n} \sum\_{i=1}^{n} \nabla_w f(w\_k, x\_i)
        $$
    *   BGD 更新规则：
        $$
        w_{k+1} = w_k - \alpha_k \left( \frac{1}{n} \sum_{i=1}^{n} \nabla_w f(w_k, x_i) \right)
        $$
    *   **缺点**：在每次迭代 (更新 $w_k$) 时，都需要处理整个数据集（所有 $n$ 个样本）来计算梯度估计，如果 $n$ 非常大，计算成本会很高。

*   **方法3：随机梯度下降 (Stochastic Gradient Descent, SGD)**
    *   SGD 更新规则：
        $$
        w_{k+1} = w_k - \alpha_k \nabla_w f(w_k, x_k)
        $$
    *   **核心思想**：在每次迭代时，只使用**一个**随机抽取的样本 $x_k$ 来计算梯度 $\nabla_w f(w_k, x_k)$。这个梯度称为**随机梯度 (stochastic gradient)**。
    *   **与 GD 比较**：用随机梯度 $\nabla_w f(w_k, x_k)$ 替代真实梯度 $E[\nabla_w f(w_k, X)]$。
    *   **与 BGD 比较**：相当于 BGD 中批大小 $n=1$ 的情况。
    *   **优点**：计算效率高，每次迭代成本低。
    *   **潜在问题**：随机梯度是对真实梯度的有噪声估计，可能导致收敛路径不稳定。

**3.4. SGD – 示例与应用 (Example and Application)**

*   **示例问题**：最小化均方误差类型的目标函数：
    $$
    \min_w J(w) = E\left[\frac{1}{2}\|w-X\|^2\right]
    $$
    其中 $f(w,X) = \frac{1}{2}\|w-X\|^2$。
    假设 $w$ 和 $X$ 是向量，$\|\cdot\|^2$ 是欧氏范数的平方。
    该函数 $f$ 关于 $w$ 的梯度是：
    $$
    \nabla_w f(w,X) = w-X.
    $$

*   **练习 1：证明最优解是 $w^\* = E[X]$**
    *   **解答**：最优解 $w^\*$ 满足 $\nabla_w J(w^\*) = 0$。
        $$
        \nabla_w J(w) = \nabla_w E\left[\frac{1}{2}\|w-X\|^2\right] = E\left[\nabla_w \frac{1}{2}\|w-X\|^2\right] = E[w-X].
        $$
        令 $E[w-X] = 0$，由于 $w$ 不是随机变量 (在期望算子 $E_X[\cdot]$ 中)，$E[w] = w$。
        所以 $w - E[X] = 0 \implies w^* = E[X]$。
        *   **意义**：均值估计问题（找到 $E[X]$）可以被表述为一个优化问题（最小化 $J(w)$）。

*   **练习 2：写出解决此问题的 GD 算法**
    *   **解答**：
        $$
        \begin{aligned}
        w_{k+1} &= w_k - \alpha_k \nabla_w J(w_k) \\
                &= w_k - \alpha_k E[\nabla_w f(w_k, X)] \\
                &= w_k - \alpha_k E[w_k - X].
        \end{aligned}
        $$

*   **练习 3：写出解决此问题的 SGD 算法**
    *   **解答**：
        $$
        w_{k+1} = w_k - \alpha_k \nabla_w f(w_k, x_k) = w_k - \alpha_k (w_k - x_k).
        $$
        其中 $x_k$ 是在第 $k$ 次迭代时从 $X$ 中随机抽取的一个样本。
    *   **观察**：这个 SGD 算法与本讲开始时介绍的增量均值估计算法形式完全相同。
    *   **结论**：因此，（增量）均值估计算法是 SGD 算法的一个特例。

**3.5. SGD – 收敛性 (Convergence of SGD)**

*   **SGD 的思想回顾**：
    GD 使用真实梯度: $w_{k+1} = w_k - \alpha_k E[\nabla_w f(w_k, X)]$
    SGD 使用随机梯度: $w_{k+1} = w_k - \alpha_k \nabla_w f(w_k, x_k)$
*   **问题**：由于随机梯度 $\nabla_w f(w_k, x_k)$ 是真实梯度 $E[\nabla_w f(w_k, X)]$ 的有噪声估计（通常 $\nabla_w f(w_k, x_k) \neq E[\nabla_w f(w_k, X)]$），SGD 是否仍能收敛到最优解 $w^*$？
*   **随机梯度与真实梯度的关系**：
    $$
    \nabla_w f(w_k, x_k) = E[\nabla_w f(w_k, X)] + \underbrace{\left( \nabla_w f(w_k, x_k) - E[\nabla_w f(w_k, X)] \right)}_{\text{噪声 } \eta_k}
    $$
    如果 $E[\nabla_w f(w_k, x_k) | w_k] = E_X[\nabla_w f(w_k, X) | w_k] = E[\nabla_w f(w_k, X)]$ （即随机梯度是真实梯度的无偏估计），那么噪声项 $E[\eta_k|w_k]=0$。

*   **证明 SGD 是 RM 算法的特例**：
    1.  **SGD 的目标**：最小化 $J(w) = E[f(w,X)]$。
    2.  **转化为求根问题**：优化问题的一阶最优条件是 $\nabla_w J(w) = 0$。令
        $$
        g(w) := \nabla_w J(w) = \nabla_w E[f(w,X)] = E[\nabla_w f(w,X)].
        $$
        则 SGD 的目标是找到 $g(w)=0$ 的根。
    3.  **定义带噪声的观测 $\tilde{g}(w_k, x_k)$**：在 SGD 中，我们使用的是随机梯度 $\nabla_w f(w_k, x_k)$。可以将其视为对 $g(w_k)$ 的带噪声观测：
        $$
        \tilde{g}(w_k, x_k) := \nabla_w f(w_k, x_k).
        $$
        则
        $$
        \tilde{g}(w\_k, x\_k) = \underbrace{E[\nabla_w f(w_k, X)]}\_{g(w\_k)} + \underbrace{\left( \nabla_w f(w\_k, x\_k) - E[\nabla_w f(w\_k, X)] \right)}_{\text{噪声 } \eta_k}.
        $$
    4.  **应用 RM 算法**：使用上述 $g(w)$ 和 $\tilde{g}(w_k, x_k)$，RM 算法的更新规则为：
        $$
        w_{k+1} = w_k - a_k \tilde{g}(w_k, x_k) = w_k - a_k \nabla_w f(w_k, x_k).
        $$
        (这里用 $a_k$ 对应 RM 的步长，与 SGD 中的 $\alpha_k$ 相同)。
    5.  **结论**：这正是 SGD 算法的更新规则。因此，SGD 是 RM 算法的一个特例。

*   **SGD 的收敛性定理**：由于 SGD 是 RM 算法的特例，其收敛性条件可以从 RM 定理推导出来。
    在 SGD 算法中，如果满足以下条件，$w_k$ 将以概率1收敛到满足 $\nabla_w E[f(w,X)] = 0$ 的根 $w^*$：
    1.  **关于目标函数 $f(w,X)$ 的条件**：存在常数 $c_1, c_2$ 使得 $0 < c_1 \le \nabla_w^2 f(w,X) \le c_2$ (对于所有 $w,X$ 成立，这里 $\nabla_w^2 f$ 指 $f$ 关于 $w$ 的 Hessian 矩阵，条件指其特征值有界且为正)。这实质上要求 $J(w)=E[f(w,X)]$ 是强凸 (strongly convex)的。
        *   讲者字幕解释：这意味着 $f$ 是严格凸的 (convexity condition)。如果 $w$ 是向量，上下界应该是矩阵。
    2.  **关于步长序列 $\{a_k\}$ (或 $\{\alpha_k\}$) 的条件**：
        $$
        \sum_{k=1}^{\infty} a_k = \infty \quad \text{且} \quad \sum_{k=1}^{\infty} a_k^2 < \infty.
        $$
    3.  **关于样本序列 $\{x_k\}$ 的条件**：$\{x_k\}_{k=1}^{\infty}$ 是从 $X$ 的分布中独立同分布 (iid) 抽取的。这保证了随机梯度是真实梯度的无偏估计，且噪声满足 RM 定理的条件。

**3.6. SGD – 收敛模式 (Convergence Pattern)**

*   **问题**：由于随机梯度是随机的，近似不准确，SGD 的收敛是缓慢的还是非常随机的（例如，会不会在接近最优解时剧烈震荡）？
*   **直观表现 (来自示例图)**：
    *   **远离最优解时**：SGD 的估计可以快速地向真实值的邻域移动，其行为类似于确定性的梯度下降 (GD)。
    *   **接近最优解时**：SGD 的估计会表现出一定的随机性（围绕最优解震荡），但总体趋势仍然是逐渐逼近真实值。
    *   *(图示描述：一个均值估计的例子，X在正方形区域均匀分布，真实均值为原点。SGD的路径从初始点开始，先是比较直接地移向原点，在原点附近开始出现抖动但仍在原点附近。)*

*   **为什么会有这样的模式？相对误差分析**：
    考虑随机梯度和真实梯度之间的**相对误差 $\delta_k$**：
    $$
    \delta_k := \frac{|\nabla_w f(w_k, x_k) - E[\nabla_w f(w_k, X)]|}{|E[\nabla_w f(w_k, X)]|}
    $$
    其中分子是绝对误差，分母是真实梯度的大小。
    可以证明 (推导可选)：
    $$
    \delta_k \le \frac{C}{|w_k - w^*|} \quad \text{或更准确地是} \quad \delta_k \le \frac{|\nabla_w f(w_k, x_k) - E[\nabla_w f(w_k, X)]|}{c|w_k - w^*|}
    $$
    其中 $c$ 是一个与目标函数二阶导数下界相关的正常数，$C$ 是某个常数。

*   **推导 (可选，来自幻灯片)**：
    1.  真实梯度的大小 $|E[\nabla_w f(w_k, X)]|$：
        由于 $E[\nabla_w f(w^*, X)] = \nabla_w J(w^*) = 0$ (因为 $w^*$ 是最优解)，
        $$
        |E[\nabla_w f(w_k, X)]| = |E[\nabla_w f(w_k, X)] - E[\nabla_w f(w^*, X)]|
        $$
        根据中值定理 (Mean Value Theorem)，对于函数 $h(w) = E[\nabla_w f(w, X)]$，有 $h(w_k) - h(w^*) = \nabla_w h(\tilde{w}_k) (w_k - w^*)$，其中 $\tilde{w}_k$ 在 $w_k$ 和 $w^*$ 之间。
        $\nabla_w h(w) = \nabla_w (E[\nabla_w f(w,X)]) = E[\nabla_w^2 f(w,X)]$.
        所以，
        $$
        |E[\nabla_w f(w_k, X)]| = |E[\nabla_w^2 f(\tilde{w}_k, X)(w_k - w^*)]|.
        $$
    2.  假设 $f$ 是严格凸的，使得其 Hessian 矩阵 $\nabla_w^2 f(w,X)$ 的最小特征值大于等于一个正常数 $c > 0$ (即 $\nabla_w^2 f(w,X) \ge cI$)。
        那么 $E[\nabla_w^2 f(\tilde{w}_k, X)]$ 也是一个正定矩阵，其最小特征值也大于等于某个 $c' \ge c$ (通常假设 $E[\nabla_w^2 f]$ 继承了 $\nabla_w^2 f$ 的性质)。
        因此，分母满足：
        $$
        |E[\nabla_w^2 f(\tilde{w}_k, X)(w_k - w^*)]| = |E[\nabla_w^2 f(\tilde{w}_k, X)] (w_k - w^*)| \ge c |w_k - w^*|.
        $$
        (这里假设 $E[\nabla_w^2 f(\tilde{w}_k, X)]$ 是一个标量 $c'$ 或其作用类似于标量乘法，或者 $c$ 是其最小特征值。)
    3.  代入相对误差 $\delta_k$ 的表达式：
        $$
        \delta_k \le \frac{|\nabla_w f(w_k, x_k) - E[\nabla_w f(w_k, X)]|}{c|w_k - w^*|}.
        $$

*   **对收敛模式的解释**：
    $$
    \delta_k \le \frac{\text{绝对梯度误差}}{c \cdot (\text{到最优解的距离 } |w_k - w^*|)}.
    $$
    *   当 $|w_k - w^*|$ **较大时** (离最优解较远)：分母较大，相对误差 $\delta_k$ 的上界较小。这意味着随机梯度 $\nabla_w f(w_k, x_k)$ 与真实梯度 $E[\nabla_w f(w_k, X)]$ 的方向和大小相对比较一致。因此，SGD 的行为类似于确定性的 GD，表现出较快的、方向较为明确的收敛。
    *   当 $|w_k - w^*|$ **较小时** (接近最优解)：分母较小，相对误差 $\delta_k$ 的上界可能较大（尽管绝对误差可能也变小了，但真实梯度本身也变小了）。这意味着随机梯度可能与真实梯度有较大偏差，导致 SGD 在 $w^*$ 的邻域表现出更多的随机性和震荡。

*   **SGD 的另一种问题表述 (讲者字幕补充)**
    *   有时 SGD 的目标函数被写为有限和的形式：$\min_w J(w) = \frac{1}{n} \sum_{i=1}^n f(w, x_i)$，其中 $\{x_i\}_{i=1}^n$ 是一组给定的数据点，不是随机变量的样本。
    *   **问题1**：这还是 SGD 吗？因为没有期望 $E[\cdot]$ 和随机变量 $X$。
    *   **问题2**：如果使用 $w_{k+1} = w_k - \alpha_k \nabla_w f(w, x_k)$，那么 $x_k$ 应该如何从数据集中选取？是按顺序，还是随机？
    *   **解答**：
        1.  **引入随机性**：可以手动引入一个随机变量 $X'$，它在给定的数据集 $\{x_i\}_{i=1}^n$ 上取值，且取每个 $x_i$ 的概率为 $1/n$ (均匀分布)。
        2.  那么原目标函数可以写成期望形式：$J(w) = E_{X'}[f(w, X')]$。
        3.  这样，使用 $w_{k+1} = w_k - \alpha_k \nabla_w f(w, x_k')$ (其中 $x_k'$ 是从 $\{x_i\}$ 中随机均匀抽取的一个样本) 就成为了标准的 SGD。
        4.  **结论**：应该从数据集中**随机均匀地**抽取样本 $x_k$ (有放回采样)，而不是按固定顺序。这样才能保证随机梯度是真实梯度（对于有限和目标函数而言）的无偏估计。

**3.7. BGD, MBGD, 和 SGD 的比较**

*   假设目标是最小化 $J(w) = E[f(w,X)]$，给定一组随机样本 $\{x_i\}_{i=1}^N$ (这里用 $N$ 表示总样本数，区别于BGD中的批大小 $n$ 或MBGD中的 $m$。但幻灯片中用 $n$ 指代总样本数，然后在BGD中也用 $n$ 作为批大小，即全批量)。我们统一理解为有一个大的数据集，大小为 $N_{total}$。

*   **批量梯度下降 (Batch Gradient Descent, BGD)**：
    $$
    w_{k+1} = w_k - \alpha_k \left( \frac{1}{N_{total}} \sum_{i=1}^{N_{total}} \nabla_w f(w_k, x_i) \right)
    $$
    *   每次迭代使用**所有** $N_{total}$ 个样本计算梯度。梯度估计最准确，最接近真实梯度。

*   **小批量梯度下降 (Mini-Batch Gradient Descent, MBGD)**：
    $$
    w_{k+1} = w_k - \alpha_k \left( \frac{1}{m} \sum_{j \in I_k} \nabla_w f(w_k, x_j) \right)
    $$
    *   $I_k$ 是从总样本中随机选取的、大小为 $m$ (mini-batch size, $1 < m < N_{total}$) 的一个子集。
    *   每次迭代使用 $m$ 个样本。是 BGD 和 SGD 的折中。

*   **随机梯度下降 (Stochastic Gradient Descent, SGD)**：
    $$
    w_{k+1} = w_k - \alpha_k \nabla_w f(w_k, x_k)
    $$
    *   $x_k$ 是在第 $k$ 次迭代时从总样本中随机抽取的一个样本 (相当于 $m=1$)。

*   **比较 MBGD 与 BGD 和 SGD**：
    *   **随机性**：SGD (m=1) > MBGD (m较小) > MBGD (m较大) > BGD ($m=N_{total}$)。随机性越小，收敛路径越平滑，但单次迭代计算量越大。
    *   **计算效率**：SGD (单次迭代最快) > MBGD > BGD (单次迭代最慢)。
    *   **MBGD 的角色**：
        *   如果 $m=1$，MBGD 变成 SGD。
        *   如果 $m=N_{total}$ (总样本数)：
            *   严格来说，MBGD (从 $N_{total}$ 个样本中有放回地抽取 $N_{total}$ 个) **不完全等同于** BGD (使用所有 $N_{total}$ 个样本各一次)。MBGD 可能会重复使用某些样本，而遗漏另一些。但如果MBGD是无放回抽取且$m=N_{total}$，则与BGD等价。通常，当 $m$ 接近 $N_{total}$ 时，其行为接近 BGD。
        *   MBGD 通过调整 $m$ 的大小，在梯度估计的准确性和计算效率之间取得平衡。

*   **示例说明 (均值估计问题)**：
    目标：计算给定 $N$ 个数 $\{x\_i\}_{I=1}^N$ 的均值 $\bar{x} = \frac{1}{N} \sum\_{i=1}^N x\_i$。
    等价优化问题：$\min\_w J(w) = \frac{1}{2N} \sum\_{i=1}^N \|w - x\_i\|^2$。
    此时，$\nabla_w f(w, x\_i) = w - x\_i$。
    *   **BGD** ($m=N$):
        $$
        w_{k+1} = w_k - \alpha_k \frac{1}{N} \sum_{i=1}^N (w_k - x_i) = w_k - \alpha_k (w_k - \bar{x}).
        $$
        如果 $\alpha_k=1$ (或者足够大使得一步收敛)，则 $w_{k+1} = \bar{x}$。讲者字幕提到若 $\alpha_k=1/k$，对于BGD，$w_{k+1}$ 直接等于 $\bar{x}$。这似乎暗示一次迭代就完成。通常BGD也需要迭代，除非问题特殊。
    *   **MBGD** (mini-batch size $m$):
        $$
        w_{k+1} = w_k - \alpha_k \frac{1}{m} \sum_{j \in I_k} (w_k - x_j) = w_k - \alpha_k (w_k - \bar{x}_k^{(m)}),
        $$
        其中 $\bar{x}_k^{(m)} = \frac{1}{m} \sum_{j \in I_k} x_j$ 是小批量样本的均值。
    *   **SGD** ($m=1$):
        $$
        w_{k+1} = w_k - \alpha_k (w_k - x_k).
        $$
    *   *(图示再次出现：SGD(m=1), MBGD(m=5), MBGD(m=50) 的收敛路径和到均值距离的曲线。$m$ 越大，路径越平滑，收敛到目标邻域越快，震荡越小。)*

---

### **4. 总结 (Summary)**

*   **均值估计 (Mean estimation)**：使用 $\{x_k\}$ (样本序列) 计算 $E[X]$
    $$
    w_{k+1} = w_k - \alpha_k(w_k - x_k) \quad (\text{例如, } \alpha_k = 1/k)
    $$
    *   一种迭代计算均值的方法。

*   **罗宾斯-蒙罗 (RM) 算法**：使用 $\{\tilde{g}(w_k, \eta_k)\}$ (带噪声的函数观测) 求解 $g(w)=0$
    $$
    w_{k+1} = w_k - a_k \tilde{g}(w_k, \eta_k)
    $$
    *   一种通用的随机迭代求根算法，无需函数 $g$ 的显式模型。

*   **随机梯度下降 (SGD) 算法**：使用 $\{\nabla_w f(w_k, x_k)\}$ (随机梯度) 最小化 $J(w) = E[f(w,X)]$
    $$
    w_{k+1} = w_k - \alpha_k \nabla_w f(w_k, x_k)
    $$
    *   一种重要的优化算法，用随机梯度近似真实梯度。

**这些结果的用途：**

1.  **理解 TD 学习**：下一讲的时序差分 (TD) 学习算法可以被视为特殊的随机近似算法，具有与本讲算法相似的迭代表达式。学习本讲内容有助于理解 TD 算法的设计和收敛性。
2.  **通用优化技术**：RM 和 SGD 是强大的优化工具，不仅在强化学习中，在机器学习和其他许多领域都有广泛应用。
