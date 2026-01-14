**已知一个复杂分布 $p(x)$，但无法直接采样**  
如何生成服从 $p(x)$ 的样本？

典型困难：

- 分布只给出 **未归一化形式**（只知道比例）
    
- 高维
    
- 条件分布容易、联合分布难
    


##  MCMC 的基本思想

**不直接采样 $p(x)$**  
而是 **构造一个马尔可夫链，使其平稳分布正好是 $p(x)$**

然后：

- 沿着马尔可夫链跑
    
- 丢掉前面未收敛的样本（burn-in）
    
- 后面的状态样本 $\approx p(x)$ 的样本


## 马尔可夫链在 MCMC 中的角色

**状态 = 一个具体的样本值**

- 一维：  
    $X_t = x_t \in \mathbb{R}$
    
- 二维：  
    $X_t = (x_t, y_t)$
    
- n 维：  
    $X_t = (x_1^{(t)}, \dots, x_n^{(t)})$

**状态不是分布，不是参数，是一个“点”**

$$  
x_0 \rightarrow x_1 \rightarrow x_2 \rightarrow \cdots  
$$

- 这是一个**相关的样本序列**
    
- 单个 $x_t$ 是一个样本
    
- 整个序列的**边际分布**逐渐逼近 $p(x)$
   

---

## 三、细致平稳条件


目标：  
让 $p(x)$ 成为马尔可夫链的平稳分布

一个**充分条件**是：

$$  
\boxed{  
p(i),P(i\to j) = p(j),P(j\to i)  
}  
$$

这叫 **Detailed Balance（细致平稳）**

直观理解：
 从 $i$ 流向 $j$ 的概率流  = 从 $j$ 流向 $i$ 的概率流

---

## 状态转移的分解

在 MH / Gibbs 中，转移通常写成：

$$  
P(i \to j) = q(j|i),\alpha(i,j)  
$$

- $q(j|i)$：提议分布（proposal）
    
- $\alpha(i,j)$：接受概率


## Metropolis–Hastings（MH）算法

### 接受率的推导逻辑

从细致平稳条件出发：
$$
p(i),q(j|i),\alpha(i,j)

p(j),q(i|j),\alpha(j,i)  
$$

为了：

- **提高接受率**
    
- **同时满足细致平稳**
    
 把比例统一放大，使最大者为 1

最终得到 **经典 MH 接受率**：

# $$  
\boxed{  
\alpha(i,j)

\min\left(  
1,
\frac{p(j)q(i|j)}{p(i)q(j|i)}  
\right)  
}  
$$


##  Metropolis–Hastings 算法流程

**Algorithm: Metropolis–Hastings**

1. 初始化状态 $X_0 = x_0$
    
2. 对 $t = 0,1,2,\dots$：
    
    1. 采样候选  
        $$  
        y \sim q(y \mid x_t)  
        $$
    2. 采样  
        $$  
        u \sim \text{Uniform}(0,1)  
        $$
    3. 若  
        $$  
        u < \alpha(x_t,y)  
        $$  
        则接受：  
        $$  
        x_{t+1} = y  
        $$  
        否则：  
        $$  
        x_{t+1} = x_t  
        $$


## Gibbs Sampling（MH 的特例）

### Gibbs 的出发点

在高维中：

- MH 接受率低
    
- 但**条件分布往往好采样**
    

 那就 **直接用条件分布做转移**

---

## 2. 二维 Gibbs Sampling 的推导逻辑

已知联合分布：  
$$  
p(x,y)  
$$

# 注意恒等式：  
$$  
p(x,y_1)p(y_2|x)

p(x,y_2)p(y_1|x)  
$$

这意味着：

# $$  
p(A),p(y_2|x_1)

p(B),p(y_1|x_1)  
$$

 **沿着 $x=x_1$ 的直线，用条件分布转移，天然满足细致平稳**

---

## 3. 二维 Gibbs 算法

**Algorithm: 2D Gibbs Sampling**

1. 初始化 $(x_0,y_0)$
    
2. 对 $t=0,1,2,\dots$：  
    1.  
    $$  
    y_{t+1} \sim p(y \mid x_t)  
    $$  
    2.  
    $$  
    x_{t+1} \sim p(x \mid y_{t+1})  
    $$
    

 **接受率恒为 1**  
 不需要 MH 的接受–拒绝步骤

---

## 4. n 维 Gibbs Sampling

对状态：  
$$  
(x_1,\dots,x_n)  
$$

依次更新每一维：

$$  
x_1^{(t+1)} \sim p(x_1 \mid x_2^{(t)},\dots,x_n^{(t)})  
$$  
$$  
x_2^{(t+1)} \sim p(x_2 \mid x_1^{(t+1)},x_3^{(t)},\dots)  
$$  
$$  
\vdots  
$$  
$$  
x_n^{(t+1)} \sim p(x_n \mid x_1^{(t+1)},\dots,x_{n-1}^{(t+1)})  
$$

---

# 六、MH vs Gibbs

| 方面      | MH  | Gibbs   |
| ------- | --- | ------- |
| 是否需要接受率 | 是   | 否       |
| 接受率     | < 1 | = 1     |
| 提议分布    | 任意  | 条件分布    |
| 实现难度    | 低   | 需能算条件分布 |
| 高维效率    | 可能差 | 通常更好    |

