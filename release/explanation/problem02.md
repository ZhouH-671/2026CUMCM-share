# 第二问：滚动预测与风险修正下的购电和储能优化模型

> 本文档面向论文写作，统一使用数学记号描述预测、风险修正、优化与结算流程。

## 1. 问题分析

第二问中，每天 0:00 需要制定当天的计划购电策略，但此时无法获得当天的实际负载和实际光伏。因此，模型必须首先利用历史数据预测未来负载和光伏，再根据预测结果制定购电和储能计划。

实际运行过程中，若计划购电量和实际光伏不能满足负载，则电池优先放电；若电池放电后仍存在缺口，则需要按正常电价的 5 倍进行紧急购电。

整体决策流程为

$$
\text{历史数据}
\rightarrow
\text{滚动预测}
\rightarrow
\text{风险修正}
\rightarrow
\text{计划购电与储能优化}
\rightarrow
\text{实际运行与费用结算}.
$$

## 2. 时间划分与符号

将一天划分为

$$
t=1,2,\ldots,T,\qquad T=144,
$$

每个时段长度为

$$
\Delta t=\frac{1}{6}\ \mathrm{h}.
$$

| 符号 | 含义 | 单位 |
|---|---|---|
| \(L_{d,t},V_{d,t}\) | 第 \(d\) 天第 \(t\) 时段实际负载和光伏功率 | kW |
| \(\widehat L_{d,t},\widehat V_{d,t}\) | 对应预测功率 | kW |
| \(p_t\) | 第 \(t\) 时段电价 | 元/kWh |
| \(g_{d,t}\) | 计划购电量 | kWh |
| \(c_{d,t},d_{d,t}\) | 充电量和放电量 | kWh |
| \(u_{d,t}\) | 计划阶段的未利用富余电量 | kWh |
| \(S_{d,t}\) | 储电量 | kWh |
| \(e_{d,t}\) | 紧急购电量 | kWh |
| \(z_{d,t}\) | 充放电状态 | 0-1 |

储能参数为

$$
\eta_c=\eta_d=0.9,\qquad
S_{\min}=1200,\qquad
S_{\max}=10800,
$$

$$
C_{\max}=D_{\max}
=\frac{2500}{3}
\approx833.33\ \mathrm{kWh}.
$$

## 3. 负载滚动预测模型

### 3.1 特征构造

负载具有短期相关性和周期变化特征。对目标日 \(d\) 的第 \(t\) 个时段，选用以下特征：

1. 前 1、2、3、7 日同时刻负载；
2. 前 3 日和前 7 日同时刻均值；
3. 最近两个相同星期几的同时刻均值；
4. 星期周期特征 \(\sin(2\pi w_d/7)\)、\(\cos(2\pi w_d/7)\)；
5. 年周期特征；
6. 日内周期特征。

记特征向量为

$$
\boldsymbol X_{d,t}^{L}
=
\left(
x_{d,t,1}^{L},
\ldots,
x_{d,t,p_L}^{L}
\right).
$$

负载预测为

$$
\widehat L_{d,t}
=
\boldsymbol X_{d,t}^{L}\boldsymbol\beta_L.
$$

### 3.2 滚动岭回归

第 \(d\) 天只使用第 \(d\) 天以前的数据训练模型，目标函数为

$$
\widehat{\boldsymbol\beta}_L
=
\arg\min_{\boldsymbol\beta}
\left[
\left\|
\boldsymbol Y_L
-
\boldsymbol X_L\boldsymbol\beta
\right\|_2^2
+
\lambda_L
\left\|
\boldsymbol\beta
\right\|_2^2
\right].
$$

其中 \(\lambda_L\) 为岭参数。负载模型使用最近 31 天作为训练窗口，使模型能够适应近期负载水平变化。

候选参数为

$$
\lambda_L\in\{0.1,1,10,100\}.
$$

在 1 月验证集 \(\mathcal V\) 上选择使 MAE 最小的参数：

$$
\lambda_L^*
=
\arg\min_{\lambda_L}
\frac{1}{|\mathcal V|T}
\sum_{d\in\mathcal V}
\sum_{t=1}^{T}
\left|
L_{d,t}
-
\widehat L_{d,t}(\lambda_L)
\right|.
$$

最终选取

$$
\lambda_L^*=100.
$$

## 4. 光伏滚动预测模型

### 4.1 物理特征与历史特征

光伏出力具有明显的季节变化。太阳赤纬定义为

$$
\delta_d
=
23.44^\circ
\sin\left[
\frac{2\pi(D_d-81)}{365}
\right],
$$

其中 \(D_d\) 为目标日在一年中的序号。

光伏特征包括：

1. 前 1、2、3、7、14 日同时刻光伏；
2. 前 3、7、14 日同时刻均值；
3. \(\sin(\delta_d)\)、\(\cos(\delta_d)\)；
4. 日内周期特征。

岭回归预测为

$$
\widehat V_{d,t}^{\mathrm{ridge}}
=
\boldsymbol X_{d,t}^{V}
\boldsymbol\beta_V,
$$

其中

$$
\widehat{\boldsymbol\beta}_V
=
\arg\min_{\boldsymbol\beta}
\left[
\left\|
\boldsymbol Y_V
-
\boldsymbol X_V\boldsymbol\beta
\right\|_2^2
+
\lambda_V
\left\|
\boldsymbol\beta
\right\|_2^2
\right].
$$

光伏模型采用扩展窗口，随着日期增加逐步加入新的历史数据。

### 4.2 与短期均值方法融合

定义最近 7 日同时刻均值为

$$
\widehat V_{d,t}^{7}
=
\frac{1}{7}
\sum_{k=1}^{7}V_{d-k,t}.
$$

最终光伏预测为

$$
\widehat V_{d,t}
=
\alpha
\widehat V_{d,t}^{7}
+
(1-\alpha)
\widehat V_{d,t}^{\mathrm{ridge}},
$$

并进行非负截断：

$$
\widehat V_{d,t}
\leftarrow
\max\left(0,\widehat V_{d,t}\right).
$$

模型中取

$$
\lambda_V^*=1,\qquad
\alpha^*=0.4.
$$

## 5. 预测净负荷与风险修正

预测净负荷电量为

$$
\widehat N_{d,t}
=
\left(
\widehat L_{d,t}
-
\widehat V_{d,t}
\right)
\Delta t.
$$

实际净负荷电量为

$$
N_{d,t}
=
\left(
L_{d,t}
-
V_{d,t}
\right)
\Delta t.
$$

预测误差定义为

$$
\varepsilon_{d,t}
=
N_{d,t}
-
\widehat N_{d,t}.
$$

由于紧急购电价格为正常电价的 5 倍，若只使用点预测可能造成较大的缺电风险。因此，对每个时段使用最近 60 天同位置误差的 80% 分位数作为风险裕度：

$$
q_{d,t}
=
Q_{0.8}
\left\{
\varepsilon_{d-1,t},
\varepsilon_{d-2,t},
\ldots,
\varepsilon_{d-60,t}
\right\}.
$$

风险修正后的计划净负荷为

$$
N_{d,t}^{\mathrm{plan}}
=
\widehat N_{d,t}
+
q_{d,t}.
$$

当 \(q_{d,t}>0\) 时，模型倾向于提前多购电，以减少五倍电价紧急购电；当 \(q_{d,t}<0\) 时，则适当降低计划购电量。

## 6. 日计划购电与储能 MILP

### 6.1 决策变量

每天优化以下变量：

$$
g_{d,t},\quad
c_{d,t},\quad
d_{d,t},\quad
u_{d,t},\quad
S_{d,t},\quad
z_{d,t}.
$$

其中 \(u_{d,t}\) 表示计划阶段未利用的富余电量。

### 6.2 目标函数

$$
\min
\sum_{t=1}^{T}
p_tg_{d,t}.
$$

### 6.3 约束条件

电量平衡：

$$
g_{d,t}+d_{d,t}-c_{d,t}-u_{d,t}
=
N_{d,t}^{\mathrm{plan}}.
$$

储能状态递推：

$$
S_{d,t}
=
S_{d,t-1}
+
\eta_cc_{d,t}
-
\frac{d_{d,t}}{\eta_d}.
$$

储能容量：

$$
S_{\min}\le S_{d,t}\le S_{\max}.
$$

充放电互斥：

$$
0\le c_{d,t}\le C_{\max}z_{d,t},
$$

$$
0\le d_{d,t}\le D_{\max}(1-z_{d,t}),
$$

$$
z_{d,t}\in\{0,1\}.
$$

非负约束：

$$
g_{d,t},c_{d,t},d_{d,t},u_{d,t}\ge0.
$$

日计划要求日初与日末储电量一致：

$$
S_{d,0}=S_{d,T}=S_{d}^{\mathrm{start}}.
$$

因此，日计划 MILP 可统一写为

$$
\begin{aligned}
\min\quad
&
\sum_{t=1}^{T}p_tg_{d,t}\\
\mathrm{s.t.}\quad
&
g_{d,t}+d_{d,t}-c_{d,t}-u_{d,t}=N_{d,t}^{\mathrm{plan}},\\
&
S_{d,t}=S_{d,t-1}+\eta_cc_{d,t}-\frac{d_{d,t}}{\eta_d},\\
&
S_{\min}\le S_{d,t}\le S_{\max},\\
&
0\le c_{d,t}\le C_{\max}z_{d,t},\\
&
0\le d_{d,t}\le D_{\max}(1-z_{d,t}),\\
&
g_{d,t},c_{d,t},d_{d,t},u_{d,t}\ge0,\\
&
z_{d,t}\in\{0,1\},\\
&
S_{d,0}=S_{d,T}=S_d^{\mathrm{start}}.
\end{aligned}
$$

## 7. 实际运行模型

计划购电量 \(g_{d,t}\) 在当天 0:00 锁定。实际运行采用附件 2 的实际负载和实际光伏。

定义储能动作前的供需差额：

$$
a_{d,t}
=
g_{d,t}
+
V_{d,t}\Delta t
-
L_{d,t}\Delta t.
$$

### 7.1 富余情况

当 \(a_{d,t}\ge0\) 时，有

$$
e_{d,t}=0,\qquad d_{d,t}=0.
$$

富余电量优先充电：

$$
c_{d,t}
=
\min
\left\{
a_{d,t},
\ C_{\max},
\ \frac{S_{\max}-S_{d,t-1}^{\mathrm{act}}}{\eta_c}
\right\}.
$$

未利用电量为

$$
u_{d,t}=a_{d,t}-c_{d,t}.
$$

### 7.2 缺电情况

当 \(a_{d,t}<0\) 时，有

$$
c_{d,t}=u_{d,t}=0.
$$

电池优先放电：

$$
d_{d,t}
=
\min
\left\{
-a_{d,t},
\ D_{\max},
\ \eta_d
\left(
S_{d,t-1}^{\mathrm{act}}-S_{\min}
\right)
\right\}.
$$

若仍有缺口，则紧急购电量为

$$
e_{d,t}
=
\max
\left\{
0,
\ -a_{d,t}-d_{d,t}
\right\}.
$$

实际储电量递推为

$$
S_{d,t}^{\mathrm{act}}
=
S_{d,t-1}^{\mathrm{act}}
+
\eta_cc_{d,t}
-
\frac{d_{d,t}}{\eta_d}.
$$

## 8. 跨日储能衔接

实际储能状态需要跨日连续：

$$
S_{d+1,0}^{\mathrm{act}}
=
S_{d,T}^{\mathrm{act}}.
$$

模型从

$$
S_{2025\text{-}01\text{-}01,0}=6000\ \mathrm{kWh}
$$

开始滚动。2025 年 1 月用于储能热启动，费用统计从

$$
2025\text{-}02\text{-}01
$$

开始。

## 9. 费用模型

计划购电费用为

$$
C_d^{\mathrm{plan}}
=
\sum_{t=1}^{T}p_tg_{d,t}.
$$

紧急购电费用为

$$
C_d^{\mathrm{em}}
=
\sum_{t=1}^{T}5p_te_{d,t}.
$$

当天总费用为

$$
C_d
=
C_d^{\mathrm{plan}}
+
C_d^{\mathrm{em}}.
$$

统计期内总费用为

$$
C_{\mathrm{total}}
=
\sum_{d\in\mathcal D}
\left(
C_d^{\mathrm{plan}}
+
C_d^{\mathrm{em}}
\right),
$$

其中 \(\mathcal D\) 为 2025-02-01 至 2025-12-31 的日期集合。

## 10. 求解流程

对每一天依次执行：

1. 使用历史数据预测负载和光伏；
2. 计算预测净负荷；
3. 计算最近 60 天预测误差的 80% 分位数；
4. 求解日计划购电与储能 MILP；
5. 按实际负载和实际光伏模拟电池运行；
6. 计算紧急购电量；
7. 计算计划购电费、紧急购电费和当天总费用；
8. 将日末储电量传递给下一天，并将当天实际结果加入历史数据。

## 11. 预测精度与优化结果

### 11.1 预测精度

| 序列 | \(R^2\) | MAE（kW） | RMSE（kW） |
|---|---:|---:|---:|
| 小区负载 | 0.9745 | 166.44 | 224.64 |
| 光伏全时段 | 0.9899 | 154.77 | 301.76 |
| 光伏 6:00-18:00 | 0.9735 | 300.46 | 426.23 |

### 11.2 购电与费用结果

| 指标 | 结果 |
|---|---:|
| 计划购电量 | 22,046,434.834858 kWh |
| 计划购电费 | 13,454,464.390255 元 |
| 实际充电量 | 6,734,511.639971 kWh |
| 实际放电量 | 5,455,069.040090 kWh |
| 紧急购电量 | 126,347.963398 kWh |
| 紧急购电费 | 769,998.124403 元 |
| 总费用 | 14,224,462.514658 元 |
| 期末储电量 | 10,672.653652 kWh |

## 12. 结果分析与模型评价

1. 负载预测的 \(R^2\) 达到 0.9745，说明滚动岭回归能够较好地刻画负载的日周期和周周期。
2. 光伏预测的整体拟合较高，但误差主要集中在白天，主要受到天气变化影响。
3. 80% 分位数风险修正使计划购电量适当增加，从而降低了紧急购电风险。
4. 日计划 MILP 通过充放电互斥约束保证储能不会同时充放电。
5. 实际储电量跨日连续，避免了每天重置储能状态导致的非物理调度。

模型的主要局限是使用点预测和分位数风险裕度，没有显式建立多场景随机优化；日计划模型强制日初与日末储电量相同，没有完整刻画储能的跨日价值。

## 13. 论文结论表述

第二问建立了“滚动时间序列预测、分位数风险修正、储能 MILP 和实际运行结算”相结合的购电优化模型。模型在预测不确定性和购电成本之间进行权衡，得到 2025 年 2 月 1 日至 12 月 31 日的总费用为

$$
C_{\mathrm{total}}=14,224,462.51\ \text{元}.
$$
