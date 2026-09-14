# 第三问：光伏预报驱动的多阶段购电和储能优化模型

> 本文档面向论文写作，统一描述光伏预报订正、风险裕度、计划购电、日内调整、实际运行和信息价值分析。

## 1. 问题分析

第三问在第二问的基础上增加了日内光伏预报更新。每天 0:00 首先制定全天计划购电量；随后在 6:00、12:00、18:00 获得新的未来 24 小时光伏预报，并根据最新预报和已发生的实际数据调整未来尚未执行时段的购电量。

实际运行时，如果最终调整购电量、实际光伏和电池放电仍不能满足负载，则产生紧急购电。决策链为

$$
0{:}00\ \text{计划}
\rightarrow
6{:}00\ \text{调整}
\rightarrow
12{:}00\ \text{调整}
\rightarrow
18{:}00\ \text{调整}
\rightarrow
\text{实际运行与紧急购电}.
$$

## 2. 数据使用原则与假设

1. 负载预测采用第二问的滚动时间序列模型。
2. 光伏预测使用附件 3 在 0:00、6:00、12:00、18:00 发布的小时级预报。
3. 实际结算使用附件 2 的实际负载和实际光伏。
4. 一天划分为 144 个 10 分钟时段。
5. 小时级光伏预报通过历史回归和日内形状恢复转换为 10 分钟预报。
6. 储能参数和第二问保持一致。

## 3. 符号说明

| 符号 | 含义 | 单位 |
|---|---|---|
| \(L_{d,t}^{\mathrm{act}},V_{d,t}^{\mathrm{act}}\) | 实际负载和实际光伏功率 | kW |
| \(\widehat L_{d,t}^{(\tau)}\) | 时刻 \(\tau\) 对未来时段 \(t\) 的负载预测 | kW |
| \(\widehat V_{d,\tau,t}\) | 时刻 \(\tau\) 对时段 \(t\) 的订正后光伏预报 | kW |
| \(p_t\) | 第 \(t\) 时段电价 | 元/kWh |
| \(g_{d,t}^{\mathrm{plan}}\) | 计划购电量 | kWh |
| \(g_{d,t}^{\mathrm{adj}}\) | 最终调整购电量 | kWh |
| \(e_{d,t}\) | 紧急购电量 | kWh |
| \(c_{d,t},d_{d,t}\) | 充电量和放电量 | kWh |
| \(u_{d,t}\) | 未利用富余电量 | kWh |
| \(S_{d,t}\) | 储电量 | kWh |
| \(z_{d,t}\) | 充放电状态 | 0-1 |

设

$$
T=144,\qquad
\Delta t=\frac{1}{6}\ \mathrm{h}.
$$

储能参数为

$$
\eta_c=\eta_d=0.9,
$$

$$
S_{\min}=1200,\qquad
S_{\max}=10800,\qquad
S_{\mathrm{target}}=6000,
$$

$$
C_{\max}=D_{\max}
=\frac{2500}{3}
\approx833.33\ \mathrm{kWh}.
$$

## 4. 负载滚动修正

0:00 的负载预测记为

$$
L_{d,t}^{\mathrm{base}}.
$$

在时刻 \(\tau\in\{6,12,18\}\)，根据已经发生的实际负载修正未来负载。令

$$
k_\tau=6\tau
$$

表示时刻 \(\tau\) 对应的时段数。已发生部分的平均预测偏差为

$$
\Delta L_\tau
=
\frac{1}{k_\tau}
\sum_{t=1}^{k_\tau}
\left(
L_{d,t}^{\mathrm{act}}
-
L_{d,t}^{\mathrm{base}}
\right).
$$

将修正量限制在未来负载均值的一定范围内：

$$
-0.2\overline L_\tau
\le
\Delta L_\tau
\le
0.2\overline L_\tau.
$$

未来负载更新为

$$
\widehat L_{d,t}^{(\tau)}
=
\max
\left(
0,
L_{d,t}^{\mathrm{base}}
+
\Delta L_\tau
\right),
\qquad t>k_\tau.
$$

已经执行的时段保持不变。

## 5. 光伏预报订正与 10 分钟展开

### 5.1 小时级水平订正

设附件 3 在时刻 \(\tau\) 对目标小时 \(H\) 的原始光伏预报为

$$
\widehat V_{d,\tau,H}^{\mathrm{raw}}.
$$

使用目标日前若干天的历史实际小时均值建立一元线性回归：

$$
\overline V_{d,H}^{\mathrm{act}}
=
a_{\tau,H}
+
b_{\tau,H}
\widehat V_{d,\tau,H}^{\mathrm{raw}}.
$$

得到订正后的小时光伏水平：

$$
\widehat V_{d,\tau,H}^{\mathrm{corr}}
=
a_{\tau,H}
+
b_{\tau,H}
\widehat V_{d,\tau,H}^{\mathrm{raw}}.
$$

该步骤用于修正附件 3 预报的系统性高估或低估。若历史样本不足或回归系数异常，则退化为原始预报。

### 5.2 小时内形状恢复

对每个整点内第 \(j\) 个 10 分钟时段，计算历史实际值与小时均值之比：

$$
r_{H,j}
=
\frac{V_{d,6H+j}^{\mathrm{act}}}
{\overline V_{d,H}^{\mathrm{act}}}.
$$

对历史比例取平均并归一化，使

$$
\sum_{j=0}^{5}\overline r_{H,j}=6.
$$

则订正后的 10 分钟光伏预报为

$$
\widehat V_{d,\tau,6H+j}
=
\widehat V_{d,\tau,H}^{\mathrm{corr}}
\overline r_{H,j}.
$$

形状比例均值为 1，因此该方法不改变小时总电量，只恢复小时内的升功率和降功率过程。

## 6. 预测净负荷与风险裕度

时刻 \(\tau\) 的预测净负荷电量为

$$
\widehat N_{d,t}^{(\tau)}
=
\left[
\widehat L_{d,t}^{(\tau)}
-
\widehat V_{d,\tau,t}
\right]\Delta t.
$$

实际净负荷电量为

$$
N_{d,t}^{\mathrm{act}}
=
\left[
L_{d,t}^{\mathrm{act}}
-
V_{d,t}^{\mathrm{act}}
\right]\Delta t.
$$

0:00 预测误差定义为

$$
\varepsilon_{d,t}
=
N_{d,t}^{\mathrm{act}}
-
\widehat N_{d,t}^{(0)}.
$$

由于 0:00 至 6:00 之间没有调整机会，而 6:00 之后存在多次调整窗口，因此采用分段风险裕度：

$$
q_t
=
\begin{cases}
Q_{\alpha_1}
\left(
\varepsilon_{d-1,t},
\ldots,
\varepsilon_{d-60,t}
\right),
&t\le36,\\[1mm]
Q_{\alpha_2}
\left(
\varepsilon_{d-1,t},
\ldots,
\varepsilon_{d-60,t}
\right),
&t>36.
\end{cases}
$$

0:00 风险修正后的计划净负荷为

$$
N_{d,t}^{\mathrm{plan}}
=
\widehat N_{d,t}^{(0)}
+
q_t.
$$

通过 67 组裕度搜索，最终取

$$
\alpha_1^*=\alpha_2^*=0.60.
$$

## 7. 0:00 计划购电模型

### 7.1 决策变量

对每个时段引入

$$
g_{d,t}^{\mathrm{plan}},\quad
c_{d,t},\quad
d_{d,t},\quad
u_{d,t},\quad
S_{d,t},\quad
z_{d,t}.
$$

引入期末储能不足变量

$$
s^{\mathrm{short}}\ge0.
$$

### 7.2 目标函数

$$
\min
\sum_{t=1}^{T}
p_tg_{d,t}^{\mathrm{plan}}
+
c_s s^{\mathrm{short}},
$$

其中

$$
c_s=5\overline p,
\qquad
\overline p
=
\frac{1}{T}
\sum_{t=1}^{T}p_t.
$$

### 7.3 约束条件

电量平衡：

$$
g_{d,t}^{\mathrm{plan}}
+
d_{d,t}
-
c_{d,t}
-
u_{d,t}
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
g_{d,t}^{\mathrm{plan}},
c_{d,t},
d_{d,t},
u_{d,t},
s^{\mathrm{short}}
\ge0.
$$

期末储能软约束为

$$
S_{d,T}
+
s^{\mathrm{short}}
\ge
S_{\mathrm{target}}.
$$

当天初始储电量为实际值：

$$
S_{d,0}=S_{d,0}^{\mathrm{act}}.
$$

## 8. 6:00、12:00、18:00 调整模型

### 8.1 调整范围

在时刻 \(\tau\) 只优化尚未执行的时段：

$$
t>k_\tau.
$$

已经执行部分保持不变：

$$
g_{d,t|\tau}^{\mathrm{adj}}
=
g_{d,t|\tau^-}^{\mathrm{adj}},
\qquad
t\le k_\tau.
$$

### 8.2 调整变量

令参考计划量为上一版计划值：

$$
g_{d,t}^{\mathrm{ref}}.
$$

定义调高量和调低量：

$$
\Delta_{d,t}^{+}
=
\max
\left(
g_{d,t}^{\mathrm{adj}}
-
g_{d,t}^{\mathrm{ref}},
0
\right),
$$

$$
\Delta_{d,t}^{-}
=
\max
\left(
g_{d,t}^{\mathrm{ref}}
-
g_{d,t}^{\mathrm{adj}},
0
\right).
$$

调整阶段的目标函数为

$$
\min
\sum_{t>k_\tau}
\left[
1.5p_t\Delta_{d,t}^{+}
+
0.5p_t\Delta_{d,t}^{-}
\right]
+
c_s s^{\mathrm{short}}.
$$

调整模型仍需满足电量平衡、储能递推、储能容量和充放电互斥约束。每次调整后只更新未来时段的最终购电计划。

每轮调整只作用于尚未执行的未来时段，因此每个时段在执行前只经历一次对应调整。最终结算时，统一按最终调整购电量相对 0:00 原始计划量的一次性差额计算，不重复计费。

## 9. 实际运行模型

实际运行使用最终调整购电量

$$
g_{d,t}^{\mathrm{adj}}.
$$

定义储能动作前的供需差额：

$$
a_{d,t}
=
g_{d,t}^{\mathrm{adj}}
+
V_{d,t}^{\mathrm{act}}\Delta t
-
L_{d,t}^{\mathrm{act}}\Delta t.
$$

### 9.1 富余情况

当 \(a_{d,t}\ge0\) 时，

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

### 9.2 缺电情况

当 \(a_{d,t}<0\) 时，

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
S_{d,t-1}^{\mathrm{act}}
-
S_{\min}
\right)
\right\}.
$$

剩余缺口由紧急购电补足：

$$
e_{d,t}
=
\max
\left\{
0,
\ -a_{d,t}-d_{d,t}
\right\}.
$$

实际储能递推为

$$
S_{d,t}^{\mathrm{act}}
=
S_{d,t-1}^{\mathrm{act}}
+
\eta_cc_{d,t}
-
\frac{d_{d,t}}{\eta_d}.
$$

跨日满足

$$
S_{d+1,0}^{\mathrm{act}}
=
S_{d,T}^{\mathrm{act}}.
$$

## 10. 费用模型

计划购电费用为

$$
C_d^{\mathrm{plan}}
=
\sum_{t=1}^{T}
p_tg_{d,t}^{\mathrm{plan}}.
$$

最终调整购电量相对原始计划购电量的差额定义为

$$
\Delta_{d,t}
=
g_{d,t}^{\mathrm{adj}}
-
g_{d,t}^{\mathrm{plan}},
$$

并分解为

$$
\Delta_{d,t}^{+}
=
\max
\left(
\Delta_{d,t},0
\right),
\qquad
\Delta_{d,t}^{-}
=
\max
\left(
-\Delta_{d,t},0
\right).
$$

调整费用为

$$
C_d^{\mathrm{adj}}
=
\sum_{t=1}^{T}
\left[
1.5p_t\Delta_{d,t}^{+}
+
0.5p_t\Delta_{d,t}^{-}
\right].
$$

紧急购电费用为

$$
C_d^{\mathrm{em}}
=
\sum_{t=1}^{T}
5p_te_{d,t}.
$$

当天总费用为

$$
C_d
=
C_d^{\mathrm{plan}}
+
C_d^{\mathrm{adj}}
+
C_d^{\mathrm{em}}.
$$

统计期内的总费用为

$$
C_{\mathrm{total}}
=
\sum_{d\in\mathcal D}
C_d,
$$

其中 \(\mathcal D\) 为 2025-02-01 至 2025-12-31 的日期集合。

## 11. 求解流程

对每一天执行：

1. 0:00 使用负载预测、附件 3 的 0:00 光伏预报和风险裕度求解计划 MILP；
2. 6:00 使用已发生数据修正负载预测，读取新的光伏预报，并优化第 37 至 144 时段；
3. 12:00 重复预测更新和调整，并优化第 73 至 144 时段；
4. 18:00 执行最后一次调整，并优化第 109 至 144 时段；
5. 使用附件 2 的实际负载和实际光伏模拟储能运行；
6. 计算紧急购电量及三类费用；
7. 将日末储能传递给下一天。

## 12. 裕度搜索结果

对夜间和白天两个风险分位进行 67 组搜索，最优结果为

$$
\alpha_1^*=0.60,\qquad
\alpha_2^*=0.60.
$$

| 指标 | 最优结果 |
|---|---:|
| 总费用 | 13,557,776.15 元 |
| 计划购电费 | 12,502,869.72 元 |
| 调整费用 | 785,257.51 元 |
| 紧急购电费 | 269,648.92 元 |
| 紧急购电量 | 47,836.98 kWh |
| 计划购电量 | 20,506,836.54 kWh |
| 弃电量 | 2,014,571.71 kWh |

费用在

$$
\alpha_1\in[0.55,0.70],
\qquad
\alpha_2\in[0.55,0.65]
$$

内形成平坦最优区域，说明最优裕度对轻微参数变化不敏感。

## 13. 是否需要增加新的预报时刻

为回答题目后半部分，采用“完美预见上界法”。定义信息价值为

$$
\mathrm{VOI}
=
C_{\mathrm{baseline}}
-
C_{\mathrm{variant}},
$$

$$
\mathrm{VOI}\%
=
\frac{\mathrm{VOI}}
{C_{\mathrm{baseline}}}
\times100\%.
$$

若把某个时刻的光伏预报替换为当天实际光伏，所得收益仍很小，则真实新增预报的收益必然更低。

主要结果为：

1. 将 6:00、12:00、18:00 的预报全部替换为完美预见，总费用仅变化约 0.09%，且统计上不显著。
2. 在 3:00、9:00、15:00、21:00 分别增加一个调整时刻并赋予完美预见，单个时刻仅节省 0.08% 至 0.44%，四个合计仅节省 0.92%。
3. 取消 18:00 调整使费用增加约 2.01%，只保留 6:00 调整使费用增加约 3.24%，完全不调整使费用增加约 10.70%。

因此，现有 0:00、6:00、12:00、18:00 四个信息时刻已经覆盖主要可用信息，无需增加其他时刻的光伏预报。

从误差角度看，净负荷预测误差的逐时段标准差约为

$$
\sigma_\varepsilon=64.97\ \mathrm{kWh},
$$

而电池单个时段可吞吐电量为

$$
C_{\max}=833.33\ \mathrm{kWh}.
$$

定义缓冲比

$$
\beta
=
\frac{\sigma_\varepsilon}{C_{\max}}
\approx0.078,
$$

说明大部分预测误差可以由储能吸收，进一步降低了增加预报时刻的边际价值。

## 14. 结果分析与结论

第三问的最终模型可以概括为

$$
\begin{aligned}
\text{0:00：}\quad
&
\text{负载预测}
+
\text{附件 3 的 0:00 光伏预报}
+
\text{风险裕度}
\rightarrow
\text{计划购电},\\
\text{6/12/18:00：}\quad
&
\text{新光伏预报}
+
\text{已发生实际数据}
\rightarrow
\text{调整未来购电},\\
\text{实际运行：}\quad
&
\text{实际负载}
+
\text{实际光伏}
+
\text{最终调整计划}
+
\text{储能}
\rightarrow
\text{紧急购电}.
\end{aligned}
$$

模型通过光伏历史订正提高附件 3 预报的 10 分钟精度，通过分段风险裕度控制计划购电的保守程度，通过多阶段调整降低紧急购电风险，并通过信息价值实验证明无需增加新的预报时刻。

最终采用 \(0.60/0.60\) 裕度时，2025 年 2 月 1 日至 12 月 31 日的总费用为

$$
C=13,557,776.15\ \text{元}.
$$

模型的主要局限是采用确定性日度滚动和分位数风险裕度，没有显式生成多场景并求解两阶段随机规划；期末储能目标采用软约束，尚未建立季节性动态价值函数。
