import os
import pandas as pd
import pulp

# ============================================================
# 1. 读取数据
# ============================================================
input_path = "../data/appendix01.xlsx"
output_path = "../result/result1.xlsx"

df = pd.read_excel(input_path)

# 根据附件，前四列分别为：
# 时间、电价（元/kWh）、小区负载（kW）、光伏发电预测功率（kW）
df = df.iloc[:, :4]
df.columns = ["时间", "电价", "小区负载", "光伏发电预测功率"]

# 删除空行
df = df.dropna(subset=["时间"]).reset_index(drop=True)

# 检查数据行数
T = len(df)
print(f"读取到 {T} 个时段")

# 本问题按 10 分钟一个时段
# 一天 24 小时 = 144 个 10 分钟时段
if T != 144:
    print(f"警告：读取到 {T} 行，不是 144 行。请检查数据是否完整。")

# ============================================================
# 2. 参数设置
# ============================================================
dt = 1 / 6  # 10 分钟 = 1/6 小时

# 电价
p = df["电价"].astype(float).values

# 小区负载功率 kW -> 电量 kWh
L = df["小区负载"].astype(float).values
ell = L * dt

# 光伏预测功率 kW -> 电量 kWh
PV = df["光伏发电预测功率"].astype(float).values
v = PV * dt

# 储能参数
eta_c = 0.9
eta_d = 0.9

S_min = 1200.0
S_max = 10800.0
S_init = 6000.0

P_c_max = 5000.0  # kW
P_d_max = 5000.0  # kW

# 每个时段最大充放电量 kWh
C_max = P_c_max * dt
D_max = P_d_max * dt

# ============================================================
# 3. 建立 MILP 模型
# ============================================================
model = pulp.LpProblem("Microgrid_Day_Ahead_Scheduling", pulp.LpMinimize)

# 时段索引
t_idx = range(T)

# 决策变量
g = pulp.LpVariable.dicts("g", t_idx, lowBound=0, cat="Continuous")  # 购电量 kWh
c = pulp.LpVariable.dicts("c", t_idx, lowBound=0, cat="Continuous")  # 充电量 kWh
d = pulp.LpVariable.dicts("d", t_idx, lowBound=0, cat="Continuous")  # 放电量 kWh
S = pulp.LpVariable.dicts("S", range(T + 1), lowBound=S_min, upBound=S_max, cat="Continuous")
z = pulp.LpVariable.dicts("z", t_idx, cat="Binary")

# 目标函数：全天购电费用最小
model += pulp.lpSum(p[t] * g[t] for t in t_idx)

# 初始储电量
model += S[0] == S_init

# 约束
for t in t_idx:
    # 电量平衡：购电 + 光伏 + 放电 = 负载 + 充电
    model += g[t] + v[t] + d[t] == ell[t] + c[t]

    # 储能递推
    model += S[t + 1] == S[t] + eta_c * c[t] - d[t] / eta_d

    # 充放电功率限制与互斥
    model += c[t] <= C_max * z[t]
    model += d[t] <= D_max * (1 - z[t])

# 日末储电量等于日初储电量
model += S[T] == S_init

# ============================================================
# 4. 求解
# ============================================================
solver = pulp.PULP_CBC_CMD(msg=True)
model.solve(solver)

print("求解状态：", pulp.LpStatus[model.status])

# ============================================================
# 5. 提取结果
# ============================================================
g_val = [pulp.value(g[t]) for t in t_idx]
c_val = [pulp.value(c[t]) for t in t_idx]
d_val = [pulp.value(d[t]) for t in t_idx]
S_val = [pulp.value(S[t]) for t in range(T + 1)]

# 全天购电量与购电费
total_g = sum(g_val)
total_cost = sum(p[t] * g_val[t] for t in t_idx)

print("全天购电量 = %.2f kWh" % total_g)
print("全天购电费 = %.2f 元" % total_cost)
print("0:00 储电量 = %.2f kWh" % S_val[0])
print("24:00 储电量 = %.2f kWh" % S_val[T])

# ============================================================
# 6. 生成输出表：
# ============================================================

def format_time(minutes):
    h = minutes // 60
    m = minutes % 60
    return f"{h}:{m:02d}"

time_labels = []
for t in range(T):
    start = (t + 1) * 10
    end = (t + 2) * 10
    time_labels.append(f"{format_time(start)}-{format_time(end)}")

result_df = pd.DataFrame({
    "时间段": time_labels,
    "购电量": g_val,
    "储能设备充电量": c_val,
    "储能设备放电量": d_val
})

# 保留合适小数位
result_df["购电量"] = result_df["购电量"].round(6)
result_df["储能设备充电量"] = result_df["储能设备充电量"].round(6)
result_df["储能设备放电量"] = result_df["储能设备放电量"].round(6)

# ============================================================
# 7. 保存结果
# ============================================================
os.makedirs("result", exist_ok=True)
result_df.to_excel(output_path, index=False)

print(f"结果已保存到：{output_path}")
