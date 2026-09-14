from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

# ===== 输入文件路径 =====
# 预测值表：第1页=load预测值，第2页=pv预测值
pred_file = ROOT / "result" / "problem02" / "prediction" / "prediction.xlsx"
# 实际值表：第1页=load实际值，第2页=pv实际值
actual_file = ROOT / "附件" / "附件2.xlsx"

# ===== 输出文件路径 =====
output_load_file = ROOT / "result" / "problem02" / "expend" / "expend_prediction_load.xlsx"
output_pv_file = ROOT / "result" / "problem02" / "expend" / "expend_prediction_pv.xlsx"

# ===== 读取数据 =====
# 预测值表
df_load_pred = pd.read_excel(pred_file, sheet_name=0, header=None)   # load 预测
df_pv_pred = pd.read_excel(pred_file, sheet_name=1, header=None)     # pv 预测

# 实际值表
df_load_actual = pd.read_excel(actual_file, sheet_name=0, header=None)  # load 实际
df_pv_actual = pd.read_excel(actual_file, sheet_name=1, header=None)    # pv 实际

# ===== 截取 B2:EO335 区域 =====
# 行：第2行到第335行 -> iloc[1:335]
# 列：B 到 EO -> iloc[:, 1:145]
load_pred_matrix = df_load_pred.iloc[1:335, 1:145]
load_actual_matrix = df_load_actual.iloc[1:335, 1:145]
pv_pred_matrix = df_pv_pred.iloc[1:335, 1:145]
pv_actual_matrix = df_pv_actual.iloc[1:335, 1:145]

# ===== 按行优先展平成一列 =====
load_pred_vector = load_pred_matrix.values.flatten()
load_actual_vector = load_actual_matrix.values.flatten()
pv_pred_vector = pv_pred_matrix.values.flatten()
pv_actual_vector = pv_actual_matrix.values.flatten()

# ===== 检查长度是否一致 =====
lengths = {
    "load预测值": len(load_pred_vector),
    "load实际值": len(load_actual_vector),
    "pv预测值": len(pv_pred_vector),
    "pv实际值": len(pv_actual_vector),
}
if len(set(lengths.values())) != 1:
    raise ValueError(f"四个向量长度不一致：{lengths}")

# ===== 生成两个结果 DataFrame =====
result_load = pd.DataFrame({
    "load预测值": load_pred_vector,
    "load实际值": load_actual_vector,
})

result_pv = pd.DataFrame({
    "pv预测值": pv_pred_vector,
    "pv实际值": pv_actual_vector,
})

# ===== 确保输出目录存在 =====
output_load_file.parent.mkdir(parents=True, exist_ok=True)
output_pv_file.parent.mkdir(parents=True, exist_ok=True)

# ===== 分别写出两个 Excel =====
result_load.to_excel(output_load_file, index=False)
result_pv.to_excel(output_pv_file, index=False)

print("完成！")
print(f"load 结果：共 {len(result_load)} 行，已保存到：{output_load_file}")
print(f"pv   结果：共 {len(result_pv)} 行，已保存到：{output_pv_file}")