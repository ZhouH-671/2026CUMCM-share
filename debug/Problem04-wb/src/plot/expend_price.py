import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# 输入文件路径
input_file = ROOT / "result" / "forecast" / "pred_price_sarima.xlsx"
# 输出文件路径
output_file = ROOT / "result" / "expend" / "expend_pred_sarima.xlsx"

# 读取两个 sheet
# sheet_name=0 表示第一个 sheet（预测值），sheet_name=1 表示第二个 sheet（实际值）
# header=None 表示不把第一行当作表头，因为我们要读取的是 B2:EO335 这种固定区域
df_pred = pd.read_excel(input_file, sheet_name=0, header=None)
df_actual = pd.read_excel(input_file, sheet_name=1, header=None)

# pandas 读取时列索引从 0 开始：
# B 列 -> 索引 1
# EO 列 -> 需要计算一下
# Excel 列号从 1 开始：A=1, B=2, ..., Z=26, AA=27, ...
# EO 列号：
# E=5, O=15 -> EO = 5*26 + 15 = 145
# 所以 EO 对应的 0-based 索引是 144
# 行：第2行到第335行 -> 0-based 索引 1 到 334（切片右开，所以用 335）

pred_matrix = df_pred.iloc[1:335, 1:145]
actual_matrix = df_actual.iloc[1:335, 1:145]

# 按行优先展平成一列
pred_vector = pred_matrix.values.flatten()
actual_vector = actual_matrix.values.flatten()

# 检查长度是否一致
if len(pred_vector) != len(actual_vector):
    raise ValueError(
        f"预测值和实际值展平后的长度不一致："
        f"{len(pred_vector)} vs {len(actual_vector)}"
    )

# 生成结果 DataFrame
result = pd.DataFrame({
    "预测值": pred_vector,
    "实际值": actual_vector
})

# 写出 Excel
result.to_excel(output_file, index=False)

print(f"完成，共 {len(result)} 行，已保存到：{output_file}")