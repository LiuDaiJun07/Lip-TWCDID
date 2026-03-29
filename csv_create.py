import os
import csv

# ======================
# 自动生成 CSV
# ======================

BASE = "/data/coding"
DATASET_FOLDER = os.path.join(BASE, "dataset")
CSV_BASE = os.path.join(BASE, "data_path_csv")

# 必须和代码一致
NAME_SOURCE = "15-0"
NAME_TARGET = "15+30"

os.makedirs(f"{CSV_BASE}/{NAME_SOURCE}_source", exist_ok=True)
os.makedirs(f"{CSV_BASE}/{NAME_TARGET}_target", exist_ok=True)

mat_files = [f for f in os.listdir(DATASET_FOLDER) if f.endswith(".npz")]
mat_files.sort()

lines = []
for i, f in enumerate(mat_files):
    lines.append([f"dataset/{f}", i])  # <-- 自动加 dataset/

csv_list = [
    f"{NAME_SOURCE}_source/train.csv",
    f"{NAME_SOURCE}_source/val.csv",
    f"{NAME_SOURCE}_source/test.csv",
    f"{NAME_SOURCE}_source/database.csv",
    f"{NAME_TARGET}_target/train.csv",
    f"{NAME_TARGET}_target/val.csv",
    f"{NAME_TARGET}_target/test.csv",
    f"{NAME_TARGET}_target/database.csv",
]

for csv_file in csv_list:
    path = os.path.join(CSV_BASE, csv_file)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(lines)

print("✅ CSV 已写入数据！")