import os
import csv

BASE = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER = os.path.join(BASE, "dataset")
CSV_BASE = os.path.join(BASE, "data_path_csv")

NAME_SOURCE = "15-0"
NAME_TARGET = "15+30"

os.makedirs(f"{CSV_BASE}/{NAME_SOURCE}_source", exist_ok=True)
os.makedirs(f"{CSV_BASE}/{NAME_TARGET}_target", exist_ok=True)

# 输入你的人名简写
key_input = input("请输入分类简写（逗号分隔）：").strip()
key_input = key_input.replace("，", ",")
keywords = [kw.strip() for kw in key_input.split(",") if kw.strip()]

# 标签从 0 开始
name_to_id = {kw: i for i, kw in enumerate(keywords)}

lines = []
for root, _, files in os.walk(DATASET_FOLDER):
    for file in files:
        if not file.endswith((".npz", ".mat")):
            continue

        full_path = os.path.join(root, file)
        rel_path = os.path.relpath(full_path, BASE)
        path_str = rel_path.replace("\\", "/")

        # 遍历所有名字，确保匹配到
        class_id = -1
        for name, idx in name_to_id.items():
            if f"/{name}/" in path_str:
                class_id = idx

        if class_id == -1:
            class_id = 0

        lines.append([rel_path, class_id])

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
    # 用utf-8
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(lines)

print("CSV 生成完成")
for name, idx in name_to_id.items():
    print(f"{name} → {idx}")
