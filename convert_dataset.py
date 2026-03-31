"""
数据集转换脚本
=============
将 data.npz / data.mat 格式的原始文件转换为本仓库可直接使用的格式，
并自动生成 train/val/test CSV 索引文件。

使用方法
-------
将你所有的原始数据文件放在某个目录下（每个文件对应一条录制），
然后运行：
    python convert_dataset.py \
        --input_dir  /path/to/raw_files \
        --output_dir ./dataset \
        --domain     source          # source 或 target
        --domain_name 15-0           # 与 data_loader.py 里的 key 对应

文件命名约定（原始文件）
----------------------
支持两种格式：
  1. data.npz  包含 bin(3000,1,1,30) 和 index([user_id, session, word_label])
  2. data.mat  包含 feature(512,128)  和 cond([user_id, ?, label, ...])

建议批量文件命名为 user{u}_sess{s}_word{w}.npz 或同名 .mat，
也可以将 user_id / session / word_label 编码在文件名里，脚本会自动解析。
"""

import os
import csv
import argparse
import random
import numpy as np
import scipy.io as sio
from pathlib import Path


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def load_raw_npz(path: str) -> dict:
    """
    读取原始 npz 文件，返回统一结构：
      {
        'raw'  : ndarray (frames, 1, 1, range_bins),  complex64
        'label': int,
        'user' : int,
      }
    """
    d = np.load(path, allow_pickle=True)
    raw   = d['bin']                   # (3000, 1, 1, 30)
    index = d['index']                 # [user_id, session, word_label]
    return {'raw': raw, 'label': int(index[2]), 'user': int(index[0])}


def load_raw_mat(path: str) -> dict:
    """
    读取原始 mat 文件，返回统一结构：
      {
        'raw'  : ndarray (frames, 128),  complex128  (已是谱图，无需再 STFT)
        'label': int,
        'user' : int,
      }
    """
    d = sio.loadmat(path)
    feat = d['feature']                # (512, 128)
    cond = d['cond'][0]                # [user_id, ?, label, ...]
    return {'raw': feat, 'label': int(cond[2]), 'user': int(cond[0])}


def convert_and_save_npz(raw_dict: dict, save_path: str):
    """
    将原始字典包装成仓库 data_loader 期望的 npz 格式并保存。

    仓库原始期望：
        np.load(path)['data'].item()['lip']['raw']
        shape: (frames, D, D, D)  用于取 [:, i-1,i-1,i-1] 等

    本脚本生成一个兼容格式，同时 data_loader.py 也会相应调整（见下文）：
        np.load(path)['data'].item()['lip']['raw']
        shape: (frames, 1, 1, range_bins) —— range_bins 在最后一维
    """
    raw = raw_dict['raw']              # (frames, 1, 1, 30) 或 (frames, 128)

    # 统一包装成仓库期望的嵌套字典对象
    data_obj = {'lip': {'raw': raw}}
    np.savez_compressed(save_path, data=np.array(data_obj, dtype=object))


# ── 主流程 ───────────────────────────────────────────────────────────────────

def build_dataset(input_dir: str, output_dir: str, domain: str, domain_name: str,
                  train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    遍历 input_dir 下所有 .npz / .mat 文件，转换后存入 output_dir，
    并生成 train.csv / val.csv / test.csv / database.csv。

    CSV 格式（与仓库一致）：
        相对路径/filename.npz, label
    """
    random.seed(seed)
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)

    # 输出目录
    data_save_dir = output_dir / 'data' / domain_name
    csv_save_dir  = output_dir / f'data_path_csv/{domain_name}_{domain}'
    data_save_dir.mkdir(parents=True, exist_ok=True)
    csv_save_dir.mkdir(parents=True, exist_ok=True)

    # 收集所有原始文件
    all_files = sorted(list(input_dir.glob('*.npz')) + list(input_dir.glob('*.mat')))
    if not all_files:
        raise FileNotFoundError(f"在 {input_dir} 下没有找到 .npz 或 .mat 文件")

    print(f"找到 {len(all_files)} 个文件，开始转换 ...")

    records = []   # [(saved_relative_path, label), ...]

    for src_path in all_files:
        ext = src_path.suffix.lower()

        # 加载原始数据
        if ext == '.npz':
            raw_dict = load_raw_npz(str(src_path))
        elif ext == '.mat':
            raw_dict = load_raw_mat(str(src_path))
        else:
            continue

        label = raw_dict['label']

        # 保存为仓库期望的 npz
        dst_filename = src_path.stem + '_converted.npz'
        dst_path     = data_save_dir / dst_filename
        convert_and_save_npz(raw_dict, str(dst_path))

        # CSV 里记录相对路径（相对于仓库根目录 root 参数）
        relative_path = str(dst_path)   # 绝对路径；训练时通过 root='' 传入完整路径也行
        records.append((relative_path, label))

        print(f"  {src_path.name}  label={label}  ->  {dst_filename}")

    # 打乱并划分
    random.shuffle(records)
    n = len(records)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    splits = {
        'train'   : records[:n_train],
        'val'     : records[n_train:n_train + n_val],
        'test'    : records[n_train + n_val:],
        'database': records,           # 全量用于检索
    }

    for split_name, split_records in splits.items():
        csv_path = csv_save_dir / f'{split_name}.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for path, label in split_records:
                writer.writerow([path, label])
        print(f"写入 {csv_path}  ({len(split_records)} 条)")

    print(f"\n✅ 完成！数据目录: {data_save_dir}")
    print(f"   CSV  目录: {csv_save_dir}")
    print(f"\n训练时请使用以下参数：")
    print(f"  --source {domain_name}  (若为 source domain)")
    print(f"  --target {domain_name}  (若为 target domain)")
    print(f"  --root   ''            (路径已写入 CSV)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',   required=True,  help='存放原始 .npz/.mat 文件的目录')
    parser.add_argument('--output_dir',  default='.',    help='输出根目录（默认当前目录）')
    parser.add_argument('--domain',      default='source', choices=['source', 'target'])
    parser.add_argument('--domain_name', default='15-0', help='与 data_loader.py dataset_dict 中的 key 对应')
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio',   type=float, default=0.15)
    parser.add_argument('--seed',        type=int,   default=42)
    args = parser.parse_args()

    build_dataset(
        input_dir   = args.input_dir,
        output_dir  = args.output_dir,
        domain      = args.domain,
        domain_name = args.domain_name,
        train_ratio = args.train_ratio,
        val_ratio   = args.val_ratio,
        seed        = args.seed,
    )
