import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.colors as col



def visualize(source_feature: torch.Tensor, target_feature: torch.Tensor,
              filename: str, source_color='r', target_color='b'):
    """
    Visualize features from different domains using t-SNE.

    Args:
        source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
        target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
        filename (str): the file name to save t-SNE
        source_color (str): the color of the source features. Default: 'r'
        target_color (str): the color of the target features. Default: 'b'

    """
    source_feature = source_feature.numpy()
    target_feature = target_feature.numpy()
    features = np.concatenate([source_feature, target_feature], axis=0)

    # map features to 2-d using TSNE
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)

    # domain labels, 1 represents source while 0 represents target
    domains = np.concatenate((np.ones(len(source_feature)), np.zeros(len(target_feature))))

    # visualize using matplotlib
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=domains, cmap=col.ListedColormap([target_color, source_color]), s=20)
    plt.xticks([])
    plt.yticks([])

    plt.savefig(filename)
    plt.savefig(filename.split('.')[0]+'.png')
    plt.savefig(filename.split('.')[0]+'.svg')
    plt.close()

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score  # 新增导入

def visualize222(source_feature: torch.Tensor, target_feature: torch.Tensor,
                 filename: str, source_color='r', target_color='b'):
    """
    Visualize features from different domains using t-SNE with Silhouette Score.
    """
    source_feature = source_feature.numpy()
    target_feature = target_feature.numpy()
    features = np.concatenate([source_feature, target_feature], axis=0)

    # t-SNE降维
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)

    # 生成域标签（源域1，目标域0）
    domains = np.concatenate((np.ones(len(source_feature)), np.zeros(len(target_feature))))

    # 计算轮廓系数（新增步骤）
    silhouette_avg = silhouette_score(X_tsne, domains)  # 使用降维后数据和域标签

    # 可视化
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    # 绘制散点图
    scatter = ax.scatter(
        X_tsne[:, 0], X_tsne[:, 1],
        c=domains,
        cmap=plt.cm.get_cmap('viridis', 2),  # 确保颜色与标签对应
        s=20,
        alpha=0.6
    )

    # 添加轮廓系数文本（关键新增部分）
    ax.text(
        0.05, 0.95,
        f'Silhouette Score: {silhouette_avg:.3f}',
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    plt.xticks([])
    plt.yticks([])
    plt.savefig(filename)
    plt.savefig(filename.split('.')[0]+'.png')
    plt.savefig(filename.split('.')[0]+'.svg')
    plt.close()



def visualize_(source_feature: torch.Tensor, target_feature: torch.Tensor,
              source_labels: torch.Tensor, target_labels: torch.Tensor,
              filename: str):
    """
    Visualize features from different domains using t-SNE with different shapes and colors for each class.

    Args:
        source_feature (torch.Tensor): features from source domain in shape (minibatch, F)
        target_feature (torch.Tensor): features from target domain in shape (minibatch, F)
        source_labels (torch.Tensor): labels of source domain samples
        target_labels (torch.Tensor): labels of target domain samples
        filename (str): the file name to save t-SNE plot
    """

    # Convert tensors to numpy arrays
    source_feature = source_feature.numpy()
    target_feature = target_feature.numpy()
    source_labels = source_labels.numpy()
    target_labels = target_labels.numpy()

    # Determine the number of classes and unique labels
    unique_source_labels = np.unique(source_labels)
    unique_target_labels = np.unique(target_labels)
    num_source_classes = len(unique_source_labels)
    num_target_classes = len(unique_target_labels)

    # Map features to 2D using t-SNE
    features = np.concatenate([source_feature, target_feature], axis=0)
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)

    # Define colors for each class
    class_colors = ["#33ccff","#33ffff", "#33ffaa", "#33ff33", "#ccff33", "#ffff33","#ffcc22","#ff3333", "#ffcc32","#ff3331"]
   
    plt.figure(figsize=(10, 10))

    # Plot source domain samples
    for i, label in enumerate(unique_source_labels):
        source_labels=source_labels.flatten()
        idx = source_labels == label
        # idx = (source_labels == label).astype(int)
        vv = len(source_labels)
        aa = X_tsne[:len(source_labels)]
        label_features = X_tsne[:len(source_labels)][idx]
        class_color = class_colors[i % num_source_classes]
        plt.scatter(label_features[:, 0], label_features[:, 1],
                    c=class_color, marker='o')

    # Plot target domain samples
    for i, label in enumerate(unique_target_labels):
        target_labels=target_labels.flatten()
        idx = target_labels == label
        # idx = (source_labels == label).astype(int)
        # a = X_tsne[len(source_labels):]
        label_features = X_tsne[len(target_labels):][idx]
        class_color = class_colors[i % num_target_classes]
        plt.scatter(label_features[:, 0], label_features[:, 1],
                    c=class_color, marker='^')

    plt.legend(loc='best')
    plt.savefig(filename, dpi=200, bbox_inches='tight')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize11(
        source_feature: torch.Tensor,
        target_feature: torch.Tensor,
        source_labels: torch.Tensor,
        target_labels: torch.Tensor,
        filename: str
):
    """
    改进说明：
    1. 修复目标域索引错误
    2. 添加特征与标签数量校验
    3. 使用全局颜色映射
    4. 添加图例支持
    """
    # 转换为NumPy数组
    source_feature = source_feature.cpu().numpy()
    target_feature = target_feature.cpu().numpy()
    source_labels = source_labels.cpu().numpy() # 确保标签为一维
    target_labels = target_labels.cpu().numpy()

    # 校验特征与标签数量
    assert len(source_feature) == len(source_labels), "Source特征与标签数量不匹配"
    assert len(target_feature) == len(target_labels), "Target特征与标签数量不匹配"

    # 合并特征并降维
    features = np.concatenate([source_feature, target_feature], axis=0)
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)

    # 使用标准颜色映射（支持10类）
    class_colors = plt.cm.tab10.colors

    plt.figure(figsize=(12, 10))

    # --- 绘制源域 ---
    source_tsne = X_tsne[:len(source_feature)]
    for label in np.unique(source_labels):
        mask = (source_labels == label)
        plt.scatter(
            source_tsne[mask, 0],
            source_tsne[mask, 1],
            c=[class_colors[int(label)]],
            marker='o',
            edgecolor='k',
            label=f'Source Class {label}'
        )

    # --- 绘制目标域 ---
    target_tsne = X_tsne[len(source_feature):]
    for label in np.unique(target_labels):
        mask = (target_labels == label)
        plt.scatter(
            target_tsne[mask, 0],
            target_tsne[mask, 1],
            c=[class_colors[int(label)]],
            marker='^',
            edgecolor='k',
            label=f'Target Class {label}'
        )

    # 图例优化（去重）
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    plt.legend(unique_labels.values(), unique_labels.keys(),
               loc='upper center', bbox_to_anchor=(0.5, -0.05),
               ncol=5, fontsize=8)

    plt.xticks([])
    plt.yticks([])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


import torch
import matplotlib

matplotlib.use('Agg')
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
def visualize1(source_feature: torch.Tensor, target_feature: torch.Tensor,
              filename: str, source_color='r', target_color='b'):
    """
    Visualize features from different domains using t-SNE.

    Args:
        source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
        target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
        filename (str): the file name to save t-SNE
        source_color (str): the color of the source features. Default: 'r'
        target_color (str): the color of the target features. Default: 'b'

    """
    source_feature = source_feature.numpy()
    target_feature = target_feature.numpy()
    features = np.concatenate([source_feature, target_feature], axis=0)

    # map features to 2-d using TSNE
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)

    # domain labels, 1 represents source while 0 represents target
    domains = np.concatenate((np.ones(len(source_feature)), np.zeros(len(target_feature))))

    # visualize using matplotlib
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=domains, cmap=col.ListedColormap([target_color, source_color]), s=20)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(filename)


import torch
import matplotlib
matplotlib.use('Agg')
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

def visualize2(
        source_feature: torch.Tensor,
        target_feature: torch.Tensor,
        source_labels: torch.Tensor,
        target_labels: torch.Tensor,
        filename: str,
        filename1: str
):
    # Convert features to NumPy
    source_feature = source_feature.cpu().numpy() if source_feature.is_cuda else source_feature.numpy()
    target_feature = target_feature.cpu().numpy() if target_feature.is_cuda else target_feature.numpy()

    # Convert labels to NumPy
    source_labels = source_labels.cpu().numpy() if source_labels.is_cuda else source_labels.numpy()
    target_labels = target_labels.cpu().numpy() if target_labels.is_cuda else target_labels.numpy()

    # 如果标签是二维数组且两个值相同，取其中一个值作为类别索引
    if source_labels.ndim == 2 and source_labels.shape[1] == 2:
        source_labels = source_labels[:, 0]
    if target_labels.ndim == 2 and target_labels.shape[1] == 2:
        target_labels = target_labels[:, 0]

    # 合并特征和标签
    features = np.concatenate([source_feature, target_feature], axis=0)
    all_labels = np.concatenate([source_labels, target_labels], axis=0)

    # t-SNE 降维到 2D
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)

    # 构造域标识：源域=1, 目标域=0
    domains = np.concatenate([
        np.ones(len(source_feature)),
        np.zeros(len(target_feature))
    ])

    # 准备画布
    fig, ax = plt.subplots(figsize=(4, 4))

    # 去掉周围边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # 使用 tab10 调色板（适合 10 个类别）
    cmap = plt.cm.get_cmap('tab10', 10)

    # 按类别循环绘制，共 10 个类别（0..9）
    for cat in range(10):
        # 找到源域中该类别的索引
        idx_source = np.where((all_labels == cat) & (domains == 1))[0]
        # 找到目标域中该类别的索引
        idx_target = np.where((all_labels == cat) & (domains == 0))[0]

        # 颜色从调色板中取
        color = cmap(cat)

        # 源域：圆形 'o'
        ax.scatter(
            X_tsne[idx_source, 0],
            X_tsne[idx_source, 1],
            c=[color],
            marker='o',
            label=f"Source_Cat_{cat}",
            edgecolors='green',
            s=100
        )

        # 目标域：正方形 's'
        ax.scatter(
            X_tsne[idx_target, 0],
            X_tsne[idx_target, 1],
            c=[color],
            marker='s',
            label=f"Target_Cat_{cat}",
            edgecolors='blue',
            alpha = 0.5,
            s=100
        )

    # 处理图例，去重
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # ax.legend(by_label.values(), by_label.keys(), loc='best')

    # 隐藏坐标轴刻度
    plt.xticks([])
    plt.yticks([])

    # 保存结果
    plt.savefig(filename)
    plt.close(fig)

import torch
import matplotlib
matplotlib.use('Agg')
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

def visualize_target_only(
        target_feature: torch.Tensor,
        target_labels: torch.Tensor,
        filename: str,
        filename1: str
):
    # Convert features to NumPy
    target_feature = target_feature.cpu().numpy() if target_feature.is_cuda else target_feature.numpy()

    # Convert labels to NumPy
    target_labels = target_labels.cpu().numpy() if target_labels.is_cuda else target_labels.numpy()

    # 如果标签是二维数组且两个值相同，取其中一个值作为类别索引
    if target_labels.ndim == 2 and target_labels.shape[1] == 2:
        target_labels = target_labels[:, 0]

    # t-SNE 降维到 2D
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(target_feature)

    # 准备画布
    fig, ax = plt.subplots(figsize=(5, 5))

    # 去掉周围边框
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    # 使用 tab10 调色板（适合 10 个类别）
    cmap = plt.cm.get_cmap('tab10', 10)

    # 按类别循环绘制，共 10 个类别（0..9）
    for cat in range(10):
        # 找到目标域中该类别的索引
        idx_target = np.where(target_labels == cat)[0]

        # 颜色从调色板中取
        color = cmap(cat)

        # 目标域：空心圆圈 'o'
        ax.scatter(
            X_tsne[idx_target, 0],
            X_tsne[idx_target, 1],
            # edgecolors=color,
            # facecolors='none',  # 确保是空心
            # marker='o',
            label=f"Target_Cat_{cat}",
            alpha=0.8,
            s=50  # 调整点的大小
        )

    # 处理图例，去重
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # ax.legend(by_label.values(), by_label.keys(), loc='best')

    # 隐藏坐标轴刻度
    plt.xticks([])
    plt.yticks([])

    # 保存结果
    plt.savefig(filename)
    plt.savefig(filename1)
    plt.close(fig)



