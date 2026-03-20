"""
Question 3: CNN for K-line Feature Recognition - Data Augmentation Techniques

使用CNN识别K线(蜡烛图)特征时，适当与不适当的数据增强技术分析及实现。

适当的增强方法:
  - 添加价格噪声 (Gaussian noise)
  - 时间窗口裁剪/滑动 (Time window cropping)
  - 成交量缩放 (Volume scaling)
  - 价格缩放/归一化变体 (Price scaling)
  - 亮度/对比度微调 (用于图像表示)
  - Dropout/Cutout (遮挡部分区域)

必须严格避免的增强方法:
  - 水平翻转 (Horizontal flip) → 逆转时间轴,未来变过去
  - 垂直翻转 (Vertical flip) → 牛市变熊市,含义完全颠倒
  - 随机旋转 (Random rotation) → 破坏时间轴和价格轴的固有方向
  - 随机平移时间轴 → 可能破坏形态完整性
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# K线数据生成与图像化
# ============================================================

def generate_kline_patterns(n_samples: int = 500, window: int = 20, seed: int = 42):
    """
    生成合成K线数据集(4类，难度较高):
    - Label 0: 上升趋势 (Bullish Trend)
    - Label 1: 下降趋势 (Bearish Trend)
    - Label 2: 顶部反转 (Top Reversal) - 先涨后跌
    - Label 3: 底部反转 (Bottom Reversal) - 先跌后涨
    高噪声使得形态不易区分,考验模型泛化能力。
    """
    np.random.seed(seed)
    data = []
    labels = []
    noise_std = 0.6

    for _ in range(n_samples):
        label = np.random.randint(0, 4)
        t = np.linspace(0, 1, window)

        if label == 0:
            trend = t * 1.5 + np.random.randn(window) * noise_std
        elif label == 1:
            trend = (1 - t) * 1.5 + np.random.randn(window) * noise_std
        elif label == 2:
            trend = -2.0 * (t - 0.4) ** 2 + 0.5 + np.random.randn(window) * noise_std
        else:
            trend = 2.0 * (t - 0.6) ** 2 - 0.2 + np.random.randn(window) * noise_std

        opens = trend + np.random.randn(window) * 0.15
        direction = 0.08 if label in (0, 3) else -0.08
        closes = opens + direction + np.random.randn(window) * 0.2
        highs = np.maximum(opens, closes) + np.abs(np.random.randn(window)) * 0.25
        lows = np.minimum(opens, closes) - np.abs(np.random.randn(window)) * 0.25
        volumes = np.random.exponential(100, window)

        sample = np.stack([opens, highs, lows, closes, volumes], axis=-1)
        data.append(sample)
        labels.append(label)

    return np.array(data), np.array(labels)


def kline_to_image(ohlcv: np.ndarray, img_size: int = 64) -> np.ndarray:
    """将OHLCV数据转为简化的K线图图像 (grayscale)"""
    img = np.zeros((img_size, img_size), dtype=np.float32)
    n_bars = ohlcv.shape[0]

    prices = ohlcv[:, :4]
    p_min, p_max = prices.min(), prices.max()
    if p_max == p_min:
        p_max = p_min + 1

    bar_width = max(1, img_size // n_bars)

    for i in range(n_bars):
        x = int(i * img_size / n_bars)
        o, h, l, c = ohlcv[i, :4]

        y_h = int((1 - (h - p_min) / (p_max - p_min)) * (img_size - 1))
        y_l = int((1 - (l - p_min) / (p_max - p_min)) * (img_size - 1))
        y_o = int((1 - (o - p_min) / (p_max - p_min)) * (img_size - 1))
        y_c = int((1 - (c - p_min) / (p_max - p_min)) * (img_size - 1))

        # 影线
        for y in range(min(y_h, y_l), max(y_h, y_l) + 1):
            if 0 <= y < img_size and 0 <= x < img_size:
                img[y, x] = 0.5

        # 实体
        body_top, body_bot = min(y_o, y_c), max(y_o, y_c)
        intensity = 1.0 if c >= o else 0.7
        for y in range(body_top, body_bot + 1):
            for dx in range(bar_width):
                if 0 <= y < img_size and 0 <= x + dx < img_size:
                    img[y, x + dx] = intensity

    return img


# ============================================================
# 数据增强方法
# ============================================================

class KLineAugmentation:
    """K线数据增强工具集"""

    # --- 适当的增强方法 ---

    @staticmethod
    def add_price_noise(ohlcv: np.ndarray, std: float = 0.05) -> np.ndarray:
        """
        [适当] 添加高斯价格噪声
        模拟市场微观结构噪声,不改变形态语义
        """
        augmented = ohlcv.copy()
        noise = np.random.randn(*ohlcv[:, :4].shape) * std
        augmented[:, :4] += noise
        augmented[:, 1] = np.maximum(augmented[:, 1], np.maximum(augmented[:, 0], augmented[:, 3]))
        augmented[:, 2] = np.minimum(augmented[:, 2], np.minimum(augmented[:, 0], augmented[:, 3]))
        return augmented

    @staticmethod
    def scale_price(ohlcv: np.ndarray, factor_range: tuple = (0.8, 1.2)) -> np.ndarray:
        """
        [适当] 价格整体缩放
        不改变形态的相对关系,仅改变绝对价格水平
        """
        augmented = ohlcv.copy()
        factor = np.random.uniform(*factor_range)
        augmented[:, :4] *= factor
        return augmented

    @staticmethod
    def scale_volume(ohlcv: np.ndarray, factor_range: tuple = (0.5, 2.0)) -> np.ndarray:
        """
        [适当] 成交量缩放
        模拟不同流动性环境
        """
        augmented = ohlcv.copy()
        factor = np.random.uniform(*factor_range)
        augmented[:, 4] *= factor
        return augmented

    @staticmethod
    def time_crop(ohlcv: np.ndarray, crop_ratio: float = 0.8) -> np.ndarray:
        """
        [适当] 时间窗口随机裁剪
        保持时间顺序不变,仅截取子序列
        """
        n = len(ohlcv)
        crop_len = max(int(n * crop_ratio), 5)
        start = np.random.randint(0, n - crop_len + 1)
        return ohlcv[start:start + crop_len]

    @staticmethod
    def image_cutout(img: np.ndarray, n_holes: int = 2, hole_size: int = 8) -> np.ndarray:
        """
        [适当] Cutout: 随机遮挡图像小区域
        增强模型对局部遮挡的鲁棒性
        """
        augmented = img.copy()
        h, w = img.shape[:2]
        for _ in range(n_holes):
            y = np.random.randint(0, h)
            x = np.random.randint(0, w)
            y1, y2 = max(0, y - hole_size // 2), min(h, y + hole_size // 2)
            x1, x2 = max(0, x - hole_size // 2), min(w, x + hole_size // 2)
            augmented[y1:y2, x1:x2] = 0
        return augmented

    @staticmethod
    def brightness_contrast(img: np.ndarray, b_range: tuple = (-0.1, 0.1),
                            c_range: tuple = (0.8, 1.2)) -> np.ndarray:
        """
        [适当] 亮度/对比度微调
        模拟不同渲染条件,不改变形态含义
        """
        brightness = np.random.uniform(*b_range)
        contrast = np.random.uniform(*c_range)
        augmented = img * contrast + brightness
        return np.clip(augmented, 0, 1)

    # --- 必须严格避免的增强方法 ---

    @staticmethod
    def horizontal_flip_FORBIDDEN(ohlcv: np.ndarray) -> np.ndarray:
        """
        [禁止!!!] 水平翻转 = 时间轴反转
        原因: 将时间序列倒序,上升趋势变为下降趋势(从结尾看),
        破坏因果关系,使模型学到不存在的形态
        """
        return ohlcv[::-1].copy()

    @staticmethod
    def vertical_flip_FORBIDDEN(ohlcv: np.ndarray) -> np.ndarray:
        """
        [禁止!!!] 垂直翻转 = 价格轴反转
        原因: 阳线变阴线,涨变跌,支撑变阻力,
        完全颠覆K线形态的语义含义
        """
        augmented = ohlcv.copy()
        price_max = augmented[:, :4].max()
        augmented[:, :4] = price_max - augmented[:, :4]
        # open↔close meaning也被反转了
        augmented[:, [0, 3]] = augmented[:, [3, 0]]
        return augmented

    @staticmethod
    def random_rotation_FORBIDDEN(img: np.ndarray, angle: float = 45) -> np.ndarray:
        """
        [禁止!!!] 随机旋转
        原因: K线图的X轴(时间)和Y轴(价格)有固定含义,
        旋转后时间不再从左到右流动,价格轴也被扭曲
        """
        from scipy.ndimage import rotate
        return rotate(img, angle, reshape=False, mode='constant', cval=0)


# ============================================================
# CNN模型
# ============================================================

class KLineCNN(nn.Module):
    """用于K线图像分类的轻量CNN"""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class KLineDataset(Dataset):
    """支持增强的K线数据集"""

    def __init__(self, ohlcv_data, labels, augment=False,
                 forbidden_augment=False, img_size=64):
        self.data = ohlcv_data
        self.labels = labels
        self.augment = augment
        self.forbidden_augment = forbidden_augment
        self.img_size = img_size
        self.aug = KLineAugmentation()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ohlcv = self.data[idx].copy()

        if self.augment:
            if np.random.random() < 0.5:
                ohlcv = self.aug.add_price_noise(ohlcv, std=0.05)
            if np.random.random() < 0.3:
                ohlcv = self.aug.scale_price(ohlcv)
            if np.random.random() < 0.3:
                ohlcv = self.aug.scale_volume(ohlcv)

        if self.forbidden_augment:
            if np.random.random() < 0.5:
                ohlcv = self.aug.horizontal_flip_FORBIDDEN(ohlcv)
            if np.random.random() < 0.5:
                ohlcv = self.aug.vertical_flip_FORBIDDEN(ohlcv)

        img = kline_to_image(ohlcv, self.img_size)

        if self.augment:
            if np.random.random() < 0.3:
                img = self.aug.image_cutout(img)
            if np.random.random() < 0.3:
                img = self.aug.brightness_contrast(img)

        img_tensor = torch.FloatTensor(img).unsqueeze(0)  # (1, H, W)
        label = torch.LongTensor([self.labels[idx]])[0]
        return img_tensor, label


# ============================================================
# 训练与评估
# ============================================================

def train_and_evaluate(use_augmentation: bool, forbidden_augmentation: bool = False,
                       epochs: int = 20):
    """
    训练CNN并评估。
    - use_augmentation: 使用适当的数据增强
    - forbidden_augmentation: 使用禁止的增强(水平/垂直翻转),用于对照实验
    """
    train_data, train_labels = generate_kline_patterns(n_samples=600, seed=42)
    test_data, test_labels = generate_kline_patterns(n_samples=400, seed=99)

    train_dataset = KLineDataset(
        train_data, train_labels,
        augment=use_augmentation,
        forbidden_augment=forbidden_augmentation,
    )
    test_dataset = KLineDataset(test_data, test_labels, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    model = KLineCNN(num_classes=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    test_accs = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for imgs, lbls in train_loader:
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, lbls in test_loader:
                preds = model(imgs).argmax(dim=1)
                correct += (preds == lbls).sum().item()
                total += len(lbls)
        acc = correct / total
        test_accs.append(acc)

    return train_losses, test_accs


def visualize_augmentations():
    """可视化各种增强方法的效果"""
    data, labels = generate_kline_patterns(n_samples=1, window=20, seed=10)
    sample = data[0]
    aug = KLineAugmentation()

    augmentations = {
        'Original': sample,
        '+ Price Noise\n(Appropriate)': aug.add_price_noise(sample, 0.1),
        '+ Price Scale\n(Appropriate)': aug.scale_price(sample, (0.7, 0.7)),
        '+ Volume Scale\n(Appropriate)': aug.scale_volume(sample, (2.0, 2.0)),
        'Horizontal Flip\n(FORBIDDEN!)': aug.horizontal_flip_FORBIDDEN(sample),
        'Vertical Flip\n(FORBIDDEN!)': aug.vertical_flip_FORBIDDEN(sample),
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (title, ohlcv) in zip(axes.flatten(), augmentations.items()):
        img = kline_to_image(ohlcv, 64)
        ax.imshow(img, cmap='gray', aspect='auto')
        color = 'red' if 'FORBIDDEN' in title else ('green' if 'Appropriate' in title else 'blue')
        ax.set_title(title, fontsize=11, fontweight='bold', color=color)
        ax.axis('off')

    fig.suptitle('K-Line Data Augmentation: Appropriate vs Forbidden Methods',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('assignment/Q3_augmentation_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_training_comparison():
    """对比三组实验: 无增强 / 适当增强 / 禁止增强(水平+垂直翻转)"""
    print("训练 CNN (无增强)...")
    loss_no, acc_no = train_and_evaluate(
        use_augmentation=False, epochs=30)

    print("训练 CNN (适当增强: 噪声+缩放+Cutout)...")
    loss_good, acc_good = train_and_evaluate(
        use_augmentation=True, epochs=30)

    print("训练 CNN (禁止增强: 水平翻转+垂直翻转)...")
    loss_bad, acc_bad = train_and_evaluate(
        use_augmentation=False, forbidden_augmentation=True, epochs=30)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(loss_no, 'gray', linewidth=1.5, label='No Augmentation')
    axes[0].plot(loss_good, 'b-', linewidth=1.5, label='Appropriate Aug.')
    axes[0].plot(loss_bad, 'r--', linewidth=1.5, label='Forbidden Aug. (H/V Flip)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Loss')
    axes[0].set_title('Training Loss Comparison (4-class K-line)')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(acc_no, 'gray', linewidth=1.5, label='No Augmentation')
    axes[1].plot(acc_good, 'b-', linewidth=1.5, label='Appropriate Aug.')
    axes[1].plot(acc_bad, 'r--', linewidth=1.5, label='Forbidden Aug. (H/V Flip)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Test Accuracy')
    axes[1].set_title('Test Accuracy Comparison (4-class K-line)')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig('assignment/Q3_training_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n最终测试准确率 (4类K线形态分类):")
    print(f"  无增强:       {acc_no[-1]:.2%}")
    print(f"  适当增强:     {acc_good[-1]:.2%}")
    print(f"  禁止增强(翻转): {acc_bad[-1]:.2%}")


if __name__ == '__main__':
    print("=" * 60)
    print("Question 3: CNN K-Line Data Augmentation")
    print("=" * 60)

    visualize_augmentations()
    plot_training_comparison()
