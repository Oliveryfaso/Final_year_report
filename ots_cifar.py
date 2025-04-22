# ots_cifar.py

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from torchvision import transforms

def extract_features(model, data_loader, device):
    """
    提取模型的特征表示。

    Args:
        model (nn.Module): 已训练的模型。
        data_loader (DataLoader): 数据加载器。
        device (torch.device): 设备信息。

    Returns:
        tuple: 特征数组和对应的标签数组。
    """
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc='提取特征', unit='batch')
        for images, lbls in progress_bar:
            images = images.to(device)
            outputs, _ = model(images)
            features.append(outputs.cpu().numpy())
            labels.extend(lbls.numpy())
    features = np.concatenate(features, axis=0)
    labels = np.array(labels)
    return features, labels

def evaluate_model(model, known_test_loader, open_set_loader, device, threshold=0.5, openmax=None, discriminator=None):
    """
    评估模型在已知测试集和开放集上的性能，结合判别器进行双重验证。

    Args:
        model (nn.Module): 已训练的模型。
        known_test_loader (DataLoader): 已知类别的测试集数据加载器。
        open_set_loader (DataLoader): 开放集数据加载器（Unknown 类别）。
        device (torch.device): 设备信息。
        threshold (float, optional): Softmax 概率阈值，低于该阈值的预测被归类为 'Unknown'。默认值为 0.5。
        openmax (OpenMax, optional): OpenMax 实例，用于调整概率分布。默认值为 None。
        discriminator (nn.Module, optional): 判别器模型，用于双重验证。默认值为 None。

    Returns:
        dict: 包含各种评估指标的字典。
    """
    model.eval()
    all_true_labels = []
    all_pred_labels = []
    if discriminator:
        discriminator.eval()

    with torch.no_grad():
        # 评估已知测试集
        print("评估已知测试集...")
        for images, labels in tqdm(known_test_loader, desc='已知测试集评估', unit='batch'):
            images = images.to(device)
            labels = labels.to(device, dtype=torch.long)
            labels = labels.clone()
            labels[labels == 0] = 0  # 'airplane' remains 0 (Unknown)
            labels[labels != 0] += 0  # 其他类别保持不变，确保标签在1-9范围内

            # 解包模型输出
            features, logits = model(images)  # 确保模型返回 (features, logits)
            features = features.cpu().numpy()
            logits = logits.cpu().numpy()
            softmax_outputs = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()

            if openmax:
                # 使用 OpenMax 调整后的概率分布
                adjusted_probs = openmax.compute_adjusted_scores(features, logits)  # 形状: (batch_size, num_classes)
                preds = np.argmax(adjusted_probs, axis=1)
            else:
                # 使用原始 Softmax 概率
                max_probs = np.max(softmax_outputs, axis=1)
                preds = np.argmax(softmax_outputs, axis=1)

                # 仅在不使用 OpenMax 时应用阈值
                preds[max_probs < threshold] = 0  # 将概率低于阈值的预测为 'Unknown' 类
                preds = preds.astype(int)

            # 记录真实标签和预测标签
            all_true_labels.extend(labels.cpu().numpy().tolist())  # 真实标签为 [0-9]
            all_pred_labels.extend(preds.tolist())  # 预测标签为 [0-9]

        # 评估开放集
        print("评估开放集...")
        for images, _ in tqdm(open_set_loader, desc='开放集评估', unit='batch'):
            images = images.to(device)

            # 同样解包
            features, logits = model(images)
            features = features.cpu().numpy()
            logits = logits.cpu().numpy()
            softmax_outputs = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()

            if openmax:
                adjusted_probs = openmax.compute_adjusted_scores(features, logits)
                preds = np.argmax(adjusted_probs, axis=1)
            else:
                max_probs = np.max(softmax_outputs, axis=1)
                preds = np.argmax(softmax_outputs, axis=1)
                preds[max_probs < threshold] = 0  # 将概率低于阈值的预测为 'Unknown' 类
                preds = preds.astype(int)

            # 如果有判别器，进一步验证 'Unknown' 类样本
            if discriminator:
                unknown_indices = np.where(preds == 0)[0]
                if len(unknown_indices) > 0:
                    unknown_images = images[unknown_indices]
                    real_or_fake, _, _ = discriminator(unknown_images)
                    real_or_fake = torch.sigmoid(real_or_fake).cpu().numpy()

                    # 根据判别器输出调整预测
                    for idx, val in zip(unknown_indices, real_or_fake):
                        if val < threshold:
                            preds[idx] = 0  # 继续保持为未知类
                        else:
                            # 这里可以选择将其重新分类为某个已知类，或者保持为未知类
                            # 例如，保持为未知类
                            preds[idx] = 0

            # 开放集真实标签为0 ('Unknown')
            all_true_labels.extend([0] * len(preds))
            all_pred_labels.extend(preds.tolist())

    # 计算评估指标
    num_classes = 10  # 包含 'Unknown' 类别
    class_labels = list(range(num_classes))
    target_names = ['Unknown', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

    classification_rep = classification_report(
        all_true_labels, all_pred_labels, digits=4, output_dict=True, zero_division=0, labels=class_labels, target_names=target_names
    )
    conf_matrix = confusion_matrix(all_true_labels, all_pred_labels, labels=class_labels)

    precision = precision_score(all_true_labels, all_pred_labels, average='macro', labels=class_labels, zero_division=0)
    recall = recall_score(all_true_labels, all_pred_labels, average='macro', labels=class_labels, zero_division=0)
    f1 = f1_score(all_true_labels, all_pred_labels, average='macro', labels=class_labels, zero_division=0)

    metrics = {
        'accuracy': accuracy_score(all_true_labels, all_pred_labels),
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'classification_report': classification_rep
    }

    print("Evaluation metrics calculated.")
    return metrics
