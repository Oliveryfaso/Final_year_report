# ots_MNIST.py

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from torchvision import transforms  # 导入 transforms

def extract_features(model, data_loader, device):
    model.eval()
    features = []
    labels = []
    # 定义 ResNet18 的标准化参数
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc='提取特征', unit='batch')
        for images, lbls in progress_bar:
            images = images.to(device)

            # # 将图像从1通道复制为3通道
            # if images.shape[1] == 1:
            #     images = images.repeat(1, 3, 1, 1)

            # # 应用标准化
            # images = normalize(images)

            outputs = model(images)
            features.append(outputs.cpu().numpy())
            labels.extend(lbls.numpy())
    features = np.concatenate(features, axis=0)
    labels = np.array(labels)
    return features, labels

# ots_MNIST.py

def evaluate_model(model, known_test_loader, open_set_loader, device, threshold=0.5, openmax=None):
    model.eval()
    all_true_labels = []
    all_pred_labels = []
    with torch.no_grad():
        # 评估已知测试集
        print("评估已知测试集...")
        for images, labels in tqdm(known_test_loader, desc='已知测试集评估', unit='batch'):
            images = images.to(device)
            labels = labels.to(device, dtype=torch.long)
            labels = labels - 1  # 调整标签范围到 [0,22]

            # 解包模型输出
            features, logits = model(images)  # 确保模型返回(features, logits)
            features = features.cpu().numpy()
            logits = logits.cpu().numpy()
            softmax_outputs = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()

            if openmax:
                # 使用OpenMax调整后的概率分布
                adjusted_probs = openmax.compute_adjusted_scores(features, logits)  # 形状: (batch_size, num_classes)
                preds = np.argmax(adjusted_probs, axis=1)
            else:
                # 使用原始Softmax概率
                max_probs = np.max(softmax_outputs, axis=1)
                preds = np.argmax(softmax_outputs, axis=1)

                # 仅在不使用OpenMax时应用阈值
                preds[max_probs < threshold] = 0  # 将概率低于阈值的预测为 'Unknown' 类
                preds = preds.astype(int)

            # 记录真实标签和预测标签
            all_true_labels.extend((labels.cpu().numpy() + 1).tolist())  # 真实标签为 [1-23]
            all_pred_labels.extend(preds.tolist())  # 预测标签为 [0-23]

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

            # 开放集真实标签为0
            all_true_labels.extend([0] * len(preds))
            all_pred_labels.extend(preds.tolist())

    # 后续评估指标计算保持不变
    num_classes = 24  # 包含 'Unknown' 类
    class_labels = list(range(num_classes))
    target_names = ['Unknown'] + [f'Class_{i}' for i in range(1, num_classes)]

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

