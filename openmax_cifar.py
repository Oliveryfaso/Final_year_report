# openmax_cifar.py

import numpy as np
import pickle
from scipy import stats
import torch.nn.functional as F  # 确保引入 F

class OpenMax:
    def __init__(self, activations_path, tail_size=20, weibull_params_path='./weibulls_cifar.pkl', alpha=0.3, num_classes=10):
        """
        初始化OpenMax类。

        Args:
            activations_path (str): 保存的激活数据（pickle文件）的路径。
            tail_size (int): 用于Weibull拟合的最大距离数量。
            weibull_params_path (str): Weibull模型参数的保存路径。
            alpha (float): 缩放因子，用于调整OpenMax得分的影响力。
            num_classes (int): 总类别数（包括 'Unknown' 类别）。
        """
        try:
            with open(activations_path, 'rb') as f:
                self.activations = pickle.load(f)
            # print(f"激活数据已从 {activations_path} 加载。")
        except FileNotFoundError:
            # print(f"激活数据文件未找到: {activations_path}")
            raise
        except Exception as e:
            # print(f"加载激活数据时发生错误: {e}")
            raise

        self.tail_size = tail_size
        self.weibull_models = {}
        self.class_means = {}
        self.weibull_params_path = weibull_params_path
        self.alpha = alpha  # 添加缩放因子
        self.num_classes = num_classes  # 总类别数

        # 存储训练时的类别标签
        self.train_labels = list(self.activations.keys())
        # print("训练时的类别标签:", self.train_labels)

        self._fit_weibulls()

    def _fit_weibulls(self):
        """
        为每个已知类别（1-num_classes-1）拟合Weibull分布，并保存Weibull模型参数。
        """
        for cls, feats in self.activations.items():
            if cls == 0:
                # 跳过 'Unknown' 类
                continue
            class_label = cls  # 类别标签保持为1-num_classes-1
            feats = np.array(feats)
            # print(f"类别 {class_label} 的样本数量: {len(feats)}")
            if feats.size == 0:
                # print(f"类别 {class_label} 的激活数据为空，跳过拟合。")
                continue
            mean_feat = np.mean(feats, axis=0)
            self.class_means[class_label] = mean_feat

            # 计算特征与均值的距离
            distances = np.linalg.norm(feats - mean_feat, axis=1)
            if len(distances) < self.tail_size:
                tail_distances = distances
                # print(f"类别 {class_label} 的样本数少于 tail_size={self.tail_size}，使用所有距离进行拟合。")
            else:
                # 取距离最大的tail_size个距离进行Weibull拟合
                tail_distances = np.sort(distances)[-self.tail_size:]

            # 拟合Weibull分布
            try:
                params = stats.weibull_min.fit(tail_distances, floc=0)
                self.weibull_models[class_label] = params
                # print(f"类别 {class_label} 的Weibull模型拟合成功。参数: shape={params[0]:.4f}, loc={params[1]:.4f}, scale={params[2]:.4f}")
            except Exception as e:
                # print(f"拟合类别 {class_label} 的Weibull模型时发生错误: {e}")
                pass

        # 保存Weibull模型参数
        try:
            with open(self.weibull_params_path, 'wb') as f:
                pickle.dump({
                    'weibull_models': self.weibull_models,
                    'class_means': self.class_means
                }, f)
            print(f"Weibull模型参数已保存到 {self.weibull_params_path}")
        except Exception as e:
            # print(f"保存Weibull模型参数时发生错误: {e}")
            raise

    def load_weibulls(self):
        """
        从保存的文件中加载Weibull模型参数。
        """
        try:
            with open(self.weibull_params_path, 'rb') as f:
                data = pickle.load(f)
                self.weibull_models = data['weibull_models']
                self.class_means = data['class_means']
            print(f"Weibull模型参数已从 {self.weibull_params_path} 加载。")
        except FileNotFoundError:
            # print(f"Weibull模型参数文件未找到: {self.weibull_params_path}")
            raise
        except Exception as e:
            # print(f"加载Weibull模型参数时发生错误: {e}")
            raise

    def _compute_distance(self, feat, cls):
        """
        计算特征与类别均值之间的欧氏距离。

        Args:
            feat (np.ndarray): 样本特征向量。
            cls (int): 类别标签。

        Returns:
            float: 距离值。
        """
        if cls not in self.class_means:
            # print(f"类别 {cls} 的均值未找到。")
            return np.inf
        mean_feat = self.class_means[cls]
        distance = np.linalg.norm(feat - mean_feat)
        # print(f"类别 {cls} 的距离: {distance:.4f}")
        return distance

    def _openmax_score(self, feat, cls):
        """
        计算OpenMax得分。

        Args:
            feat (np.ndarray): 样本特征向量。
            cls (int): 类别标签。

        Returns:
            float: OpenMax得分。
        """
        distance = self._compute_distance(feat, cls)
        if cls not in self.weibull_models:
            # print(f"类别 {cls} 的Weibull模型未找到，OpenMax得分设为0。")
            return 0.0
        weibull_params = self.weibull_models[cls]
        cdf = stats.weibull_min.cdf(distance, *weibull_params)
        openmax_score = 1 - cdf
        # print(f"类别 {cls}: 距离={distance:.4f}, Weibull CDF={cdf:.4f}, OpenMax得分={openmax_score:.4f}")
        return openmax_score

    def compute_adjusted_scores(self, features, logits):
        """
        使用OpenMax调整分类器的输出。

        Args:
            features (np.ndarray): 样本的特征向量，形状为 (num_samples, feature_dim)。
            logits (np.ndarray): 分类器的logits输出，形状为 (num_samples, num_classes)。

        Returns:
            np.ndarray: 调整后的概率分布，包含一个“Unknown”类别。
        """
        adjusted_scores = []
        # 使用PyTorch的softmax实现，确保数值稳定性
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exps / np.sum(exps, axis=1, keepdims=True)

        for i, feat in enumerate(features):
            cls = np.argmax(logits[i])
            original_probs = probs[i].copy()

            if cls == 0:
                # 如果预测为 'Unknown' 类，保持原有概率分布
                adjusted_score = probs[i]
                # print(f"样本 {i}: 预测为 'Unknown' 类，概率保持不变。")
            elif 1 <= cls <= (self.num_classes -1):
                class_label = int(cls)  # 确保是Python的int类型
                if class_label not in self.weibull_models:
                    # print(f"样本 {i}: 类别 {class_label} 在Weibull模型中未找到，保持原有概率分布。")
                    adjusted_score = probs[i]
                else:
                    openmax_score = self._openmax_score(feat, class_label)
                    # 确保 openmax_score 在 [0,1] 范围内
                    openmax_score = np.clip(openmax_score, 0.0, 1.0)
                    # 应用缩放因子 alpha
                    adjusted_openmax = self.alpha * openmax_score
                    # 调整预测类别的概率
                    probs[i, cls] *= (1 - adjusted_openmax)
                    # 将OpenMax得分分配给 'Unknown' 类
                    probs[i, 0] += adjusted_openmax
                    # 形成新的概率分布
                    probs[i] /= np.sum(probs[i])
                    adjusted_score = probs[i]
                    # print(f"样本 {i}: 类别 {class_label}, OpenMax 得分: {openmax_score:.4f}, 调整后 'Unknown' 概率: {adjusted_openmax:.4f}")
                    # print(f"样本 {i}: 原始概率: {original_probs[:5]}..., 调整后概率: {adjusted_score[:5]}...")

            else:
                # 无效的类别，保持原有概率分布
                # print(f"样本 {i}: 预测类别 {cls} 超出范围 [0,{self.num_classes -1}]，保持原有概率分布。")
                adjusted_score = probs[i]

            adjusted_scores.append(adjusted_score)

        # 打印调整后的概率分布前5个样本
        # print("调整后的概率分布（前5个样本）:")
        # print(adjusted_scores[:5])

        return np.array(adjusted_scores)
