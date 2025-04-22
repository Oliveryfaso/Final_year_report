# main_MNIST.py

import time
import pickle
import os
from data_loader_MNIST import load_sign_mnist_data
from model_convgan_MNIST import KwayNetwork, Generator, Discriminator, weights_init_normal
from train_MNIST import train_kway_network, train_gan
from ots_MNIST import evaluate_model
import torch
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pixelcnn_MNIST import PixelCNN, discretized_mix_logistic_loss_c1
from openmax_MNIST import OpenMax
from train_pixelcnn_MNIST import train_pixelcnn_pp
from sklearn.metrics import classification_report

from model_convgan_MNIST import Generator_cWGAN_GP, Critic_cWGAN_GP, weights_init_normal
from train_MNIST import train_cwgan_gp


def save_results(metrics, timestamp, class_names, save_dir='evaluation_results'):
    os.makedirs(save_dir, exist_ok=True)

    # 保存文本结果
    result_filename = os.path.join(save_dir, f"result_{timestamp}.txt")
    with open(result_filename, 'w') as f:
        f.write(f"Training and Evaluation Results - {timestamp}\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision (Macro Average): {metrics['precision']:.4f}\n")
        f.write(f"Recall (Macro Average): {metrics['recall']:.4f}\n")
        f.write(f"F1-Score (Macro Average): {metrics['f1_score']:.4f}\n")
        f.write(f"Confusion Matrix:\n{metrics['confusion_matrix']}\n")
        f.write("Classification Report:\n")
        f.write(classification_report_to_string(metrics['classification_report']))
    print(f"Results saved to {result_filename}")

    # 保存混淆矩阵图像，文件名包含时间戳
    confusion_matrix_path = os.path.join(save_dir, f"confusion_matrix_{timestamp}.png")
    plot_confusion_matrix(metrics['confusion_matrix'], classes=class_names, save_path=confusion_matrix_path)
    print(f"混淆矩阵热力图已保存到 {confusion_matrix_path}。")

    # 保存各类别F1分数图像，文件名包含时间戳
    f1_scores_path = os.path.join(save_dir, f"class_f1_scores_{timestamp}.png")
    plot_class_f1_scores(metrics['classification_report'], classes=class_names, save_path=f1_scores_path)
    print(f"各类别的F1分数图已保存到 {f1_scores_path}。")


def classification_report_to_string(report):
    report_str = ""
    for class_label, metrics in report.items():
        if class_label in ['accuracy', 'macro avg', 'weighted avg']:
            report_str += f"{class_label}: {metrics}\n"
        else:
            report_str += f"Class {class_label}:\n"
            for metric, value in metrics.items():
                report_str += f"  {metric}: {value}\n"
    return report_str


def plot_confusion_matrix(cm, classes, save_path='evaluation_results/confusion_matrix.png'):
    plt.figure(figsize=(12, 10))
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()


def plot_class_f1_scores(class_report, classes, save_path='evaluation_results/class_f1_scores.png'):
    # 提取每个类别的F1分数
    f1_scores = []
    for cls in classes:
        cls_str = str(cls)  # sklearn的classification_report keys是字符串
        if cls_str in class_report:
            f1_scores.append(class_report[cls_str]['f1-score'])
        else:
            f1_scores.append(0.0)  # 如果某个类别没有被预测，则F1分数为0

    plt.figure(figsize=(12, 6))
    sns.barplot(x=classes, y=f1_scores, palette='viridis')
    plt.xlabel('Classes')
    plt.ylabel('F1 Score')
    plt.title('F1 Scores per Class')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def check_model_parameters(model, model_name="Model"):
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"{model_name} parameter {name} has NaNs.")


def main():
    torch.autograd.set_detect_anomaly(True)
    # 设置参数
    batch_size = 64
    known_labels = list(range(1, 24))  # 已知标签为1-23，包含23
    unknown_label = 0  # 定义未知类标签
    train_csv = './data/handfigure/sign_mnist_train.csv'
    test_csv = './data/handfigure/sign_mnist_test.csv'
    combined_csv = './data/handfigure/sign_mnist_combined.csv'

    os.makedirs('./data/handfigure/', exist_ok=True)
    os.makedirs('./models/', exist_ok=True)  # 确保模型目录存在

    # 配置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1. 加载数据集并划分已知集和开放集
    print("加载数据集并划分已知集和开放集...")
    try:
        train_loader, test_loader, open_set_loader = load_sign_mnist_data(
            batch_size=batch_size,
            known_labels=known_labels,
            # unknown_label=unknown_label,
            combined_csv=combined_csv,
            save_combined=True,  # 设置为True以保存并覆盖合并后的CSV
            train_csv=train_csv,
            test_csv=test_csv,
            device=device  # 传递设备信息
        )
    except Exception as e:
        print(f"加载数据集时发生错误: {e}")
        return

    # 2. 训练 PixelCNN
    print("训练 PixelCNN 模型...")
    # train_pixelcnn_pp()
    print("PixelCNN 模型训练完成。")

    # 3. 初始化并加载 PixelCNN
    print("加载训练好的 PixelCNN 模型...")
    pixelcnn_pp = PixelCNN(
        nr_filters=64,  # 保持 MNIST 的参数
        nr_resnet=5,
        nr_logistic_mix=10,
        disable_third=False,
        dropout_p=0.3,
        n_channel=1,
        image_wh=28
    ).to(device)

    pixelcnn_pp_pth_path = './models/pixelcnn_pp.pth'
    if os.path.exists(pixelcnn_pp_pth_path):
        try:
            pixelcnn_pp.load_state_dict(torch.load(pixelcnn_pp_pth_path, map_location=device))
            pixelcnn_pp.eval()
            print("PixelCNN++ 模型已加载。")
        except RuntimeError as e:
            print(f"加载 PixelCNN++ 时发生错误: {e}")
            return
    else:
        print("PixelCNN++ 模型文件未找到，请确保已训练 PixelCNN++。")
        return

    # # 4. 初始化 GAN
    # print("初始化 GAN 网络...")
    # generator = Generator(nz=100, ngf=64, nc=1, num_classes=24).to(device)
    # discriminator = Discriminator(nc=1, ndf=64, num_classes=24, input_size=28).to(device)

    # print("应用权重初始化...")
    # generator.apply(weights_init_normal)
    # discriminator.apply(weights_init_normal)

    # # 检查生成器和判别器参数
    # check_model_parameters(generator, "生成器")
    # check_model_parameters(discriminator, "判别器")

    # # 5. 训练 GAN
    # generator_pth_path = './models/generator.pth'
    # discriminator_pth_path = './models/discriminator.pth'

    # gan_loaded = False
    # if os.path.exists(generator_pth_path) and os.path.exists(discriminator_pth_path):
    #     try:
    #         generator.load_state_dict(torch.load(generator_pth_path, map_location=device))
    #         discriminator.load_state_dict(torch.load(discriminator_pth_path, map_location=device))
    #         generator.eval()
    #         discriminator.eval()
    #         gan_loaded = True
    #         print("GAN 模型已加载。")
    #     except RuntimeError as e:
    #         print(f"加载 GAN 模型时发生错误: {e}")
    #         print("将重新训练 GAN 模型。")
    #         gan_loaded = False
    # else:
    #     print("GAN 模型文件未找到，将进行训练。")

    # if not gan_loaded:
    #     print("开始训练 GAN 网络...")
    #     train_gan(
    #         generator=generator,
    #         discriminator=discriminator,
    #         data_loader=train_loader,
    #         device=device,
    #         pixelcnn_pp=pixelcnn_pp,
    #         epochs=120,  # 保持 MNIST 的训练轮数
    #         learning_rate=4e-4,  # 保持 MNIST 的学习率
    #         nz=100,
    #         save_dir='generated_images',
    #         loss_dir='loss_curves_gan',
    #         epsilon=1e-4  # 保持 MNIST 的 epsilon 设置
    #     )
    #     torch.save(generator.state_dict(), generator_pth_path)
    #     torch.save(discriminator.state_dict(), discriminator_pth_path)
    #     print("GAN 网络训练完成，模型已保存。")
    # else:
    #     print("GAN 网络已成功加载。")

    print("初始化 cWGAN-GP 的生成器(Generator)和Critic...")
    generator = Generator_cWGAN_GP(nz=100, num_classes=24, ngf=64, nc=1).to(device)
    critic = Critic_cWGAN_GP(num_classes=24, ndf=64, nc=1, image_size=28).to(device)

    print("应用权重初始化...")
    generator.apply(weights_init_normal)
    critic.apply(weights_init_normal)

    # 检查生成器和判别器参数
    check_model_parameters(generator, "cWGAN-GP Generator")
    check_model_parameters(critic, "cWGAN-GP Critic")

    generator_pth_path = './models/generator_cwgan_gp.pth'
    critic_pth_path = './models/critic_cwgan_gp.pth'

    gan_loaded = False
    if os.path.exists(generator_pth_path) and os.path.exists(critic_pth_path):
        try:
            generator.load_state_dict(torch.load(generator_pth_path, map_location=device))
            generator.nz = 100  # 假设训练时使用的nz=100
            critic.load_state_dict(torch.load(critic_pth_path, map_location=device))
            generator.eval()
            critic.eval()
            gan_loaded = True
            print("cWGAN-GP 模型已加载。")
        except RuntimeError as e:
            print(f"加载 cWGAN-GP 模型时发生错误: {e}")
            print("将重新训练 cWGAN-GP 模型。")
            gan_loaded = False
    else:
        print("cWGAN-GP 模型文件未找到，将进行训练。")

    if not gan_loaded:
        print("开始训练 cWGAN-GP 网络...")
        # 调用我们新的训练函数 train_cwgan_gp
        train_cwgan_gp(
            generator=generator,
            critic=critic,
            dataloader=train_loader,
            device=device,
            nz=100,
            num_epochs=120,      # 你可以根据需求调整epoch
            lr=1e-5,
            beta1=0.5,
            beta2=0.9,
            lambda_gp=50,
            n_critic=3
        )
        # 训练结束后保存
        torch.save(generator.state_dict(), generator_pth_path)
        torch.save(critic.state_dict(), critic_pth_path)
        print("cWGAN-GP 网络训练完成，模型已保存。")
    else:
        print("cWGAN-GP 网络已成功加载。")

    # 6. 初始化并加载 K-way 网络
    print("初始化 K-way 网络...")
    kway_network = KwayNetwork(num_classes=24).to(device)  # num_classes=24，包括 Unknown 类
    print("应用权重初始化...")
    kway_network.apply(weights_init_normal)
    check_model_parameters(kway_network, "K-way 网络")

    kway_pth_path = './models/kway_network.pth'
    activations_path = './models/activations.pkl'  # 激活数据保存路径
    # loss_save_path = './models/kway_losses.pkl'  # 损失历史保存路径

    # 加载或训练 K-way 网络
    kway_loaded = False
    if os.path.exists(kway_pth_path) and os.path.exists(activations_path):
        try:
            kway_network.load_state_dict(torch.load(kway_pth_path, map_location=device))
            kway_network.eval()
            kway_loaded = True
            print("K-way 网络已加载。")
        except RuntimeError as e:
            print(f"加载 K-way 网络时发生错误: {e}")
            print("将重新训练 K-way 网络。")
            kway_loaded = False
    else:
        print("K-way 网络文件或激活数据未找到，将进行训练。")

    if not kway_loaded:
        print("开始训练 K-way 网络...")
        train_kway_network(
            model=kway_network,
            data_loader=train_loader,
            generator=generator,
            device=device,
            epochs=30,  # 保持 MNIST 的训练轮数
            learning_rate=1e-5,  # 保持 MNIST 的学习率
            activation_save_path=activations_path,
            # loss_save_path=loss_save_path
        )
        torch.save(kway_network.state_dict(), kway_pth_path)
        print("K-way 网络训练完成，模型已保存。")
    else:
        print("K-way 网络已成功加载。")

    # 7. 初始化 OpenMax
    print("初始化 OpenMax...")
    openmax = OpenMax(
        activations_path=activations_path,
        tail_size=30,
        weibull_params_path='./models/weibulls.pkl',
        alpha=0.495,  # 保持 MNIST 的 alpha 设置
        # num_classes=24
    )
    print("OpenMax 初始化完成。")

    # 8. 生成用于 OpenMax 的“未知”类样本
    print("生成用于 OpenMax 的未知类样本...")
    kway_network.eval()
    generator.eval()
    num_openmax_samples = 1000
    # 修复后代码
    nz = generator.nz
    fake_labels_openmax = torch.zeros(num_openmax_samples, dtype=torch.long, device=device)
    noise_openmax = torch.randn(num_openmax_samples, nz, device=device)  # 正确：2D噪声 [batch, nz]

    with torch.no_grad():
        fake_images_openmax = generator(noise_openmax, fake_labels_openmax)
        # 逆标准化（根据 MNIST 的标准化参数，如果有的话）
        # 如果 MNIST 数据集没有标准化，请忽略此步骤或根据实际情况调整
        # 例如：
        # fake_images_openmax = fake_images_openmax * 0.5 + 0.5
        # 这里只是一个示例，具体调整请根据实际数据处理流程
        # 提取特征
        features_openmax, _ = kway_network(fake_images_openmax)

    features_openmax = features_openmax.cpu().numpy()

    # 将生成的“未知”类特征添加到 OpenMax 激活中
    for feat in features_openmax:
        openmax.activations[0].append(feat)
    # 重新拟合 Weibull 模型
    openmax._fit_weibulls()
    # 保存更新后的 Weibull 参数
    with open(openmax.weibull_params_path, 'wb') as f:
        pickle.dump({
            'weibull_models': openmax.weibull_models,
            'class_means': openmax.class_means
        }, f)
    print("OpenMax Weibull 模型已更新。")

    # 9. 评估模型
    print("评估模型在开放集和已知测试集上的表现...")
    try:
        threshold = 0.96  # 根据需要调整阈值
        print("开始评估模型...")
        metrics = evaluate_model(
            model=kway_network,
            known_test_loader=test_loader,
            open_set_loader=open_set_loader,
            device=device,
            threshold=threshold,
            openmax=openmax,
            # discriminator=discriminator  # 如果需要双重验证，可以传递判别器
        )
        print("模型评估完成。")
    except AttributeError as e:
        print(f"评估模型时发生错误: {e}")
        print("请确保模型类定义中包含必要的属性。")
        return
    except Exception as e:
        print(f"评估模型时发生错误: {e}")
        return

    # 打印评估结果
    print("\n=== 评估指标 ===")
    print(f"准确率: {metrics['accuracy']:.4f}")
    print(f"精确率 (宏平均): {metrics['precision']:.4f}")
    print(f"召回率 (宏平均): {metrics['recall']:.4f}")
    print(f"F1 分数 (宏平均): {metrics['f1_score']:.4f}")
    print("\n混淆矩阵:")
    print(metrics['confusion_matrix'])
    print("\n分类报告:")
    print(pd.DataFrame(metrics['classification_report']).transpose())

    # 定义所有类别的名称
    # 确保类别0为 'Unknown'，类别1-23为实际已知类别
    class_names = ['Unknown'] + [f'Class_{i}' for i in range(1, 24)]  # 根据实际类名修改

    # 保存评估结果到文件，保存在一个文件夹中，并绘制图像
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    save_results(metrics, timestamp, class_names, save_dir='evaluation_results')

    print("训练和评估完成。")


if __name__ == "__main__":
    main()
