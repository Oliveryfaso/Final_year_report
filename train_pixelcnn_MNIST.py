# train_pixelcnn_MNIST.py

import torch
from torch import optim
from torch.utils.data import DataLoader
from pixelcnn_MNIST import PixelCNN, discretized_mix_logistic_loss_c1
from data_loader_MNIST import SignMNISTDataset
from torchvision import transforms
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


def train_pixelcnn_pp():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 使用与main.py一致的参数训练PixelCNN++
    model = PixelCNN(nr_filters=64, nr_resnet=5, nr_logistic_mix=10, disable_third=False, dropout_p=0.5, n_channel=1,
                     image_wh=28).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = SignMNISTDataset(csv_file='./data/handfigure/sign_mnist_combined.csv', transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

    model.train()
    epochs = 50
    loss_history = []

    for epoch in range(epochs):
        running_loss = 0.0
        progress_bar = tqdm(enumerate(dataloader), desc=f"Epoch [{epoch + 1}/{epochs}]", unit="batch",
                            total=len(dataloader))
        for i, (images, _) in progress_bar:
            images = images.to(device)
            if images.dim() == 3:
                images = images.unsqueeze(1)
            elif images.dim() != 4:
                print(f"Unexpected image dimensions: {images.dim()}D, expected 4D.")
                continue

            optimizer.zero_grad()
            output = model(images)
            loss = discretized_mix_logistic_loss_c1(images, output, sum_all=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / (i + 1)
            progress_bar.set_postfix(loss=avg_loss)

        epoch_avg_loss = running_loss / len(dataloader)
        loss_history.append(epoch_avg_loss)
        print(f"Epoch [{epoch + 1}/{epochs}] Loss: {epoch_avg_loss:.4f}")

    os.makedirs('./models', exist_ok=True)
    torch.save(model.state_dict(), './models/pixelcnn_pp.pth')
    print("PixelCNN++ 模型训练完成并已保存。")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), loss_history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('PixelCNN Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss_curve.png')
    plt.close()
    print("训练过程损失曲线已保存为 training_loss_curve.png")


if __name__ == "__main__":
    train_pixelcnn_pp()
