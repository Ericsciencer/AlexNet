import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ----------------------
# 1. 适配CIFAR-10的AlexNet定义
# ----------------------
class AlexNet_CIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet_CIFAR10, self).__init__()
        # 特征提取部分（卷积+ReLU+池化，调整第一个卷积核适配CIFAR-10的32x32输入）
        self.features = nn.Sequential(
            # 第1层：卷积 -> ReLU -> 最大池化
            nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=2), # 输入(3,32,32) -> 输出(96,32,32)
            nn.ReLU(inplace=True),  # inplace=True节省内存
            nn.MaxPool2d(kernel_size=3, stride=2), # 输出(96,15,15)
            
            # 第2层：卷积 -> ReLU -> 最大池化
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2), # 输出(256,15,15)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # 输出(256,7,7)
            
            # 第3层：卷积 -> ReLU
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1), # 输出(384,7,7)
            nn.ReLU(inplace=True),
            
            # 第4层：卷积 -> ReLU
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),# 输出(384,7,7)
            nn.ReLU(inplace=True),
            
            # 第5层：卷积 -> ReLU -> 最大池化
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),# 输出(256,7,7)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2) # 输出(256,3,3)
        )
        
        # 分类器部分（全连接+Dropout，防止过拟合）
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),  # 随机丢弃50%神经元，防止过拟合
            nn.Linear(256 * 3 * 3, 4096), # 输入维度：256*3*3空间尺寸
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes) # 输出10类
        )

    def forward(self, x):
        x = self.features(x) # 提取特征：(batch,3,32,32) -> (batch,256,3,3)
        x = torch.flatten(x, 1) # 展平：保留batch维度，从第1维开始展平 -> (batch,256*3*3)
        x = self.classifier(x) # 分类：(batch,256*3*3) -> (batch,10)
        return x

# ----------------------
# 2. 数据加载与预处理（CIFAR-10官方归一化参数）
# ----------------------
def get_data_loaders(batch_size=64):
    # 预处理：转张量 + 归一化（均值和标准差是CIFAR-10数据集的官方统计值）
    transform = transforms.Compose([
        transforms.ToTensor(), # 把PIL图像转成PyTorch张量(0-1)
        transforms.Normalize((0.4914, 0.4822, 0.4465), # RGB三通道的均值
                             (0.2023, 0.1994, 0.2010)) # RGB三通道的标准差
    ])

    # 下载并加载训练集（50000张图）
    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    # 下载并加载测试集（10000张图）
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )

    # 构建DataLoader：批量加载数据，shuffle=True打乱训练集顺序
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# ----------------------
# 3. 训练函数
# ----------------------
def train(model, train_loader, criterion, optimizer, device):
    model.train()  # 开启训练模式（Dropout、BatchNorm生效）
    total_loss = 0.0 
    correct = 0  
    total = 0 

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device) # 把数据移到GPU/CPU
        optimizer.zero_grad() # 清空上一步的梯度（防止累加）
        outputs = model(images) # 前向传播：得到预测结果
        loss = criterion(outputs, labels) # 计算损失：预测值 vs 真实值
        loss.backward()  # 反向传播：计算梯度
        optimizer.step() # 更新权重：SGD优化器一步

        # 计算当前batch的准确率
        _, predicted = torch.max(outputs.data, 1) # 取概率最大的类别作为预测结果
        total += labels.size(0)  # 累加总样本数（labels.size(0)=batch_size）
        correct += (predicted == labels).sum().item() # 累加预测正确的样本数
        total_loss += loss.item() * images.size(0) # 累加总损失（loss.item()是平均损失，乘batch_size得总损失）
    
    # 计算整个epoch的平均损失和平均准确率
    avg_train_loss = total_loss / len(train_loader.dataset) # len(train_loader.dataset)=50000（训练集总样本数）
    avg_train_acc = correct / total
    return avg_train_loss, avg_train_acc

# ----------------------
# 4. 测试函数
# ----------------------
def test(model, test_loader, device):
    model.eval() # 开启测试模式（Dropout、BatchNorm关闭）
    correct = 0
    total = 0
    with torch.no_grad(): # 关闭梯度计算（节省内存+加速）
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# ----------------------
# 5. 主程序
# ----------------------
if __name__ == '__main__':
    # 基础配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 优先用GPU
    batch_size = 64
    lr = 0.01 # 学习率
    num_epochs = 15 # 训练轮数

    # 初始化核心组件
    model = AlexNet_CIFAR10(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss() # 交叉熵损失函数（分类任务常用）
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)  # SGD优化器（带动量+权重衰减）
    train_loader, test_loader = get_data_loaders(batch_size)

    # 初始化列表，存储每个epoch的3个核心指标（用于绘图）
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    # 训练循环
    print(f"Training on {device}...")
    for epoch in range(num_epochs):
        # 训练一个epoch，接收训练损失+训练准确率
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        # 测试当前模型在测试集上的准确率
        test_acc = test(model, test_loader, device)
        
        # 把当前epoch的指标存入列表
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        
        # 打印训练日志
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

    # 保存模型权重
    torch.save(model.state_dict(), 'alexnet_cifar10.pth')
    print("Model saved as alexnet_cifar10.pth")

    # ----------------------
    # 可视化绘图
    # ----------------------
    epochs = range(1, num_epochs + 1) # x轴：1-10个epoch
    plt.figure(figsize=(10, 7)) # 设置画布大小

    # 绘制3条曲线
    plt.plot(epochs, train_loss_list, 'b-', linewidth=2, label='train loss')  # 蓝色实线：训练损失
    plt.plot(epochs, train_acc_list, 'm--', linewidth=2, label='train acc')   # 紫色虚线：训练准确率
    plt.plot(epochs, test_acc_list, 'g--', linewidth=2, label='test acc')     # 绿色虚线：测试准确率

    
    plt.xlabel('epoch', fontsize=18)                           
    plt.xticks(range(2, 11, 2))                              
    plt.ylim(0, 2.4)                                      
    plt.grid(True)                                 
    plt.legend(loc='upper right', fontsize=18)         
    plt.title('AlexNet CIFAR-10 Training Metrics', fontsize=16)

    
    plt.savefig('alexnet_training_curve.png', dpi=300, bbox_inches='tight')
    plt.show()