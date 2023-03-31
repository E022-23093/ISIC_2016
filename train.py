from model.unet_model import UNet
from utils.dataset import Loader
from torch import optim
import torch.nn as nn
import torch
import tqdm

def train_net(net, device, data_path, epochs=10, batch_size=3, lr=0.00001):
    # 加载训练集
    dataset = I=Loader(data_path)
    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=batch_size, 
                                               shuffle=True)
    # 定义RMSprop算法
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # 定义Loss算法
    criterion = nn.BCEWithLogitsLoss()
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')
    # 训练epochs次
    for epoch in range(epochs):
        # 训练模式
        net.train()
        loss_train = 0.0
        # 按照batch_size开始训练
        for i,data in enumerate(train_loader):
            optimizer.zero_grad()
            # 将数据拷贝到device中
            image,label=data

            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            # 使用网络参数，输出预测结果
            pred = net(image)

            # 计算loss     
            loss = criterion(pred, label)
            loss_train += loss

            loss.backward()
            optimizer.step()
            
            if i % 10 ==0:
                print(i)
            
        print('Epoch:',epoch,',Loss:', loss_train)
        if loss_train < best_loss:
            best_loss = loss_train
            torch.save(net.state_dict(), 'best_model.pth')

if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # 加载网络，图片单通道1，分类为1。
    net = UNet(n_channels=1, n_classes=1)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 指定训练集地址，开始训练
    data_path = r"..\dataset_transformed\ISBI2016_ISIC_Part3B_Training_Data"
    train_net(net, device, data_path)