import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random

class Loader(Dataset):
    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = sorted([
        os.path.join(data_path, fname)
        for fname in os.listdir(data_path)
        if fname.endswith(".jpg")])

        
    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        # 根据image_path生成label_path
        label_path = image_path.replace('.jpg', '_Segmentation.png')
        # 读取训练图片和标签图片
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)
        # 将数据转为单通道的图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        # 处理标签，将像素值为255的改为1
        if label.max() > 1:
            label = label / 255
            
        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)

    
if __name__ == "__main__":
    dataset = Loader(r"..\dataset_transformed\ISBI2016_ISIC_Part3B_Training_Data")
    print("数据个数：", len(dataset))
    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=2, 
                                               shuffle=True)
    for image, label in train_loader:
        print(image.shape)
        print(label.shape)
        print(image.dtype)
        break