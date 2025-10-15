import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, folderA, folderB, folderC):
        self.folderA = folderA
        self.folderB = folderB
        self.folderC = folderC

        # 获取文件名列表
        self.file_names = [f for f in os.listdir(folderA) if f.endswith('.png')]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        # 获取文件名
        file_name = self.file_names[idx]

        transform = transforms.Compose([
            transforms.Resize([256,256]),
            transforms.PILToTensor(),  # 转换为Tensor
        ])

        # 读取图像和标签
        imgA = Image.open(os.path.join(self.folderA, file_name)).convert('RGB')
        imgB = Image.open(os.path.join(self.folderB, file_name)).convert('RGB')
        label = Image.open(os.path.join(self.folderC, file_name)).convert('L')

        # 应用变换（如果有）
        imgA = transform(imgA)/255.0
        imgB = transform(imgB)/255.0
        
        new_size = (int(256), int(256))
        label = label.resize(new_size, resample=Image.NEAREST)

        label = transform(label)
        label = label.squeeze(0)

        return imgA, imgB, label


if __name__ == '__main__':

    folderA = '/home/hfz/data/MSRS/vi'
    folderB = '/home/hfz/data/MSRS/ir'
    folderC = '/home/hfz/data/MSRS/Segmentation_labels'

    # 定义变换

    # 创建数据集E:\HFZ\INN_MoE\INN_MoE\load_Seg2.py
    dataset = CustomDataset(folderA, folderB, folderC)

    # 测试 __getitem__ 方法
    for index in range(100):
        imgA, imgB, label = dataset[index]

        unique_values = torch.unique(label)
        print(unique_values)