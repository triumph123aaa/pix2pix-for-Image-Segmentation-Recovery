from torch.utils.data.dataset import Dataset
import torchvision.transforms as transform
from PIL import Image
import cv2
import random
import glob


class CreateDatasets(Dataset):
    def __init__(self, ori_imglist,img_size):
        self.ori_imglist = ori_imglist
        self.transform = transform.Compose([
            transform.ToTensor(),
            transform.Resize((img_size, img_size)),
            transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.ori_imglist)

    def __getitem__(self, item):
        ori_img = cv2.imread(self.ori_imglist[item])
        ori_img = ori_img[:, :, ::-1]
        real_img = Image.open(self.ori_imglist[item].replace('.png', '.jpg'))
        ori_img = self.transform(ori_img.copy())
        real_img = self.transform(real_img)
        return ori_img, real_img


def split_data(dir_root):
    random.seed(0)
    ori_img = glob.glob(dir_root + '/*.png')
    k = 0.2
    train_ori_imglist = []
    val_ori_imglist = []
    sample_data = random.sample(population=ori_img, k=int(k * len(ori_img)))
    for img in ori_img:
        if img in sample_data:
            val_ori_imglist.append(img)
        else:
            train_ori_imglist.append(img)
    return train_ori_imglist, val_ori_imglist


if __name__ == '__main__':
    a, b= split_data('../data')
