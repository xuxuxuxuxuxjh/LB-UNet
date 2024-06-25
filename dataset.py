from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image


class NPY_datasets(Dataset):
    def __init__(self, path_Data, config, train=True):
        super(NPY_datasets, self)
        self.status = train
        if train:
            images_list = os.listdir(path_Data+'train/images/')
            masks_list = os.listdir(path_Data+'train/masks/')
            points_list = os.listdir(path_Data+'train/points_boundary2/')
            images_list = sorted(images_list)
            masks_list = sorted(masks_list)
            points_list = sorted(points_list)
            self.data = []
            # print(len(images_list), len(masks_list), len(points_list))
            for i in range(len(images_list)):
                img_path = path_Data+'train/images/' + images_list[i]
                mask_path = path_Data+'train/masks/' + masks_list[i]
                points_path = path_Data+'train/points_boundary2/' + points_list[i]
                self.data.append([img_path, mask_path, points_path])
            self.transformer = config.train_transformer
        else:
            images_list = os.listdir(path_Data+'val/images/')
            masks_list = os.listdir(path_Data+'val/masks/')
            images_list = sorted(images_list)
            masks_list = sorted(masks_list)
            # print(len(images_list), len(masks_list))
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data+'val/images/' + images_list[i]
                mask_path = path_Data+'val/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = config.test_transformer
        
    def __getitem__(self, indx):
        if self.status == True:
            img_path, msk_path, pnt_path = self.data[indx]
            img = np.array(Image.open(img_path).convert('RGB'))
            msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255
            pnt = np.expand_dims(np.array(Image.open(pnt_path).convert('L')), axis=2) / 255
            img, msk, pnt = self.transformer((img, msk, pnt))
            return img, msk, pnt 
        else:
            img_path, msk_path = self.data[indx]
            img = np.array(Image.open(img_path).convert('RGB'))
            msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255
            img, msk = self.transformer((img, msk))
            return img, msk

    def __len__(self):
        return len(self.data)
        
    
class Test_datasets(Dataset):
    def __init__(self, path_Data, config):
        super(Test_datasets, self)
        images_list = os.listdir(path_Data+'test/images/')
        masks_list = os.listdir(path_Data+'test/masks/')
        images_list = sorted(images_list)
        masks_list = sorted(masks_list)
        self.data = []
        for i in range(len(images_list)):
            img_path = path_Data+'test/images/' + images_list[i]
            mask_path = path_Data+'test/masks/' + masks_list[i]
            self.data.append([img_path, mask_path])
        self.transformer = config.test_transformer
        
    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]
        img = np.array(Image.open(img_path).convert('RGB'))
        msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255
        img, msk = self.transformer((img, msk))
        return img, msk

    def __len__(self):
        return len(self.data)
        
    