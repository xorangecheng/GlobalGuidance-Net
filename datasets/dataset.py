import numpy as np
import os
import os.path
import torch
import torch.utils.data as data
from PIL import Image
import pandas as pd
from torchvision.datasets.folder import default_loader
from glob import glob
from sklearn.model_selection import train_test_split
def make_dataset(root):
    nlist=range(1,624)

    return ([(os.path.join(root, 'images', str(img_name)+'.png'), os.path.join(root, 'seg', str(img_name)+'.png')) for img_name in
            nlist])
def make_dataset2(root):
    return [(os.path.join(root, 'images', img_name), os.path.join(root, 'seg', img_name.split('.')[0]+'_mask.png')) for img_name in
            os.listdir(os.path.join(root, 'images'))]
    


class ImageFolder(data.Dataset):
    def __init__(self, root, joint_transform=None, transform=None, target_transform=None):
        self.root  = root
        self.imgs = make_dataset(root)
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index]
        #print(img_path)
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path).convert('L')

        if self.joint_transform is not None:

            img, target = self.joint_transform(img, target)


        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        #print(target.shape)
        return img, target

    def __len__(self):
        return len(self.imgs)


    


class ImageFolder2(data.Dataset):
    def __init__(self, root, joint_transform=None, transform=None, target_transform=None):
        self.root  = root
        self.imgs = make_dataset2(root)
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path).convert('L')
        if self.joint_transform is not None:
            img, target = self.joint_transform(img, target)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


# def make_dataset_split(root):
#     img_paths=glob(root+'/images/*')
#     mask_paths=glob(root+'seg/*')
#     train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = \
#         train_test_split(img_paths, mask_paths, test_size=0.1, random_state=20)
#     for img_name in train_img_paths.basename

#     return 

# class ImageFolder(data.Dataset):
#     def __init__(self, root,joint_transform=None, transform=None, target_transform=None,val=True):
#         self.root = root
#         # self.split=split
#         self.train_imgs,self.val_imgs=make_dataset_split(root)
#         # self.imgs = make_dataset(root)
#         self.joint_transform = joint_transform
#         self.transform = transform
#         self.target_transform = target_transform
#         self.val=val

#     def __getitem__(self, index):
#         if self.val:
#             img_path, gt_path = self.train_imgs[index]
#             print('val',img_path, gt_path )
#         else:
#             img_path, gt_path = self.val_imgs[index]
#             print(img_path,gt_path)

#         img = Image.open(img_path).convert('RGB')
#         # print(img_path)
#         target = Image.open(gt_path)
#         # print(gt_path)
#         if self.joint_transform is not None:
#             img, target = self.joint_transform(img, target)

#         if self.transform is not None:
#             img = self.transform(img)
#             # print(img.size())
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#             # print(target.size())

#         return img, target

#     def __len__(self):
#         if self.val:
#             l=len(self.train_imgs)
#         else:
#             l=len(self.val_imgs)

#         return l




# TODO: Make target_field optional for unannotated datasets.
class CSVDataset(data.Dataset):
    def __init__(self, root, csv_file, image_field, target_field,
                 loader=default_loader, transform=None,
                 target_transform=None, add_extension=None,
                 limit=None, random_subset_size=None,
                 split=None):
        self.root = root
        self.loader = loader
        self.image_field = image_field
        self.target_field = target_field
        self.transform = transform
        self.target_transform = target_transform
        self.add_extension = add_extension

        self.data = pd.read_csv(csv_file, sep=None)

        # Split
        if split is not None:
            with open(split, 'r') as f:
                selected_images = f.read().splitlines()
            self.data = self.data[self.data[image_field].isin(selected_images)]
            self.data = self.data.reset_index()

        # Calculate class weights for WeightedRandomSampler
        self.class_counts = dict(self.data['label'].value_counts())
        self.class_weights = {label: max(self.class_counts.values()) / count
                              for label, count in self.class_counts.items()}
        self.sampler_weights = [self.class_weights[cls]
                                for cls in self.data['label']]
        self.class_weights_list = [self.class_weights[k]
                                   for k in sorted(self.class_weights)]

        if random_subset_size:
            self.data = self.data.sample(n=random_subset_size)
            self.data = self.data.reset_index()

        if type(limit) == int:
            limit = (0, limit)
        if type(limit) == tuple:
            self.data = self.data[limit[0]:limit[1]]
            self.data = self.data.reset_index()

        classes = list(self.data[self.target_field].unique())
        classes.sort()
        self.class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.classes = classes

        print('Found {} images from {} classes.'.format(len(self.data),
                                                        len(classes)))
        for class_name, idx in self.class_to_idx.items():
            n_images = dict(self.data[self.target_field].value_counts())
            print("    Class '{}' ({}): {} images.".format(
                class_name, idx, n_images[class_name]))

    def __getitem__(self, index):
        path = os.path.join(self.root,
                            self.data.loc[index, self.image_field])
        if self.add_extension:
            path = path + self.add_extension
        sample = self.loader(path)
        target = self.class_to_idx[self.data.loc[index, self.target_field]]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.data)


class CSVDatasetWithName(CSVDataset):
    """
    CSVData that also returns image names.
    """

    def __getitem__(self, i):
        """
        Returns:
            tuple(tuple(PIL image, int), str): a tuple
            containing another tuple with an image and
            the label, and a string representing the
            name of the image.
        """
        name = self.data.loc[i, self.image_field]
        return super().__getitem__(i), name


from sklearn.model_selection import StratifiedKFold 
from torch.utils import data
import torch


def cross_validate(model, dataset, splits, epochs, dataloader_params):
    """
    Does cross validation for a model.
    @param model: An instance of a model to be evaluated.
    @param dataset: A torch.utils.data.Dataset dataset.
    @param splits: The number of cross validation folds.
    @param epochs: The number of epochs per cross validation fold.
    @dataloader_params: parameters to be passed to the torch.utils.data.DataLoader class.
    Typically
    params = {
        'batch_size': 100,
        'shuffle': True, 
        'num_workers': 4,
    }
    """
    skf = StratifiedKFold(n_splits=splits)
    metrics_to_avg_values = {}
    fold = 0

    for train_idx, test_idx in skf.split(dataset.targets, dataset.targets):
        print("\nCross validation fold %d" %fold)

        model.apply(weights_init)
        dataset.set_active_data(train_idx)
        train_generator = data.DataLoader(dataset, **dataloader_params)

        for epoch in range(epochs):
            model.train_epoch(train_generator, epoch)

        test_inputs = dataset.inputs[test_idx]
        test_targets = dataset.targets[test_idx]

        metrics_to_values = model.test(test_inputs, test_targets)

        for metric, value in metrics_to_values.items():
            if metric not in metrics_to_avg_values:
                metrics_to_avg_values[metric] = 0

            metrics_to_avg_values[metric] += value/splits

        fold += 1

    print("\n########################################")
    print("Cross validation with %d folds complete." % splits)
    for metric, value in metrics_to_avg_values.items():
        print('Average {0}: {1:10.4f}'.format(metric, value))