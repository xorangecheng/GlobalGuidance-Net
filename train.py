import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from datasets.dataset import ImageFolder
from modeling.deeplab_dense import *
import utils.imgs
import utils.training as train_utils
from datasets import skin
from datasets import joint_transforms
from utils.lr_scheduler import LR_Scheduler
from torch.utils.data import DataLoader,SubsetRandomSampler
from misc import AvgMeter, check_mkdir
from torch.autograd import Variable
import datetime
from PIL import Image
import cv2
import scipy
from misc import check_mkdir
from misc import crf_refine
from tqdm import tqdm

from sklearn.model_selection import KFold
import copy
from torch import autograd

def count_parameters(model):
    print('parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

class edge_loss(nn.Module):
  def __init__(self):
    super(edge_loss,self).__init__()

    self.pool=nn.MaxPool2d(3,stride=1,padding=1)
    self.loss=nn.MSELoss().cuda()

  def forward(self,x,target):

    t_smoothed=self.pool(target)
    edge2=torch.abs(target-t_smoothed).cuda()
    edge_loss=self.loss(x,edge2)

    return edge_loss


class DiceLoss(nn.Module):
  def __init__(self):
    super(DiceLoss, self).__init__()
    self.sigmoid=nn.Sigmoid()
 
  def forward(self, input, target):
    N = target.size(0)
    smooth = 1e-5
    input=self.sigmoid(input)
 
    input_flat = input.view(N, -1)
    target_flat = target.view(N, -1)
 
    intersection = input_flat * target_flat
 
    loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
    loss = 1 - loss.sum() / N
 
    return loss

def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)

train_path=('../data/breast/breast_full/breast_tumor_seg/')
RESULTS_PATH = './results/'
WEIGHTS_PATH = './ultra_weights/dense1'



batch_size = 4

ckpt_path = './breast_weights'
exp_name = 'dgnlb'
args = {
    'iter_num': 13400,
    'train_batch_size': 4,
    'lr': 1e-3,
    'lr_step': 6700,
    'lr_decay': 10,
    'weight_decay': 1e-4,
    'momentum': 0.9,
    'last_iter':0
}


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

target_transform = transforms.ToTensor()
to_pil = transforms.ToPILImage()

joint_transforms=joint_transforms.Compose([
  joint_transforms.FreeScale((256,256)),
    # joint_transforms.RandomCrop(224), # commented for fine-tuning
    joint_transforms.RandomHorizontallyFlip(),
    joint_transforms.RandomRotate(10),

    ])
val_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_target_transform = transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()])

bce_logit = nn.BCEWithLogitsLoss().cuda()
e_loss=edge_loss()
d_loss=DiceLoss()
log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')

def validate( val_loader,model, criterion):
    losses = AvgMeter()
    dices=AvgMeter()

    # switch to evaluate mode

    model.eval()
    
    # time_taken=[]

    with torch.no_grad():
        torch.cuda.empty_cache()
        for i, (input, target) in tqdm(enumerate(val_loader),total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()
            start_time=time.time()

            
            output = model(input)[0]
            loss = criterion(output, target)
            dice = dice_coef(output, target)

            losses.update(loss.item(), input.size(0))
            dices.update(dice.item(), input.size(0))

    print('loss', losses.avg,'dice', dices.avg)
    return(dices.avg)
    



def main():
  net = DeepLab(num_classes=1,
                        backbone='daf_ds',
                        output_stride=16,
                        mm='dgnlb',
                        sync_bn='auto',
                        freeze_bn=False).cuda()
  count_parameters(net)
  optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}
         ], momentum=args['momentum'])
  init_state = copy.deepcopy(net.state_dict())
  init_state_opt = copy.deepcopy(optimizer.state_dict())

  check_mkdir(ckpt_path)
  check_mkdir(os.path.join(ckpt_path, exp_name))
  print(os.path.join(ckpt_path, exp_name))
  open(log_path, 'w').write(str(args) + '\n\n')
  train_set = ImageFolder(train_path,joint_transforms, transform, target_transform)
  test_set=ImageFolder(train_path,None, val_transform, val_target_transform)
  # print(train_set[0].size())
  cv=KFold(n_splits=7, random_state=42, shuffle=True)
  fold=0
  re=[]
  
  for train_idx,test_idx in cv.split(train_set):
    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name,str(fold)))
    net.load_state_dict(init_state)
    optimizer.load_state_dict(init_state_opt)

    print("\nCross validation fold %d" %fold)
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler=SubsetRandomSampler(test_idx)
    train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=12, shuffle=False,sampler=train_sampler)
    test_loader = DataLoader(test_set, batch_size=1, num_workers=12, shuffle=False,sampler=test_sampler)
    results=train(net,optimizer,train_loader,test_loader,fold,test_idx)
    re.append(results)
    fold+=1
    torch.cuda.empty_cache()
  print(re)
  print(exp_name)



def train(net,optimizer,train_loader,test_loader,fold,test_idx):
    
    curr_iter = args['last_iter']
    # print(curr_iter)
    total_iter=args['last_iter']
    best_dice=0
    trigger = 0


    while True:
      train_loss_record = AvgMeter()
      loss0_record,loss1_record, loss2_record,loss3_record,loss4_record, loss5_record = AvgMeter(), AvgMeter(),AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
      
      with autograd.detect_anomaly():
        # time_taken=[]
        for i, data in enumerate(train_loader):
            if curr_iter == args['lr_step']:
                optimizer.param_groups[0]['lr'] = 2 * optimizer.param_groups[0]['lr'] / args['lr_decay']
                optimizer.param_groups[1]['lr'] = optimizer.param_groups[1]['lr'] / args['lr_decay']
                curr_iter=0
                net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, str(fold),'bestmodel.pth')))

            inputs, labels = data
            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda()
            labels = Variable(labels).cuda()
            start_time=time.time()
            # print(labels.size())

            optimizer.zero_grad()
            outputs0,outputs1,outputs2,outputs3,outputs4,outputs5,e0,e1,e2,e3,e4= net(inputs)

            loss0 = bce_logit(outputs0, labels)
            loss1 = bce_logit(outputs1, labels)
            loss2 = bce_logit(outputs2, labels)
            loss3 = bce_logit(outputs3, labels)
            loss4 = bce_logit(outputs4, labels)

            loss11 = e_loss(e1, labels)
            loss22 = e_loss(e2, labels)
            loss33 = e_loss(e3, labels)
            loss44 = e_loss(e4, labels)
            loss00 = e_loss(e0,labels)

            loss000 = d_loss(outputs0, labels)
            loss111 = d_loss(outputs1, labels)
            loss222 = d_loss(outputs2, labels)
            loss333 = d_loss(outputs3, labels)
            loss444 = d_loss(outputs4, labels)

           

            loss =  loss0+loss1 + loss2 + loss3 + loss4 +loss00+loss11 + loss22 + loss33 + loss44 +loss000+loss111+loss222+loss333+loss444
            loss.backward()
            optimizer.step()

            train_loss_record.update(loss.item(), batch_size)
            loss0_record.update(loss0.item(), batch_size)
            curr_iter +=1
            total_iter+=1

            log = '[iter %d], [train loss %.5f], [loss0 %.5f], [lr %.13f]' % \
                  (total_iter, train_loss_record.avg, loss0_record.avg, optimizer.param_groups[1]['lr'])

            open(log_path, 'a').write(log + '\n')

            if total_iter % 134==0:
                print('[iter %d]' % total_iter)
                dice=validate(test_loader,net,criterion=bce_logit)
                trigger +=1
                if trigger>=30:
                  print("=> early stopping")
                  print(test_idx)
                  results=test(fold,test_idx)
                  return (results)
                if dice>best_dice:
                  best_dice=dice
                  print("=> saved best model")
                  torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name,str(fold), 'bestmodel.pth'))
                  trigger=0


            if total_iter > args['iter_num']:
              print(test_idx)
              results=test(fold,test_idx)
              return(results)


def test(fold,test_idx):
    scores=[]
    Dices=[]
    ACs=[]
    SEs=[]
    SPs=[]
    Precisions=[]
    Adbs=[]
    Confms=[]
    testing_root='../data/breast/breast_full/breast_tumor_seg/'
    gt_path='../data/breast/breast_full//breast_tumor_seg/seg/'
    model = DeepLab(num_classes=1,
                        backbone='daf_ds',
                        output_stride=16,
                        mm='dgnlb',
                        sync_bn='auto',
                        freeze_bn=True).cuda()
    model.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, str(fold),'bestmodel.pth')))

    img_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    to_pil = transforms.ToPILImage()
    model.eval()
    with torch.no_grad():
      for idx in tqdm(test_idx):
          check_mkdir(os.path.join(ckpt_path, exp_name, str(fold), 'prediction' ))
          img_name=str(idx+1)+'.png'
          img = Image.open(os.path.join(testing_root, 'images', img_name))
          h,w=img.size
          img_var = Variable(img_transform(img).unsqueeze(0)).cuda()
          pro=torch.sigmoid(model(img_var)[0]).data.squeeze(0).cpu()
          pro = np.array(to_pil(pro))     
          pro=crf_refine(np.array(img.resize((256,256))),pro)
          pro[pro>=128]=255
          pro[pro<128]=0
          ((Image.fromarray(pro)).resize((h,w),Image.ANTIALIAS)).save(os.path.join(ckpt_path, exp_name, str(fold), 'prediction', img_name))
          pro=cv2.resize(pro,(h,w))/255
          gt=cv2.imread(os.path.join(gt_path,img_name),0)
          pro[pro>=0.5]=1
          pro[pro<0.5]=0
          gt[gt>=0.5]=1
          gt[gt<0.5]=0
          TP=float(np.sum(np.logical_and(pro==1,gt==1)))
          TN=float(np.sum(np.logical_and(pro==0,gt==0)))
          FP=float(np.sum(np.logical_and(pro==1,gt==0)))
          FN=float(np.sum(np.logical_and(pro==0,gt==1)))
          JA=TP/((TP+FN+FP)+1e-5)
          AC=(TP+TN)/((TP+FP+TN+FN+1e-5))
          DI=2*TP/((2*TP+FN+FP+1e-5))
          SE=TP/(TP+FN+1e-5)
          SP=TN/((TN+FP)+1e-5)
          precision=TP/(TP+FP+1e-5)
          #     adb=evaluation.asd(pre,gt)
          #     confm=(3*DI-2)/DI
          Dices.append(DI)
          scores.append(JA)
          ACs.append(AC)
          SEs.append(SE)
          SPs.append(SP)
          Precisions.append(precision)

      print('JA',np.mean(scores), 
              'DI',np.mean(Dices),
              'acc',np.mean(ACs),
              'Recall',np.mean(SEs),
              'SP',np.mean(SPs),
              'precision',np.mean(Precisions))
      return('JA',np.mean(scores), 
              'DI',np.mean(Dices),
              'acc',np.mean(ACs),
              'Recall',np.mean(SEs),
              'SP',np.mean(SPs),
              'precision',np.mean(Precisions))



if __name__ == '__main__':
    main()
