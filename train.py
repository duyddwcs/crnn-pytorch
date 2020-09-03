from __future__ import print_function
from __future__ import division

import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.nn import CTCLoss
from torch.autograd import Variable

from collections import OrderedDict 
import numpy as np
import os
import utils
import dataset

from config import *
import models.crnn as crnn


if not os.path.exists(expr_dir):
    os.makedirs(expr_dir)

cudnn.benchmark = True

if torch.cuda.is_available() and not cuda:
    print("WARNING: You have a CUDA device")

# create train dataset
train_dataset = dataset.lmdbDataset(root=trainRoot)

if not random_sample:
    sampler = dataset.randomSequentialSampler(train_dataset, batchSize)
else:
    sampler = None
    
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batchSize,
    shuffle=True, sampler=sampler,
    num_workers=int(workers),
    collate_fn=dataset.my_collate_fn(imgH=imgH, imgW=imgW, keep_ratio=keep_ratio))

# create test dataset
test_dataset = dataset.lmdbDataset(
    root=valRoot, transform=dataset.resizeNormalize((100, 32)))

# class = number of characters in alphabet + 1
# image channel = 1
nclass = len(alphabet) + 1
nc = 1

# define loss
converter = utils.strLabelConverter(alphabet)
criterion = CTCLoss()


# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


crnn = crnn.CRNN(imgH, nc, nclass, nh)
crnn.apply(weights_init)
print(crnn)

# init image, label and label`s length
image = torch.FloatTensor(batchSize, 3, imgH, imgH)
text = torch.IntTensor(batchSize * 5)
length = torch.IntTensor(batchSize)

if cuda:
    crnn.cuda()
    crnn = torch.nn.DataParallel(crnn, device_ids=range(ngpu))
    image = image.cuda()
    criterion = criterion.cuda()
    
if pretrained != '':
    """
    every time save checkpoint on pretrained model: 'module' + state_dict key
    """
    print('loading pretrained model from %s' % pretrained)
    state_dict = torch.load(pretrained)
    state_dict_rename = OrderedDict()
    
    # remove 'module.' in state_dict key
    for k, v in state_dict.items():
        name = k[7:] 
        state_dict_rename[name] = v
    crnn.load_state_dict(state_dict_rename)

# wrap image, label and label`s length with Variable class allows pytorch to gradiate
image = Variable(image)
text = Variable(text)
length = Variable(length)

# loss averager
loss_avg = utils.averager()

# setup optimizer
if adam:
    optimizer = optim.Adam(crnn.parameters(), lr=lr,
                           betas=(beta1, 0.999))
elif adadelta:
    optimizer = optim.Adadelta(crnn.parameters())
else:
    optimizer = optim.RMSprop(crnn.parameters(), lr=lr)


def validation(net, dataset, criterion, max_iter=100):
    print("---"*20)
    print('Start val')

    for p in crnn.parameters():
        p.requires_grad = False
    
    # change model to eval mode
    net.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=opt.batchSize, num_workers=int(workers))
    
    # returns an iterator for the data_loader
    val_iter = iter(data_loader)

    i = 0
    n_correct = 0
    loss_avg = utils.averager()

    max_iter = min(max_iter, len(data_loader))
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        
        # get image and label for the validaton
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        # copy cpu_images to image
        utils.loadData(image, cpu_images)
        # encode label to number
        t, l = converter.encode(cpu_texts)
        # copy label and label`s to t and l
        utils.loadData(text, t)
        utils.loadData(length, l)
        
        # image (1x1x32x100)
        preds = crnn(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        # compute cost
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)

        _, preds = preds.max(2)
        # preds = preds.squeeze(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        for pred, target in zip(sim_preds, cpu_texts):
            # if pred == target.lower():     # for case insensitive
            if pred == target:              # for case sensitive
                n_correct += 1

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:n_test_disp]
    
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))
        
    # compute accuracy
    accuracy = n_correct / float(max_iter * batchSize)
    print('Val loss: %f, accuray: %f' % (loss_avg.val(), accuracy))


def trainBatch(net, criterion, optimizer):
    data = train_iter.next()
    
    cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    utils.loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts)
    utils.loadData(text, t)
    utils.loadData(length, l)

    preds = crnn(image)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    cost = criterion(preds, text, preds_size, length) / batch_size
    
    crnn.zero_grad()
    cost.backward()
    optimizer.step()
    
    return cost


if __name__ == "__main__":
    for epoch in range(nepoch):
        print(epoch)
        train_iter = iter(train_loader)
        i = 0
    
        while i < len(train_loader):
            for p in crnn.parameters():
                p.requires_grad = True
                crnn.train()

            cost = trainBatch(crnn, criterion, optimizer)
            loss_avg.add(cost)
            i += 1
        
            if i % displayInterval == 0:
                print('[%d/%d][%d/%d] Loss: %f' %
                      (epoch, nepoch, i, len(train_loader), loss_avg.val()))
                loss_avg.reset()

            if i % valInterval == 0:
                print(i)
                validation(crnn, test_dataset, criterion)
                
                # do checkpointing
                if i % saveInterval == 0:
                    print(i)
                    torch.save(
                        crnn.state_dict(), '{0}/netCRNN_synth90k_{1}_{2}.pth'.format(expr_dir, epoch, i))
                    
            