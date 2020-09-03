# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

import torch
import torch.utils.data
from torch.autograd import Variable

import utils
import dataset
import models.crnn as crnn
from config import *

test_dataset = dataset.lmdbDataset(
    root=testRoot, transform=dataset.resizeNormalize((100, 32)))

converter = utils.strLabelConverter(alphabet)

model = crnn.CRNN(imgH, nc, nclass, nh)
model = torch.nn.DataParallel(model).cuda()
model.load_state_dict(torch.load(model_path))

# random initial
image = torch.FloatTensor(1, 3, 3, 4)
text = torch.IntTensor(5)
length = torch.IntTensor(1)

image = Variable(image)
text = Variable(text)
length = Variable(length)


def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def evaluate(net, dataset):
    editing_distance = 0
    count = 0
    
    for p in model.parameters():
        p.requires_grad = False
    
    net.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=True, num_workers=int(workers))
    
    val_iter = iter(data_loader)
    for i in range(len(data_loader)):
        data = val_iter.next()
    
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        
        t, l = converter.encode(cpu_texts)    
        utils.loadData(text, t)
        utils.loadData(length, l)
        
        preds = model(image)    
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        gt = converter.decode(t.data, l.data, raw=False)
        
        if sim_preds[2:len(gt)-1] == gt[2:len(gt)-1]:
            count += 1
        
        editing_distance += levenshtein_distance(sim_preds[2:len(gt)-1], gt[2:len(gt)-1])

    editing_distance /= len(data_loader)
    accuracy = count / len(data_loader)
      
    print("Number of sample in dataset: ", len(data_loader))
    print("Number of positive prediction: ", count)
    print("Editing distance on dataset: ", editing_distance)
    print("Accuracy on dataset: ", accuracy)


if __name__ == "__main__":
    evaluate(model, test_dataset)
    