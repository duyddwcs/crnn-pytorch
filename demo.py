import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
import cv2
import imutils
from collections import OrderedDict 

import models.crnn as crnn
from config import *


model_path = './'
img_path = './images/image.png'            

model = crnn.CRNN(32, 1, 92, 256)
print(model)
if torch.cuda.is_available():
    # model = model.cuda()
    model = torch.nn.DataParallel(model).cuda()
    
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path))


## remove 'module.' in state_dict keys
# state_dict = torch.load(model_path)
# state_dict_rename = OrderedDict()
# for k, v in state_dict.items():
#     name = k[7:] # remove `module.`
#     state_dict_rename[name] = v
    
# model.load_state_dict(state_dict_rename)


converter = utils.strLabelConverter(alphabet)

transformer = dataset.resizeNormalize((100, 32))
image = Image.open(img_path).convert('L')
image = transformer(image)

if torch.cuda.is_available():
    image = image.cuda()
image = image.view(1, *image.size())
image = Variable(image)

model.eval()
preds = model(image)

_, preds = preds.max(2)
preds = preds.transpose(1, 0).contiguous().view(-1)

preds_size = Variable(torch.IntTensor([preds.size(0)]))
raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
print('%-20s' % (sim_pred[2:len(sim_pred) - 1]))

image = cv2.imread(img_path)
image = imutils.resize(image, width=400)
cv2.imshow("original", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
