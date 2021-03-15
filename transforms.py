import torch
import random
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF




def pad_to_square(img, boxes, pad_value=0, normalized_labels=True):
    w, h = img.size
    w_factor, h_factor = (w,h) if normalized_labels else (1, 1)
    
    dim_diff = np.abs(h - w)
    pad1= dim_diff // 2
    pad2= dim_diff - pad1
    
    if h<=w:
        left, top, right, bottom= 0, pad1, 0, pad2
    else:
        left, top, right, bottom= pad1, 0, pad2, 0
    padding= (left, top, right, bottom)

    img_padded = TF.pad(img, padding=padding, fill=pad_value)
    w_padded, h_padded = img_padded.size
            
    x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
    y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
    x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
    y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)    
    
    x1 += padding[0] # left
    y1 += padding[1] # top
    x2 += padding[2] # right
    y2 += padding[3] # bottom
            
    boxes[:, 1] = ((x1 + x2) / 2) / w_padded
    boxes[:, 2] = ((y1 + y2) / 2) / h_padded
    boxes[:, 3] *= w_factor / w_padded
    boxes[:, 4] *= h_factor / h_padded

    return img_padded, boxes    


def hflip(image, labels):
    image = TF.hflip(image)
    labels[:, 1] = 1.0 - labels[:, 1]
    return image, labels



def transformer(image, labels, params):
    if params["pad2square"] is True:
        image,labels= pad_to_square(image, labels)
    
    image = TF.resize(image, params["target_size"])

    if random.random() < params["p_hflip"]:
        image, labels=hflip(image,labels)

    image=TF.to_tensor(image)
    targets = torch.zeros((len(labels), 6)) # additional dimension for slicing batch of targets by images first columns indexer
    targets[:, 1:] = torch.from_numpy(labels)
    
    return image, targets


def transformer2(image, boxes, labels, params):
    labels = labels-1
    labels = torch.hstack([labels.reshape(-1, 1), boxes])

    if params["pad2square"] is True:
        image, labels = pad_to_square(image, labels)
    
    image = TF.resize(image, params["target_size"])

    if random.random() < params["p_hflip"]:
        image, labels=hflip(image,labels)

    image=TF.to_tensor(image)
    targets = torch.zeros((len(labels), 6)) # additional dimension for slicing batch of targets by images first columns indexer
    targets[:, 1:] = labels
    
    return image, targets


def load_img_for_inference(path, target_size=(416, 416)):
    img = Image.open(path, mode='r').convert('RGB')
    w, h = img.size
    dim_diff = np.abs(h - w)
    pad1= dim_diff // 2
    pad2= dim_diff - pad1
    if h<=w:
        left, top, right, bottom= 0, pad1, 0, pad2
    else:
        left, top, right, bottom= pad1, 0, pad2, 0
    padding= (left, top, right, bottom)
    img_padded = TF.pad(img, padding=padding, fill=0)
    
    image = TF.resize(img_padded, target_size)
    image=TF.to_tensor(image)
    return image