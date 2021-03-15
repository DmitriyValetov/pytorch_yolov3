import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_pil_image

from utils import rescale_bbox



def show_img_bbox(img, targets, names):
    """
    img: torch.tensor or pil image
    targets: torch tensor or numpy of shape: (n, 5) class, xc,yc,w,h
    names: list of str
    """

    COLORS = np.random.randint(0, 255, size=(80, 3),dtype="uint8")
    fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 16)

    if torch.is_tensor(img):
        img=to_pil_image(img)
    if torch.is_tensor(targets):
        targets=targets.numpy()[:,1:]
        
    W, H = img.size
    draw = ImageDraw.Draw(img)
    
    for tg in targets:
        id_=int(tg[0])
        bbox=tg[1:]
        bbox=rescale_bbox(bbox,W,H)
        xc,yc,w,h=bbox
        
        color = [int(c) for c in COLORS[id_]]
        name = names[id_]
        
        draw.rectangle(((xc-w/2, yc-h/2), (xc+w/2, yc+h/2)),outline=tuple(color),width=3)
        draw.text((xc-w/2,yc-h/2),name, font=fnt, fill=(255,255,255,0))
    plt.figure(figsize=(10,10))
    plt.imshow(np.array(img))        


def show_img_bbox2(img, boxes, labels, names):
    labels = labels.reshape(-1, 1)
    targets = torch.hstack([torch.zeros_like(labels), labels, boxes])
    show_img_bbox(img, targets, names)