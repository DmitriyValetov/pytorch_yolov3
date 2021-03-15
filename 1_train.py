import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau


from yolov3.model import Darknet, train_val
from utils import load_maps
from datasets import PascalVOCDataset, collate_fn2
from transforms import transformer2




def train(
    parameters,
):
    """
    Training.
    """
    label_map, rev_label_map, label_color_map = load_maps(os.path.join(parameters['data_folder'], 'label_map.json'))
    n_classes = len(label_map)  # number of different types of objects

    trans_params_train={
        "target_size" : [parameters['img_size']]*2,
        "pad2square": True,
        "p_hflip" : 1.0,
        "normalized_labels": True,
    }
    trans_params_val={
        "target_size" : [parameters['img_size']]*2,
        "pad2square": True,
        "p_hflip" : 0.0,
        "normalized_labels": True,
    }
    train_ds = PascalVOCDataset('trial_dataset_dumps', 'train', transformer2, trans_params_train, return_yolo=True)
    val_ds = PascalVOCDataset('trial_dataset_dumps', 'test', transformer2, trans_params_val, return_yolo=True)

    train_dl = DataLoader(
        train_ds,
        batch_size=parameters['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn2,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=parameters['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn2,
    )

    # Initialize model or load checkpoint
    if parameters['checkpoint_path'] is None or not os.path.exists(parameters['checkpoint_path']):
        start_epoch = 0
        model = Darknet(n_classes=n_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=parameters['lr']) 
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min',factor=0.5, patience=20,verbose=1)

    else:
        checkpoint = torch.load(parameters['checkpoint_path'])
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        lr_scheduler = checkpoint['lr_scheduler']


    model.to(parameters['device'])
    scaled_anchors=[model.module_list[ind][0].scaled_anchors for ind in model.yolo_layers_inds]

    mse_loss = nn.MSELoss(reduction="sum")
    bce_loss = nn.BCELoss(reduction="sum")
    params_loss={
        "scaled_anchors" : scaled_anchors,
        "ignore_thres": 0.5,
        "mse_loss": mse_loss,
        "bce_loss": bce_loss,
        "num_yolos": 3,
        "num_anchors": 3,
        "obj_scale": 1,
        "noobj_scale": 100,
        "device": model.device
    } 
    params_train={
        "num_epochs": 20,
        "optimizer": optimizer,
        "params_loss": params_loss,
        "train_dl": train_dl,
        "val_dl": val_dl,
        "sanity_check": True,
        "lr_scheduler": lr_scheduler,
        "path2weights": parameters['checkpoint_path'],
    }
    model, loss_hist = train_val(model, params_train, start_epoch)





if __name__ == '__main__':
      # folder with data files
    parameters = {
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'img_size': 416, # should be like n*8
        'batch_size': 8,
        'lr': 10**-3,
    }
    parameters['data_folder'] = 'trial_dataset_dumps'
    parameters['checkpoint_path'] = os.path.join(parameters['data_folder'], "checkpoint_yolov3.pkl")

    train(
        parameters,
    )
