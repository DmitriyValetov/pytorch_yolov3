import os
import copy
import torch
from torch import nn

from .model_archs import name2config
from .loss import get_loss_batch

def save_checkpoint(epoch, model, optimizer, lr_scheduler, filename):
    """
    Save model checkpoint.

    :param epoch: epoch number
    :param model: model
    :param optimizer: optimizer
    """
    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer,
             'lr_scheduler': lr_scheduler}
    torch.save(state, filename+'_')
    if os.path.exists(filename):
        os.remove(filename)
    os.rename(filename+'_', filename)


def get_model(filename):
    return torch.load(filename)['model']

def parse_model_config(path2file=None, model_name=None):
    if isinstance(model_name, str):
        lines = name2config[model_name].split('\n')
    else:
        with open(path2file, 'r') as cfg_file:
            lines = cfg_file.read().split('\n')

    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines] 
    
    blocks_list = []
    for line in lines:
        # start of a new block
        if line.startswith('['): 
            blocks_list.append({})
            blocks_list[-1]['type'] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            value = value.strip()
            blocks_list[-1][key.rstrip()] = value.strip()

    return blocks_list


def create_layers(
    n_channels,
    n_classes,
    img_size,
    blocks_list, 
    device='cpu'
):
    channels_list = [n_channels]
    module_list = nn.ModuleList()
    yolo_layers_inds = []

    for layer_ind, layer_dict in enumerate(blocks_list):
        modules = nn.Sequential()
        
        if layer_dict["type"] in ["convolutional", 'convolutional_before_yolo']:
            if layer_dict["type"] == 'convolutional_before_yolo':
                filters = (n_classes+5)*3  
            else:
                filters = int(layer_dict["filters"])

            kernel_size = int(layer_dict["size"])
            pad = (kernel_size - 1) // 2
            bn=layer_dict.get("batch_normalize",0)    
            
            
            conv2d= nn.Conv2d(
                        in_channels=channels_list[-1],
                        out_channels=filters,
                        kernel_size=kernel_size,
                        stride=int(layer_dict["stride"]),
                        padding=pad,
                        bias=not bn)
            modules.add_module("conv_{0}".format(layer_ind), conv2d)
            
            if bn:
                bn_layer = nn.BatchNorm2d(filters,momentum=0.9, eps=1e-5)
                modules.add_module("batch_norm_{0}".format(layer_ind), bn_layer)
                
                
            if layer_dict["activation"] == "leaky":
                activn = nn.LeakyReLU(0.1)
                modules.add_module("leaky_{0}".format(layer_ind), activn)
                
        elif layer_dict["type"] == "upsample":
            stride = int(layer_dict["stride"])
            upsample = nn.Upsample(scale_factor = stride)
            modules.add_module("upsample_{}".format(layer_ind), upsample) 
            

        elif layer_dict["type"] == "shortcut": # to sum
            backwards=int(layer_dict["from"])
            filters = channels_list[1:][backwards]
            modules.add_module("shortcut_{}".format(layer_ind), EmptyLayer())
            
        elif layer_dict["type"] == "route": # to cat
            layers = [int(x) for x in layer_dict["layers"].split(",")]
            filters = sum([channels_list[1:][l] for l in layers])
            modules.add_module("route_{}".format(layer_ind), EmptyLayer())
            
        elif layer_dict["type"] == "yolo":
            anchors = [int(a) for a in layer_dict["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]

            mask = [int(m) for m in layer_dict["mask"].split(",")]
            
            anchors = [anchors[i] for i in mask]
            
            yolo_layer = YOLOLayer(anchors, n_classes, device, img_size)
            yolo_layers_inds.append(layer_ind)
            modules.add_module("yolo_{}".format(layer_ind), yolo_layer)
            
        module_list.append(modules)       
        channels_list.append(filters)

    return module_list, yolo_layers_inds     



class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()
        
        
class YOLOLayer(nn.Module):

    def __init__(self, anchors, num_classes, device, img_dim=416):
        super(YOLOLayer, self).__init__()
        self.device = device
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.img_dim = img_dim
        self.grid_size = 0 
        
    def to(self, device):
        self.device = device
        if hasattr(self, 'grid_x'):
            self.grid_x = self.grid_x.to(device)
            self.grid_y = self.grid_y.to(device)
            self.anchor_w = self.anchor_w.to(device)
            self.anchor_h = self.anchor_h.to(device)
            self.scaled_anchors = self.scaled_anchors.to(device)

        
    def forward(self, x_in): # n, chnls, grid_w, grid_h
        batch_size = x_in.size(0)
        grid_size = x_in.size(2)
        
        prediction=x_in.view(batch_size, self.num_anchors, self.num_classes + 5, grid_size, grid_size) # +5: obj score, x,y,w,h
        prediction=prediction.permute(0, 1, 3, 4, 2)
        prediction=prediction.contiguous()
        
        obj_score = torch.sigmoid(prediction[..., 4]) 
        pred_cls = torch.sigmoid(prediction[..., 5:]) 
        
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size)
            
        pred_boxes=self.transform_outputs(prediction) 
        
        output = torch.cat(
            (
                pred_boxes.view(batch_size, -1, 4),
                obj_score.view(batch_size, -1, 1),
                pred_cls.view(batch_size, -1, self.num_classes),
            ), -1,)
        return output        
    
    
        
    def compute_grid_offsets(self, grid_size):
        self.grid_size = grid_size
        self.stride = self.img_dim / self.grid_size
        
        self.grid_x = torch.arange(grid_size, device=self.device).repeat(1, 1, grid_size, 1 ).type(torch.float32)
        self.grid_y = torch.arange(grid_size, device=self.device).repeat(1, 1, grid_size, 1).transpose(3, 2).type(torch.float32)
        
        scaled_anchors=[(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors]
        self.scaled_anchors=torch.tensor(scaled_anchors,device=self.device)
        
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))
        
        
        
    def transform_outputs(self,prediction):
        x = torch.sigmoid(prediction[..., 0]) # Center x
        y = torch.sigmoid(prediction[..., 1]) # Center y
        w = prediction[..., 2] # Width
        h = prediction[..., 3] # Height

        pred_boxes = torch.zeros_like(prediction[..., :4]).to(self.device)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h
        
        return pred_boxes * self.stride             



class Darknet(nn.Module):
    def __init__(
        self,
        n_channels=3,
        n_classes=80,
        img_size=416,
        model_name="yolov3", 
        config_path=None, 
        device='cpu'
    ):
        super(Darknet, self).__init__()
        self.device = device
        self.n_channels = n_channels
        self.n_classes  = n_classes
        self.img_size   = img_size

        assert any([not inp is None for inp in [model_name, config_path]])
        self.blocks_list = parse_model_config(config_path, model_name)
        self.module_list, self.yolo_layers_inds = create_layers(
            self.n_channels, 
            self.n_classes,  
            self.img_size,   
            self.blocks_list, 
            self.device, 
        )
        self.init_scaled_anchors()
    
    def init_scaled_anchors(self):
        """
            TODO: to be removed by better arch
        """
        dummy_img=torch.rand(1,3,416,416).to(self.device)
        with torch.no_grad():
            dummy_out_cat, dummy_out = self.forward(dummy_img)


    def to(self, device):
        self.device = device
        super(Darknet, self).to(device)
        for yolo_ind in self.yolo_layers_inds:
            self.module_list[yolo_ind][0].to(device)

    def forward(self, x):
        img_dim = x.shape[2]
        layer_outputs, yolo_outputs = [], []
        
        for block, module in zip(self.blocks_list, self.module_list):
            if block["type"] in ["convolutional", "upsample", "maxpool", 'convolutional_before_yolo']:
                x = module(x)        
                
            elif block["type"] == "shortcut":
                layer_ind = int(block["from"])
                x = layer_outputs[-1] + layer_outputs[layer_ind]

            elif block["type"] == "yolo":
                x = module[0](x)
                yolo_outputs.append(x)
            
            elif block["type"] == "route":
                x = torch.cat([layer_outputs[int(l_i)] 
                               for l_i in block["layers"].split(",")], 1)
            layer_outputs.append(x)
        
        yolo_out_cat = torch.cat(yolo_outputs, 1)
        return yolo_out_cat, yolo_outputs    



def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']


def loss_epoch(model,params_loss,dataset_dl,sanity_check=False,opt=None):
    running_loss=0.0
    len_data=len(dataset_dl.dataset)
    running_metrics= {}
    
    for xb, yb in dataset_dl:
        yb=yb.to(model.device)
        _,output=model(xb.to(model.device))
        loss_b=get_loss_batch(output, yb, params_loss, opt)
        running_loss+=loss_b
        if sanity_check is True:
            break 
    loss=running_loss/float(len_data)
    return loss


def train_val(model, params, start_epoch):
    num_epochs=params["num_epochs"]
    params_loss=params["params_loss"]
    opt=params["optimizer"]
    train_dl=params["train_dl"]
    val_dl=params["val_dl"]
    sanity_check=params["sanity_check"]
    lr_scheduler=params["lr_scheduler"]
    path2weights=params["path2weights"]
    
    
    loss_history={
        "train": [],
        "val": [],
    }
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss=float('inf') 
    
    for epoch in range(start_epoch, start_epoch+num_epochs):
        current_lr=get_lr(opt)
        print('Epoch {}/{}, current lr={}'.format(epoch, start_epoch+num_epochs - 1, current_lr)) 
        model.train()
        train_loss=loss_epoch(model,params_loss,train_dl,sanity_check,opt)
        loss_history["train"].append(train_loss)
        print("train loss: %.6f" %(train_loss))    
        
        model.eval()
        with torch.no_grad():
            val_loss=loss_epoch(model,params_loss,val_dl,sanity_check)
        loss_history["val"].append(val_loss)
        print("val loss: %.6f" %(val_loss))
        
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            print(path2weights)
            save_checkpoint(epoch, model, opt, lr_scheduler, path2weights)
            print("Copied best model weights!")
            
        lr_scheduler.step(val_loss)
        if current_lr != get_lr(opt):
            print("Loading best model weights!")
            model.load_state_dict(best_model_wts)

        print("-"*10) 
    model.load_state_dict(best_model_wts)
    return model, loss_history            