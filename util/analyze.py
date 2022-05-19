import torch
import torch.nn.functional as F
import numpy as np
import argparse
from torch.utils.data import DataLoader
from protors.protors import ProtoRS
from util.log import Log

def analyze_output_shape(model: ProtoRS, trainloader: DataLoader, log: Log, device):
    with torch.no_grad():
        # Get a batch of training data
        xs, ys = next(iter(trainloader))
        xs, ys = xs.to(device), ys.to(device)
        log.log_message("Image input shape: "+str(xs[0,:,:,:].shape))
        log.log_message("Features output shape (without 1x1 conv layer): "+str(model.net(xs).shape))
        log.log_message("Convolutional output shape (with 1x1 conv layer): "+str(model.add_on(model.net(xs)).shape))
        log.log_message("Prototypes shape: "+str(model.prototype_layer.prototype_vectors.shape))

def log_learning_rates(optimizer, args: argparse.Namespace, log: Log):
    log.log_message("Learning rate net: "+str(optimizer.param_groups[0]['lr']))
    if 'densenet121' in args.net or 'resnet50' in args.net:
        log.log_message("Learning rate block: "+str(optimizer.param_groups[1]['lr']))
        log.log_message("Learning rate net 1x1 conv: "+str(optimizer.param_groups[2]['lr']))
        log.log_message("Learning rate prototypes: "+str(optimizer.param_groups[3]['lr']))
    else:
        log.log_message("Learning rate net 1x1 conv: "+str(optimizer.param_groups[1]['lr']))
        log.log_message("Learning rate prototypes: "+str(optimizer.param_groups[2]['lr']))
    log.log_message("Learning rate rule set: "+str(optimizer.param_groups[-1]['lr']))
    