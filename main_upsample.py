from protors.protors import ProtoRS
from util.log import Log

from util.args import get_args, save_args, get_optimizer
from util.data import get_dataloaders
from util.init import init_model
from util.net import get_network, freeze
from util.save import *
from util.analyze import *
from protors.train import train_epoch
from protors.test import eval
from protors.project import project
from protors.upsample import upsample

import torch
from shutil import copy
from copy import deepcopy

def upsample_prototypes(model:ProtoRS = None, 
                        project_info = None, 
                        device = None, 
                        args = None, 
                        log:Log = None):
    args = args or get_args()
    if device is None:
        if not args.disable_cuda and torch.cuda.is_available():
            # device = torch.device('cuda')
            device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
        else:
            device = torch.device('cpu')

    if log is None: log = Log('/tmp/protors_log') # dummy logger

    _, projectloader, testloader, classes, num_channels = get_dataloaders(args) # Obtain the dataloader used for projection

    if model is None: # model needs to be loaded
        '''
        LOAD MODEL
        '''
        # Create a convolutional network based on arguments and add 1x1 conv layer
        features_net, add_on_layers = get_network(num_channels, args)
        # Create a ProtoRS model
        model = ProtoRS(num_classes=len(classes),
                        feature_net = features_net,
                        args = args,
                        add_on_layers = add_on_layers)
        model = model.to(device=device)
        # Determine which optimizer should be used to update the model parameters
        optimizer, params_to_freeze, params_to_train = get_optimizer(model, args)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=args.milestones, gamma=args.gamma)
        model, epoch = init_model(model, optimizer, scheduler, device, args)

    if project_info is None: # model was not projected beforehand
        # Perform projection without saving
        project_info, model = project(model, projectloader, device, args, log)

    eval_info = eval(model, testloader, 'projected', device, log)
    print("Projected model's accuracy: {0}".format(eval_info['test_accuracy']))

    # Upsample prototypes
    print('Upscaling prototypes ...')
    project_info = upsample(model, project_info, projectloader, 'projected', args, log)
    print('Upscaling finished.')
    # Visualize
    return project_info

if __name__ == '__main__':
    args = get_args()
    upsample_prototypes(args=args)
    
    