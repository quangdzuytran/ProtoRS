import os
import argparse
import pickle
import numpy as np
import random
import torch
import torch.optim

"""
    Utility functions for handling parsed arguments

"""
def get_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser('Train a ProtoRS model')
    parser.add_argument('--dataset',
                        type=str,
                        default='CUB-200-2011',
                        help='Data set on which the ProtoRS model should be trained')
    parser.add_argument("--structure",
                        type=str,
                        default='256@128',
                        help='Set the number of nodes in the logical layers. '
                         'E.g., 10@64, 10@64@32@16.')
    parser.add_argument("--num_prototypes",
                        type=int,
                        default=512,
                        help='Number of prototypes to be learned')
    parser.add_argument('--net',
                        type=str,
                        default='resnet50_inat',
                        help='Base network used in the model. Pretrained network on iNaturalist is only available for resnet50_inat (default). Others are pretrained on ImageNet. Options are: resnet18, resnet34, resnet50, resnet50_inat, resnet101, resnet152, densenet121, densenet169, densenet201, densenet161, vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg13_bn, vgg16_bn or vgg19_bn')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='Batch size when training the model using minibatch gradient descent')
    parser.add_argument('--epochs',
                        type=int,
                        default=100,
                        help='The number of epochs the model should be trained')
    parser.add_argument('--optimizer',
                        type=str,
                        default='AdamW',
                        help='The optimizer that should be used when training the model')
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='The optimizer learning rate for training the prototypes')
    parser.add_argument('--lr_block',
                        type=float,
                        default=0.001,
                        help='The optimizer learning rate for training the 1x1 conv layer and last conv layer of the underlying neural network (applicable to resnet50 and densenet121)')
    parser.add_argument('--lr_net',
                        type=float,
                        default=1e-5,
                        help='The optimizer learning rate for the underlying neural network')
    parser.add_argument('--lr_rule',
                        type=float,
                        default=0.001,
                        help='The optimizer learning rate for the logical layers')
    parser.add_argument('--momentum',
                        type=float,
                        default=0.9,
                        help='The optimizer momentum parameter (only applicable to SGD)')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.0,
                        help='Weight decay used in the optimizer')
    parser.add_argument('--disable_cuda',
                        action='store_true',
                        help='Flag that disables GPU usage if set')
    parser.add_argument('--log_dir',
                        type=str,
                        default='./runs/run_protors',
                        help='The directory in which train progress should be logged')
    parser.add_argument('--W1',
                        type=int,
                        default = 1,
                        help='Width of the prototype. Correct behaviour of the model with W1 != 1 is not guaranteed')
    parser.add_argument('--H1',
                        type=int,
                        default = 1,
                        help='Height of the prototype. Correct behaviour of the model with H1 != 1 is not guaranteed')
    parser.add_argument('--num_features',
                        type=int,
                        default = 256,
                        help='Depth of the prototype and therefore also depth of convolutional output')
    parser.add_argument('--milestones',
                        type=str,
                        default='',
                        help='The milestones for the MultiStepLR learning rate scheduler')
    parser.add_argument('--gamma',
                        type=float,
                        default=0.5,
                        help='The gamma for the MultiStepLR learning rate scheduler. Needs to be 0<=gamma<=1')
    parser.add_argument('--state_dict_dir_net',
                        type=str,
                        default='',
                        help='The directory containing a state dict with a pretrained backbone network')
    parser.add_argument('--state_dict_dir_model',
                        type=str,
                        default='',
                        help='The directory containing a state dict (checkpoint) with a pretrained protors. Note that training further from a checkpoint does not seem to work correctly. Evaluating a trained protors does work.')
    parser.add_argument('--freeze_epochs',
                        type=int,
                        default = 30,
                        help='Number of epochs where pretrained features_net will be frozen'
                        )
    parser.add_argument('--dir_for_saving_images',
                        type=str,
                        default='upsampling_results',
                        help='Directory for saving the prototypes, patches and heatmaps')
    parser.add_argument('--upsample_threshold',
                        type=float,
                        default=0.98,
                        help='Threshold (between 0 and 1) for visualizing the nearest patch of an image after upsampling. The higher this threshold, the larger the patches.')
    parser.add_argument('--disable_pretrained',
                        action='store_true',
                        help='When set, the backbone network is initialized with random weights instead of being pretrained on another dataset). When not set, resnet50_inat is initalized with weights from iNaturalist2017. Other networks are initialized with weights from ImageNet'
                        )
    parser.add_argument('--estimated_grad',
                        action='store_true',
                        help='Flag that uses estimated gradient.'
                        )
    parser.add_argument('--soft_epochs',
                        type=int,
                        default=50,
                        help='Number of epochs where soft threshold (sigmoid) will be used')
    parser.add_argument('projection_cycle',
                        type=int,
                        default=10,
                        help='Cycle (in epochs) for prototype projection after model is unfrozen')
    
    args = parser.parse_args()
    args.milestones = get_milestones(args)
    args.rule_file = args.log_dir + '/ruleset.txt'
    return args

"""
    Parse the milestones argument to get a list
    :param args: The arguments given
    """
def get_milestones(args: argparse.Namespace):
    if args.milestones != '':
        milestones_list = args.milestones.split(',')
        for m in range(len(milestones_list)):
            milestones_list[m]=int(milestones_list[m])
    else:
        milestones_list = []
    return milestones_list

def save_args(args: argparse.Namespace, directory_path: str) -> None:
    """
    Save the arguments in the specified directory as
        - a text file called 'args.txt'
        - a pickle file called 'args.pickle'
    :param args: The arguments to be saved
    :param directory_path: The path to the directory where the arguments should be saved
    """
    # If the specified directory does not exists, create it
    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)
    # Save the args in a text file
    with open(directory_path + '/args.txt', 'w') as f:
        for arg in vars(args):
            val = getattr(args, arg)
            if isinstance(val, str):  # Add quotation marks to indicate that the argument is of string type
                val = f"'{val}'"
            f.write('{}: {}\n'.format(arg, val))
    # Pickle the args for possible reuse
    with open(directory_path + '/args.pickle', 'wb') as f:
        pickle.dump(args, f)                                                                               
    

def load_args(directory_path: str) -> argparse.Namespace:
    """
    Load the pickled arguments from the specified directory
    :param directory_path: The path to the directory from which the arguments should be loaded
    :return: the unpickled arguments
    """
    with open(directory_path + '/args.pickle', 'rb') as f:
        args = pickle.load(f)
    return args

def get_optimizer(model, args: argparse.Namespace) -> torch.optim.Optimizer:
    """
    Construct the optimizer as dictated by the parsed arguments
    :param model: The model that should be optimized
    :param args: Parsed arguments containing hyperparameters. The '--optimizer' argument specifies which type of
                 optimizer will be used. Optimizer specific arguments (such as learning rate and momentum) can be passed
                 this way as well
    :return: the optimizer corresponding to the parsed arguments, parameter set that can be frozen, and parameter set of the net that will be trained
    """

    optim_type = args.optimizer
    #create parameter groups
    params_to_freeze = []
    params_to_train = []

    # set up optimizer
    if 'resnet50' in args.net: 
        # freeze resnet50 except last convolutional layer
        for name, param in model.net.named_parameters():
            if 'layer4.2' not in name:
                params_to_freeze.append(param)
            else:
                params_to_train.append(param)
   
        if optim_type == 'SGD':
            paramlist = [
                {"params": params_to_freeze, "lr": args.lr_net, "weight_decay_rate": args.weight_decay, "momentum": args.momentum},
                {"params": params_to_train, "lr": args.lr_block, "weight_decay_rate": args.weight_decay, "momentum": args.momentum},
                {"params": model.add_on.parameters(), "lr": args.lr_block, "weight_decay_rate": args.weight_decay, "momentum": args.momentum},
                {"params": model.prototype_layer.parameters(), "lr": args.lr, "weight_decay_rate": 0, "momentum": 0}]
            for layer in model.mllp.layer_list:
                paramlist.append({"params": layer.parameters(), "lr": args.lr_rule, "weight_decay_rate": args.weight_decay, "momentum": args.momentum})
        
        else:
            paramlist = [
                {"params": params_to_freeze, "lr": args.lr_net, "weight_decay_rate": args.weight_decay},
                {"params": params_to_train, "lr": args.lr_block, "weight_decay_rate": args.weight_decay},
                {"params": model.add_on.parameters(), "lr": args.lr_block, "weight_decay_rate": args.weight_decay},
                {"params": model.prototype_layer.parameters(), "lr": args.lr, "weight_decay_rate": 0}]
            for layer in model.mllp.layer_list:
                paramlist.append({"params": layer.parameters(), "lr": args.lr_rule, "weight_decay_rate": args.weight_decay})              
    
    elif args.net == 'densenet121':
        # freeze densenet121 except last convolutional layer
        for name, param in model.net.named_parameters():
            if 'denseblock4' not in name and 'norm5' not in name:
                params_to_freeze.append(param)
            else:
                params_to_train.append(param)
        
        paramlist = [
            {"params": params_to_freeze, "lr": args.lr_net, "weight_decay_rate": args.weight_decay},
            {"params": params_to_train, "lr": args.lr_block, "weight_decay_rate": args.weight_decay},
            {"params": model.add_on.parameters(), "lr": args.lr_block, "weight_decay_rate": args.weight_decay},
            {"params": model.prototype_layer.parameters(), "lr": args.lr, "weight_decay_rate": 0}]
        for layer in model.mllp.layer_list:
            paramlist.append({"params": layer.parameters(), "lr": args.lr_rule, "weight_decay_rate": args.weight_decay})

    else:
        paramlist = [
            {"params": model.net.parameters(), "lr": args.lr_net, "weight_decay_rate": args.weight_decay},
            {"params": model.add_on.parameters(), "lr": args.lr_block, "weight_decay_rate": args.weight_decay},
            {"params": model.prototype_layer.parameters(), "lr": args.lr, "weight_decay_rate": 0}]
        for layer in model.mllp.layer_list:
            paramlist.append({"params": layer.parameters(), "lr": args.lr_rule, "weight_decay_rate": args.weight_decay})
    
    if optim_type == 'SGD':
        return torch.optim.SGD(paramlist,
                               lr=args.lr,
                               momentum=args.momentum), params_to_freeze, params_to_train
    if optim_type == 'Adam':
        return torch.optim.Adam(paramlist, lr=args.lr, eps=1e-07), params_to_freeze, params_to_train
    if optim_type == 'AdamW':
        return torch.optim.AdamW(paramlist, lr=args.lr, eps=1e-07, weight_decay=args.weight_decay), params_to_freeze, params_to_train

    raise Exception('Unknown optimizer argument given!')


