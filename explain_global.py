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
import argparse

def get_args_explain() -> argparse.Namespace:
    parser = argparse.ArgumentParser('Load and globally explain a ProtoRS model.')
    parser.add_argument('--dataset',
                        type=str,
                        default='CUB-200-2011',
                        help='Data set on which the ProtoRS model was trained on')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='Batch size for projecting, testing, and detecting dead nodes')
    parser.add_argument('--disable_cuda',
                        action='store_true',
                        help='Flag that disables GPU usage if set')
    parser.add_argument('--state_dict_dir_model',
                        type=str,
                        default='',
                        help='The directory containing a state dict (checkpoint) with a pretrained protors. Note that training further from a checkpoint does not seem to work correctly. Evaluating a trained protors does work.')
    parser.add_argument('--dir_for_saving_images',
                        type=str,
                        default='upsampling_results',
                        help='Directory for saving the prototypes, patches and heatmaps')
    parser.add_argument('--upsample_threshold',
                        type=float,
                        default=0.98,
                        help='Threshold (between 0 and 1) for visualizing the nearest patch of an image after upsampling. The higher this threshold, the larger the patches.')
    parser.add_argument('--no_reeval',
                        action='store_true',
                        help='Not to reevaluate model after loading')
    parser.add_argument('--no_upsample',
                        action='store_true',
                        help='Not to perform prototype upsampling')
    parser.add_argument('--no_ruleprint',
                        action='store_true',
                        help='Not to perform rule printing')
    args = parser.parse_args()
    args.log_dir = args.state_dict_dir_model
    args.dir_for_saving_images  = 'upsampling_results'
    args.rule_file = args.state_dict_dir_model + '/ruleset.txt'
    return args

def explain_global(model:ProtoRS = None, 
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

    trainloader, projectloader, testloader, classes, num_channels = get_dataloaders(args) # Obtain the dataloader used for projection

    if model is None: # model needs to be loaded
        # load model
        if args.disable_cuda or not torch.cuda.is_available():
            model = torch.load(args.state_dict_dir_model+'/model.pth', map_location=device)
        else:
            model = torch.load(args.state_dict_dir_model+'/model.pth')
        model.to(device=device)

    # Reevaluate
    if not hasattr(args, 'no_reeval') or hasattr(args, 'no_reeval') and not args.no_reeval: 
        eval_info = eval(model, testloader, 'projected', device, log)
        print("Loaded model's accuracy: {0}".format(eval_info['test_accuracy']))
    print(model.mllp.layer_list)

    #  Upsample prototypes
    if not hasattr(args, 'no_upsample') or hasattr(args, 'no_upsample') and not args.no_upsample:
        # Project
        if project_info is None: # model was not projected beforehand
            # Perform projection without saving
            project_info, model = project(model, projectloader, device, args, log)
            eval_info = eval(model, testloader, 'projected', device, log)
            print("Projected model's accuracy: {0}".format(eval_info['test_accuracy']))

        # Upsample
        print('Upscaling prototypes ...')
        project_info = upsample(model, project_info, projectloader, 'projected', args, log)
        print('Upscaling finished.')

    # Print rules
    if not hasattr(args, 'no_ruleprint') or hasattr(args, 'no_ruleprint') and not args.no_ruleprint:
        print('Printing rule set ...')
        with open(args.rule_file, 'w') as rule_file:
            model.rule_print(classes, trainloader, device=device, file=rule_file)
        print('Rule set printed at {}'.format(args.rule_file))
    
        # Save extracted model
        model.save(f'{args.state_dict_dir_model}/'+ 'rules_extracted')

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    args = get_args_explain()
    explain_global(args=args)