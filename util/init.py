import argparse
import torch
from protors.protors import ProtoRS
import os
import pickle

def load_state(directory_path: str, device):
    with open(directory_path + '/model.pkl', 'rb') as f:
        model = pickle.load(f)
        state = torch.load(directory_path + '/model_state.pth', map_location=device)
        model.load_state_dict(state)
    return model

def init_model(model: ProtoRS, optimizer, scheduler, device, args: argparse.Namespace):
    epoch = 1
    mean = 0.5
    std = 0.1
    # load trained protors if flag is set

    # NOTE: TRAINING FURTHER FROM A CHECKPOINT DOESN'T SEEM TO WORK CORRECTLY. EVALUATING A TRAINED PROTOTREE FROM A CHECKPOINT DOES WORK. 
    if args.state_dict_dir_model != '':
        if not args.disable_cuda and torch.cuda.is_available():
            device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
        else:
            device = torch.device('cpu')
        if args.resume:
            if args.disable_cuda or not torch.cuda.is_available():
            # model = load_state(args.state_dict_dir_model, device)
                model.load_state_dict(torch.load(args.state_dict_dir_model+'/model_state.pth', map_location=device))
            else:
                model.load_state_dict(torch.load(args.state_dict_dir_model+'/model_state.pth'))
            model.to(device=device)

            try:
                epoch = int(args.state_dict_dir_model.split('epoch_')[-1]) + 1
            except:
                epoch=args.epochs+1
            print("Train further from epoch: ", epoch, flush=True)
            optimizer.load_state_dict(torch.load(args.state_dict_dir_model+'/optimizer_state.pth', map_location=device))

            if epoch>args.freeze_epochs:
                for parameter in model.net.parameters():
                    parameter.requires_grad = True
            # if epoch>args.soft_epochs:
            #     model.binarize_layer.hard_threshold = True
            
            if os.path.isfile(args.state_dict_dir_model+'/scheduler_state.pth'):
                # scheduler.load_state_dict(torch.load(args.state_dict_dir_model+'/scheduler_state.pth'))
                # print(scheduler.state_dict(),flush=True)
                scheduler.last_epoch = epoch - 2
                scheduler._step_count = epoch - 1
                scheduler.step()
        else:
            if args.disable_cuda or not torch.cuda.is_available():
            # model = load_state(args.state_dict_dir_model, device)
                model = torch.load(args.state_dict_dir_model+'/model.pth', map_location=device)
            else:
                model = torch.load(args.state_dict_dir_model+'/model.pth')
            model.to(device=device)
            
            optimizer.load_state_dict(torch.load(args.state_dict_dir_model+'/optimizer_state.pth', map_location=device))
            
            epoch=args.epochs+1
            
            if os.path.isfile(args.state_dict_dir_model+'/scheduler_state.pth'):
                # scheduler.load_state_dict(torch.load(args.state_dict_dir_model+'/scheduler_state.pth'))
                # print(scheduler.state_dict(),flush=True)
                scheduler.last_epoch = epoch - 1
                scheduler._step_count = epoch
    
    elif args.state_dict_dir_net != '': # load pretrained conv network
        # initialize prototypes
        # torch.nn.init.normal_(model.prototype_layer.prototype_vectors, mean=mean, std=std)
        #strict is False so when loading pretrained model, ignore the linear classification layer
        model.net.load_state_dict(torch.load(args.state_dict_dir_net+'/model_state.pth'), strict=False)
        model.add_on.load_state_dict(torch.load(args.state_dict_dir_net+'/model_state.pth'), strict=False) 
        model.fc.apply(init_weights_xavier)
    
    else:
        with torch.no_grad():
            # initialize prototypes
            # torch.nn.init.normal_(model.prototype_layer.prototype_vectors, mean=mean, std=std)
            model.add_on.apply(init_weights_xavier)
            model.fc.apply(init_weights_xavier)
    return model, epoch

def init_weights_xavier(m):
    if type(m) == torch.nn.Conv2d:
        torch.nn.init.xavier_normal_(m.weight, gain=torch.nn.init.calculate_gain('sigmoid'))

def init_weights_kaiming(m):
    if type(m) == torch.nn.Conv2d:
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')