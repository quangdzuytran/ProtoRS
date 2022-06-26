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
from explain_global import explain_global

import torch
from shutil import copy
from copy import deepcopy

def run_model(args=None):
    args = args or get_args()
    # Create a logger
    log = Log(args.log_dir)
    print("Log dir: ", args.log_dir, flush=True)
    # Create a csv log for storing the test accuracy, mean train accuracy and mean loss for each epoch
    log.create_log('log_epoch_overview', 'epoch', 'test_acc', 'mean_train_acc', 'mean_train_crossentropy_loss_during_epoch')
    # Log the run arguments
    save_args(args, log.metadata_dir)
    if not args.disable_cuda and torch.cuda.is_available():
        # device = torch.device('cuda')
        device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
    else:
        device = torch.device('cpu')
        
    # Log which device was actually used
    log.log_message('Device used: '+str(device))

    # Create a log for logging the loss values
    log_prefix = 'log_train_epochs'
    log_loss = log_prefix+'_losses'
    log.create_log(log_loss, 'epoch', 'batch', 'loss', 'batch_train_acc')

    # Obtain the dataset and dataloaders
    trainloader, projectloader, testloader, classes, num_channels = get_dataloaders(args)
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

    model.save(f'{log.checkpoint_dir}/model_init')
    log.log_message("Number of prototypes: " + str(args.num_prototypes))
    analyze_output_shape(model, trainloader, log, device)

    best_train_acc = 0.
    best_test_acc = 0.

    if epoch < args.epochs + 1:
        '''
            TRAIN AND EVALUATE MODEL
        '''
        if args.resume:
            epoch -= 1
            if epoch >= args.projection_start and epoch != args.epochs and epoch % args.projection_cycle == 0:
                _, model = project(model, projectloader, device, args, log)
                eval_info = eval(model, testloader, epoch, device, log)
                original_test_acc = eval_info['test_accuracy']
                best_test_acc = save_best_test_model(model, optimizer, scheduler, best_test_acc, eval_info['test_accuracy'], log)
                log.log_values('log_epoch_overview', epoch, eval_info['test_accuracy'], "n.a.", "n.a.")
            epoch += 1
            
        for epoch in range(epoch, args.epochs + 1):
            log.log_message("\nEpoch %s"%str(epoch))
            # Freeze (part of) network for some epochs if indicated in args
            freeze(model, epoch, params_to_freeze, params_to_train, args, log)
            log_learning_rates(optimizer, args, log)

            # Changing between soft and hard threshold
            if epoch == args.soft_epochs + 1:
                model.binarize_layer.hard_threshold = True
            if model.binarize_layer.hard_threshold:
                log.log_message("Threshold: Hard")
            else:
                log.log_message("Threshold: Soft")
            
            # Train model
            train_info = train_epoch(model, trainloader, optimizer, epoch, device, log, log_prefix)
            save_model(model, optimizer, scheduler, epoch, log, args)
            best_train_acc = save_best_train_model(model, optimizer, scheduler, best_train_acc, train_info['train_accuracy'], log)
            
            # Evaluate model
            if args.epochs>100:
                if epoch%10==0 or epoch==args.epochs:
                    eval_info = eval(model, testloader, epoch, device, log)
                    original_test_acc = eval_info['test_accuracy']
                    best_test_acc = save_best_test_model(model, optimizer, scheduler, best_test_acc, eval_info['test_accuracy'], log)
                    log.log_values('log_epoch_overview', epoch, eval_info['test_accuracy'], train_info['train_accuracy'], train_info['loss'])
                else:
                    log.log_values('log_epoch_overview', epoch, "n.a.", train_info['train_accuracy'], train_info['loss'])
            else:
                eval_info = eval(model, testloader, epoch, device, log)
                original_test_acc = eval_info['test_accuracy']
                best_test_acc = save_best_test_model(model, optimizer, scheduler, best_test_acc, eval_info['test_accuracy'], log)
                log.log_values('log_epoch_overview', epoch, eval_info['test_accuracy'], train_info['train_accuracy'], train_info['loss'])
            
            # Project prototypes
            if epoch >= args.projection_start and epoch != args.epochs and epoch % args.projection_cycle == 0:
                _, model = project(model, projectloader, device, args, log)
                eval_info = eval(model, testloader, epoch, device, log)
                original_test_acc = eval_info['test_accuracy']
                best_test_acc = save_best_test_model(model, optimizer, scheduler, best_test_acc, eval_info['test_accuracy'], log)
                log.log_values('log_epoch_overview', epoch, eval_info['test_accuracy'], "n.a.", "n.a.")
            
            scheduler.step()
 
    else: #model was loaded and not trained, so evaluate only
        '''
            EVALUATE MODEL
        ''' 
        eval_info = eval(model, testloader, epoch, device, log)
        original_test_acc = eval_info['test_accuracy']
        best_test_acc = save_best_test_model(model, optimizer, scheduler, best_test_acc, eval_info['test_accuracy'], log)
        log.log_values('log_epoch_overview', epoch, eval_info['test_accuracy'], "n.a.", "n.a.")
    
    check = model.prototype_layer.prototype_vectors.lt(0).any() or model.prototype_layer.prototype_vectors.gt(1).any()
    if check:
        print("OUT OF RANGE PROTOTYPE VECTORS!!!")
        print("Min:", model.prototype_layer.prototype_vectors.min())
        print("Max:", model.prototype_layer.prototype_vectors.max())
    else:
        print("PROTOTYPE VECTORS ARE OKAY!!!")
    
    '''
        EVALUATE AND ANALYZE TRAINED MODEL
    '''
    log.log_message("Training Finished. Best training accuracy was %s, best test accuracy was %s\n"%(str(best_train_acc), str(best_test_acc)))
    trained_model = deepcopy(model)
    
    # Detect dead nodes
    '''
        PRUNE
    '''
    # Prune the model
    '''
        PROJECT
    '''
    # Project prototypes
    name = 'projected'
    projection_info, model = project(model, projectloader, device, args, log)
    projected_model = deepcopy(model)
    save_model_description(model, optimizer, scheduler, name, log)
    eval_info = eval(model, testloader, name, device, log)
    projected_test_acc = eval_info['test_accuracy']
    log.log_values('log_epoch_overview', name, projected_test_acc, "n.a.", "n.a.")

    # Upscaling and printing rule set
    explain_global(model, projection_info, device, args, log)

    # TODO: Visualize
    
    return trained_model.to('cpu'), projected_model.to('cpu'), original_test_acc, projected_test_acc, projection_info

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    args = get_args()
    run_model(args)
    
    