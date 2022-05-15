from numpy import NaN
from tqdm import tqdm
import argparse
from copy import deepcopy
import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader

from protors.protors import ProtoRS

from util.log import Log

def train_epoch(model: ProtoRS, 
                train_loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                epoch: int,
                device,
                log: Log = None,
                log_prefix: str = 'log_train_epochs',
                progress_prefix: str = 'Train Epoch'
                ) -> dict:
    
    # Make sure the model is in eval mode
    model.eval()
    # Store info about the procedure
    train_info = dict()
    total_loss = 0.
    total_acc = 0.
    # Create a log if required
    log_loss = f'{log_prefix}_losses'

    nr_batches = float(len(train_loader))

    # Show progress on progress bar
    train_iter = tqdm(enumerate(train_loader),
                    total=len(train_loader),
                    desc=progress_prefix+' %s'%epoch,
                    ncols=0)
    # Iterate through the data set to update leaves, prototypes and network
    for i, (xs, ys) in train_iter:
        # Make sure the model is in train mode
        model.train()
        # Reset the gradients
        optimizer.zero_grad()

        xs, ys = xs.to(device), ys.to(device)

        ys_onehot = F.one_hot(ys, num_classes=model.num_classes)

        # Perform a forward pass through the network
        ys_pred_cont, ys_pred_disc = model.forward(xs)

        # Learn prototypes and network with gradient descent. 
        # If disable_derivative_free_leaf_optim, leaves are optimized with gradient descent as well.
        # Compute the loss
        ys_prob = torch.softmax(ys_pred_disc, dim=1)
        loss = F.cross_entropy(ys_pred_disc, ys)
        loss_grad = (ys_prob - ys_onehot) / nr_batches
        # Compute the gradient
        ys_pred_cont.backward(loss_grad)
        # Update model parameters
        optimizer.step()

        for layer in model.mllp.layer_list[:-1]:
            layer.clip()
        
        # Count the number of correct classifications  
        ys_pred_max = torch.argmax(ys_prob, dim=1)      
        correct = torch.sum(torch.eq(ys_pred_max, ys))
        acc = correct.item() / float(len(xs))

        train_iter.set_postfix_str(
            f'Batch [{i + 1}/{len(train_loader)}], Loss: {loss.item():.3f}, Acc: {acc:.3f}'
        )
        # Compute metrics over this batch
        total_loss+=loss.item()
        total_acc+=acc

        if log is not None:
            log.log_values(log_loss, epoch, i + 1, loss.item(), acc)

    train_info['loss'] = total_loss/float(i+1)
    train_info['train_accuracy'] = total_acc/float(i+1)
    return train_info
