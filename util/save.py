import torch
import argparse
from protors.protors import ProtoRS
from util.log import Log

def save_model(model: ProtoRS, optimizer, scheduler, epoch: int, log: Log, args: argparse.Namespace):
    # model.eval()
    # Save latest model
    model.save(f'{log.checkpoint_dir}/latest')
    model.save_state(f'{log.checkpoint_dir}/latest')
    torch.save(optimizer.state_dict(), f'{log.checkpoint_dir}/latest/optimizer_state.pth')
    torch.save(scheduler.state_dict(), f'{log.checkpoint_dir}/latest/scheduler_state.pth')

    # Save model every 10 epochs
    if epoch == args.epochs or epoch%10==0:
        model.save(f'{log.checkpoint_dir}/epoch_{epoch}')
        model.save_state(f'{log.checkpoint_dir}/epoch_{epoch}')
        torch.save(optimizer.state_dict(), f'{log.checkpoint_dir}/epoch_{epoch}/optimizer_state.pth')
        torch.save(scheduler.state_dict(), f'{log.checkpoint_dir}/epoch_{epoch}/scheduler_state.pth')

def save_best_train_model(model: ProtoRS, optimizer, scheduler, best_train_acc: float, train_acc: float, log: Log):
    # model.eval()
    if train_acc > best_train_acc:
        best_train_acc = train_acc
        model.save(f'{log.checkpoint_dir}/best_train_model')
        model.save_state(f'{log.checkpoint_dir}/best_train_model')
        torch.save(optimizer.state_dict(), f'{log.checkpoint_dir}/best_train_model/optimizer_state.pth')
        torch.save(scheduler.state_dict(), f'{log.checkpoint_dir}/best_train_model/scheduler_state.pth')
    return best_train_acc

def save_best_test_model(model: ProtoRS, optimizer, scheduler, best_test_acc: float, test_acc: float, log: Log):
    # model.eval()
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        model.save(f'{log.checkpoint_dir}/best_test_model')
        model.save_state(f'{log.checkpoint_dir}/best_test_model')
        torch.save(optimizer.state_dict(), f'{log.checkpoint_dir}/best_test_model/optimizer_state.pth')
        torch.save(scheduler.state_dict(), f'{log.checkpoint_dir}/best_test_model/scheduler_state.pth')
    return best_test_acc

def save_model_description(model: ProtoRS, optimizer, scheduler, description: str, log: Log):
    # model.eval()
    # Save model with description
    model.save(f'{log.checkpoint_dir}/'+description)
    model.save_state(f'{log.checkpoint_dir}/'+description)
    torch.save(optimizer.state_dict(), f'{log.checkpoint_dir}/'+description+'/optimizer_state.pth')
    torch.save(scheduler.state_dict(), f'{log.checkpoint_dir}/'+description+'/scheduler_state.pth')
    
