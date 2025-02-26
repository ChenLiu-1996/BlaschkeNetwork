'''
Adapted from https://github.com/ZiyaoLi/fast-kan/tree/master/examples
'''

from types import SimpleNamespace
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.BN1d import BlaschkeNetwork1d
from nn_utils.scheduler import LinearWarmupCosineAnnealingLR
from nn_utils.seed import seed_everything
from nn_utils.log import log, count_parameters


def load_mnist(args):
    # Load MNIST
    transform_train = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1307],
                              std=[0.3081])
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1307],
                              std=[0.3081])
    ])
    train_set = torchvision.datasets.MNIST(
        root="../data", train=True, download=True, transform=transform_train
    )
    val_set = torchvision.datasets.MNIST(
        root="../data", train=False, download=True, transform=transform_test
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--layers',
                        type=int,
                        default=5)
    parser.add_argument('--lr',
                        help='Learning rate.',
                        type=float,
                        default=1e-2)
    parser.add_argument('--batch-size',
                        type=int,
                        default=256)
    parser.add_argument('--num-epoch',
                        type=int,
                        default=100)
    parser.add_argument('--loss-recon-coeff',
                        type=float,
                        default=1.0)
    parser.add_argument('--num-workers',
                        type=int,
                        default=8)
    parser.add_argument('--random-seed',
                        type=int,
                        default=1)
    args = SimpleNamespace(**vars(parser.parse_args()))

    model_save_path = f'../checkpoints/mnist/BN1d_{args.layers}_lr_{args.lr}_epoch_{args.num_epoch}-seed_{args.random_seed}/model_best_val_acc.ckpt'
    results_dir = f'../results/mnist/BN1d_{args.layers}_lr_{args.lr}_epoch_{args.num_epoch}-seed_{args.random_seed}/'
    log_dir = os.path.join(results_dir, 'log.txt')

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    seed_everything(args.random_seed)
    train_loader, val_loader = load_mnist(args)

    model = BlaschkeNetwork1d(layers=args.layers, signal_dim=28*28*1, patch_size=14)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,
                                              warmup_start_lr=args.lr * 1e-3,
                                              warmup_epochs=min(10, args.num_epoch//5),
                                              max_epochs=args.num_epoch)

    log('Training begins.', filepath=log_dir)
    log(f'Number of total parameters: {count_parameters(model)}', filepath=log_dir)
    log(f'Number of trainable parameters: {count_parameters(model, trainable_only=True)}', filepath=log_dir)

    loss_fn_pred = nn.CrossEntropyLoss()
    train_acc_list, val_acc_list = [], []
    train_loss_recon_list, train_loss_pred_list, val_loss_recon_list, val_loss_pred_list = [], [], [], []

    best_acc = 0
    with tqdm(range(args.num_epoch)) as pbar:
        for epoch in pbar:
            # Train
            model.train()
            train_loss, train_loss_recon, train_loss_pred, train_acc = 0, 0, 0, 0
            for i, (images, y_true) in enumerate(train_loader):
                optimizer.zero_grad()
                images = images.view(images.shape[0], -1)
                images = images.to(device)
                y_pred, residual_signals_sqsum = model(images)
                loss_recon = residual_signals_sqsum.mean()
                loss_pred = loss_fn_pred(y_pred, y_true.to(device))
                loss = loss_recon * args.loss_recon_coeff + loss_pred
                loss.backward()
                optimizer.step()
                accuracy = (y_pred.argmax(dim=1) == y_true.to(device)).float().mean()

                train_loss_recon += loss_recon.item()
                train_loss_pred += loss_pred.item()
                train_loss += loss.item()
                train_acc += (
                    (y_pred.argmax(dim=1) == y_true.to(device)).float().mean().item()
                )
            train_loss_recon /= len(train_loader)
            train_loss_pred /= len(train_loader)
            train_loss /= len(train_loader)
            train_acc /= len(train_loader)
            train_acc_list.append(train_acc)
            train_loss_recon_list.append(train_loss_recon)
            train_loss_pred_list.append(train_loss_pred)

            # Validation
            model.eval()
            val_loss, val_loss_recon, val_loss_pred, val_acc = 0, 0, 0, 0
            with torch.no_grad():
                for i, (images, y_true) in enumerate(val_loader):
                    images = images.view(images.shape[0], -1)
                    images = images.to(device)
                    y_pred, residual_signals_sqsum = model(images)
                    loss_recon = residual_signals_sqsum.mean()
                    loss_pred = loss_fn_pred(y_pred, y_true.to(device))
                    loss = loss_recon * args.loss_recon_coeff + loss_pred

                    val_loss_recon += loss_recon.item()
                    val_loss_pred += loss_pred.item()
                    val_loss += loss.item()
                    val_acc += (
                        (y_pred.argmax(dim=1) == y_true.to(device)).float().mean().item()
                    )
            val_loss_recon /= len(val_loader)
            val_loss_pred /= len(val_loader)
            val_loss /= len(val_loader)
            val_acc /= len(val_loader)
            val_acc_list.append(val_acc)
            val_loss_recon_list.append(val_loss_recon)
            val_loss_pred_list.append(val_loss_pred)

            # Update learning rate.
            scheduler.step()

            # Update progress bar.
            pbar.set_postfix(tr_recon=f'{train_loss_recon:.5f}', tr_pred=f'{train_loss_pred:.3f}',
                             val_recon=f'{val_loss_recon:.5f}', val_pred=f'{val_loss_pred:.3f}',
                             train_acc=f'{train_acc:.5f}', val_acc=f'{val_acc:.5f}', lr=optimizer.param_groups[0]['lr'])
            log_string = f'Epoch [{epoch}/{args.num_epoch}]. Train recon loss = {train_loss_recon:.5f}, pred loss = {train_loss_pred:.3f}, acc = {train_acc:.5f}.'
            log_string += f'\nValidation recon loss = {val_loss_recon:.5f}, pred loss = {val_loss_pred:.3f}, acc = {val_acc:.5f}.'
            log(log_string, filepath=log_dir, to_console=False)

            # Save best model.
            if val_acc > best_acc:
                best_acc = val_acc
                best_model = model.state_dict()
                torch.save(best_model, model_save_path)
                log('Model weights successfully saved for best validation accuracy.', filepath=log_dir, to_console=False)

            # Save stats.
            results_stats_save_path = os.path.join(results_dir, 'results_stats.npz')

            np.savez(results_stats_save_path,
                    train_acc_list=np.array(train_acc_list).astype(np.float16),
                    val_acc_list=np.array(val_acc_list).astype(np.float16),
                    train_loss_pred_list=np.array(train_loss_pred_list).astype(np.float16),
                    train_loss_recon_list=np.array(train_loss_recon_list).astype(np.float16),
                    val_loss_pred_list=np.array(val_loss_pred_list).astype(np.float16),
                    val_loss_recon_list=np.array(val_loss_recon_list).astype(np.float16),
            )

