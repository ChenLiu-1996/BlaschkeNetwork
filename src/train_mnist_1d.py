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


def load_mnist(args):
    # Load MNIST
    transform_train = transforms.Compose([
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
                        help='Example: `32 16 10`',
                        nargs='+',
                        type=int)
    parser.add_argument('--proj-learnable',
                        action='store_true')
    parser.add_argument('--lr',
                        help='Learning rate.',
                        type=float,
                        default=3e-3)
    parser.add_argument('--batch-size',
                        type=int,
                        default=256)
    parser.add_argument('--num-epoch',
                        type=int,
                        default=100)
    parser.add_argument('--num-workers',
                        type=int,
                        default=8)
    parser.add_argument('--random-seed',
                        type=int,
                        default=1)
    args = SimpleNamespace(**vars(parser.parse_args()))

    model_save_path = f'../checkpoints/mnist/BN1d_{args.layers}_proj-learnable-{args.proj_learnable}_lr_{args.lr}_epoch_{args.num_epoch}-seed_{args.random_seed}/model_best_val_acc.ckpt'
    results_dir = f'../results/mnist/BN1d_{args.layers}_proj-learnable-{args.proj_learnable}-lr_{args.lr}_epoch_{args.num_epoch}-seed_{args.random_seed}/'
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    seed_everything(args.random_seed)
    train_loader, val_loader = load_mnist(args)

    image_channels = 1
    model = BlaschkeNetwork1d(layers_hidden=[image_channels, *args.layers])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,
                                              warmup_start_lr=args.lr * 1e-3,
                                              warmup_epochs=min(10, args.num_epoch//5),
                                              max_epochs=args.num_epoch)
    criterion = nn.CrossEntropyLoss()
    train_acc_list, val_acc_list = [], []

    best_acc = 0
    with tqdm(range(args.num_epoch)) as pbar:
        for epoch in pbar:
            # Train
            model.train()
            train_loss, train_acc = 0, 0
            for i, (images, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                images = images.view(images.shape[0], image_channels, -1)
                images = images.to(device)
                output = model(images)
                output = torch.real(output)
                loss = criterion(output, labels.to(device))
                loss.backward()
                optimizer.step()
                accuracy = (output.argmax(dim=1) == labels.to(device)).float().mean()

                train_loss += criterion(output, labels.to(device)).item()
                train_acc += (
                    (output.argmax(dim=1) == labels.to(device)).float().mean().item()
                )
            train_loss /= len(train_loader)
            train_acc /= len(train_loader)
            train_acc_list.append(train_acc)

            # Validation
            model.eval()
            val_loss, val_acc = 0, 0
            with torch.no_grad():
                for i, (images, labels) in enumerate(val_loader):
                    images = images.view(images.shape[0], image_channels, -1)
                    images = images.to(device)
                    output = model(images)
                    output = torch.real(output)
                    val_loss += criterion(output, labels.to(device)).item()
                    val_acc += (
                        (output.argmax(dim=1) == labels.to(device)).float().mean().item()
                    )
            val_loss /= len(val_loader)
            val_acc /= len(val_loader)
            val_acc_list.append(val_acc)

            # Save best model.
            if val_acc > best_acc:
                best_acc = val_acc
                best_model = model.state_dict()
                torch.save(best_model, model_save_path)

            # Update learning rate.
            scheduler.step()

            # Update progress bar.
            pbar.set_postfix(train_loss=f'{train_loss:.5f}', val_loss=f'{val_loss:.5f}',
                             train_acc=f'{train_acc:.5f}', val_acc=f'{val_acc:.5f}', lr=optimizer.param_groups[0]['lr'])

            # Save stats.
            results_stats_save_path = os.path.join(results_dir, 'results_stats.npz')

            np.savez(results_stats_save_path,
                    train_acc_list=np.array(train_acc_list).astype(np.float16),
                    val_acc_list=np.array(val_acc_list).astype(np.float16))

