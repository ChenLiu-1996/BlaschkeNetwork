from types import SimpleNamespace
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve

from models.BN1d import BlaschkeNetwork1d
from nn_utils.scheduler import LinearWarmupCosineAnnealingLR
from nn_utils.seed import seed_everything
from nn_utils.log import log, count_parameters
from dataset.ecg_datasets import get_ecg_dataset


def load_ptbxl(args):
    data_path = os.path.join(args.data_dir, 'PTBXL')

    data_split_dir = os.path.join(args.data_dir, 'data_split', 'ptbxl', args.subset)
    train_csv_path = os.path.join(data_split_dir, f'ptbxl_{args.subset}_train.csv')
    val_csv_path = os.path.join(data_split_dir, f'ptbxl_{args.subset}_val.csv')
    test_csv_path = os.path.join(data_split_dir, f'ptbxl_{args.subset}_test.csv')

    train_set = get_ecg_dataset(data_path, train_csv_path, mode='train', dataset_name='ptbxl', ratio=args.training_percentage)
    val_set = get_ecg_dataset(data_path, val_csv_path, mode='val', dataset_name='ptbxl')
    test_set = get_ecg_dataset(data_path, test_csv_path, mode='test', dataset_name='ptbxl')

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader, test_loader, train_set.num_classes


def train_epoch(train_loader, model, optimizer, loss_fn_pred, num_classes):
    train_loss_recon, train_loss_pred, train_acc, train_auroc = 0, 0, 0, 0
    y_true_arr, y_pred_arr = None, None

    for (x, y_true) in train_loader:
        optimizer.zero_grad()
        x = x.to(device)
        y_pred, residual_signals_sqsum = model(x)
        loss_recon = residual_signals_sqsum.mean()
        loss_pred = loss_fn_pred(y_pred, y_true.to(device))
        loss = loss_recon * args.loss_recon_coeff + loss_pred
        loss.backward()
        optimizer.step()

        train_loss_recon += loss_recon.item()
        train_loss_pred += loss_pred.item()

        if y_true_arr is None:
            y_true_arr = y_true.detach().cpu().numpy()
            y_pred_arr = y_pred.detach().cpu().numpy()
        else:
            y_true_arr = np.vstack((y_true_arr, y_true.detach().cpu().numpy()))
            y_pred_arr = np.vstack((y_pred_arr, y_pred.detach().cpu().numpy()))

    train_loss_recon /= len(train_loader)
    train_loss_pred /= len(train_loader)

    acc_by_class, auroc_by_class = [], []
    for class_idx in range(num_classes):
        precision, recall, thresholds = precision_recall_curve(y_true_arr[:, class_idx], y_pred_arr[:, class_idx])
        numerator = 2 * recall * precision
        denom = recall + precision
        f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom!=0))
        max_f1_thresh = thresholds[np.argmax(f1_scores)]

        acc_by_class.append(accuracy_score(y_true_arr[:, class_idx], (y_pred_arr[:, class_idx] > max_f1_thresh).astype(int)))
        auroc_by_class.append(roc_auc_score(y_true_arr[:, class_idx], y_pred_arr[:, class_idx], average='macro'))

    train_acc = np.mean(acc_by_class)
    train_auroc = np.mean(auroc_by_class)

    return train_loss_recon, train_loss_pred, train_acc, train_auroc

@torch.no_grad()
def infer(loader, model, loss_fn_pred, num_classes):
    avg_loss_recon, avg_loss_pred, acc, auroc = 0, 0, 0, 0
    y_true_arr, y_pred_arr = None, None

    for x, y_true in loader:
        x = x.to(device)
        y_pred, residual_signals_sqsum = model(x)
        loss_recon = residual_signals_sqsum.mean()
        loss_pred = loss_fn_pred(y_pred, y_true.to(device))

        avg_loss_recon += loss_recon.item()
        avg_loss_pred += loss_pred.item()

        if y_true_arr is None:
            y_true_arr = y_true.detach().cpu().numpy()
            y_pred_arr = y_pred.detach().cpu().numpy()
        else:
            y_true_arr = np.vstack((y_true_arr, y_true.detach().cpu().numpy()))
            y_pred_arr = np.vstack((y_pred_arr, y_pred.detach().cpu().numpy()))

    avg_loss_recon /= len(loader)
    avg_loss_pred /= len(loader)

    acc_by_class, auroc_by_class = [], []
    for class_idx in range(num_classes):
        precision, recall, thresholds = precision_recall_curve(y_true_arr[:, class_idx], y_pred_arr[:, class_idx])
        numerator = 2 * recall * precision
        denom = recall + precision
        f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom!=0))
        max_f1_thresh = thresholds[np.argmax(f1_scores)]

        acc_by_class.append(accuracy_score(y_true_arr[:, class_idx], (y_pred_arr[:, class_idx] > max_f1_thresh).astype(int)))
        auroc_by_class.append(roc_auc_score(y_true_arr[:, class_idx], y_pred_arr[:, class_idx], average='macro'))

    acc = np.mean(acc_by_class)
    auroc = np.mean(auroc_by_class)
    return avg_loss_recon, avg_loss_pred, acc, auroc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--lr', help='Learning rate.', type=float, default=1e-1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-epoch', type=int, default=100)
    parser.add_argument('--loss-recon-coeff', type=float, default=1.0)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--random-seed', type=int, default=1)
    parser.add_argument('--subset', type=str, default='form')
    parser.add_argument('--patch-size', type=int, default=50)
    parser.add_argument('--training-percentage', type=int, default=100)
    parser.add_argument('--data-dir', type=str, default='$ROOT_DIR/data/')
    args = SimpleNamespace(**vars(parser.parse_args()))

    ROOT_DIR = '/'.join(os.path.realpath(__file__).split('/')[:-2])
    args.data_dir = args.data_dir.replace('$ROOT_DIR', ROOT_DIR)

    model_save_path = f'../checkpoints/ECG_PTBXL/subset={args.subset}-percentage_{args.training_percentage}_BN1d_{args.layers}_lr_{args.lr}_epoch_{args.num_epoch}-seed_{args.random_seed}/model_best_val_auroc.ckpt'
    results_dir = f'../results/ECG_PTBXL/subset={args.subset}-percentage_{args.training_percentage}_BN1d_{args.layers}_lr_{args.lr}_epoch_{args.num_epoch}-seed_{args.random_seed}/'
    log_dir = os.path.join(results_dir, 'log.txt')

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    seed_everything(args.random_seed)
    train_loader, val_loader, test_loader, num_classes = load_ptbxl(args)

    model = BlaschkeNetwork1d(layers=args.layers, signal_len=5000, patch_size=args.patch_size, num_channels=12, out_classes=num_classes)
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

    loss_fn_pred = nn.BCEWithLogitsLoss()
    train_acc_list, train_auroc_list, val_acc_list, val_auroc_list = [], [], [], []
    train_loss_recon_list, train_loss_pred_list, val_loss_recon_list, val_loss_pred_list = [], [], [], []

    best_auroc = 0
    with tqdm(range(args.num_epoch)) as pbar:
        for epoch in pbar:
            # Train
            model.train()
            train_loss_recon, train_loss_pred, train_acc, train_auroc = \
                train_epoch(train_loader=train_loader, model=model, optimizer=optimizer, loss_fn_pred=loss_fn_pred, num_classes=num_classes)
            train_loss_recon_list.append(train_loss_recon)
            train_loss_pred_list.append(train_loss_pred)
            train_acc_list.append(train_acc)
            train_auroc_list.append(train_auroc)

            # Validation
            model.eval()
            val_loss_recon, val_loss_pred, val_acc, val_auroc = \
                infer(loader=val_loader, model=model, loss_fn_pred=loss_fn_pred, num_classes=num_classes)
            val_loss_recon_list.append(val_loss_recon)
            val_loss_pred_list.append(val_loss_pred)
            val_acc_list.append(val_acc)
            val_auroc_list.append(val_auroc)

            # Update learning rate.
            scheduler.step()

            # Update progress bar.
            pbar.set_postfix(tr_recon=f'{train_loss_recon:.5f}', tr_pred=f'{train_loss_pred:.3f}',
                             val_recon=f'{val_loss_recon:.5f}', val_pred=f'{val_loss_pred:.3f}',
                             tr_acc=f'{100 * train_acc:.2f}', val_acc=f'{100 * val_acc:.2f}',
                             tr_auroc=f'{100 * train_auroc:.2f}', val_auroc=f'{100 * val_auroc:.2f}',
                             lr=optimizer.param_groups[0]['lr'])
            log_string = f'Epoch [{epoch}/{args.num_epoch}]. Train recon loss = {train_loss_recon:.5f}, pred loss = {train_loss_pred:.3f}, acc = {100 * train_acc:.3f}, auroc = {100 * train_auroc:.3f}.'
            log_string += f'\nValidation recon loss = {val_loss_recon:.5f}, pred loss = {val_loss_pred:.3f}, acc = {100 * val_acc:.3f}, auroc = {100 * val_auroc:.3f}.'
            log(log_string, filepath=log_dir, to_console=False)

            # Save best model.
            if val_auroc > best_auroc:
                best_auroc = val_auroc
                best_model = model.state_dict()
                torch.save(best_model, model_save_path)
                log('Model weights successfully saved for best validation AUROC.', filepath=log_dir, to_console=False)

            # Save stats.
            results_stats_save_path = os.path.join(results_dir, 'results_stats.npz')

            np.savez(results_stats_save_path,
                     train_acc_list=100*np.array(train_acc_list).astype(np.float16),
                     val_acc_list=100*np.array(val_acc_list).astype(np.float16),
                     train_auroc_list=100*np.array(train_auroc_list).astype(np.float16),
                     val_auroc_list=100*np.array(val_auroc_list).astype(np.float16),
                     train_loss_pred_list=np.array(train_loss_pred_list).astype(np.float16),
                     train_loss_recon_list=np.array(train_loss_recon_list).astype(np.float16),
                     val_loss_pred_list=np.array(val_loss_pred_list).astype(np.float16),
                     val_loss_recon_list=np.array(val_loss_recon_list).astype(np.float16),
            )

    # Testing
    model.eval()
    test_loss_recon, test_loss_pred, test_acc, test_auroc = \
        infer(loader=test_loader, model=model, loss_fn_pred=loss_fn_pred, num_classes=num_classes)

    log_string = f'\nTest recon loss = {test_loss_recon:.5f}, pred loss = {test_loss_pred:.3f}, acc = {100 * test_acc:.3f}, auroc = {100 * test_auroc:.3f}.'
    log(log_string, filepath=log_dir, to_console=False)
