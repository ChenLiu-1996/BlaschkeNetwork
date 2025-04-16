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
from analytical.analytical_decomposition import blaschke_decomposition


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


def train_epoch(train_loader, model, optimizer, loss_fn_pred, num_classes, device):
    train_loss_pred, train_loss_recon, train_loss_indicator, train_acc, train_auroc = 0, 0, 0, 0, 0
    y_true_arr, y_pred_arr = None, None

    for (x, y_true) in train_loader:
        optimizer.zero_grad()
        x = x.to(device)
        y_pred, residual_sqnorm, weighting_coeffs = model(x)

        loss_recon = residual_sqnorm.mean()
        loss_indicator = (weighting_coeffs * (1 - weighting_coeffs)).mean()
        loss_pred = loss_fn_pred(y_pred, y_true.to(device))

        # if args.direct_supervision:
        #     signal = x.detach().cpu().numpy()
        #     true_scale, true_blaschke_factor, _ = blaschke_decomposition(signal=signal,  num_blaschke_iters=args.layers, fourier_poly_order=signal.shape[-1], oversampling_rate=2, lowpass_order=1, carrier_freq=0)
        #     assert np.all(np.diff(true_scale, axis=-1) == 0)
        #     true_scale = true_scale[:, :, :, 0]

        loss = loss_pred + loss_recon * args.loss_recon_coeff + loss_indicator * args.loss_indicator_coeff
        loss.backward()
        optimizer.step()

        train_loss_pred += loss_pred.item()
        train_loss_recon += loss_recon.item()
        train_loss_indicator += loss_indicator.item()

        if y_true_arr is None:
            y_true_arr = y_true.detach().cpu().numpy()
            y_pred_arr = y_pred.detach().cpu().numpy()
        else:
            y_true_arr = np.vstack((y_true_arr, y_true.detach().cpu().numpy()))
            y_pred_arr = np.vstack((y_pred_arr, y_pred.detach().cpu().numpy()))

    train_loss_pred /= len(train_loader)
    train_loss_recon /= len(train_loader)
    train_loss_indicator /= len(train_loader)

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

    return train_loss_pred, train_loss_recon, train_loss_indicator, train_acc, train_auroc

@torch.no_grad()
def infer(loader, model, loss_fn_pred, num_classes, device):
    avg_loss_pred, avg_loss_recon, avg_loss_indicator, acc, auroc = 0, 0, 0, 0, 0
    y_true_arr, y_pred_arr = None, None
    residual_by_iter = 0

    for x, y_true in loader:
        x = x.to(device)
        y_pred, residual_sqnorm, weighting_coeffs = model(x)

        loss_pred = loss_fn_pred(y_pred, y_true.to(device))
        loss_recon = residual_sqnorm.mean()
        loss_indicator = (weighting_coeffs * (1 - weighting_coeffs)).mean()

        avg_loss_pred += loss_pred.item()
        avg_loss_recon += loss_recon.item()
        avg_loss_indicator += loss_indicator.item()
        residual_by_iter += residual_sqnorm.detach().cpu().numpy()

        if y_true_arr is None:
            y_true_arr = y_true.detach().cpu().numpy()
            y_pred_arr = y_pred.detach().cpu().numpy()
        else:
            y_true_arr = np.vstack((y_true_arr, y_true.detach().cpu().numpy()))
            y_pred_arr = np.vstack((y_pred_arr, y_pred.detach().cpu().numpy()))

    avg_loss_recon /= len(loader)
    avg_loss_pred /= len(loader)
    avg_loss_indicator /= len(loader)
    residual_by_iter /= len(loader)

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
    return avg_loss_pred, avg_loss_recon, avg_loss_indicator, residual_by_iter, acc, auroc

def main(args):
    # Log the config.
    config_str = 'Config: \n'
    args_dict = args.__dict__
    for key in args_dict.keys():
        config_str += '%s: %s\n' % (key, args_dict[key])
    config_str += '\nTraining History:'
    log(config_str, filepath=args.log_path, to_console=False)

    seed_everything(args.random_seed)
    train_loader, val_loader, test_loader, num_classes = load_ptbxl(args)

    model = BlaschkeNetwork1d(
        signal_len=5000,
        num_channels=12,
        layers=args.layers,
        num_roots=args.roots,
        detach_by_iter=args.detach_by_iter,
        patch_size=args.patch_size,
        out_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    log(f'Number of total parameters: {count_parameters(model)}', filepath=args.log_path)
    log(f'Number of trainable parameters: {count_parameters(model, trainable_only=True)}', filepath=args.log_path)

    loss_fn_pred = nn.BCEWithLogitsLoss()

    if not os.path.isfile(args.model_save_path):
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,
                                                warmup_start_lr=args.lr * 1e-3,
                                                warmup_epochs=min(20, args.epoch//5),
                                                max_epochs=args.epoch)

        log('Training begins.', filepath=args.log_path)
        best_auroc = 0
        train_acc_list, train_auroc_list, val_acc_list, val_auroc_list = [], [], [], []
        train_loss_pred_list, train_loss_recon_list, train_loss_indicator_list = [], [], []
        val_loss_pred_list, val_loss_recon_list, val_loss_indicator_list = [], [], []
        with tqdm(range(args.epoch)) as pbar:
            for epoch in pbar:
                # Training.
                model.train()
                train_loss_pred, train_loss_recon, train_loss_indicator, train_acc, train_auroc = \
                    train_epoch(train_loader=train_loader, model=model, optimizer=optimizer, loss_fn_pred=loss_fn_pred, num_classes=num_classes, device=device)
                train_loss_pred_list.append(train_loss_pred)
                train_loss_recon_list.append(train_loss_recon)
                train_loss_indicator_list.append(train_loss_indicator)
                train_acc_list.append(train_acc)
                train_auroc_list.append(train_auroc)

                # Validation.
                model.eval()
                val_loss_pred, val_loss_recon, val_loss_indicator, val_residual_by_iter, val_acc, val_auroc = \
                    infer(loader=val_loader, model=model, loss_fn_pred=loss_fn_pred, num_classes=num_classes, device=device)
                val_loss_pred_list.append(val_loss_pred)
                val_loss_recon_list.append(val_loss_recon)
                val_loss_indicator_list.append(val_loss_indicator)
                val_acc_list.append(val_acc)
                val_auroc_list.append(val_auroc)

                # Update learning rate.
                scheduler.step()

                # Update progress bar.
                pbar.set_postfix(
                    tr_pred=f'{train_loss_pred:.3f}', tr_recon=f'{train_loss_recon:.5f}', tr_idc=f'{train_loss_indicator:.5f}',
                    val_pred=f'{val_loss_pred:.3f}', val_recon=f'{val_loss_recon:.5f}', val_idc=f'{val_loss_indicator:.5f}',
                    tr_acc=f'{100 * train_acc:.2f}', val_acc=f'{100 * val_acc:.2f}',
                    tr_auroc=f'{100 * train_auroc:.2f}', val_auroc=f'{100 * val_auroc:.2f}',
                    lr=optimizer.param_groups[0]['lr'])
                log_string = f'Epoch [{epoch + 1}/{args.epoch}]. Train pred loss = {train_loss_pred:.3f}, recon loss = {train_loss_recon:.5f}, indicator loss = {train_loss_indicator:.5f}, acc = {100 * train_acc:.3f}, auroc = {100 * train_auroc:.3f}.'
                log_string += f'\nValidation pred loss = {val_loss_pred:.3f}, recon loss = {val_loss_recon:.5f}, indicator loss = {val_loss_indicator:.5f}, acc = {100 * val_acc:.3f}, auroc = {100 * val_auroc:.3f}, val_residual_by_iter = {val_residual_by_iter}.'
                log(log_string, filepath=args.log_path, to_console=False)

                # Save best model.
                if val_auroc > best_auroc:
                    best_auroc = val_auroc
                    torch.save(model.state_dict(), args.model_save_path)
                    log(f'Model weights with the best validation AUROC is saved to {args.model_save_path}.', filepath=args.log_path, to_console=False)

                # Save stats.
                np.savez(
                    args.results_stats_save_path,
                    train_acc_list=100*np.array(train_acc_list).astype(np.float16),
                    val_acc_list=100*np.array(val_acc_list).astype(np.float16),
                    train_auroc_list=100*np.array(train_auroc_list).astype(np.float16),
                    val_auroc_list=100*np.array(val_auroc_list).astype(np.float16),
                    train_loss_pred_list=np.array(train_loss_pred_list).astype(np.float16),
                    train_loss_recon_list=np.array(train_loss_recon_list).astype(np.float16),
                    train_loss_indicator_list=np.array(train_loss_indicator_list).astype(np.float16),
                    val_loss_pred_list=np.array(val_loss_pred_list).astype(np.float16),
                    val_loss_recon_list=np.array(val_loss_recon_list).astype(np.float16),
                    val_loss_indicator_list=np.array(val_loss_indicator_list).astype(np.float16),
                    val_residual_by_iter=np.array(val_residual_by_iter).astype(np.float16),
                )

    # Testing.
    model.load_state_dict(torch.load(args.model_save_path, map_location=device))
    log(f'Model weights is loaded from {args.model_save_path}.', filepath=args.log_path, to_console=False)

    log('Testing begins.', filepath=args.log_path)
    model.eval()
    test_loss_pred, test_loss_recon, test_loss_indicator, test_residual_by_iter, test_acc, test_auroc = \
        infer(loader=test_loader, model=model, loss_fn_pred=loss_fn_pred, num_classes=num_classes, device=device)

    log_string = f'\nTest pred loss = {test_loss_pred:.3f}, recon loss = {test_loss_recon:.5f}, indicator loss = {test_loss_indicator:.5f}, acc = {100 * test_acc:.3f}, auroc = {100 * test_auroc:.3f}, test_residual_by_iter = {test_residual_by_iter}.'
    log(log_string, filepath=args.log_path, to_console=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--roots', type=int, default=16)
    parser.add_argument('--detach-by-iter', action='store_true')                  # Independently optimize Blaschke decomposition per iteration.
    parser.add_argument('--direct-supervision', action='store_true')              # Use the analytical Blaschke coeffs to supervise training.
    parser.add_argument('--lr', help='Learning rate.', type=float, default=1e-2)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epoch', type=int, default=40)
    parser.add_argument('--loss-recon-coeff', type=float, default=1.0)
    parser.add_argument('--loss-indicator-coeff', type=float, default=1e-2)       # Blaschke root selectors should be (almost) either 0 or 1.
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--random-seed', type=int, default=1)
    parser.add_argument('--subset', type=str, default='super_class')
    parser.add_argument('--patch-size', type=int, default=50)
    parser.add_argument('--training-percentage', type=int, default=100)
    parser.add_argument('--data-dir', type=str, default='$ROOT_DIR/data/')
    args = SimpleNamespace(**vars(parser.parse_args()))

    ROOT_DIR = '/'.join(os.path.realpath(__file__).split('/')[:-2])
    args.data_dir = args.data_dir.replace('$ROOT_DIR', ROOT_DIR)

    curr_run_identifier = f'ECG_PTBXL/subset={args.subset}-{args.training_percentage}%_BN1d-L{args.layers}_R{args.roots}_DS-{args.direct_supervision}_detach-{args.detach_by_iter}_patch-{args.patch_size}_reconCoeff-{args.loss_recon_coeff}_indicatorCoeff-{args.loss_indicator_coeff}_lr-{args.lr}_epoch-{args.epoch}_seed-{args.random_seed}'
    args.results_dir = os.path.join(ROOT_DIR, 'results', curr_run_identifier)
    args.log_path = os.path.join(args.results_dir, 'log.txt')
    args.model_save_path = os.path.join(args.results_dir, 'model.pty')
    args.results_stats_save_path = os.path.join(args.results_dir, 'results_stats.npz')

    os.makedirs(args.results_dir, exist_ok=True)

    main(args)