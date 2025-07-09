from types import SimpleNamespace
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve
from einops import rearrange
from matplotlib import pyplot as plt

from models.BN1d import BlaschkeNetwork1d
from nn_utils.scheduler import LinearWarmupCosineAnnealingLR
from nn_utils.seed import seed_everything
from nn_utils.log import log, count_parameters
from dataset.ecg_datasets import get_ecg_dataset
from analytical.analytical_decomposition import blaschke_decomposition, display_blaschke_product


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


def binary_entropy(batched_coeffs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    '''
    Binary (self-)entropy.
    Assuming range of input to be [0, 1].
    '''
    entropy = - batched_coeffs * torch.log(batched_coeffs + eps) + \
              - (1 - batched_coeffs) * torch.log((1 - batched_coeffs) + eps)
    return entropy


@torch.no_grad()
def plot_signal_approx(signal_complex, s_arr, B_prod_arr, mode: str, epoch_idx: int, batch_idx: int):
    signal_complex = signal_complex.detach().cpu().numpy()
    s_arr = s_arr.detach().numpy()
    B_prod_arr = B_prod_arr.detach().numpy()
    assert len(signal_complex.shape) == 3
    assert len(s_arr.shape) == len(B_prod_arr.shape) == 4
    assert signal_complex.shape[0] == 1
    assert signal_complex.shape[1] == B_prod_arr.shape[1] == 1
    assert signal_complex.shape[2] == B_prod_arr.shape[2]
    blaschke_order = B_prod_arr.shape[-1]

    signal_complex = rearrange(signal_complex, 'b c l -> (b c) l').squeeze(0)
    s_arr = rearrange(s_arr, 'b c 1 i -> (b c 1) i 1').squeeze(0)
    B_prod_arr = rearrange(B_prod_arr, 'b c l i -> (b c) i l').squeeze(0)
    time_arr = np.arange(signal_complex.shape[-1])

    fig, ax = plt.subplots(blaschke_order, blaschke_order + 2, figsize = (4 * blaschke_order + 8, 4 * blaschke_order))
    if blaschke_order == 1:
        ax = ax[np.newaxis, :]
    for total_order in range(blaschke_order):
        ax[total_order, 0].plot(time_arr, signal_complex.real, label = 'original signal', color='firebrick', alpha=0.8)
        ax[total_order, 0].legend(loc='lower left')
        ax[total_order, 0].spines['top'].set_visible(False)
        ax[total_order, 0].spines['right'].set_visible(False)

    for total_order in range(1, blaschke_order + 1):
        for curr_order in range(1, total_order + 1):
            ax[total_order - 1, curr_order].hlines(np.abs(s_arr[curr_order-1]), xmin=time_arr.min(), xmax=time_arr.max(), label = f'$s_{curr_order}$', color='darkblue', linestyle='--')
            ax[total_order - 1, curr_order].plot(time_arr, (B_prod_arr[curr_order-1] * s_arr[curr_order-1]).real, label = f'$s_{curr_order}$ * ${display_blaschke_product(curr_order)}$', color='darkgreen', alpha=0.6)
            ax[total_order - 1, curr_order].legend(loc='lower left')
            ax[total_order - 1, curr_order].spines['top'].set_visible(False)
            ax[total_order - 1, curr_order].spines['right'].set_visible(False)

    final = 0
    for curr_order in range(1, blaschke_order + 1):
        final += (B_prod_arr[curr_order-1] * s_arr[curr_order-1]).real
        ax[curr_order - 1, blaschke_order + 1].plot(time_arr, signal_complex.real, label = 'original signal', color='firebrick', alpha=0.8)
        ax[curr_order - 1, blaschke_order + 1].plot(time_arr, final, label = 'reconstruction', color='skyblue', alpha=0.9)
        ax[curr_order - 1, blaschke_order + 1].plot(time_arr, final - signal_complex.real, label = 'residual', color='gray', alpha=1.0)
        ax[curr_order - 1, blaschke_order + 1].set_title(f'Reconstruction Error: {np.power(np.abs(final - signal_complex.real), 2).mean():.5f}')
        ax[curr_order - 1, blaschke_order + 1].legend(loc='lower left')
        ax[curr_order - 1, blaschke_order + 1].spines['top'].set_visible(False)
        ax[curr_order - 1, blaschke_order + 1].spines['right'].set_visible(False)

    # Remove axes from unused subplots
    for i in range(blaschke_order):
        for j in range(blaschke_order + 2):
            if (j > i + 1) and (j != blaschke_order + 1):
                ax[i, j].axis('off')

    if mode != 'test':
        save_path = os.path.join(args.plot_folder, mode,
                                 f'epoch_{str(epoch_idx + 1).zfill(3)}_batch_{str(batch_idx + 1).zfill(5)}.png')
    else:
        save_path = os.path.join(args.plot_folder, 'test.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout(pad=2)
    fig.savefig(save_path)
    plt.close(fig)
    return


def train_epoch(train_loader, model, optimizer, loss_fn_pred, num_classes, device, epoch_idx):
    train_loss_pred, train_loss_recon, train_loss_orth, train_loss_smoothness, train_loss_direct, train_acc, train_auroc = 0, 0, 0, 0, 0, 0, 0
    y_true_arr, y_pred_arr = None, None
    plot_freq = max(len(train_loader) // args.n_plot_per_epoch, 1)

    for batch_idx, (x, y_true) in enumerate(train_loader):
        should_plot = batch_idx % plot_freq == 0

        optimizer.zero_grad()
        x = x.to(device)
        y_pred, residual_sqnorm_by_iter, pred_scale, pred_blaschke_factor, orth_loss, smoothness_loss = model(x)

        loss_pred = loss_fn_pred(y_pred, y_true.to(device))
        if args.only_final_iter:
            loss_recon = residual_sqnorm_by_iter[-1].mean() * args.loss_recon_coeff
        else:
            loss_recon = residual_sqnorm_by_iter.mean() * args.loss_recon_coeff
        loss_orth = orth_loss * args.loss_orth_coeff
        loss_smoothness = smoothness_loss * args.loss_smoothness_coeff

        loss_direct = torch.zeros(1).to(device)
        if args.loss_direct_coeff > 0:
            signal = x.detach().cpu().numpy()
            true_scale, true_blaschke_factor, _, _ = blaschke_decomposition(
                signal=signal,
                num_blaschke_iters=args.layers,
                fourier_poly_order=signal.shape[-1],
                oversampling_rate=2,
                lowpass_order=1,
                carrier_freq=0)
            assert np.all(np.diff(true_scale, axis=-1) == 0)
            true_scale = true_scale[:, :, :, 0]
            true_scale = rearrange(true_scale, 'i b c -> b c i')
            true_blaschke_factor = rearrange(true_blaschke_factor, 'i b c l -> b c l i')
            assert len(pred_blaschke_factor.shape) == 5
            assert pred_blaschke_factor.shape[3] == 2
            pred_blaschke_factor = pred_blaschke_factor[:, :, :, 0, :] + 1j * pred_blaschke_factor[:, :, :, 1, :]
            loss_direct_scale = (torch.from_numpy(true_scale).to(device) - pred_scale).abs().pow(2).mean()
            loss_direct_b_factor = (torch.from_numpy(true_blaschke_factor).to(device) - pred_blaschke_factor).abs().pow(2).mean()
            loss_direct = (loss_direct_scale + loss_direct_b_factor) * args.loss_direct_coeff

        loss = loss_pred + loss_recon + loss_orth + train_loss_smoothness + loss_direct
        loss.backward()
        optimizer.step()

        train_loss_pred += loss_pred.item()
        train_loss_recon += loss_recon.item()
        train_loss_orth += loss_orth.item()
        train_loss_smoothness += loss_smoothness.item()
        train_loss_direct += loss_direct.item()

        if y_true_arr is None:
            y_true_arr = y_true.detach().cpu().numpy()
            y_pred_arr = y_pred.detach().cpu().numpy()
        else:
            y_true_arr = np.vstack((y_true_arr, y_true.detach().cpu().numpy()))
            y_pred_arr = np.vstack((y_pred_arr, y_pred.detach().cpu().numpy()))

        if should_plot:
            with torch.no_grad():
                signal_complex, s_arr, B_prod_arr = model.test_approximate(x[:1, :1, :])
            plot_signal_approx(signal_complex, s_arr, B_prod_arr, mode='train', epoch_idx=epoch_idx, batch_idx=batch_idx)

    train_loss_pred /= len(train_loader)
    train_loss_recon /= len(train_loader)
    train_loss_orth /= len(train_loader)
    train_loss_smoothness /= len(train_loader)
    train_loss_direct /= len(train_loader)

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

    return train_loss_pred, train_loss_recon, train_loss_orth, train_loss_smoothness, train_loss_direct, train_acc, train_auroc

@torch.no_grad()
def infer(loader, model, loss_fn_pred, num_classes, device, epoch_idx):
    avg_loss_pred, avg_loss_recon, acc, auroc = 0, 0, 0, 0
    y_true_arr, y_pred_arr = None, None
    residual_by_iter, mean_scale_by_iter = 0, 0
    plot_freq = max(len(loader) // args.n_plot_per_epoch, 1)

    for batch_idx, (x, y_true) in enumerate(loader):
        should_plot = batch_idx % plot_freq == 0
        x = x.to(device)
        y_pred, residual_sqnorm_by_iter, scale_by_iter, _, _, _ = model(x)

        loss_pred = loss_fn_pred(y_pred, y_true.to(device))
        if args.only_final_iter:
            loss_recon = residual_sqnorm_by_iter[-1].mean() * args.loss_recon_coeff
        else:
            loss_recon = residual_sqnorm_by_iter.mean() * args.loss_recon_coeff

        avg_loss_pred += loss_pred.item()
        avg_loss_recon += loss_recon.item() * args.loss_recon_coeff
        residual_by_iter += residual_sqnorm_by_iter.detach().cpu().numpy()
        mean_scale_by_iter += scale_by_iter.mean(dim=(0,1)).abs().detach().cpu().numpy()

        if y_true_arr is None:
            y_true_arr = y_true.detach().cpu().numpy()
            y_pred_arr = y_pred.detach().cpu().numpy()
        else:
            y_true_arr = np.vstack((y_true_arr, y_true.detach().cpu().numpy()))
            y_pred_arr = np.vstack((y_pred_arr, y_pred.detach().cpu().numpy()))

        if should_plot:
            with torch.no_grad():
                signal_complex, s_arr, B_prod_arr = model.test_approximate(x[:1, :1, :])
            if epoch_idx is None:
                mode = 'test'
            else:
                mode = 'val'
            plot_signal_approx(signal_complex, s_arr, B_prod_arr, mode=mode, epoch_idx=epoch_idx, batch_idx=batch_idx)

    avg_loss_pred /= len(loader)
    avg_loss_recon /= len(loader)
    residual_by_iter /= len(loader)
    mean_scale_by_iter /= len(loader)

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
    return avg_loss_pred, avg_loss_recon, acc, auroc, residual_by_iter, mean_scale_by_iter


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
        detach_by_iter=args.detach_by_iter,
        patch_size=args.patch_size,
        out_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    log(f'Number of total parameters: {count_parameters(model)}', filepath=args.log_path)
    log(f'Number of trainable parameters: {count_parameters(model, trainable_only=True)}', filepath=args.log_path)

    loss_fn_pred = nn.BCEWithLogitsLoss()

    if not os.path.isfile(args.model_save_path):
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_start_lr=args.lr * 1e-3,
            warmup_epochs=min(20, args.epoch//5),
            max_epochs=args.epoch)

        log('Training begins.', filepath=args.log_path)
        best_auroc = 0
        train_acc_list, train_auroc_list, val_acc_list, val_auroc_list = [], [], [], []
        train_loss_pred_list, train_loss_recon_list = [], []
        val_loss_pred_list, val_loss_recon_list = [], []
        with tqdm(range(args.epoch)) as pbar:
            for epoch_idx, epoch in enumerate(pbar):
                # Training.
                model.train()
                train_loss_pred, train_loss_recon, train_loss_orth, train_loss_smoothness, train_loss_direct, train_acc, train_auroc = \
                    train_epoch(train_loader=train_loader, model=model, optimizer=optimizer, loss_fn_pred=loss_fn_pred, num_classes=num_classes, device=device, epoch_idx=epoch_idx)
                train_loss_pred_list.append(train_loss_pred)
                train_loss_recon_list.append(train_loss_recon)
                train_acc_list.append(train_acc)
                train_auroc_list.append(train_auroc)

                # Validation.
                model.eval()
                val_loss_pred, val_loss_recon, val_acc, val_auroc, val_residual_by_iter, val_mean_scale_by_iter = \
                    infer(loader=val_loader, model=model, loss_fn_pred=loss_fn_pred, num_classes=num_classes, device=device, epoch_idx=epoch_idx)
                val_loss_pred_list.append(val_loss_pred)
                val_loss_recon_list.append(val_loss_recon)
                val_acc_list.append(val_acc)
                val_auroc_list.append(val_auroc)

                # Update learning rate.
                scheduler.step()

                # Update progress bar.
                pbar.set_postfix(
                    tr_pred=f'{train_loss_pred:.3f}', tr_recon=f'{train_loss_recon:.5f}',
                    val_pred=f'{val_loss_pred:.3f}', val_recon=f'{val_loss_recon:.5f}',
                    tr_acc=f'{100 * train_acc:.2f}', val_acc=f'{100 * val_acc:.2f}',
                    tr_auroc=f'{100 * train_auroc:.2f}', val_auroc=f'{100 * val_auroc:.2f}',
                    lr=optimizer.param_groups[0]['lr'])
                log_string = f'Epoch [{epoch + 1}/{args.epoch}]. Train pred loss = {train_loss_pred:.3f}, recon loss = {train_loss_recon:.5f}, orthogonality loss = {train_loss_orth:.5f}, smoothness loss = {train_loss_smoothness:.5f}, direct loss = {train_loss_direct:.5f}, acc = {100 * train_acc:.3f}, auroc = {100 * train_auroc:.3f}.'
                log_string += f'\nValidation pred loss = {val_loss_pred:.3f}, recon loss = {val_loss_recon:.5f}, acc = {100 * val_acc:.3f}, auroc = {100 * val_auroc:.3f}, val_residual_by_iter = {val_residual_by_iter}, mean scale = {val_mean_scale_by_iter}.'
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
                    val_loss_pred_list=np.array(val_loss_pred_list).astype(np.float16),
                    val_loss_recon_list=np.array(val_loss_recon_list).astype(np.float16),
                    val_residual_by_iter=np.array(val_residual_by_iter).astype(np.float16),
                    val_mean_scale_by_iter=np.array(val_mean_scale_by_iter).astype(np.float16),
                )

    # Testing.
    model.load_state_dict(torch.load(args.model_save_path, map_location=device))
    log(f'Model weights is loaded from {args.model_save_path}.', filepath=args.log_path, to_console=False)

    log('Testing begins.', filepath=args.log_path)
    model.eval()
    test_loss_pred, test_loss_recon, test_acc, test_auroc, test_residual_by_iter, test_mean_scale_by_iter = \
        infer(loader=test_loader, model=model, loss_fn_pred=loss_fn_pred, num_classes=num_classes, device=device, epoch_idx=None)

    log_string = f'\nTest pred loss = {test_loss_pred:.3f}, recon loss = {test_loss_recon:.5f}, acc = {100 * test_acc:.3f}, auroc = {100 * test_auroc:.3f}, test_residual_by_iter = {test_residual_by_iter}, mean scale = {test_mean_scale_by_iter}.'
    log(log_string, filepath=args.log_path, to_console=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=str, default='super_class')
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--detach-by-iter', action='store_true')                  # Independently optimize Blaschke decomposition per iteration.
    parser.add_argument('--only-final-iter', action='store_true')                 # Only penalize Blaschke decomposition in the final iteration.
    parser.add_argument('--lr', help='Learning rate.', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--n-plot-per-epoch', type=int, default=1)
    parser.add_argument('--loss-recon-coeff', type=float, default=1)
    parser.add_argument('--loss-orth-coeff', type=float, default=0)
    parser.add_argument('--loss-smoothness-coeff', type=float, default=0)
    parser.add_argument('--loss-direct-coeff', type=float, default=0)             # Use the analytical Blaschke coeffs to supervise training.
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--random-seed', type=int, default=1)
    parser.add_argument('--patch-size', type=int, default=1)
    parser.add_argument('--training-percentage', type=int, default=100)
    parser.add_argument('--data-dir', type=str, default='$ROOT_DIR/data/')
    args = SimpleNamespace(**vars(parser.parse_args()))

    ROOT_DIR = '/'.join(os.path.realpath(__file__).split('/')[:-2])
    args.data_dir = args.data_dir.replace('$ROOT_DIR', ROOT_DIR)

    curr_run_identifier = f'ECG_PTBXL/subset={args.subset}--{args.training_percentage}%_BN1d-L{args.layers}_detach-{args.detach_by_iter}_final-{args.only_final_iter}_patch-{args.patch_size}_reconCoeff-{args.loss_recon_coeff}_orthCoeff-{args.loss_orth_coeff}_smoothnessCoeff-{args.loss_orth_coeff}_directCoeff-{args.loss_direct_coeff}_lr-{args.lr}_epoch-{args.epoch}_seed-{args.random_seed}'
    args.results_dir = os.path.join(ROOT_DIR, 'results', curr_run_identifier)
    args.log_path = os.path.join(args.results_dir, 'log.txt')
    args.model_save_path = os.path.join(args.results_dir, 'model.pty')
    args.results_stats_save_path = os.path.join(args.results_dir, 'results_stats.npz')
    args.plot_folder = os.path.join(ROOT_DIR, 'results', curr_run_identifier, 'figures')

    os.makedirs(args.results_dir, exist_ok=True)

    main(args)