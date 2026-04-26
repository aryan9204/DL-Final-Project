#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_log_rows(log_path):
    rows = []
    with open(log_path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith('INFO:'):
                _, _, payload = line.partition('{')
                line = '{' + payload if payload else ''
            if not line.startswith('{'):
                continue
            rows.append(json.loads(line))
    return rows


def extract_series(rows, row_type, key):
    xs = []
    ys = []
    for row in rows:
        if row.get('type') != row_type:
            continue
        if row.get(key) is None or row.get('epoch') is None:
            continue
        xs.append(row['epoch'])
        ys.append(row[key])
    return xs, ys


def save_plot(output_path, title, series):
    plt.figure(figsize=(8, 5))
    for label, xs, ys, style in series:
        if xs and ys:
            plt.plot(xs, ys, style, label=label)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(title)
    plt.grid(True, alpha=0.3)
    if any(xs for _, xs, _, _ in series):
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Create training and validation loss plots from a trainer log.')
    parser.add_argument('--log', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--label', default='')
    args = parser.parse_args()

    rows = parse_log_rows(args.log)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or Path(args.log).stem

    train_loss = extract_series(rows, 'train-epoch', 'loss')
    val_loss = extract_series(rows, 'val-epoch', 'loss')
    val_test_loss = extract_series(rows, 'val-epoch', 'test_loss')

    save_plot(
        output_dir / 'loss_overview.png',
        'Loss',
        [
            (f'{label} train', *train_loss, '-'),
            (f'{label} val', *val_loss, '-'),
            (f'{label} val prior/test', *val_test_loss, '--'),
        ],
    )

    train_recon = extract_series(rows, 'train-epoch', 'recon_loss')
    val_recon = extract_series(rows, 'val-epoch', 'recon_loss')
    if train_recon[0] or val_recon[0]:
        save_plot(
            output_dir / 'reconstruction_loss.png',
            'Reconstruction Loss',
            [
                (f'{label} train recon', *train_recon, '-'),
                (f'{label} val recon', *val_recon, '--'),
            ],
        )

    train_kld = extract_series(rows, 'train-epoch', 'kld_loss')
    val_kld = extract_series(rows, 'val-epoch', 'kld_loss')
    if train_kld[0] or val_kld[0]:
        save_plot(
            output_dir / 'kl_divergence_loss.png',
            'KL Divergence Loss',
            [
                (f'{label} train KL', *train_kld, '-'),
                (f'{label} val KL', *val_kld, '--'),
            ],
        )


if __name__ == '__main__':
    main()
