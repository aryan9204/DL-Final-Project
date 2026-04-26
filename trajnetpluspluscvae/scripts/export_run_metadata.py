#!/usr/bin/env python3

import argparse
import csv
import json
import os
from pathlib import Path

import torch


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


def first_row(rows, row_type):
    for row in rows:
        if row.get('type') == row_type:
            return row
    return {}


def last_row(rows, row_type):
    for row in reversed(rows):
        if row.get('type') == row_type:
            return row
    return {}


def best_value(rows, row_type, key):
    values = [row.get(key) for row in rows if row.get('type') == row_type and row.get(key) is not None]
    if not values:
        return None
    return min(values)


def checkpoint_epoch(checkpoint_path):
    state_path = checkpoint_path + '.state'
    if not os.path.exists(state_path):
        return None
    state = torch.load(state_path, map_location='cpu')
    return state.get('epoch')


def load_existing_rows(csv_path):
    if not csv_path.exists():
        return []
    with csv_path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def write_rows(csv_path, rows, fieldnames):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description='Export one run summary row into a CSV file.')
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--family', required=True, choices=('lstm', 'cvae'))
    parser.add_argument('--variant', required=True, choices=('vanilla', 'social'))
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--log', required=True)
    parser.add_argument('--command', required=True)
    parser.add_argument('--status', default='completed')
    parser.add_argument('--metadata-csv', required=True)
    parser.add_argument('--plots-dir', required=False, default='')
    args = parser.parse_args()

    rows = parse_log_rows(args.log)
    process_row = first_row(rows, 'process')
    train_epoch_row = last_row(rows, 'train-epoch')
    val_epoch_row = last_row(rows, 'val-epoch')

    record = {
        'run_name': args.run_name,
        'family': args.family,
        'variant': args.variant,
        'status': args.status,
        'checkpoint_path': args.checkpoint,
        'checkpoint_epoch': checkpoint_epoch(args.checkpoint),
        'state_path': args.checkpoint + '.state',
        'log_path': args.log,
        'plots_dir': args.plots_dir,
        'command': args.command,
        'seed': process_row.get('args', {}).get('seed'),
        'path': process_row.get('args', {}).get('path'),
        'epochs_requested': process_row.get('args', {}).get('epochs'),
        'epochs_completed': train_epoch_row.get('epoch'),
        'batch_size': process_row.get('args', {}).get('batch_size'),
        'lr': process_row.get('args', {}).get('lr'),
        'step_size': process_row.get('args', {}).get('step_size'),
        'sample': process_row.get('args', {}).get('sample'),
        'augment': process_row.get('args', {}).get('augment'),
        'normalize_scene': process_row.get('args', {}).get('normalize_scene'),
        'augment_noise': process_row.get('args', {}).get('augment_noise'),
        'obs_dropout': process_row.get('args', {}).get('obs_dropout'),
        'k_modes': process_row.get('args', {}).get('k'),
        'noise_dim': process_row.get('args', {}).get('noise_dim'),
        'alpha_kld': process_row.get('args', {}).get('alpha_kld'),
        'kld_anneal_epochs': process_row.get('args', {}).get('kld_anneal_epochs'),
        'train_loss_final': train_epoch_row.get('loss'),
        'train_recon_loss_final': train_epoch_row.get('recon_loss'),
        'train_kld_loss_final': train_epoch_row.get('kld_loss'),
        'train_kld_weight_final': train_epoch_row.get('kld_weight'),
        'val_loss_final': val_epoch_row.get('loss'),
        'val_test_loss_final': val_epoch_row.get('test_loss'),
        'val_recon_loss_final': val_epoch_row.get('recon_loss'),
        'val_kld_loss_final': val_epoch_row.get('kld_loss'),
        'val_kld_weight_final': val_epoch_row.get('kld_weight'),
        'best_val_loss': best_value(rows, 'val-epoch', 'loss'),
        'best_val_test_loss': best_value(rows, 'val-epoch', 'test_loss'),
        'started_at': process_row.get('asctime'),
        'ended_at': val_epoch_row.get('asctime') or train_epoch_row.get('asctime'),
    }

    csv_path = Path(args.metadata_csv)
    existing_rows = load_existing_rows(csv_path)
    existing_rows = [row for row in existing_rows if row.get('run_name') != args.run_name]
    existing_rows.append(record)

    fieldnames = list(record.keys())
    write_rows(csv_path, existing_rows, fieldnames)


if __name__ == '__main__':
    main()
