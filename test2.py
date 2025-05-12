#!/usr/bin/env python3
"""
Custom test script for PARSeq Jawi OCR model with hard‑coded charset.
Evaluates a checkpoint on a given LMDB split and computes accuracy, normalized edit distance (1-NED), average confidence, and average label length.
Usage:
    python test.py \
        --checkpoint path/to/ckpt.ckpt \
        --data_root data \
        --split val \
        --batch_size 256 \
        --num_workers 4 \
        --device cuda
"""
import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from strhub.data.dataset import LmdbDataset
from strhub.models.utils import load_from_checkpoint

# === HARD‑CODED CHARSET ===
HARD_CODED_CHARSET = (" 0123456789۰۱۲٢۳۴۵۶۷۸۹اآأؤإءئۓۂئےۍېىيےیبپڀتٹثٿجچحخدڈذڎرڑزژسشصضطظعغفقڤڠݢکكڭگڬلمنںوۏههةۃۀہھڽضئکڤݢۏ-‌!\"#$%&'()*+,./:;<=>?@[\\]^_`{|}~")


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j - 1], dp[j])
            prev = cur
    return dp[n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to .ckpt file')
    parser.add_argument('--data_root', default='data', help='Root directory containing splits')
    parser.add_argument('--split', choices=['train','val','test'], default='val', help='Which dataset split to evaluate')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using hard-coded charset (length={len(HARD_CODED_CHARSET)}): {HARD_CODED_CHARSET}")

    # Load the model with the same test charset
    model = load_from_checkpoint(
        args.checkpoint,
        charset_test=HARD_CODED_CHARSET
    ).to(device).eval()

    # Retrieve max label length from model hyperparameters
    max_label_len = getattr(model.hparams, 'max_label_length', None)
    if max_label_len is None:
        raise AttributeError("Model checkpoint missing 'max_label_length' in hparams.")

    # Prepare LMDB dataset loader
    lmdb_path = os.path.join(args.data_root, args.split, 'jawi')
    if not os.path.isdir(lmdb_path):
        raise FileNotFoundError(f"LMDB path not found: {lmdb_path}")

    dataset = LmdbDataset(
        root=lmdb_path,
        charset=HARD_CODED_CHARSET,
        max_label_len=max_label_len
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dataset.collate
    )


    # Initialize accumulators
    total = correct = 0
    total_ned = 0.0
    total_conf = 0.0
    total_len = 0

    # Run evaluation
    for batch_idx, (imgs, labels) in enumerate(tqdm(loader, desc=f"Evaluating '{args.split}' split")):
        imgs = imgs.to(device)
        out = model.test_step((imgs, labels), batch_idx)['output']

        total += out.num_samples
        correct += out.correct
        total_ned += out.ned
        total_conf += out.confidence
        total_len += out.label_length

    # Final metrics
    accuracy = 100.0 * correct / total
    one_minus_ned = 100.0 * (1 - (total_ned / total))
    mean_conf = 100.0 * (total_conf / total)
    avg_len = total_len / total

    # Display
    print("\n=== Test Results ===")
    print(f"Total samples    : {total}")
    print(f"Accuracy         : {accuracy:.2f}%")
    print(f"1 - NED          : {one_minus_ned:.2f}%")
    print(f"Avg confidence   : {mean_conf:.2f}%")
    print(f"Avg label length : {avg_len:.2f}")


if __name__ == '__main__':
    main()
