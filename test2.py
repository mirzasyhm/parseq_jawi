#!/usr/bin/env python3
"""
Robust test script for PARSeq Jawi OCR using SceneTextDataModule to ensure consistent image resizing.
Evaluates a checkpoint on the 'jawi' LMDB split and computes accuracy, 1-NED, average confidence, and average label length.
Usage:
    python test.py \
        --checkpoint path/to/ckpt.ckpt \
        --data_root data \
        --batch_size 512 \
        --num_workers 4 \
        --device cuda
"""
import argparse
from dataclasses import dataclass
import sys
from tqdm import tqdm

import torch

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint

# === HARD-CODED CHARSET (training & testing) ===
HARD_CODED_CHARSET = (" 0123456789۰۱۲٢۳۴۵۶۷۸۹اآأؤإءئۓۂئےۍېىيےیبپڀتٹثٿجچحخدڈذڎرڑزژسشصضطظعغفقڤڠݢکكڭگڬلمنںوۏههةۃۀہھڽضئکڤݢۏ-‌!\"#$%&'()*+,./:;<=>?@[\\]^_`{|}~")

@dataclass
class Result:
    dataset: str
    num_samples: int
    accuracy: float
    ned: float
    confidence: float
    label_length: float


def print_table(results: list[Result], file=None):
    w = max(len(r.dataset) for r in results + [Result('Combined',0,0,0,0,0)])
    header = f"| {'Dataset':<{w}} | # samples | Accuracy | 1 - NED | Confidence | Label Length |"
    sep    = f"|:{'-'*w}---------------------------------------------------"
    print(header, file=file)
    print(sep, file=file)
    # Combined
    comb = Result('Combined',0,0,0,0,0)
    for r in results:
        comb.num_samples += r.num_samples
        comb.accuracy    += r.num_samples * r.accuracy
        comb.ned         += r.num_samples * r.ned
        comb.confidence  += r.num_samples * r.confidence
        comb.label_length+= r.num_samples * r.label_length
        print(f"| {r.dataset:<{w}} | {r.num_samples:>9} | {r.accuracy:>8.2f} | {r.ned:>7.2f} | {r.confidence:>10.2f} | {r.label_length:>12.2f} |", file=file)
    # finalize combined averages
    comb.accuracy     /= comb.num_samples
    comb.ned          /= comb.num_samples
    comb.confidence   /= comb.num_samples
    comb.label_length /= comb.num_samples
    print(f"|:{'-'*w}:|-----------|----------|---------|------------|--------------|", file=file)
    print(f"| {comb.dataset:<{w}} | {comb.num_samples:>9} | {comb.accuracy:>8.2f} | {comb.ned:>7.2f} | {comb.confidence:>10.2f} | {comb.label_length:>12.2f} |", file=file)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Loading checkpoint on {device} with charset len={len(HARD_CODED_CHARSET)}")

    # load model with correct charset
    model = load_from_checkpoint(
        args.checkpoint,
        charset_test=HARD_CODED_CHARSET
    ).eval().to(device)
    hp = model.hparams

    # prepare datamodule: 'jawi' is under TEST_CUSTOM
    dm = SceneTextDataModule(
        args.data_root,
        '_',
        hp.img_size,
        hp.max_label_length,
        hp.charset_train,
        HARD_CODED_CHARSET,
        args.batch_size,
        args.num_workers,
        False
    )

    # iterate over 'jawi' split only
    test_sets = {name: dl for name, dl in dm.test_dataloaders(SceneTextDataModule.TEST_CUSTOM).items() if name=='jawi'}
    results = []
    for name, loader in test_sets.items():
        total=correct=0
        ned=conf=lbl_len=0
        for batch_idx, (imgs, labels) in enumerate(tqdm(loader, desc=name)):
            out = model.test_step((imgs.to(device), labels), batch_idx)['output']
            total    += out.num_samples
            correct  += out.correct
            ned      += out.ned
            conf     += out.confidence
            lbl_len  += out.label_length
        results.append(Result(name, total, 100*correct/total, 100*(1-ned/total), 100*(conf/total), lbl_len/total))

    # print
    print_table(results, file=sys.stdout)

if __name__=='__main__':
    main()
