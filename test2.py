#!/usr/bin/env python3
import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from strhub.data.dataset import LmdbDataset
from strhub.models.utils import load_from_checkpoint

# Your exact charset from training:
CHARSET = (" 0123456789۰۱۲٢۳۴۵۶۷۸۹اآأؤإءئۓۂئےۍېىيےیبپڀتٹثٿجچحخدڈذڎرڑزژسشصضطظعغفقڤڠݢکكڭگڬلمنںوۏههةۃۀہھڽضئکڤݢۏ-‌!\"#$%&'()*+,./:;<=>?@[\\]^_`{|}~")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data_root', default='data')
    p.add_argument('--split', choices=['train','val','test'], default='val')
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"Using charset ({len(CHARSET)} chars)")

    # 1) Load model
    model = load_from_checkpoint(
        args.checkpoint,
        charset_test=CHARSET
    ).to(device).eval()
    hp = model.hparams

    # 2) Build the exact same transforms you trained with
    transform = Compose([
        Resize((hp.img_size, hp.img_size)),
        ToTensor(),
        Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
    ])

    # 3) Direct LMDBDataset (no filtering or unicode normalization)
    lmdb_path = os.path.join(args.data_root, args.split, 'jawi')
    ds = LmdbDataset(
        root=lmdb_path,
        charset=CHARSET,
        max_label_len=hp.max_label_length,
        remove_whitespace=False,
        normalize_unicode=False,
        transform=transform
    )
    loader = DataLoader(ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: (torch.stack([x[0] for x in b]), [x[1] for x in b])
    )

    # 4) Debug print first 30 GT vs PR
    print("\n--- First 30 GT vs PR ---")
    seen = 0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        # raw forward + decode
        probs = model(imgs).softmax(-1)
        preds, _ = model.tokenizer.decode(probs)
        for gt, pr in zip(labels, preds):
            if seen < 30:
                print(f"{seen+1:02d}: GT='{gt}' | PR='{pr}'")
                seen += 1
            else:
                break
        if seen >= 30:
            break

    # 5) Full evaluation via test_step
    total = correct = 0
    total_ned = total_conf = total_len = 0
    for batch_idx, (imgs, labels) in enumerate(tqdm(loader, desc="Evaluating")):
        out = model.test_step((imgs.to(device), labels), batch_idx)['output']
        total      += out.num_samples
        correct    += out.correct
        total_ned  += out.ned
        total_conf += out.confidence
        total_len  += out.label_length

    print("\n=== Final Results ===")
    print(f"Accuracy       : {100*correct/total:.2f}%")
    print(f"1 - NED        : {100*(1 - total_ned/total):.2f}%")
    print(f"Avg confidence : {100*(total_conf/total):.2f}%")
    print(f"Avg label len  : {total_len/total:.2f}")

if __name__ == '__main__':
    main()
