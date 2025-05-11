import os
import sys
# Add your project root (where strhub/ lives) to Python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import torch
from strhub.models.utils import load_from_checkpoint
from strhub.data.dataset import LmdbDataset
from torch.utils.data import DataLoader
import lmdb
import io
from PIL import Image

def check_checkpoint_quality(ckpt_path, lmdb_path, max_label_len=25, sample_count=5):
    # Load model
    model = load_from_checkpoint(ckpt_path).eval().cuda()
    # Print model stats
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters   : {total_params:,}")
    print(f"Trainable params   : {trainable:,}")
    # Charset size
    charset = getattr(model.model, "charset", None)
    print(f"Charset length     : {len(charset) if charset is not None else 'N/A'}")

    # Load dataset to see how many samples and show some predictions
    ds = LmdbDataset(lmdb_path, charset, max_label_len)
    print(f"Dataset length seen by LmdbDataset: {len(ds)}")

    # DataLoader for a few samples
    loader = DataLoader(ds, batch_size=1, num_workers=2, shuffle=False)
    print("\nSample predictions:")
    for i, (img_tensor, label) in enumerate(loader):
        if i >= sample_count:
            break
        # Convert tensor back to PIL for prediction if needed
        img = Image.fromarray((img_tensor[0].permute(1,2,0).cpu().numpy() * 255).astype('uint8'))
        pred = model.predict([img])[0]
        print(f"{i+1}. GT='{label[0]}'  ->  PRED='{pred}'")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help='Path to model checkpoint')
    parser.add_argument('--lmdb', default='data/test/jawi', help='Path to test LMDB')
    parser.add_argument('--max_len', type=int, default=25, help='Max label length')
    parser.add_argument('--samples', type=int, default=5, help='Number of samples to inspect')
    args = parser.parse_args()
    check_checkpoint_quality(args.checkpoint, args.lmdb, args.max_len, args.samples)