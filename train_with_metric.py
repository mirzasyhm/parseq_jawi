Skip to content
You said:
is there anyway to speed up this code:

#!/usr/bin/env python3
"""a modified version of CRNN torch repository https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py"""
import io
import os

import fire
import lmdb
import numpy as np
from PIL import Image


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    img = Image.open(io.BytesIO(imageBin)).convert('RGB')
    return np.prod(img.size) > 0


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(inputPath, gtFile, outputPath, checkValid=True):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        gtFile     : list of image path and label
        checkValid : if true, check the validity of every image
    """
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=850737418)

    cache = {}
    cnt = 1

    with open(gtFile, 'r', encoding='utf-8') as f:
        data = f.readlines()

    nSamples = len(data)
    for i, line in enumerate(data):
        imagePath, label = line.strip().split(maxsplit=1)
        imagePath = os.path.join(inputPath, imagePath)
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            try:
                img = Image.open(io.BytesIO(imageBin)).convert('RGB')
            except IOError as e:
                with open(outputPath + '/error_image_log.txt', 'a') as log:
                    log.write('{}-th image data occured error: {}, {}\n'.format(i, imagePath, e))
                continue
            if np.prod(img.size) == 0:
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d'.encode() % cnt
        labelKey = 'label-%09d'.encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt - 1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    env.close()
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    fire.Fire(createDataset)

"""
python create_lmdb_dataset.py 
    ./dataset/data/test
    ./dataset/data/test/labels.txt 
    ./dataset/data_jawi_lmdb/test
    --checkValid=True

"""
ChatGPT said:

Thought for 18 seconds
You’re IO-bound in two places: image decoding/validation and LMDB commits. A few simple changes can often buy you 2×–5× speedups:

Use a faster JPEG decoder
Replace Pillow with pillow-simd (a drop-in replacement) or use cv2.imdecode, which is quite a bit faster for large batches of images:

pip uninstall pillow
pip install pillow-simd
Batch your LMDB writes more aggressively
Currently you flush every 1 000 samples. If you have RAM to spare, bump that to 5 000–10 000. Fewer transactions → less fsync overhead.

Tune LMDB for speed (at the cost of crash safety)

env = lmdb.open(
    outputPath,
    map_size=850737418,
    writemap=True,     # use write-map for zero-copy writes
    map_async=True,    # let OS flush pages in background
    metasync=False,    # don’t fsync metadata on commit
    sync=False         # don’t fsync data on every commit
)
These flags can safely be used during offline dataset construction.

Parallelize image loading/validation
The GIL mostly sits idle in I/O-heavy code. You can farm out the open/read/verify step to a thread pool, then feed results back into your LMDB writer in the main thread. E.g.:

from concurrent.futures import ThreadPoolExecutor, as_completed

def load_and_check(line):
    imagePath, label = line.strip().split(maxsplit=1)
    with open(os.path.join(inputPath, imagePath), 'rb') as f:
        data = f.read()
    if checkValid:
        try:
            # use .verify() instead of full convert if you just want to know it decodes
            img = Image.open(io.BytesIO(data))
            img.verify()  
        except Exception:
            return None
    return data, label

# ...
with ThreadPoolExecutor(max_workers=8) as pool:
    futures = [pool.submit(load_and_check, line) for line in data]
    for res in as_completed(futures):
        if res.result() is None:
            continue
        imgBin, label = res.result()
        # assemble your cache here, flush in batches as before
Skip validation on data you trust
If your source images are already clean, call your script with --checkValid=False.

Putting it all together, you might end up with something like:

env = lmdb.open(outputPath,
                map_size=2<<40,
                writemap=True,
                map_async=True,
                metasync=False,
                sync=False)

batch_size = 5000
cache = {}
cnt = 1

with ThreadPoolExecutor(max_workers=8) as pool:
    for future in pool.map(load_and_check, data, chunksize=128):
        if future is None: 
            continue
        imgBin, label = future
        cache[f'image-{cnt:09d}'.encode()] = imgBin
        cache[f'label-{cnt:09d}'.encode()] = label.encode()
        if cnt % batch_size == 0:
            with env.begin(write=True) as txn:
                for k, v in cache.items():
                    txn.put(k, v)
            cache.clear()
            print(f'Flushed {cnt}/{len(data)}')
        cnt += 1

# final flush...
cache[b'num-samples'] = str(cnt-1).encode()
with env.begin(write=True) as txn:
    for k, v in cache.items():
        txn.put(k, v)
In practice you should see several-fold speedups just by combining:

Pillow-simd

Bigger LMDB batches + tuned flags

Simple threading for reads

Give it a try!

You said:
/content/parseq_jawi
/content/parseq_jawi/train.py:64: DeprecationWarning: torch.get_autocast_gpu_dtype() is deprecated. Please use torch.get_autocast_dtype('cuda') instead. (Triggered internally at ../torch/csrc/autograd/init.cpp:787.)
  config.trainer.precision = 'bf16-mixed' if torch.get_autocast_gpu_dtype() is torch.bfloat16 else '16-mixed'
/usr/local/lib/python3.11/dist-packages/timm/models/helpers.py:7: FutureWarning: Importing from timm.models.helpers is deprecated, please import via timm.models
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.models", FutureWarning)
  | Name             | Type           | Params | Mode 
------------------------------------------------------------
0 | model            | PARSeq         | 23.9 M | train
1 | model.encoder    | Encoder        | 21.4 M | train
2 | model.decoder    | Decoder        | 2.4 M  | train
3 | model.head       | Linear         | 49.7 K | train
4 | model.text_embed | TokenEmbedding | 50.3 K | train
5 | model.dropout    | Dropout        | 0      | train
------------------------------------------------------------
23.9 M    Trainable params
0         Non-trainable params
23.9 M    Total params
95.435    Total estimated model params size (MB)
287       Modules in train mode
0         Modules in eval mode
Using 16bit Automatic Mixed Precision (AMP)
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
You are using a CUDA device ('NVIDIA A100-SXM4-40GB') that has Tensor Cores. To properly utilize them, you should set torch.set_float32_matmul_precision('medium' | 'high') which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
2025-05-12 10:37:55.538379: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable TF_ENABLE_ONEDNN_OPTS=0.
2025-05-12 10:37:55.555637: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1747046275.577962    6465 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1747046275.584556    6465 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2025-05-12 10:37:55.606255: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Loading train_dataloader to estimate number of stepping batches.
[2025-05-12 10:37:58,358][strhub.data.dataset][INFO] - dataset root:	/content/parseq_jawi/data/train/jawi
[2025-05-12 10:37:58,539][strhub.data.dataset][INFO] - 	lmdb:	.	num samples: 159170
Sanity Checking: |          | 0/? [00:00<?, ?it/s][2025-05-12 10:37:58,710][strhub.data.dataset][INFO] - dataset root:	/content/parseq_jawi/data/val
[2025-05-12 10:37:58,734][strhub.data.dataset][INFO] - 	lmdb:	jawi	num samples: 19896
Epoch 0:  14% 60/415 [00:15<01:29,  3.96it/s, v_num=.]Error executing job with overrides: []
Traceback (most recent call last):
  File "/content/parseq_jawi/train.py", line 108, in main
    trainer.fit(model, datamodule=datamodule, ckpt_path=config.ckpt_path)
  File "/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/trainer.py", line 561, in fit
    call._call_and_handle_interrupt(
  File "/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/call.py", line 48, in _call_and_handle_interrupt
    return trainer_fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/trainer.py", line 599, in _fit_impl
    self._run(model, ckpt_path=ckpt_path)
  File "/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/trainer.py", line 1012, in _run
    results = self._run_stage()
              ^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/trainer.py", line 1056, in _run_stage
    self.fit_loop.run()
  File "/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/fit_loop.py", line 216, in run
    self.advance()
  File "/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/fit_loop.py", line 455, in advance
    self.epoch_loop.run(self._data_fetcher)
  File "/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/training_epoch_loop.py", line 150, in run
    self.advance(data_fetcher)
  File "/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/training_epoch_loop.py", line 320, in advance
    batch_output = self.automatic_optimization.run(trainer.optimizers[0], batch_idx, kwargs)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/optimization/automatic.py", line 192, in run
    self._optimizer_step(batch_idx, closure)
  File "/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/optimization/automatic.py", line 270, in _optimizer_step
    call._call_lightning_module_hook(
  File "/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/call.py", line 176, in _call_lightning_module_hook
    output = fn(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/pytorch_lightning/core/module.py", line 1302, in optimizer_step
    optimizer.step(closure=optimizer_closure)
  File "/usr/local/lib/python3.11/dist-packages/pytorch_lightning/core/optimizer.py", line 154, in step
    step_output = self._strategy.optimizer_step(self._optimizer, closure, **kwargs)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/pytorch_lightning/strategies/strategy.py", line 239, in optimizer_step
    return self.precision_plugin.optimizer_step(optimizer, model=model, closure=closure, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/pytorch_lightning/plugins/precision/amp.py", line 79, in optimizer_step
    closure_result = closure()
                     ^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/optimization/automatic.py", line 146, in __call__
    self._result = self.closure(*args, **kwargs)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/optimization/automatic.py", line 131, in closure
    step_output = self._step_fn()
                  ^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/optimization/automatic.py", line 319, in _training_step
    training_step_output = call._call_strategy_hook(trainer, "training_step", *kwargs.values())
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/call.py", line 328, in _call_strategy_hook
    output = fn(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/pytorch_lightning/strategies/strategy.py", line 391, in training_step
    return self.lightning_module.training_step(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/content/parseq_jawi/strhub/models/parseq/system.py", line 171, in training_step
    tgt = self.tokenizer.encode(labels, self._device)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/content/parseq_jawi/strhub/data/utils.py", line 114, in encode
    batch = [
            ^
  File "/content/parseq_jawi/strhub/data/utils.py", line 115, in <listcomp>
    torch.as_tensor([self.bos_id] + self._tok2ids(y) + [self.eos_id], dtype=torch.long, device=device)
                                    ^^^^^^^^^^^^^^^^
  File "/content/parseq_jawi/strhub/data/utils.py", line 55, in _tok2ids
    return [self._stoi[s] for s in tokens]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/content/parseq_jawi/strhub/data/utils.py", line 55, in <listcomp>
    return [self._stoi[s] for s in tokens]
            ~~~~~~~~~~^^^
KeyError: ' '

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
Epoch 0:  14% 60/415 [00:21<02:04,  2.85it/s, v_num=.]
ChatGPT said:

Thought for 17 seconds
That KeyError on ' ' means your PARSeq tokenizer doesn’t know how to map a literal space character—so somewhere in your labels you have a space, but your vocabulary (_stoi) never got an entry for it. You have two clean ways to fix it:

1) Add space to your charset/vocab
If you really want to keep inter-word spaces in your Jawi labels, you need to include a space in the alphabet you pass to LmdbDataset (and hence into the tokenizer’s _stoi). For example, wherever you currently do:

charset = "0123456789۰۱۲٣...ابپڀتٹ..."
just prepend (or append) a space:

charset = " " + "0123456789۰۱۲٣...ابپڀتٹ..."
Then rebuild your LMDB (so the new num-samples, label-*-* entries are written with that updated charset), and re-run training. Now when the tokenizer sees ' ', self._stoi[' '] will exist.

2) Strip out or normalize spaces in your labels
If your task doesn’t require modeling whitespace (e.g. you only care about the concatenated glyph sequence), you can simply remove all spaces before you put labels into LMDB:

# inside createDataset(), right after reading `label`:
label = label.replace(" ", "")
That way none of your stored labels ever contains ' ', and you won’t hit a KeyError.

Quick patch in your data‐utils
As a belt-and-suspenders approach you could also guard against unknown tokens in strhub/data/utils.py:

# around line 55 in _tok2ids(...)
def _tok2ids(self, tokens):
    ids = []
    for s in tokens:
        if s not in self._stoi:
            # either skip it, or map to a special `<unk>` or space‐id
            continue
        ids.append(self._stoi[s])
    return ids
—but it’s cleaner to either include space in your vocab or strip it upstream.

Next steps:

Decide if you need spaces semantically.

If yes, add " " to your charset and regenerate your LMDB.

If not, strip them before caching.

That will eliminate the KeyError: ' ' and get you back to training.

You said:
!python test.py outputs/parseq/2025-05-12_10-45-42/checkpoints/epoch=17-step=7255-val_accuracy=99.9347-val_NED=99.9909.ckpt

This is the result I get:

Additional keyword arguments: {'charset_test': ' 0123456789۰۱۲٢۳۴۵۶۷۸۹اآأؤإءئۓۂئےۍېىيےیبپڀتٹثٿجچحخدڈذڎرڑزژسشصضطظعغفقڤڠݢکكڭگڬلمنںوۏههةۃۀہھڽضئکڤݢۏ-\u200c!"#$%&\'()*+,./:;<=>?@[\\]^_{|}~'}
/usr/local/lib/python3.11/dist-packages/timm/models/helpers.py:7: FutureWarning: Importing from timm.models.helpers is deprecated, please import via timm.models
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.models", FutureWarning)
jawi: 100% 39/39 [00:13<00:00,  2.92it/s]
Custom set:
| Dataset  | # samples | Accuracy | 1 - NED | Confidence | Label Length |
|:--------:|----------:|---------:|--------:|-----------:|-------------:|
| jawi     |     19897 |     0.00 |    0.52 |      99.92 |         6.51 |
|----------|-----------|----------|---------|------------|--------------|
| Combined |     19897 |     0.00 |    0.52 |      99.92 |         6.51 |


since i get the accuracy 0. i think there might something wrong with the test.py
ChatGPT said:

Thought for 16 seconds
It turns out nothing is wrong with PyTorch Lightning or your checkpoint – your test script is simply never matching any of your ground-truth labels, so your “accuracy” column stays at 0.00%. Here are the most common culprits and how to fix them:

1) Mismatched charset_test
In your test output you see:

Additional keyword arguments: {'charset_test': ' 0123456789…~'}  
Notice the leading space in that string. If your tokenizer’s _stoi doesn’t get the exact same ordering of characters that you used at training time, every decoded sequence will either be empty or garbage, so it will never exactly equal the GT label.

✔️ Fix: Make absolutely sure that the charset_test you pass to test.py is byte-for-byte identical to the one you used when you created your LMDB and trained your model (no extra leading/trailing spaces or invisible Unicode characters).

Easiest check: copy-paste your training-time charset string directly into your test command:

python test.py \
  outputs/…/epoch=17-…ckpt \
  --charset_test "0123456789۰۱۲٢۳…ڽضئک…" 
2) Verify that your LMDB “test” split is actually being read
Your test run logs show:

jawi: 100% 39/39 [00:13<00:00,  2.92it/s] 
Custom set:
| Dataset  | # samples | Accuracy | …
| jawi     |     19897 |     0.00 | …
That tells us you’ve pointed at a dataset of 19 897 samples, but Lightning may still be using your validation split or an entirely empty set under the hood.

✔️ Fix: In your test.py, make sure you’re calling:

trainer.test(model, datamodule=datamodule)
and not trainer.validate. And confirm that your datamodule.test_dataloader() is actually loading from data/val/jawi (or wherever your test LMDB is).

3) Dump a few raw predictions for sanity
Add a tiny debug loop at the top of your test_step (or right after you call .predict() in your script) to print out the first 5 (image, GT, pred) triplets:

# inside your test loop, before accumulating metrics:
if batch_idx < 5:
    for gt, pred in zip(labels, outputs):   # or however you unpack them
        print("GT >", gt)
        print("PR >", pred)
If you see either:

PR > "" (empty string)

or some string of the wrong characters

then you know right away that the decoding step is failing (usually charset mismatch).

4) As a quick hack, strip spaces (or unknown chars) at decode time
If you don’t actually care about preserving spaces in your Jawi OCR, you can post-process predictions before comparing:

clean = lambda s: s.replace(" ", "")
preds = [ clean(p) for p in preds ]
gts   = [ clean(g) for g in gts   ]
Then compute accuracy = sum(p==g)/len(g) on those cleaned lists. If your accuracy suddenly jumps above zero, you’ve confirmed it really was just a character-mapping issue.

In summary
Double-check your charset_test string – it must exactly match what you used in training.

Ensure you’re really loading your intended test LMDB in test.py.

Print out a handful of (GT, pred) pairs to see what your model is spitting out.

Once your charset and data split are correctly wired up, your “Accuracy” column will come back to life.

You said:
this is my current test.py:

#!/usr/bin/env python3
# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import string
import sys
from dataclasses import dataclass

from tqdm import tqdm

import torch

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args


@dataclass
class Result:
    dataset: str
    num_samples: int
    accuracy: float
    ned: float
    confidence: float
    label_length: float


def print_results_table(results: list[Result], file=None):
    w = max(map(len, map(getattr, results, ['dataset'] * len(results))))
    w = max(w, len('Dataset'), len('Combined'))
    print('| {:<{w}} | # samples | Accuracy | 1 - NED | Confidence | Label Length |'.format('Dataset', w=w), file=file)
    print('|:{:-<{w}}:|----------:|---------:|--------:|-----------:|-------------:|'.format('----', w=w), file=file)
    c = Result('Combined', 0, 0, 0, 0, 0)
    for res in results:
        c.num_samples += res.num_samples
        c.accuracy += res.num_samples * res.accuracy
        c.ned += res.num_samples * res.ned
        c.confidence += res.num_samples * res.confidence
        c.label_length += res.num_samples * res.label_length
        print(
            f'| {res.dataset:<{w}} | {res.num_samples:>9} | {res.accuracy:>8.2f} | {res.ned:>7.2f} '
            f'| {res.confidence:>10.2f} | {res.label_length:>12.2f} |',
            file=file,
        )
    c.accuracy /= c.num_samples
    c.ned /= c.num_samples
    c.confidence /= c.num_samples
    c.label_length /= c.num_samples
    print('|-{:-<{w}}-|-----------|----------|---------|------------|--------------|'.format('----', w=w), file=file)
    print(
        f'| {c.dataset:<{w}} | {c.num_samples:>9} | {c.accuracy:>8.2f} | {c.ned:>7.2f} '
        f'| {c.confidence:>10.2f} | {c.label_length:>12.2f} |',
        file=file,
    )


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')")
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--cased', action='store_true', default=False, help='Cased comparison')
    parser.add_argument('--punctuation', action='store_true', default=False, help='Check punctuation')
    parser.add_argument('--std', action='store_true', default=False, help='Evaluate on standard benchmark datasets')
    parser.add_argument('--new', action='store_true', default=False, help='Evaluate on new benchmark datasets')
    parser.add_argument('--custom', action='store_true', default=True, help='Evaluate on custom personal datasets')
    parser.add_argument('--rotation', type=int, default=0, help='Angle of rotation (counter clockwise) in degrees.')
    parser.add_argument('--device', default='cuda')
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)

    charset_test = (" 0123456789۰۱۲٢۳۴۵۶۷۸۹اآأؤإءئۓۂئےۍېىيےیبپڀتٹثٿجچحخدڈذڎرڑزژسشصضطظعغفقڤڠݢکكڭگڬلمنںوۏههةۃۀہھڽضئکڤݢۏ-‌!\"#$%&'()*+,./:;<=>?@[\\]^_`{|}~")
    kwargs.update({'charset_test': charset_test})
    print(f'Additional keyword arguments: {kwargs}')

    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
    hp = model.hparams
    datamodule = SceneTextDataModule(
        args.data_root,
        '_unused_',
        hp.img_size,
        hp.max_label_length,
        hp.charset_train,
        hp.charset_test,
        args.batch_size,
        args.num_workers,
        False,
        rotation=args.rotation,
    )

    test_set = tuple()
    if args.std:
        test_set = SceneTextDataModule.TEST_BENCHMARK_SUB + SceneTextDataModule.TEST_BENCHMARK
    if args.custom:
        test_set += SceneTextDataModule.TEST_CUSTOM
    if args.new:
        test_set += SceneTextDataModule.TEST_NEW
    test_set = sorted(set(test_set))

    results = {}
    max_width = max(map(len, test_set))
    for name, dataloader in datamodule.test_dataloaders(test_set).items():
        total = 0
        correct = 0
        ned = 0
        confidence = 0
        label_length = 0
        for imgs, labels in tqdm(iter(dataloader), desc=f'{name:>{max_width}}'):
            res = model.test_step((imgs.to(model.device), labels), -1)['output']
            total += res.num_samples
            correct += res.correct
            ned += res.ned
            confidence += res.confidence
            label_length += res.label_length
        accuracy = 100 * correct / total
        mean_ned = 100 * (1 - ned / total)
        mean_conf = 100 * confidence / total
        mean_label_length = label_length / total
        results[name] = Result(name, total, accuracy, mean_ned, mean_conf, mean_label_length)

    result_groups = dict()
    if args.std:
        result_groups.update({'Benchmark (Subset)': SceneTextDataModule.TEST_BENCHMARK_SUB})
        result_groups.update({'Benchmark': SceneTextDataModule.TEST_BENCHMARK})
    if args.custom:
        result_groups.update({'Custom': SceneTextDataModule.TEST_CUSTOM})
    if args.new:
        result_groups.update({'New': SceneTextDataModule.TEST_NEW})
    with open(args.checkpoint + '.log.txt', 'w') as f:
        for out in [f, sys.stdout]:
            for group, subset in result_groups.items():
                print(f'{group} set:', file=out)
                print_results_table([results[s] for s in subset], out)
                print('\n', file=out)


if __name__ == '__main__':
    main()

ChatGPT said:

Thought for 10 seconds
Your test harness is very likely loading the wrong character set into the DataModule—so every prediction is being compared against a mismatched label–decoder mapping and you end up with 0% exact matches. There are two quick fixes:

1) Make the DataModule use your charset_test, not the old one baked into hparams
Right now you do:

# after loading model…
hp = model.hparams
datamodule = SceneTextDataModule(
    args.data_root,
    '_unused_',
    hp.img_size,
    hp.max_label_length,
    hp.charset_train,    # ✅ training charset
    hp.charset_test,     # ❌ STILL pointing at the *old* test charset
    ...
)
but you just overrode charset_test for the model in parse_model_args(unknown) → load_from_checkpoint(...), so hp.charset_test is stale. Instead, explicitly pass your updated charset_test variable into the DataModule:

-    datamodule = SceneTextDataModule(
-        args.data_root,
-        '_unused_',
-        hp.img_size,
-        hp.max_label_length,
-        hp.charset_train,
-        hp.charset_test,
-        args.batch_size,
-        args.num_workers,
-        False,
-        rotation=args.rotation,
-    )
+    datamodule = SceneTextDataModule(
+        args.data_root,
+        '_unused_',
+        hp.img_size,
+        hp.max_label_length,
+        hp.charset_train,
+        charset_test,            # ← use your local variable here
+        args.batch_size,
+        args.num_workers,
+        False,
+        rotation=args.rotation,
+    )
That way both model and data loader agree on exactly the same symbol-to-id mapping.

2) Double-check for invisible characters in your literal
I also spot that your literal has a zero-width non-joiner (\u200c) right after the hyphen:

charset_test = (" 012…9…-‌!\"#$%…~")
                   ^^^^^
                   hyphen + ZWNJ
If your training charset didn’t include that ZWNJ (or included it in a different position), you’ll silently mis-index every single trailing character. Easiest fix is to copy-paste exactly the charset string from your training script into your test.py, then remove any stray whitespace/ZWNJ around it.

3) (Optional) Print a few GT vs. prediction pairs
To confirm that you’ve fixed it, sprinkle into your testing loop:

for batch_idx, (imgs, labels) in enumerate(dataloader):
    out = model.test_step((imgs.to(model.device), labels), -1)['output']
    if batch_idx == 0:
        for gt, pred in zip(labels, out.preds):
            print("GT >", gt)
            print("PR >", pred)
    # accumulate metrics...
If you now see non-empty PR > strings that match your GT >, your accuracy will jump back up.

Once you align both data and model on the very same charset_test, your 0.00% accuracy will turn into the real performance you saw on validation.

You said:
it still got 0%. At this point, lets just create our own test.py
ChatGPT said:

Thought for 59 seconds
I’ve drafted a standalone test.py that:

Loads your checkpoint with the exact charset_test you trained on

Reads the LMDB split directly via LmdbDataset

Uses the model’s built-in test_step for consistency

Computes exact-match accuracy, normalized edit‐distance (1 – NED), average confidence, and average label length

Prints a concise summary

Save this alongside your other scripts, fill in --charset with your original training string, and run:

python test.py \
  --checkpoint outputs/.../epoch=17-...ckpt \
  --data_root data \
  --split val \
  --charset "0123456789۰۱۲٢..." \
  --batch_size 256 \
  --num_workers 4 \
  --device cuda
That should bypass any charset/DataModule mismatches and give you real performance numbers. Let me know how it goes!

You said:
lets hard coded the charset
ChatGPT said:

Thought for 15 seconds
I’ve embedded the full HARD_CODED_CHARSET constant at the top and removed any command-line charset flag. Now the model and dataset loader both use exactly that same string, so there can’t be any mismatch. Just run:

python test.py \
  --checkpoint outputs/.../epoch=17-step=7255-val_accuracy=99.9347-val_NED=99.9909.ckpt \
  --data_root data \
  --split val \
  --batch_size 256 \
  --num_workers 4 \
  --device cuda
and you should finally see non-zero accuracy. Let me know how it behaves!

You said:
Traceback (most recent call last):
  File "/content/parseq_jawi/test.py", line 152, in <module>
    main()
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/content/parseq_jawi/test.py", line 84, in main
    kwargs = parse_model_args(unknown)
             ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/content/parseq_jawi/strhub/models/utils.py", line 101, in parse_model_args
    name, value = arg.split('=', maxsplit=1)
    ^^^^^^^^^^^
ValueError: not enough values to unpack (expected 2, got 1)


do you need any information from me?
ChatGPT said:
You’re still invoking the original test.py (which expects key=value args for parse_model_args). Our new standalone script doesn’t call into parse_model_args at all, so you’ll need to:

Overwrite your /content/parseq_jawi/test.py with the code from the canvas (the version that hard-codes HARD_CODED_CHARSET and takes only the usual flags: --checkpoint, --data_root, --split, etc.).

Save and then run it exactly as:

python test.py \
  --checkpoint outputs/.../epoch=17-step=7255-val_accuracy=99.9347-val_NED=99.9909.ckpt \
  --data_root data \
  --split val \
  --batch_size 256 \
  --num_workers 4 \
  --device cuda
That script has no dependency on parse_model_args, so you won’t get that unpacking error anymore. Let me know once you’ve swapped it over!

You said:
Using hard‑coded charset (length=129):  0123456789۰۱۲٢۳۴۵۶۷۸۹اآأؤإءئۓۂئےۍېىيےیبپڀتٹثٿجچحخدڈذڎرڑزژسشصضطظعغفقڤڠݢکكڭگڬلمنںوۏههةۃۀہھڽضئکڤݢۏ-‌!"#$%&'()*+,./:;<=>?@[\]^_{|}~
/usr/local/lib/python3.11/dist-packages/timm/models/helpers.py:7: FutureWarning: Importing from timm.models.helpers is deprecated, please import via timm.models
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.models", FutureWarning)
Exception ignored in: <function LmdbDataset.__del__ at 0x7da22dd48cc0>
Traceback (most recent call last):
  File "/content/parseq_jawi/strhub/data/dataset.py", line 80, in __del__
    if self._env is not None:
       ^^^^^^^^^
AttributeError: 'LmdbDataset' object has no attribute '_env'
Traceback (most recent call last):
  File "/content/parseq_jawi/test2.py", line 114, in <module>
    main()
  File "/content/parseq_jawi/test2.py", line 69, in main
    dataset = LmdbDataset(
              ^^^^^^^^^^^^
TypeError: LmdbDataset.__init__() missing 1 required positional argument: 'max_label_len'
ChatGPT said:

Thought for 21 seconds
I’ve added the missing max_label_len parameter—pulled directly from model.hparams.max_label_length—into the LmdbDataset constructor. Overwrite your current test.py with this version, then rerun. This should clear the TypeError and let the evaluation proceed.

You said:
Using hard-coded charset (length=129):  0123456789۰۱۲٢۳۴۵۶۷۸۹اآأؤإءئۓۂئےۍېىيےیبپڀتٹثٿجچحخدڈذڎرڑزژسشصضطظعغفقڤڠݢکكڭگڬلمنںوۏههةۃۀہھڽضئکڤݢۏ-‌!"#$%&'()*+,./:;<=>?@[\]^_{|}~
/usr/local/lib/python3.11/dist-packages/timm/models/helpers.py:7: FutureWarning: Importing from timm.models.helpers is deprecated, please import via timm.models
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.models", FutureWarning)
Traceback (most recent call last):
  File "/content/parseq_jawi/test2.py", line 120, in <module>
    main()
  File "/content/parseq_jawi/test2.py", line 83, in main
    collate_fn=dataset.collate
               ^^^^^^^^^^^^^^^
AttributeError: 'LmdbDataset' object has no attribute 'collate'
ChatGPT said:
I’ve replaced the broken dataset.collate call with an inline collate_fn that stacks the image tensors and aggregates labels into a list. Please overwrite your test.py with this version and rerun. Let me know if it proceeds without errors!

You said:
Using hard-coded charset (length=128): 0123456789۰۱۲٢۳۴۵۶۷۸۹اآأؤإءئۓۂئےۍېىيےیبپڀتٹثٿجچحخدڈذڎرڑزژسشصضطظعغفقڤڠݢکكڭگڬلمنںوۏههةۃۀہھڽضئکڤݢۏ-‌!"#$%&'()*+,./:;<=>?@[\]^_{|}~
/usr/local/lib/python3.11/dist-packages/timm/models/helpers.py:7: FutureWarning: Importing from timm.models.helpers is deprecated, please import via timm.models
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.models", FutureWarning)
Evaluating 'test' split:   0% 0/78 [00:00<?, ?it/s]
Traceback (most recent call last):
  File "/content/parseq_jawi/test2.py", line 129, in <module>
    main()
  File "/content/parseq_jawi/test2.py", line 103, in main
    for batch_idx, (imgs, labels) in enumerate(tqdm(loader, desc=f"Evaluating '{args.split}' split")):
  File "/usr/local/lib/python3.11/dist-packages/tqdm/std.py", line 1181, in __iter__
    for obj in iterable:
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 701, in __next__
    data = self._next_data()
           ^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1465, in _next_data
    return self._process_data(data)
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1491, in _process_data
    data.reraise()
  File "/usr/local/lib/python3.11/dist-packages/torch/_utils.py", line 715, in reraise
    raise exception
TypeError: Caught TypeError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 351, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 55, in fetch
    return self.collate_fn(data)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/content/parseq_jawi/test2.py", line 84, in collate_fn
    imgs = torch.stack([b[0] for b in batch], dim=0)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: expected Tensor as element 0 in argument 0, but got Image
ChatGPT said:

Thought for 51 seconds
I’ve updated collate_fn to convert each PIL image into a tensor via torchvision.transforms.ToTensor() before stacking. This ensures the DataLoader outputs batched tensors, and labels remain a list of strings. Please overwrite your test.py with this code, rerun, and let me know if you still get errors or see non-zero accuracy!

You said:
here is lmdbdataset for you to understand:
# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import glob
import io
import logging
import unicodedata
from pathlib import Path, PurePath
from typing import Callable, Optional, Union

import lmdb
from PIL import Image

from torch.utils.data import ConcatDataset, Dataset

from strhub.data.utils import CharsetAdapter

log = logging.getLogger(__name__)


def build_tree_dataset(root: Union[PurePath, str], *args, **kwargs):
    try:
        kwargs.pop('root')  # prevent 'root' from being passed via kwargs
    except KeyError:
        pass
    root = Path(root).absolute()
    log.info(f'dataset root:\t{root}')
    datasets = []
    for mdb in glob.glob(str(root / '**/data.mdb'), recursive=True):
        mdb = Path(mdb)
        ds_name = str(mdb.parent.relative_to(root))
        ds_root = str(mdb.parent.absolute())
        dataset = LmdbDataset(ds_root, *args, **kwargs)
        log.info(f'\tlmdb:\t{ds_name}\tnum samples: {len(dataset)}')
        datasets.append(dataset)
    return ConcatDataset(datasets)


class LmdbDataset(Dataset):
    """Dataset interface to an LMDB database.

    It supports both labelled and unlabelled datasets. For unlabelled datasets, the image index itself is returned
    as the label. Unicode characters are normalized by default. Case-sensitivity is inferred from the charset.
    Labels are transformed according to the charset.
    """

    def __init__(
        self,
        root: str,
        charset: str,
        max_label_len: int,
        min_image_dim: int = 0,
        remove_whitespace: bool = False,
        normalize_unicode: bool = False,
        unlabelled: bool = False,
        transform: Optional[Callable] = None,
    ):
        self._env = None
        self.root = root
        self.unlabelled = unlabelled
        self.transform = transform
        self.labels = []
        self.filtered_index_list = []
        self.num_samples = self._preprocess_labels(
            charset, remove_whitespace, normalize_unicode, max_label_len, min_image_dim
        )

    def __del__(self):
        if self._env is not None:
            self._env.close()
            self._env = None

    def _create_env(self):
        return lmdb.open(
            self.root, max_readers=1, readonly=True, create=False, readahead=False, meminit=False, lock=False
        )

    @property
    def env(self):
        if self._env is None:
            self._env = self._create_env()
        return self._env

    def _preprocess_labels(self, charset, remove_whitespace, normalize_unicode, max_label_len, min_image_dim):
        #charset_adapter = CharsetAdapter(charset)
        charset_adapter = lambda label: label
        with self._create_env() as env, env.begin() as txn:
            num_samples = int(txn.get('num-samples'.encode()))
            if self.unlabelled:
                return num_samples
            for index in range(num_samples):
                index += 1  # lmdb starts with 1
                label_key = f'label-{index:09d}'.encode()
                label = txn.get(label_key).decode()
                # Normally, whitespace is removed from the labels.
                if remove_whitespace:
                    label = ''.join(label.split())
                # Normalize unicode composites (if any) and convert to compatible ASCII characters
                if normalize_unicode:
                    label = unicodedata.normalize('NFKD', label).encode('ascii', 'ignore').decode()
                # Filter by length before removing unsupported characters. The original label might be too long.
                if len(label) > max_label_len:
                    continue
                label = charset_adapter(label)
                # We filter out samples which don't contain any supported characters
                #if not label:
                #    continue
                # Filter images that are too small.
                if min_image_dim > 0:
                    img_key = f'image-{index:09d}'.encode()
                    buf = io.BytesIO(txn.get(img_key))
                    w, h = Image.open(buf).size
                    if w < self.min_image_dim or h < self.min_image_dim:
                        continue
                self.labels.append(label)
                self.filtered_index_list.append(index)
        return len(self.labels)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        if self.unlabelled:
            label = index
        else:
            label = self.labels[index]
            index = self.filtered_index_list[index]

        img_key = f'image-{index:09d}'.encode()
        with self.env.begin() as txn:
            imgbuf = txn.get(img_key)
        buf = io.BytesIO(imgbuf)
        img = Image.open(buf).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label
ChatGPT said:
Now the dataset uses transform=ToTensor(), so each __getitem__ returns a tensor directly. The collate function then safely stacks them and packages labels as a list. Overwrite your test.py with this and rerun; it should load images correctly and produce real metrics. Let me know the output!

You said:
Using hard-coded charset (len=129):  0123456789۰۱۲٢۳۴۵۶۷۸۹اآأؤإءئۓۂئےۍېىيےیبپڀتٹثٿجچحخدڈذڎرڑزژسشصضطظعغفقڤڠݢکكڭگڬلمنںوۏههةۃۀہھڽضئکڤݢۏ-‌!"#$%&'()*+,./:;<=>?@[\]^_{|}~
/usr/local/lib/python3.11/dist-packages/timm/models/helpers.py:7: FutureWarning: Importing from timm.models.helpers is deprecated, please import via timm.models
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.models", FutureWarning)
Eval 'test':   0% 0/78 [00:00<?, ?it/s]
Traceback (most recent call last):
  File "/content/parseq_jawi/test2.py", line 110, in <module>
    main()
  File "/content/parseq_jawi/test2.py", line 85, in main
    for batch_idx, (imgs, labels) in enumerate(tqdm(loader, desc=f"Eval '{args.split}'")):
  File "/usr/local/lib/python3.11/dist-packages/tqdm/std.py", line 1181, in __iter__
    for obj in iterable:
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 701, in __next__
    data = self._next_data()
           ^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1465, in _next_data
    return self._process_data(data)
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1491, in _process_data
    data.reraise()
  File "/usr/local/lib/python3.11/dist-packages/torch/_utils.py", line 715, in reraise
    raise exception
RuntimeError: Caught RuntimeError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 351, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 55, in fetch
    return self.collate_fn(data)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/content/parseq_jawi/test2.py", line 67, in collate_fn
    imgs = torch.stack(imgs, dim=0)
           ^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: stack expects each tensor to be equal size, but got [3, 123, 176] at entry 0 and [3, 123, 203] at entry 1



here is the test.py that they use before, maybe you can get more idea
#!/usr/bin/env python3
# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import string
import sys
from dataclasses import dataclass

from tqdm import tqdm

import torch

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args


@dataclass
class Result:
    dataset: str
    num_samples: int
    accuracy: float
    ned: float
    confidence: float
    label_length: float


def print_results_table(results: list[Result], file=None):
    w = max(map(len, map(getattr, results, ['dataset'] * len(results))))
    w = max(w, len('Dataset'), len('Combined'))
    print('| {:<{w}} | # samples | Accuracy | 1 - NED | Confidence | Label Length |'.format('Dataset', w=w), file=file)
    print('|:{:-<{w}}:|----------:|---------:|--------:|-----------:|-------------:|'.format('----', w=w), file=file)
    c = Result('Combined', 0, 0, 0, 0, 0)
    for res in results:
        c.num_samples += res.num_samples
        c.accuracy += res.num_samples * res.accuracy
        c.ned += res.num_samples * res.ned
        c.confidence += res.num_samples * res.confidence
        c.label_length += res.num_samples * res.label_length
        print(
            f'| {res.dataset:<{w}} | {res.num_samples:>9} | {res.accuracy:>8.2f} | {res.ned:>7.2f} '
            f'| {res.confidence:>10.2f} | {res.label_length:>12.2f} |',
            file=file,
        )
    c.accuracy /= c.num_samples
    c.ned /= c.num_samples
    c.confidence /= c.num_samples
    c.label_length /= c.num_samples
    print('|-{:-<{w}}-|-----------|----------|---------|------------|--------------|'.format('----', w=w), file=file)
    print(
        f'| {c.dataset:<{w}} | {c.num_samples:>9} | {c.accuracy:>8.2f} | {c.ned:>7.2f} '
        f'| {c.confidence:>10.2f} | {c.label_length:>12.2f} |',
        file=file,
    )


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')")
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--cased', action='store_true', default=False, help='Cased comparison')
    parser.add_argument('--punctuation', action='store_true', default=False, help='Check punctuation')
    parser.add_argument('--std', action='store_true', default=False, help='Evaluate on standard benchmark datasets')
    parser.add_argument('--new', action='store_true', default=False, help='Evaluate on new benchmark datasets')
    parser.add_argument('--custom', action='store_true', default=True, help='Evaluate on custom personal datasets')
    parser.add_argument('--rotation', type=int, default=0, help='Angle of rotation (counter clockwise) in degrees.')
    parser.add_argument('--device', default='cuda')
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)

    charset_test = (" 0123456789۰۱۲٢۳۴۵۶۷۸۹اآأؤإءئۓۂئےۍېىيےیبپڀتٹثٿجچحخدڈذڎرڑزژسشصضطظعغفقڤڠݢکكڭگڬلمنںوۏههةۃۀہھڽضئکڤݢۏ-‌!\"#$%&'()*+,./:;<=>?@[\\]^_{|}~")
    kwargs.update({'charset_test': charset_test})
    print(f'Additional keyword arguments: {kwargs}')

    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
    hp = model.hparams
    datamodule = SceneTextDataModule(
        args.data_root,
        '_unused_',
        hp.img_size,
        hp.max_label_length,
        hp.charset_train,
        charset_test,
        args.batch_size,
        args.num_workers,
        False,
        rotation=args.rotation,
    )

    test_set = tuple()
    if args.std:
        test_set = SceneTextDataModule.TEST_BENCHMARK_SUB + SceneTextDataModule.TEST_BENCHMARK
    if args.custom:
        test_set += SceneTextDataModule.TEST_CUSTOM
    if args.new:
        test_set += SceneTextDataModule.TEST_NEW
    test_set = sorted(set(test_set))

    results = {}
    max_width = max(map(len, test_set))
    for name, dataloader in datamodule.test_dataloaders(test_set).items():
        total = 0
        correct = 0
        ned = 0
        confidence = 0
        label_length = 0
        for imgs, labels in tqdm(iter(dataloader), desc=f'{name:>{max_width}}'):
            res = model.test_step((imgs.to(model.device), labels), -1)['output']
            total += res.num_samples
            correct += res.correct
            ned += res.ned
            confidence += res.confidence
            label_length += res.label_length
        accuracy = 100 * correct / total
        mean_ned = 100 * (1 - ned / total)
        mean_conf = 100 * confidence / total
        mean_label_length = label_length / total
        results[name] = Result(name, total, accuracy, mean_ned, mean_conf, mean_label_length)

    result_groups = dict()
    if args.std:
        result_groups.update({'Benchmark (Subset)': SceneTextDataModule.TEST_BENCHMARK_SUB})
        result_groups.update({'Benchmark': SceneTextDataModule.TEST_BENCHMARK})
    if args.custom:
        result_groups.update({'Custom': SceneTextDataModule.TEST_CUSTOM})
    if args.new:
        result_groups.update({'New': SceneTextDataModule.TEST_NEW})
    with open(args.checkpoint + '.log.txt', 'w') as f:
        for out in [f, sys.stdout]:
            for group, subset in result_groups.items():
                print(f'{group} set:', file=out)
                print_results_table([results[s] for s in subset], out)
                print('\n', file=out)


if __name__ == '__main__':
    main()
ChatGPT said:

Thought for 23 seconds
I’ve fully rewritten the test script to reuse SceneTextDataModule—ensuring that every image is resized/padded exactly as in training—while hard-coding your charset. It pulls in only the jawi split from TEST_CUSTOM, runs test_step for each batch, and prints an exact-match table. Overwrite your test.py with this version and rerun:

python test.py \
  --checkpoint outputs/.../epoch=17-step=7255-val_accuracy=99.9347-val_NED=99.9909.ckpt \
  --data_root data \
  --batch_size 512 \
  --num_workers 4 \
  --device cuda
This should resolve the variable image sizes and give you a correct accuracy. Let me know how it goes!

You said:
can you print top 30 of the output so I can see whether it read correctly or not
ChatGPT said:
I added a “debug” pass over the ‘jawi’ loader to print the first 30 ground-truth vs. predicted strings before running the full evaluation. Run it, and you’ll see lines like:

--- First 30 GT vs Prediction ---
01: GT='...' | PR='...'
02: GT='...' | PR='...'
...
30: GT='...' | PR='...'
This will confirm whether your dataset is being read and decoded correctly. Let me know what you see!

You said:
Loading checkpoint on cuda with charset len=129
/usr/local/lib/python3.11/dist-packages/timm/models/helpers.py:7: FutureWarning: Importing from timm.models.helpers is deprecated, please import via timm.models
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.models", FutureWarning)

--- First 30 GT vs Prediction ---
Printing Debug:   0% 0/78 [00:01<?, ?it/s]
Traceback (most recent call last):
  File "/content/parseq_jawi/test2.py", line 132, in <module>
    main()
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/content/parseq_jawi/test2.py", line 105, in main
    preds = out.preds if hasattr(out, 'preds') else out.predictions
                                                    ^^^^^^^^^^^^^^^
AttributeError: 'BatchResult' object has no attribute 'predictions'
ChatGPT said:

Thought for 28 seconds
I’ve enhanced the debug section to list out all BatchResult attributes on the first batch and handle missing preds/predictions gracefully. Run it again, and you’ll see something like:

BatchResult attributes: [...]
01: GT='…' | PR='…'
…
That will reveal exactly what field holds your model outputs so we can print the top 30 correctly. Let me know what shows up!

You said:
Loading checkpoint on cuda with charset len=128
/usr/local/lib/python3.11/dist-packages/timm/models/helpers.py:7: FutureWarning: Importing from timm.models.helpers is deprecated, please import via timm.models
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.models", FutureWarning)
--- First 30 GT vs Prediction ---
Printing Debug:   0% 0/78 [00:00<?, ?it/s]BatchResult attributes: ['confidence', 'correct', 'label_length', 'loss', 'loss_numel', 'ned', 'num_samples']
ERROR: BatchResult has no 'preds' or 'predictions'
Printing Debug:   0% 0/78 [00:01<?, ?it/s]
ChatGPT said:

Thought for 27 seconds
I’ve replaced the debug block to simply print the first 30 ground-truth labels loaded by the DataModule. This will confirm your LMDB split is being read correctly:

--- First 30 GT labels (dataset read check) ---
01: GT='...' 
02: GT='...'
...
30: GT='...'
Run it again and verify the labels match what you expect. Once that’s confirmed, we can tackle printing predictions.

You said:
maybe read.py can help us to do the prediction:

#!/usr/bin/env python3
# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

from PIL import Image

import torch

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')")
    parser.add_argument('--images', nargs='+', help='Images to read')
    parser.add_argument('--device', default='cuda')
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)
    print(f'Additional keyword arguments: {kwargs}')

    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
    img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)

    for fname in args.images:
        # Load image and prepare for input
        image = Image.open(fname).convert('RGB')
        image = img_transform(image).unsqueeze(0).to(args.device)

        p = model(image).softmax(-1)
        pred, p = model.tokenizer.decode(p)
        print(f'{fname}: {pred[0]}')


if __name__ == '__main__':
    main()
ChatGPT said:

Thought for 24 seconds
I refactored the script to:

Use SceneTextDataModule for data loading and transforms.

In the debug section, forward each batch through the model directly (model(imgs)), apply softmax, then call model.tokenizer.decode to get raw predictions.

Print the first 30 GT vs. PR pairs for verification.

Finally, run the original test_step-based evaluation for full metrics.

Save this as your test.py and run. You’ll now get real predictions printed alongside the GT labels. Let me know the results!

You said:
it seems like our problem is extracting the ground truth:

--- First 30 GT vs Prediction ---
01: GT='' | PR='ڤاتوق'
02: GT='' | PR='دڤاتوق'
03: GT='' | PR='بيتالمال'
04: GT='' | PR='ڤتوڠوت'
05: GT='' | PR='مماتوتکن'
06: GT='' | PR='ڤاو'
07: GT='' | PR='دڤاءوق'
08: GT='' | PR='ڤاءوس'
09: GT='' | PR='ڤاءوس'
10: GT='' | PR='ڤاءوت'
11: GT='' | PR='برڤاءوت'
12: GT='' | PR='برباج'
13: GT='' | PR='ڤماءوت'
14: GT='' | PR='ڤرڤاءوتن'
15: GT='' | PR='ترڤاءوت'
16: GT='' | PR='برباج'
17: GT='' | PR='ڤاوه'
18: GT='-' | PR='برڤايا-ڤايا'
19: GT='' | PR='ڤايه٢'
20: GT='' | PR='ڤاير٢'
21: GT='' | PR='دباجاءي'
22: GT='' | PR='برڤايو'
23: GT='' | PR='ممايوکن'
24: GT='' | PR='ڤايوق'
25: GT='' | PR='ڤايوڠن'
26: GT='' | PR='ڤير'
27: GT='' | PR='ڤچاچل'
28: GT='' | PR='برڤچه'
29: GT='' | PR='دڤچهکن'
30: GT='' | PR='ممچه'
ChatGPT said:

Thought for 25 seconds
Let’s bypass the DataModule entirely and look straight at what your LMDB actually contains. Drop this little snippet into a cell or script and point it at your val/test LMDB:

import lmdb

LMDB_PATH = "data/val/jawi"  # ← adjust this to your split path
env = lmdb.open(LMDB_PATH, readonly=True, lock=False)
with env.begin() as txn:
    print("--- First 30 raw LMDB labels ---")
    for i in range(1, 31):
        key = f"label-{i:09d}".encode()
        val = txn.get(key)
        s = val.decode("utf-8", errors="replace") if val is not None else None
        print(f"{i:02d}: {repr(s)}")
This will print exactly what was stored under label-000000001…label-000000030. If those all come back empty (''), then the problem happened at dataset creation — your labels.txt or the createDataset run is writing blanks. If you see the proper Jawi strings there, then we know the loader is stripping them out somewhere.

Run that and let me know what the raw LMDB labels look like.

You said:
--- First 30 raw LMDB labels ---
01: 'ڤاتوق'
02: 'دڤاتوق'
03: 'بيتالمال'
04: 'ڤتوڠوت'
05: 'مماتوتکن'
06: 'ڤاو'
07: 'دڤاءوق'
08: 'ڤاءوس'
09: 'ڤاءوس'
10: 'ڤاءوت'
11: 'برڤاءوت'
12: 'برباج'
13: 'ڤماءوت'
14: 'ڤرڤاءوتن'
15: 'ترڤاءوت'
16: 'برباج'
17: 'ڤاوه'
18: 'برڤايا-ڤايا'
19: 'ڤايه٢'
20: 'ڤاير٢'
21: 'دباجاءي'
22: 'برڤايو'
23: 'ممايوکن'
24: 'ڤايوق'
25: 'ڤايوڠن'
26: 'ڤير'
27: 'ڤچاچل'
28: 'برڤچه'
29: 'دڤچهکن'
30: 'ممچه'
ChatGPT said:

Thought for 14 seconds
Great—your LMDB itself is fine. The blanks came from the SceneTextDataModule’s default filtering (it strips or normalizes labels). Let’s bypass it entirely and use LmdbDataset directly (which, by default, won’t remove or normalize anything). Here’s a minimal test.py that:

Loads the checkpoint with your hard-coded charset

Uses LmdbDataset pointing at data/{split}/jawi with remove_whitespace=False and normalize_unicode=False

Applies the same img_size transform you trained with

Prints the first 30 GT | PR pairs

Computes full accuracy, 1–NED, confidence, and label length

Save and run this instead of the DataModule version:

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
CHARSET = (
    "0123456789۰۱۲٢۳۴۵۶۷۸۹"
    "اآأؤإءئۓۂئےۍېىيےیبپڀتٹثٿجچحخدڈذڎرڑزژسشصضطظعغفقڤڠݢکكڭگڬلمنںوۏههةۃۀہھڽضئکڤݢۏ"
    "-\u200c!\"#$%&'()*+,./:;<=>?@[\\]^_`{|}~"
)

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
Why this works

No DataModule filtering: We set remove_whitespace=False and normalize_unicode=False so Jawi labels remain intact.

Exact resize/normalize: We reconstruct the same img_transform the model saw during training (you may need to tweak the Resize and Normalize mean/std to match your original).

Direct decode: We take model(imgs) → softmax → tokenizer.decode to get true predictions.

Run it, and you’ll see non‐empty GT strings and matching PRs if everything’s wired correctly.

You said:
Using charset (129 chars)
/usr/local/lib/python3.11/dist-packages/timm/models/helpers.py:7: FutureWarning: Importing from timm.models.helpers is deprecated, please import via timm.models
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.models", FutureWarning)

--- First 30 GT vs PR ---
Traceback (most recent call last):
  File "/content/parseq_jawi/test2.py", line 95, in <module>
    main()
  File "/content/parseq_jawi/test2.py", line 63, in main
    for imgs, labels in loader:
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 701, in __next__
    data = self._next_data()
           ^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1465, in _next_data
    return self._process_data(data)
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py", line 1491, in _process_data
    data.reraise()
  File "/usr/local/lib/python3.11/dist-packages/torch/_utils.py", line 715, in reraise
    raise exception
TypeError: Caught TypeError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 351, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
            ~~~~~~~~~~~~^^^^^
  File "/content/parseq_jawi/strhub/data/dataset.py", line 147, in __getitem__
    img = self.transform(img)
          ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torchvision/transforms/transforms.py", line 95, in __call__
    img = t(img)
          ^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1747, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torchvision/transforms/transforms.py", line 354, in forward
    return F.resize(img, self.size, self.interpolation, self.max_size, self.antialias)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torchvision/transforms/functional.py", line 477, in resize
    return F_pil.resize(img, size=output_size, interpolation=pil_interpolation)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torchvision/transforms/_functional_pil.py", line 250, in resize
    return img.resize(tuple(size[::-1]), interpolation)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/PIL/Image.py", line 2316, in resize
    return self._new(self.im.resize(size, resample, box))
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: 'list' object cannot be interpreted as an integer
ChatGPT said:
The error is coming from our hand-rolled Compose/Resize block: in your checkpoint hp.img_size is already a two-element list or tuple, so doing Resize((hp.img_size, hp.img_size)) winds up passing a nested list to PIL.Image.resize, which chokes on “list cannot be interpreted as an integer.”

Quick fix: ditch the manual Resize/Compose/Normalize and instead reuse exactly the same transform pipeline that your model was trained with:

from strhub.data.module import SceneTextDataModule

# … after loading your model and grabbing hp = model.hparams …

# Replace your Compose/Resize block with:
transform = SceneTextDataModule.get_transform(hp.img_size)
Then when you instantiate LmdbDataset, just pass that:

ds = LmdbDataset(
    root=lmdb_path,
    charset=CHARSET,
    max_label_len=hp.max_label_length,
    remove_whitespace=False,
    normalize_unicode=False,
    transform=transform
)
This one-liner gives you the correct resizing, padding, tensor conversion, and normalization exactly as in training, and will eliminate the “list object” TypeError. After making that swap, your loader should iterate without errors and you’ll see non-empty GT | PR pairs.

You said:
can you do so it print the loss and learning rate? here:

#!/usr/bin/env python3
# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.model_summary import summarize

from strhub.data.module import SceneTextDataModule
from strhub.models.base import BaseSystem
from strhub.models.utils import get_pretrained_weights


# Copied from OneCycleLR
def _annealing_cos(start, end, pct):
    'Cosine anneal from start to end as pct goes from 0.0 to 1.0.'
    cos_out = math.cos(math.pi * pct) + 1
    return end + (start - end) / 2.0 * cos_out


def get_swa_lr_factor(warmup_pct, swa_epoch_start, div_factor=25, final_div_factor=1e4) -> float:
    """Get the SWA LR factor for the given swa_epoch_start. Assumes OneCycleLR Scheduler."""
    total_steps = 1000  # Can be anything. We use 1000 for convenience.
    start_step = int(total_steps * warmup_pct) - 1
    end_step = total_steps - 1
    step_num = int(total_steps * swa_epoch_start) - 1
    pct = (step_num - start_step) / (end_step - start_step)
    return _annealing_cos(1, 1 / (div_factor * final_div_factor), pct)


@hydra.main(config_path='configs', config_name='main', version_base='1.2')
def main(config: DictConfig):
    trainer_strategy = 'auto'
    with open_dict(config):
        # Resolve absolute path to data.root_dir
        config.data.root_dir = hydra.utils.to_absolute_path(config.data.root_dir)
        # Special handling for GPU-affected config
        gpu = config.trainer.get('accelerator') == 'gpu'
        devices = config.trainer.get('devices', 0)
        if gpu:
            # Use mixed-precision training
            config.trainer.precision = 'bf16-mixed' if torch.get_autocast_gpu_dtype() is torch.bfloat16 else '16-mixed'
        if gpu and devices > 1:
            # Use DDP with optimizations
            trainer_strategy = DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True)
            # Scale steps-based config
            config.trainer.val_check_interval //= devices
            if config.trainer.get('max_steps', -1) > 0:
                config.trainer.max_steps //= devices

    # Special handling for PARseq
    if config.model.get('perm_mirrored', False):
        assert config.model.perm_num % 2 == 0, 'perm_num should be even if perm_mirrored = True'

    model: BaseSystem = hydra.utils.instantiate(config.model)
    # If specified, use pretrained weights to initialize the model
    if config.pretrained is not None:
        m = model.model if config.model._target_.endswith('PARSeq') else model
        m.load_state_dict(get_pretrained_weights(config.pretrained))
    print(summarize(model, max_depth=2))

    datamodule: SceneTextDataModule = hydra.utils.instantiate(config.data)

    checkpoint = ModelCheckpoint(
        monitor='val_accuracy',
        mode='max',
        save_top_k=3,
        save_last=True,
        filename='{epoch}-{step}-{val_accuracy:.4f}-{val_NED:.4f}',
    )
    swa_epoch_start = 0.75
    swa_lr = config.model.lr * get_swa_lr_factor(config.model.warmup_pct, swa_epoch_start)
    swa = StochasticWeightAveraging(swa_lr, swa_epoch_start)
    cwd = (
        HydraConfig.get().runtime.output_dir
        if config.ckpt_path is None
        else str(Path(config.ckpt_path).parents[1].absolute())
    )
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        logger=TensorBoardLogger(cwd, '', '.'),
        strategy=trainer_strategy,
        enable_model_summary=False,
        callbacks=[checkpoint, swa],
    )
    trainer.fit(model, datamodule=datamodule, ckpt_path=config.ckpt_path)


if __name__ == '__main__':
    main()
ChatGPT said:

Thought for 10 seconds
I’ve added a PrintMetricsCallback that hooks into on_train_batch_end, grabbing the batch loss and the optimizer’s current learning rate, and prints them each step. It’s registered alongside your existing checkpoint and SWA callbacks. Save this as train_with_metrics.py and run it the same way—now you’ll see:

[Batch 0] loss=2.0456 lr=1.000000e-03
[Batch 1] loss=1.9823 lr=9.980001e-04
...
in your training logs.

You said:
print also the accuracy and epoch, the loss is also calculated at epoch
ChatGPT said:

Thought for 9 seconds

Answer in chat instead
I’ve extended PrintMetricsCallback with:

on_train_epoch_end: prints epoch number, average training loss (train_loss), accuracy (train_accuracy or fallback to_







Search

Deep research

Create image



No file chosenNo file chosen
ChatGPT can make mistakes. Check important info.


Train With Metrics
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
78
79
80
81
82
83
84
85
86
87
88
89
90
91
92
93
94
95
96
97
98
99
100
101
102
103
104
105
106
107
108
109
110
111
112
113
114
115
116
117
118
119
120
121
122
123
124
125
126
127
128
129
130
131
132
#!/usr/bin/env python3
# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.model_summary import summarize

from strhub.data.module import SceneTextDataModule
from strhub.models.base import BaseSystem
from strhub.models.utils import get_pretrained_weights

# Copied from OneCycleLR

def _annealing_cos(start, end, pct):
    'Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0.'
    cos_out = math.cos(math.pi * pct) + 1
    return end + (start - end) / 2.0 * cos_out


def get_swa_lr_factor(warmup_pct, swa_epoch_start, div_factor=25, final_div_factor=1e4) -> float:
    """Get the SWA LR factor for the given `swa_epoch_start`. Assumes OneCycleLR Scheduler."""
    total_steps = 1000
    start_step = int(total_steps * warmup_pct) - 1
    end_step = total_steps - 1
    step_num = int(total_steps * swa_epoch_start) - 1
    pct = (step_num - start_step) / (end_step - start_step)
    return _annealing_cos(1, 1 / (div_factor * final_div_factor), pct)


class PrintMetricsCallback(Callback):
    """Callback to print batch & epoch metrics including loss, accuracy, learning rate."""
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        loss = outputs['loss'] if isinstance(outputs, dict) and 'loss' in outputs else None
        optim = trainer.optimizers[0]
        current_lr = optim.param_groups[0].get('lr', None)
        print(f"[Batch {batch_idx}] loss={loss:.4f} lr={current_lr:.6e}")

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics
        avg_loss = metrics.get('train_loss') or metrics.get('loss')
        avg_acc = metrics.get('train_accuracy') or metrics.get('train_acc') or metrics.get('val_accuracy')
        optim = trainer.optimizers[0]
        lr = optim.param_groups[0].get('lr', None)
        print(f"[Epoch {epoch}] avg_loss={avg_loss:.4f} accuracy={avg_acc:.2f} lr={lr:.6e}")

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics
        val_acc = metrics.get('val_accuracy')
        print(f"[Epoch {epoch}] val_accuracy={val_acc:.2f}")


@hydra.main(config_path='configs', config_name='main', version_base='1.2')
def main(config: DictConfig):
    trainer_strategy = 'auto'
    with open_dict(config):
        config.data.root_dir = hydra.utils.to_absolute_path(config.data.root_dir)
        gpu = config.trainer.get('accelerator') == 'gpu'
        devices = config.trainer.get('devices', 0)
        if gpu:
            config.trainer.precision = 'bf16-mixed' if torch.get_autocast_gpu_dtype() is torch.bfloat16 else '16-mixed'
        if gpu and devices > 1:
            trainer_strategy = DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True)
            config.trainer.val_check_interval //= devices
            if config.trainer.get('max_steps', -1) > 0:
                config.trainer.max_steps //= devices

    if config.model.get('perm_mirrored', False):
        assert config.model.perm_num % 2 == 0, 'perm_num should be even if perm_mirrored = True'

    model: BaseSystem = hydra.utils.instantiate(config.model)
    if config.pretrained is not None:
        m = model.model if config.model._target_.endswith('PARSeq') else model
        m.load_state_dict(get_pretrained_weights(config.pretrained))
    print(summarize(model, max_depth=2))

    datamodule: SceneTextDataModule = hydra.utils.instantiate(config.data)

    checkpoint = ModelCheckpoint(
        monitor='val_accuracy',
        mode='max',
        save_top_k=3,
        save_last=True,
        filename='{epoch}-{step}-{val_accuracy:.4f}-{val_NED:.4f}',
    )
    swa_epoch_start = 0.75
    swa_lr = config.model.lr * get_swa_lr_factor(config.model.warmup_pct, swa_epoch_start)
    swa = StochasticWeightAveraging(swa_lr, swa_epoch_start)

    metrics_cb = PrintMetricsCallback()

    cwd = (
        HydraConfig.get().runtime.output_dir
        if config.ckpt_path is None
        else str(Path(config.ckpt_path).parents[1].absolute())
    )
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        logger=TensorBoardLogger(cwd, '', '.'),
        strategy=trainer_strategy,
        enable_model_summary=False,
        callbacks=[metrics_cb, checkpoint, swa],
    )
    trainer.fit(model, datamodule=datamodule, ckpt_path=config.ckpt_path)


if __name__ == '__main__':
    main()

ChatGPT is still generating a response...
