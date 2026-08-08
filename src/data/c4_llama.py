import os
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset


# Same 8/1024 allenai/c4 en train shards as c4_slice.py, but tokenized with the
# Llama-2 tokenizer so absolute losses are on the SAME per-token scale as the
# QuEST paper's runs (their src/data/c4.py hardcodes meta-llama/Llama-2-7b-hf;
# that repo is gated, so we load hf-internal-testing/llama-tokenizer — the
# ungated copy of the identical sentencepiece model, vocab 32000, BOS id 1).
# Tokenization mirrors their process() exactly: add_special_tokens=True
# (BOS prepended per document, no EOS appended).
N_TRAIN_SHARDS = 8

_hf_tknzr = None


def _get_tokenizer():
    global _hf_tknzr
    if _hf_tknzr is None:
        _hf_tknzr = AutoTokenizer.from_pretrained(
            "hf-internal-testing/llama-tokenizer"
        )
    return _hf_tknzr


def get_c4_llama_data(datasets_dir, num_proc=40):
    C4L_DATA_PATH = os.path.join(datasets_dir, "c4llama/")
    if not os.path.exists(os.path.join(C4L_DATA_PATH, "train.bin")):
        os.makedirs(C4L_DATA_PATH, exist_ok=True)
        dataset = load_dataset(
            "allenai/c4",
            data_files={
                "train": [
                    f"en/c4-train.{i:05d}-of-01024.json.gz"
                    for i in range(N_TRAIN_SHARDS)
                ],
                "val": ["en/c4-validation.00000-of-00008.json.gz"],
            },
        )

        hf_tknzr = _get_tokenizer()

        def process(example):
            ids = hf_tknzr.encode(
                text=example["text"],
                add_special_tokens=True,
                padding=False,
                truncation=False,
            )
            return {"ids": ids, "len": len(ids)}

        tokenized = {
            name: dset.map(
                process,
                remove_columns=[c for c in dset.column_names if c != "ids"],
                desc=f"tokenizing c4llama {name}",
                num_proc=num_proc,
            )
            for name, dset in dataset.items()
        }

        for split, dset in tokenized.items():
            arr_len = np.sum(dset["len"])
            filename = os.path.join(C4L_DATA_PATH, f"{split}.bin")
            dtype = np.uint16
            arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
            total_batches = min(1024, len(dset))

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
                batch = dset.shard(
                    num_shards=total_batches, index=batch_idx, contiguous=True
                ).with_format("numpy")
                arr_batch = np.concatenate(batch["ids"])
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()

    return {
        "train": os.path.join(C4L_DATA_PATH, "train.bin"),
        "val": os.path.join(C4L_DATA_PATH, "val.bin"),
    }
