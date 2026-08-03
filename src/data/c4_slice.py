import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset


tknzr = tiktoken.get_encoding("gpt2")

# 8 of allenai/c4 en's 1024 train shards ~= 1.2B gpt2 tokens -- comparable to
# the redpajama/minipile tier budgets. gpt2 tokenizer (NOT the gated Llama-2
# one that src/data/c4.py uses) keeps the 50304 vocab consistent with all
# other QMC-SR experiment datasets.
N_TRAIN_SHARDS = 8


def get_c4_slice_data(datasets_dir, num_proc=40):
    C4S_DATA_PATH = os.path.join(datasets_dir, "c4slice/")
    if not os.path.exists(os.path.join(C4S_DATA_PATH, "train.bin")):
        os.makedirs(C4S_DATA_PATH, exist_ok=True)
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

        def process(example):
            ids = tknzr.encode_ordinary(example["text"])
            ids.append(tknzr.eot_token)
            return {"ids": ids, "len": len(ids)}

        tokenized = {
            name: dset.map(
                process,
                remove_columns=[c for c in dset.column_names if c != "ids"],
                desc=f"tokenizing c4slice {name}",
                num_proc=num_proc,
            )
            for name, dset in dataset.items()
        }

        for split, dset in tokenized.items():
            arr_len = np.sum(dset["len"])
            filename = os.path.join(C4S_DATA_PATH, f"{split}.bin")
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
        "train": os.path.join(C4S_DATA_PATH, "train.bin"),
        "val": os.path.join(C4S_DATA_PATH, "val.bin"),
    }
