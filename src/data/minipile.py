import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset


tknzr = tiktoken.get_encoding("gpt2")


def get_minipile_data(datasets_dir, num_proc=40):
    """MiniPile (JeanKaddour/minipile): ~1.5B gpt2 tokens of curated,
    deduplicated Pile data. Uses the dataset's own splits: train -> train.bin,
    test (10k docs) -> val.bin (the 500-doc validation split is too small for
    our eval batches)."""
    MP_DATA_PATH = os.path.join(datasets_dir, "minipile/")
    if not os.path.exists(os.path.join(MP_DATA_PATH, "train.bin")):
        os.makedirs(MP_DATA_PATH, exist_ok=True)
        dataset = load_dataset("JeanKaddour/minipile")

        splits = {"train": dataset["train"], "val": dataset["test"]}

        def process(example):
            ids = tknzr.encode_ordinary(example["text"])
            ids.append(tknzr.eot_token)
            return {"ids": ids, "len": len(ids)}

        tokenized = {
            name: dset.map(
                process,
                remove_columns=["text"],
                desc=f"tokenizing minipile {name}",
                num_proc=num_proc,
            )
            for name, dset in splits.items()
        }

        for split, dset in tokenized.items():
            arr_len = np.sum(dset["len"])
            filename = os.path.join(MP_DATA_PATH, f"{split}.bin")
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
        "train": os.path.join(MP_DATA_PATH, "train.bin"),
        "val": os.path.join(MP_DATA_PATH, "val.bin"),
    }
