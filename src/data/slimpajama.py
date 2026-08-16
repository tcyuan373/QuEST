from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset
import os


tknzr = tiktoken.get_encoding("gpt2")


def get_slimpajama_data(datasets_dir, num_proc=40):
    from data.c4_slice import _bin_exists_nfs_robust

    SPJ_DATA_PATH = os.path.join(datasets_dir, "slimpajama6B/")
    train_bin = os.path.join(SPJ_DATA_PATH, "train.bin")
    # same NFS existence-race + single-builder guard as c4_slice.py: a
    # requeue storm must never trigger a parallel rebuild over live memmaps
    # (this dataset has already been corrupted once -- see
    # rebuild_slimpajama_full.py history)
    if not _bin_exists_nfs_robust(train_bin, SPJ_DATA_PATH):
        lock = os.path.join(datasets_dir, "slimpajama6B.build.lock")
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            print(f"slimpajama: waiting on build lock {lock}")
            import time

            while os.path.exists(lock):
                time.sleep(30)
            assert os.path.exists(train_bin), (
                "slimpajama build lock released but train.bin still missing"
            )
            return {
                "train": train_bin,
                "val": os.path.join(SPJ_DATA_PATH, "val.bin"),
            }
        try:
            os.makedirs(SPJ_DATA_PATH, exist_ok=True)
            _build_slimpajama(SPJ_DATA_PATH, num_proc)
        finally:
            os.remove(lock)

    return {
        "train": train_bin,
        "val": os.path.join(SPJ_DATA_PATH, "val.bin"),
    }


def _build_slimpajama(SPJ_DATA_PATH, num_proc):
    dataset = load_dataset("DKYoon/SlimPajama-6B")

    split_dataset = dataset["train"].train_test_split(
        test_size=0.0005, seed=2357, shuffle=True
    )
    split_dataset["val"] = split_dataset.pop("test")

    def process(example):
        ids = tknzr.encode_ordinary(
            example["text"]
        )  # encode_ordinary ignores any special tokens
        ids.append(
            tknzr.eot_token
        )  # add the end of text token, e.g. 50256 for gpt2 bpe
        out = {"ids": ids, "len": len(ids)}
        return out

    # tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=["text"],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"])
        filename = os.path.join(SPJ_DATA_PATH, f"{split}.bin")
        dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
        total_batches = min(1024, len(dset))

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
            # Batch together samples for faster write
            batch = dset.shard(
                num_shards=total_batches, index=batch_idx, contiguous=True
            ).with_format("numpy")
            arr_batch = np.concatenate(batch["ids"])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()


def get_slimpajama_chunk1(datasets_dir, num_proc=40):
    SPJ_DATA_PATH = os.path.join(datasets_dir, "slimpajama6B/")
    SPJ_CHUNK_1_DATA_PATH = os.path.join(SPJ_DATA_PATH, "chunk1")
    if not os.path.exists(os.path.join(SPJ_CHUNK_1_DATA_PATH, "train.bin")):
        os.makedirs(SPJ_DATA_PATH, exist_ok=True)
        dataset = load_dataset("cerebras/SlimPajama-627B", split="train/chunk1")

        split_dataset = dataset["train"].train_test_split(
            test_size=0.0005, seed=2357, shuffle=True
        )
        split_dataset["val"] = split_dataset.pop("test")

        def process(example):
            ids = tknzr.encode_ordinary(
                example["text"]
            )  # encode_ordinary ignores any special tokens
            ids.append(
                tknzr.eot_token
            )  # add the end of text token, e.g. 50256 for gpt2 bpe
            out = {"ids": ids, "len": len(ids)}
            return out

        # tokenize the dataset
        tokenized = split_dataset.map(
            process,
            remove_columns=["text"],
            desc="tokenizing the splits",
            num_proc=num_proc,
        )

        # concatenate all the ids in each dataset into one large file we can use for training
        for split, dset in tokenized.items():
            arr_len = np.sum(dset["len"])
            filename = os.path.join(SPJ_DATA_PATH, f"{split}.bin")
            dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
            arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
            total_batches = min(1024, len(dset))

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
                # Batch together samples for faster write
                batch = dset.shard(
                    num_shards=total_batches, index=batch_idx, contiguous=True
                ).with_format("numpy")
                arr_batch = np.concatenate(batch["ids"])
                # Write into mmap
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()

    return {
        "train": os.path.join(SPJ_DATA_PATH, "train.bin"),
        "val": os.path.join(SPJ_DATA_PATH, "val.bin"),
    }
