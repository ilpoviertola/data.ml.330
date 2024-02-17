import os
from multiprocessing import cpu_count

from datasets import load_dataset
import tiktoken
from tqdm import tqdm
import numpy as np


if __name__ == "__main__":
    num_proc = cpu_count() // 2

    save_dir = (
        "/home/hdd/data/emotions/gpt2_tokenized"  # change this to your own directory
    )
    assert os.path.exists(save_dir), f"{save_dir} does not exist"

    dataset = load_dataset("dair-ai/emotion", num_proc=num_proc)

    enc = tiktoken.get_encoding("gpt2")

    def process(item):
        ids = enc.encode_ordinary(
            item["text"]
        )  # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token)  # add the end of text token, e.g. 50256 for gpt2 bpe
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        out = {"ids": ids, "len": len(ids)}
        return out

    tokenized = dataset.map(
        process,
        remove_columns=["text"],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        filename = os.path.join(save_dir, f"{split}.bin")
        dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
        total_batches = 1024

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
