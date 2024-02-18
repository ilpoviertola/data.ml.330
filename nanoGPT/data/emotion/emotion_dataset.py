from datasets import load_dataset
from torch.utils.data import Dataset
import tiktoken
import torch.nn.functional as F
import torch


# copied from https://huggingface.co/datasets/dair-ai/emotion
LABEL_MAP = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}


class EmotionDataset(Dataset):
    def __init__(self, split, block_size):
        self.data = load_dataset("dair-ai/emotion", "split", split=split)
        self.block_size = block_size - 1  # subtract 1 for the OFF token
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.pad_token = self.tokenizer.max_token_value + 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.tokenizer.encode_ordinary(self.data[idx]["text"])
        text.append(self.tokenizer.eot_token)  # add the end of text token
        text = torch.tensor(text)
        non_padded_len = len(text)
        # pad the text to the block size so default collate function works
        text = F.pad(
            text,
            (0, self.block_size - text.shape[-1]),
            mode="constant",
            value=self.pad_token,
        )
        text = text[: self.block_size]  # truncate in case the text is too long
        label = self.data[idx]["label"]
        item = {
            "text": text,
            "label": label,
            "label_str": LABEL_MAP[label],
            "non_padded_len": non_padded_len,
        }
        return item
