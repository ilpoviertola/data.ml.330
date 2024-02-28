"""Tää on muuten perseestä :D... Miks kukaan tekis tälläin?"""

import os

import torch
from torch.utils.data import DataLoader

from model import GPTConfig, GPT, GPT4SentimentAnalysis
from data.emotion.emotion_dataset import EmotionDataset, LABEL_MAP


def test():
    # -----------------------------------------------------------------------------
    # config
    # -----------------------------------------------------------------------------
    out_dir = "./out-emotion"
    device = torch.device("cpu")
    block_size = 120
    # model
    n_layer = 12
    n_head = 12
    n_embd = 768
    dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
    bias = False  # do we use bias inside LayerNorm and Linear layers?
    model_type = "gpt4sentimentanalysis"
    # -----------------------------------------------------------------------------
    config_keys = [
        k
        for k, v in globals().items()
        if not k.startswith("_") and isinstance(v, (int, float, bool, str))
    ]
    exec(
        open("nanoGPT/configurator.py").read()
    )  # overrides from command line or config file
    config = {k: globals()[k] for k in config_keys}  # will be useful for logging

    # -----------------------------------------------------------------------------
    # model
    # -----------------------------------------------------------------------------

    model_args = dict(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        bias=bias,
        vocab_size=None,
        dropout=dropout,
    )  # start with model_args from command line

    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, "ckpt-baseline.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    print(model_type)
    model = (
        GPT(gptconf) if model_type.lower() == "gpt" else GPT4SentimentAnalysis(gptconf)
    )
    state_dict = checkpoint["model"]
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]

    print(f"model loaded from {out_dir}")
    print(f"\twas trained for {iter_num} iterations")
    print(f"\tbest validation loss: {best_val_loss:.4f}")

    # -----------------------------------------------------------------------------
    # dataset
    # -----------------------------------------------------------------------------
    dataset = EmotionDataset("test", block_size)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # -----------------------------------------------------------------------------
    # inference (test)
    # -----------------------------------------------------------------------------
    model.eval()
    correct_preds = 0
    for idx, batch in enumerate(dataloader):
        text = batch["text"].to(device)
        label_str = batch["label_str"]
        with torch.no_grad():
            logits, _ = model(text)
        # get the most likely next token
        pred_label = logits.argmax(-1).item()
        if pred_label == batch["label"]:
            correct_preds += 1
        if idx % 25 == 0:
            print(f"{idx}/{len(dataset)}")
            # print every 25th example
            print(f"\ttrue label: {label_str[0]}, pred label: {LABEL_MAP[pred_label]}")

    print(f"total accuracy: {(correct_preds / len(dataset)*100):.2f} %")


if __name__ == "__main__":
    test()
