# for classifying emotions

out_dir = "out-emotion"
eval_interval = 100  # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10  # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False  # override via command line if you like
wandb_project = "emotion"
wandb_run_name = "nano-gpt"

dataset = "emotion"
gradient_accumulation_steps = 1
batch_size = 64
block_size = 120  # context of up to 240 previous characters (max tweet length)

# baby GPT model :)
model_type = "gpt4sentimentanalysis"
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-4  # with baby networks can afford to go a bit higher
max_iters = 5_000
lr_decay_iters = 5_000  # make equal to max_iters usually
min_lr = 1e-5  # learning_rate / 10 usually
beta2 = 0.99  # make a bit bigger because number of tokens per iter is small

warmup_iters = 100  # not super necessary potentially
