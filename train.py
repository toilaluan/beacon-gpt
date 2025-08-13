from modeling_beacon_gpt import BeaconGPT
from torch.nn.attention.flex_attention import create_block_mask
from datasets import load_dataset
import tiktoken
import torch
import time
import wandb

wandb.init(project="beacon-gpt", entity="toilaluan")

tokenizer = tiktoken.get_encoding("gpt2")
print(tokenizer.n_vocab)

ds = load_dataset("alkibijad/fineweb-edu-sample-10BT-gpt2tokenized", streaming=True)


def stream_input_ids(ds, max_seq_len, device):
    shard_iter = iter(ds)
    data_iter = iter(ds[shard_iter])
    ids = []
    while len(ids) < max_seq_len:
        try:
            ids.extend(next(data_iter)["values"])
        except StopIteration:
            shard_iter = next(shard_iter)
            data_iter = iter(ds[shard_iter])
    ids = ids[:max_seq_len]
    ids = torch.tensor([ids], dtype=torch.long, device=device)
    return ids


MAX_SEQ_LEN = 2048

example = stream_input_ids(ds, 128, "cpu")

print(example.shape)

model = BeaconGPT(
    vocab_size=tokenizer.n_vocab,
    hidden_size=768,
    n_layer=12,
    n_head=12,
    max_seq_len=MAX_SEQ_LEN,
)

# count the number of parameters
print(sum(p.numel() for p in model.parameters()))

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
model = torch.compile(model)

prefix_test = "Hello, "

prefix_test_ids = torch.tensor(
    [tokenizer.encode(prefix_test)], dtype=torch.long, device="cpu"
)
print(prefix_test_ids.shape)

device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device)
prefix_test_ids = prefix_test_ids.to(device)

mask = create_block_mask(
    lambda b, h, q_idx, kv_idx: q_idx >= kv_idx,
    B=None,
    H=None,
    Q_LEN=MAX_SEQ_LEN,
    KV_LEN=MAX_SEQ_LEN,
    device=device,
)

for i in range(1000):
    data = stream_input_ids(ds, MAX_SEQ_LEN, device)
    start_time = time.time()
    logits, loss = model(data, data, mask)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if i % 50 == 0:
        print(i, loss.item(), time.time() - start_time)
        wandb.log({"loss": loss.item()})
    if i % 100 == 0:
        output = model.generate(prefix_test_ids, max_new_tokens=16, device=device)
        print(prefix_test, tokenizer.decode(output))
