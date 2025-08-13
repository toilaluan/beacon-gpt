from modeling_beacon_gpt import BeaconGPT
from torch.nn.attention.flex_attention import create_block_mask
from datasets import load_dataset
import tiktoken
import torch
import time

tokenizer = tiktoken.get_encoding("gpt2")
print(tokenizer.n_vocab)

ds = load_dataset("alkibijad/fineweb-edu-sample-10BT-gpt2tokenized", streaming=True)
ds_iter = iter(ds["train_000001"])


def stream_input_ids(ds_iter, max_seq_len, device):
    ids = []
    while len(ids) < max_seq_len:
        ids.extend(next(ds_iter)["values"])
    ids = ids[:max_seq_len]
    ids = torch.tensor([ids], dtype=torch.long, device=device)
    return ids


MAX_SEQ_LEN = 1024

example = stream_input_ids(ds_iter, 128, "cpu")

print(example.shape)

model = BeaconGPT(
    vocab_size=tokenizer.n_vocab,
    hidden_size=64,
    n_layer=4,
    n_head=4,
    max_seq_len=MAX_SEQ_LEN,
)

# count the number of parameters
print(sum(p.numel() for p in model.parameters()))

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
# model = torch.compile(model)

prefix_test = "Hello,"

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
    data = stream_input_ids(ds_iter, MAX_SEQ_LEN, device)
    start_time = time.time()
    logits, loss = model(data, data, mask)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(i, loss, time.time() - start_time)
    if i % 100 == 0:
        output = model.generate(prefix_test_ids, max_new_tokens=16, device=device)
        print(tokenizer.decode(output))
