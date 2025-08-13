from modeling_beacon_gpt import BeaconGPT
from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset
import tiktoken
import torch
import torch.distributed as dist

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


example = stream_input_ids(ds_iter, 128, "cpu")

print(example.shape)

model = BeaconGPT(
    vocab_size=tokenizer.n_vocab,
    hidden_size=64,
    n_layer=4,
    n_head=4,
    max_seq_len=128,
)
model.to("cpu")
example = example.to("cpu")
print(model)
logits, loss = model(example, example)
print(logits.shape)
print(loss)
