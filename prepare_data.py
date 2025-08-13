from datasets import load_dataset
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

ds = load_dataset("alkibijad/fineweb-edu-sample-10BT-gpt2tokenized", streaming=True)

iter_ds = iter(ds["train_000001"])

for i in range(10):
    ids = next(iter_ds)["values"]
    print(len(ids))
    print(tokenizer.decode(ids))
    print("-" * 100)
