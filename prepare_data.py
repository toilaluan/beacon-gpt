from datasets import load_dataset
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

ds = load_dataset(
    "alkibijad/fineweb-edu-sample-10BT-gpt2tokenized", streaming=False, num_proc=24
)
