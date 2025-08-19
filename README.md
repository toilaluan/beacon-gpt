# Reimplement Activation Beacon: Sparse Attention + KV Checkpointing

<img width="582" height="275" alt="image" src="https://github.com/user-attachments/assets/19949705-d1e8-4cc7-9974-8944a66e2b77" />

With the release of `flex_attention` (pytorch), it's easier to implement scalable training for sparse attention
I also want to pretraining the method instead of tuning from instruct model (original work)
Current implementation has some techniques reference to https://github.com/KellerJordan/modded-nanogpt
