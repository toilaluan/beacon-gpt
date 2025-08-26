# Reimplement Activation Beacon: Sparse Attention + KV Checkpointing

<img width="582" height="275" alt="image" src="https://github.com/user-attachments/assets/19949705-d1e8-4cc7-9974-8944a66e2b77" />

With the release of `flex_attention` (pytorch), it's easier to implement scalable training for sparse attention
I also want to pretraining the method instead of tuning from instruct model (original work)
Current implementation has some techniques reference to https://github.com/KellerJordan/modded-nanogpt

### Update 1
I successfully run exps on GPT2 architecture. In about 1b tokens, it shows that we can do 16x compress without losing much ppl and 8x compress even achieve better loss convergence.
<img width="1237" height="642" alt="image" src="https://github.com/user-attachments/assets/d1a78fc0-d476-4c3e-b2f6-bf1bb87b127e" />



<img width="640" height="512" alt="image" src="./attention_scores.png" />
