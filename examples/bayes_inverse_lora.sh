uv run -m examples.bayes_inverse \
--method lora \
--max_seq_len 512 \
--batch_size 6 \
--lora_dim 32 \
--lora_sigma 1.0 \
--num_iterations 100 \
--learning_rate 1e-7\
