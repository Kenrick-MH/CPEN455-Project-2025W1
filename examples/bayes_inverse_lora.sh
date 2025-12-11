uv run -m examples.bayes_inverse \
--method lora \
--max_seq_len 256 \
--batch_size 8 \
--lora_dim 24 \
--lora_sigma 2.0 \
--num_iterations 500
