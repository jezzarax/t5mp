Major parts of the code are not mine, just adapted from https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling

```
ptlm train-model \
 --output_dir="./t5mumo" \
 --model_type="t5" \
 --config_name="./t5mumo" \
 --tokenizer_name="./t5mumo" \
 --dataset_name="wikitext" \
 --dataset_config_name="wikitext-103-v1" \
 --max_seq_length="512" \
 --per_device_train_batch_size="32" \
 --per_device_eval_batch_size="32" \
 --adafactor \
 --learning_rate="0.005" \
 --weight_decay="0.001" \
 --warmup_steps="2000" \
 --overwrite_output_dir \
 --logging_steps="500" \
 --save_steps="10000" \
 --eval_steps="2500"
```