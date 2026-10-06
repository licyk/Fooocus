These tokenizers are copied from ComfyUI commit
`d49e888586dd8ae012c0667b33466b815fee07f7`, without vocabulary changes:

- `qwen/`: `comfy/text_encoders/qwen25_tokenizer/`, originally
  [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B).
- `t5/`: `comfy/text_encoders/t5_tokenizer/`, originally
  [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl).

Qwen and T5 tokenizer assets are licensed under Apache-2.0; see
`../../ldm/anima/LICENSE-APACHE-2.0`. No neural text encoder weights are bundled.
The Qwen3 text encoder uses the Qwen2 tokenizer vocabulary. T5 supplies token IDs
to Anima's adapter; it does not load a T5 model.
