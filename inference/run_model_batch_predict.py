import ray
import ray.data
import pandas as pd
import torch

class PredictCallable:
    def __init__(self, model_id: str, amp_enabled, amp_dtype, max_new_tokens):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import intel_extension_for_pytorch as ipex
        import os
        self.amp_enabled = amp_enabled
        self.amp_dtype = amp_dtype
        self.max_new_tokens = max_new_tokens
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            low_cpu_mem_usage=True,
            cache_dir="/mnt/DP_disk3/GPTJ_Model"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = model.eval()
        # to channels last
        model = model.to(memory_format=torch.channels_last)
        # to ipex
        self.model = ipex.optimize(model, dtype=amp_dtype, inplace=True)

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        tokenized = self.tokenizer(
            list(batch["prompt"]), return_tensors="pt"
        )
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask

        gen_tokens = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=0.9,
            max_length=self.max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return pd.DataFrame(
            self.tokenizer.batch_decode(gen_tokens), columns=["responses"]
        )

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser('GPT-J generation script', add_help=False)
    parser.add_argument('--precision', default='bf16', type=str, help="fp32 or bf16")
    parser.add_argument('--model', default='EleutherAI/gpt-j-6B', type=str, help="model name or path")
    parser.add_argument('--max-new-tokens', default=100, type=int, help="output max new tokens")
    args = parser.parse_args()
    amp_enabled = True if args.precision != "fp32" else False
    amp_dtype = torch.bfloat16 if args.precision != "fp32" else torch.float32

    ray.init(address="auto")
    prompt = (
        "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
        "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
        "researchers was the fact that the unicorns spoke perfect English."
    )
    ds = ray.data.from_pandas(pd.DataFrame([prompt] * 10, columns=["prompt"]))
    preds = (
        ds
        .repartition(100)
        .map_batches(
            PredictCallable,
            batch_size=4,
            fn_constructor_kwargs=dict(
                model_id=args.model,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                max_new_tokens=args.max_new_tokens
            ),
            compute="actors"
        )
    )
    print(preds.take_all())