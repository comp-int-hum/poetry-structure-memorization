import argparse
from transformers import AutoTokenizer, LlamaForCausalLM
import torch
import json
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help=".txt input")
    parser.add_argument("--window", type=int, default=50, help="window (in token size) to provide to the model")
    parser.add_argument("--max_tokens", type=int, default=50, help="max new tokens to generate")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--output", dest="output", help="Output files")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--random_state", type=int, default=20)
    args, rest = parser.parse_known_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    with open(args.input, "rt") as p_in:
        poem = p_in.read()

    tokenized = tokenizer(poem, max_length=args.window, stride=48, truncation=True, padding=True, return_overflowing_tokens=True, return_tensors="pt").input_ids
    model = LlamaForCausalLM.from_pretrained(args.model, device_map="auto")
    batched = torch.split(tokenized, args.batch_size)
    with open(args.output, "wt") as jout:
        for batch in batched:
            stems = tokenizer.batch_decode(batch, clean_up_tokenization_spaces=False, skip_special_tokens=True) 
            res = model.generate(batch.to(args.device), max_new_tokens=args.max_tokens, min_new_tokens=args.max_tokens, return_dict_in_generate=True, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            generated = tokenizer.batch_decode(res.sequences[:, batch.shape[1]:], clean_up_tokenization_spaces=False, skip_special_tokens=True)
            for stem, gen in zip(stems, generated):
                jout.write(json.dumps({"stem": stem, "cont": gen})+"\n")



    
