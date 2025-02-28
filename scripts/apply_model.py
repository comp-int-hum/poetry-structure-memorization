import argparse
from transformers import AutoTokenizer, LlamaForCausalLM
import torch
import json
from Levenshtein import distance

def turn_on_dropout(model, dropout=None):
    """
    Turn on dropout for all modules in the model.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()
            if dropout:
                module.p = dropout
    return model

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help=".txt input")
    parser.add_argument("--window", type=int, default=50, help="window (in token size) to provide to the model")
    parser.add_argument("--max_tokens", type=int, default=50, help="max new tokens to generate")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--output", dest="output", help="Output files")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--random_state", type=int, default=20)
    parser.add_argument("--num_dropout_samples", type=int, default=10)
    args, rest = parser.parse_known_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    with open(args.input, "rt") as p_in:
        poem = p_in.read()
        poem = poem.replace("'", "’").replace('"', '“').replace('"', '”')

    tokenized = tokenizer(poem, max_length=args.window, stride=49, truncation=True, padding=True, return_overflowing_tokens=True, return_tensors="pt", add_special_tokens=False).input_ids
    targets = tokenized[args.window:]
    targets = torch.cat((targets, torch.full((1,50), tokenizer.pad_token_id)),axis=0)

    model = LlamaForCausalLM.from_pretrained(args.model, device_map="auto")
    model.eval()
    model.config.use_cache = False
    batched = torch.split(tokenized, args.batch_size)
    batched_targets = torch.split(targets, args.batch_size)
    with open(args.output, "wt") as jout:
        for batch, b_t in zip(batched, batched_targets):
            stems = tokenizer.batch_decode(batch, clean_up_tokenization_spaces=False, skip_special_tokens=True)
            targets = tokenizer.batch_decode(b_t, clean_up_tokenization_spaces=False, skip_special_tokens=True)
            model.eval()
            res = model.generate(batch.to(args.device), max_new_tokens=args.max_tokens, min_new_tokens=args.max_tokens, return_dict_in_generate=True, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            generated = tokenizer.batch_decode(res.sequences[:, batch.shape[1]:], clean_up_tokenization_spaces=False, skip_special_tokens=True)

            dropout_results = []
            model = turn_on_dropout(model)
            for _ in range(args.num_dropout_samples):
                res = model.generate(batch.to(args.device), max_new_tokens=args.max_tokens, min_new_tokens=args.max_tokens, return_dict_in_generate=True, do_sample=False, pad_token_id=tokenizer.eos_token_id)
                generated = tokenizer.batch_decode(res.sequences[:, batch.shape[1]:], clean_up_tokenization_spaces=False, skip_special_tokens=True)
                dropout_results.append(generated)

            dropout_per_sample = list(zip(*dropout_results))
            
            for stem, nd_gen, target, dropouts in zip(stems, generated, targets, dropout_per_sample):
                nd_ld = distance(nd_gen, target)
                dropout_objs = [{"gen": d, "LD": distance(d, target)} for d in dropouts]
                output_obj = {
                    "stem": stem,
                    "target": target,
                    "no_dropout": {"gen": nd_gen, "LD": nd_ld},
                    "dropout": dropout_objs
                }
                jout.write(json.dumps(output_obj) + "\n")



    
