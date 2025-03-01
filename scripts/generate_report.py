import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="jsonl results file") 
    parser.add_argument("--output", help="Output files")
    args, rest = parser.parse_known_args()

    df_ls = []
    with open(args.input, "rt") as jin:
        for line in jin:
            jline = json.loads(line)
            df_l = {"nd":jline["no_dropout"]["LD"]}
            for i, dp in enumerate(jline["dropout"]):
                df_l["d"+str(i)] = dp["LD"]
            df_ls.append(df_l)

    df = pd.DataFrame.from_dict(df_ls)
    print(df)
    df.plot()
    plt.xlabel("Step")
    plt.ylabel("LD")
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    
