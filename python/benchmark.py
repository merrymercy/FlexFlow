import argparse
import os
import pickle

import numpy as np


def run_cmd(cmd):
    print(cmd)
    os.system(cmd)


def write_tsv(heads, values, filename, print_line=True):
    """Write tsv data to a file."""
    with open(filename, "a") as fout:
        fout.write("\t".join(values) + "\n")

    if print_line:
        line = ""
        for i in range(len(heads)):
            line += heads[i] + ": " + values[i] + "  "
        print(line)


def benchmark_mlp_one_case(case):
    batch_size, seq_len, hidden_size, num_layers = case[:4]
    strategy = case[4]
    num_nodes = case[5]

    batch_size, seq_len = batch_size * seq_len, 1
    if num_nodes > 1:
        mpi_base = f"mpirun --prefix /opt/amazon/openmpi/ --mca btl_tcp_if_exclude lo,docker0 --mca oob_tcp_if_exclude lo,docker0 -npernode 1 "\
                   f"--hostfile hostfile -n {num_nodes} "
    else:
        mpi_base = ""

    base = f"$FF_HOME/python/flexflow_python $FF_HOME/examples/python/native/mlp.py "\
           f"-ll:py 1 -ll:cpu 4 -ll:gpu 4 -ll:fsize 14000 -ll:zsize 20000 --nodes {num_nodes} "\
           f"--batch-size {batch_size} --seq-len {seq_len} --hidden-size {hidden_size} "\
           f"--num-layers {num_layers} --control-replication "

    tune_suffix = "--export mlp_stra.txt --search-budget 100000 --enable-parameter-parallel -enable-attribute-parallel --search-alpha 0.05 "
    replay_suffix = "--import mlp_stra.txt "

    run_cmd("rm -rf tmp.pkl")

    if strategy == "dp":
        run_cmd(mpi_base + base)
    elif strategy == "opt":
        run_cmd(mpi_base + base + tune_suffix)
        run_cmd(mpi_base + base + replay_suffix)
    else:
        raise ValueError(f"Invalid strategy: {strategy}")
    costs, = pickle.load(open("tmp.pkl", "rb"))

    # Log benchmark results
    heads = ["Type", "Case", "Strategy", "Mean Time", "Std Time"]
    values = ["mlp", str(case[:4]), strategy,
             f"{np.mean(costs):.4f}", f"{np.std(costs):.4f}"]
    write_tsv(heads, values, "result_mlp.tsv")


benchmark_mlp_suite = [
    # Batch size, seq_len, hidden size, num_layers, strategy, num_nodes
    (16,          1024,    2304,        4,          "dp",     1),
    (16,          1024,    2304,        4,          "opt",    1),
    (8,           256,     5760,        4,          "dp",     1),
    (8,           256,     5760,        4,          "opt",    1),

    (16,          1024,    2304,        4,          "dp",      2),
    (16,          1024,    2304,        4,          "opt",     2),
    (8,           256,     5760,        4,          "dp",      2),
    (8,           256,     5760,        4,          "opt",     2),

    (32,          1024,    2304,        4,          "dp",     2),
    (64,          1024,    2304,        4,          "dp",     2),
]

def benchmark_mlp():
    for case in benchmark_mlp_suite:
        benchmark_mlp_one_case(case)


if __name__ == "__main__":
    benchmark_mlp()

