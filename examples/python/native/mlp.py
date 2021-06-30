import argparse
import os
import pickle
import time

import numpy as np

from accuracy import ModelAccuracy
from flexflow.core import *
from flexflow.keras.datasets import mnist

def top_level_task():
    ffconfig = FFConfig()
    print("Python API batchSize(%d) workersPerNodes(%d) numNodes(%d)" % (
        ffconfig.batch_size, ffconfig.workers_per_node, ffconfig.num_nodes))
    ffmodel = FFModel(ffconfig)

    batch_size = ffconfig.batch_size
    seq_len = args.seq_len
    hidden_size = args.hidden_size
    num_layers = args.num_layers

    input_tensor = ffmodel.create_tensor([batch_size * seq_len, hidden_size], DataType.DT_FLOAT)

    #t = ffmodel.reshape(input_tensor, (batch_size * seq_len, hidden_size))
    t = input_tensor
    for i in range(num_layers):
        t = ffmodel.dense(t, hidden_size * 4)
        t = ffmodel.dense(t, hidden_size)
    #t = ffmodel.reshape(t, (batch_size, seq_len, hidden_size))

    optimizer = SGDOptimizer(ffmodel, 0.001)
    ffmodel.optimizer = optimizer
    ffmodel.compile(loss_type=LossType.LOSS_MEAN_SQUARED_ERROR_AVG_REDUCE,
                    metrics=[], comp_mode=CompMode.TRAINING)

    # Data loader
    num_samples = batch_size * 4
    x_train = np.random.randn(num_samples * seq_len, hidden_size).astype("float32")
    y_train = np.random.randn(num_samples * seq_len, hidden_size).astype("float32")
    label_tensor = ffmodel.label_tensor
    dataloader_input = ffmodel.create_data_loader(input_tensor, x_train)
    dataloader_label = ffmodel.create_data_loader(label_tensor, y_train)
    dataloader_input.reset()
    dataloader_input.next_batch(ffmodel)
    dataloader_label.reset()
    dataloader_label.next_batch(ffmodel)

    ffmodel.init_layers()

    def one_batch():
        #ffconfig.begin_trace(200)
        ffmodel.forward()
        ffmodel.zero_gradients()
        ffmodel.backward()
        ffmodel.update()
        #ffconfig.end_trace(200)

    warmup = 2
    repeat = 3
    number = 5

    # Warmup
    for i in range(warmup):
        one_batch()

    # Benchmark
    costs = []
    for i in range(repeat):
        ts_start = ffconfig.get_current_time()
        for j in range(number):
            one_batch()
        ts_end = ffconfig.get_current_time()
        run_time = 1e-6 * (ts_end - ts_start) / number

        costs.append(run_time)

    # Log results
    pickle.dump((costs,), open("tmp.pkl", "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--num-layers", type=int)
    args, unknown = parser.parse_known_args()
    top_level_task()

