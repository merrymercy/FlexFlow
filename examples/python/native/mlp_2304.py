import argparse
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
    hidden_dim = 2304
    num_layers = 4

    input_tensor = ffmodel.create_tensor([batch_size, hidden_dim], DataType.DT_FLOAT)
    t = input_tensor
    for i in range(num_layers):
        t = ffmodel.dense(t, hidden_dim * 4)
        t = ffmodel.dense(t, hidden_dim)

    optimizer = SGDOptimizer(ffmodel, 0.001)
    ffmodel.optimizer = optimizer
    ffmodel.compile(loss_type=LossType.LOSS_MEAN_SQUARED_ERROR_AVG_REDUCE,
                    metrics=[], comp_mode=CompMode.TRAINING)

    # Data loader
    num_samples = batch_size * 10
    x_train = np.random.randn(num_samples, hidden_dim).astype("float32")
    y_train = np.random.randn(num_samples, hidden_dim).astype("float32")
    label_tensor = ffmodel.label_tensor
    dataloader_input = ffmodel.create_data_loader(input_tensor, x_train)
    dataloader_label = ffmodel.create_data_loader(label_tensor, y_train)
    dataloader_input.reset()
    dataloader_input.next_batch(ffmodel)
    dataloader_label.reset()
    dataloader_label.next_batch(ffmodel)

    ffmodel.init_layers()

    def one_batch():
        ffmodel.forward()
        ffmodel.zero_gradients()
        ffmodel.backward()
        ffmodel.update()

    warmup = 2
    number = 10

    # Warmup
    for i in range(warmup):
        one_batch()

    # Benchmark
    ts_start = ffconfig.get_current_time()
    for i in range(number):
        one_batch()
    ts_end = ffconfig.get_current_time()

    # Log results
    run_time = 1e-6 * (ts_end - ts_start) / number
    print(f"Time: {run_time:.3f} second")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args, unknown = parser.parse_known_args()
    top_level_task()
    print("")

