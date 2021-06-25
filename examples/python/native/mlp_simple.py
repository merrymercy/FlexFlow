import argparse
import time

import numpy as np

from flexflow.core import *

def top_level_task():
    ffconfig = FFConfig()
    print("Python API batchSize(%d) workersPerNodes(%d) numNodes(%d)" % (
        ffconfig.batch_size, ffconfig.workers_per_node, ffconfig.num_nodes))
    ffmodel = FFModel(ffconfig)

    batch_size = ffconfig.batch_size
    hidden_dim = 2304

    input_tensor = ffmodel.create_tensor([batch_size, hidden_dim], DataType.DT_FLOAT)
    t = input_tensor

    t = ffmodel.dense(t, hidden_dim)
    t = ffmodel.dense(t, hidden_dim)
    t = ffmodel.dense(t, 10)

    optimizer = SGDOptimizer(ffmodel, 0.001)
    ffmodel.optimizer = optimizer
    ffmodel.compile(loss_type=LossType.LOSS_SPARSE_CATEGORICAL_CROSSENTROPY,
                    metrics=[MetricsType.METRICS_ACCURACY,
                             MetricsType.METRICS_SPARSE_CATEGORICAL_CROSSENTROPY])

    label_tensor = ffmodel.label_tensor
    num_samples = batch_size * 12
    x_train = np.random.randn(num_samples, hidden_dim).astype("float32")
    y_train = np.random.randn(num_samples, 10).astype("int32")
    dataloader_input = ffmodel.create_data_loader(input_tensor, x_train)
    dataloader_label = ffmodel.create_data_loader(label_tensor, y_train)

    ffmodel.init_layers()
    ffmodel.fit(x=dataloader_input, y=dataloader_label, epochs=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args, unknown = parser.parse_known_args()
    top_level_task()

