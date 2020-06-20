# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A demo script showing how to use the uisrnn package on toy data."""

import numpy as np
from functools import partial
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp

mp = mp.get_context('forkserver')

import uisrnn

SAVED_MODEL_NAME = 'saved_model.uisrnn'
NUM_WORKERS = 2


def diarization_experiment(model_args, training_args, inference_args):
    """Experiment pipeline.

  Load data --> train model --> test model --> output result

  Args:
    model_args: model configurations
    training_args: training configurations
    inference_args: inference configurations
  """
    # data loading
    train_data = np.load('./data/toy_training_data.npz', allow_pickle=True)
    test_data = np.load('./data/toy_testing_data.npz', allow_pickle=True)
    train_sequence = train_data['train_sequence']
    train_cluster_id = train_data['train_cluster_id']
    test_sequences = test_data['test_sequences'].tolist()
    test_cluster_ids = test_data['test_cluster_ids'].tolist()

    # model init
    model = uisrnn.UISRNN(model_args)
    # model.load(SAVED_MODEL_NAME) # to load a checkpoint
    # tensorboard writer init
    writer = SummaryWriter()

    # training
    for epoch in range(training_args.epochs):
        stats = model.fit(train_sequence, train_cluster_id, training_args)
        # add to tensorboard
        for loss, cur_iter in stats:
            for loss_name, loss_value in loss.items():
                writer.add_scalar('loss/' + loss_name, loss_value, cur_iter)
        # save the mdoel
        model.save(SAVED_MODEL_NAME)

    # testing
    predicted_cluster_ids = []
    test_record = []
    # predict sequences in parallel
    model.rnn_model.share_memory()
    pool = mp.Pool(NUM_WORKERS, maxtasksperchild=None)
    pred_gen = pool.imap(
        func=partial(model.predict, args=inference_args),
        iterable=test_sequences)
    # collect and score predicitons
    for idx, predicted_cluster_id in enumerate(pred_gen):
        accuracy = uisrnn.compute_sequence_match_accuracy(
            test_cluster_ids[idx], predicted_cluster_id)
        predicted_cluster_ids.append(predicted_cluster_id)
        test_record.append((accuracy, len(test_cluster_ids[idx])))
        print('Ground truth labels:')
        print(test_cluster_ids[idx])
        print('Predicted labels:')
        print(predicted_cluster_id)
        print('-' * 80)

    # close multiprocessing pool
    pool.close()
    # close tensorboard writer
    writer.close()

    print('Finished diarization experiment')
    print(uisrnn.output_result(model_args, training_args, test_record))


def main():
    """The main function."""
    model_args, training_args, inference_args = uisrnn.parse_arguments()
    diarization_experiment(model_args, training_args, inference_args)


if __name__ == '__main__':
    main()
