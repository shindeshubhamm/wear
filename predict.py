import warnings

warnings.filterwarnings('ignore', category=UserWarning)
import argparse
import datetime
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils.torch_utils import InertialDataset
from utils.os_utils import Logger
from utils.data_utils import unwindow_inertial_data
from inertial_baseline.AttendAndDiscriminate import AttendAndDiscriminate
from inertial_baseline.DeepConvLSTM import DeepConvLSTM

# constants
window_size = 50
window_overlap = 50
input_dim = 12
conv_kernels = 64
conv_kernel_size = 11
lstm_units = 1024
lstm_layers = 1
dropout = 0.5
batch_size = 100
column_names = [
    'sbj_id',
    'right_arm_acc_x',
    'right_arm_acc_y',
    'right_arm_acc_z',
    'right_leg_acc_x',
    'right_leg_acc_y',
    'right_leg_acc_z',
    'left_leg_acc_x',
    'left_leg_acc_y',
    'left_leg_acc_z',
    'left_arm_acc_x',
    'left_arm_acc_y',
    'left_arm_acc_z',
    'label',
]

labels_dict = {
    0: 'null',
    1: 'jogging',
    2: 'jogging (rotating arms)',
    3: 'jogging (skipping)',
    4: 'jogging (sidesteps)',
    5: 'jogging (butt-kicks)',
    6: 'stretching (triceps)',
    7: 'stretching (lunging)',
    8: 'stretching (shoulders)',
    9: 'stretching (hamstrings)',
    10: 'stretching (lumbar rotation)',
    11: 'push-ups',
    12: 'push-ups (complex)',
    13: 'sit-ups',
    14: 'sit-ups (complex)',
    15: 'burpees',
    16: 'lunges',
    17: 'lunges (complex)',
    18: 'bench-dips',
}


def main(args):
    model_name: str = args.model_name
    ckpt_path: str = args.ckpt_path
    testdata_path: str = args.test_data


    if not model_name:
        raise ValueError('model_name cannot be empty.')
    if not ckpt_path:
        raise ValueError('checkpoint cannot be empty.')
    if not testdata_path:
        raise ValueError('test_data cannot be empty.')


    # load all csv files from testdata_path
    test_data = np.empty((0, input_dim + 1))
    testdata_dir = os.fsencode(testdata_path)
    for file in os.listdir(testdata_dir):
        sbj_filename = os.fsdecode(file)
        if sbj_filename.endswith('.csv'):
            temp = pd.read_csv(os.path.join(testdata_path, sbj_filename), index_col=False).fillna(0).to_numpy()
            test_data = np.append(test_data, temp, axis=0)

    # match dimension
    zero_column = np.zeros((test_data.shape[0], 1))
    test_data = np.concatenate((test_data, zero_column), axis=1)

    # create dataset
    test_dataset = InertialDataset(test_data, window_size, window_overlap)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    # select model
    model = None
    if model_name == 'deepconvlstm':
        model = DeepConvLSTM(
            input_dim,  # should match the input dimension used during training
            19,  # num_classes should match the number of classes used during training
            window_size,  # should match the window size used during training
            conv_kernels,
            conv_kernel_size,
            lstm_units,
            lstm_layers,
            dropout,
        )
    else:
        raise ValueError('Unsupported model name.')

    # load model state and set to evaluation mode
    checkpoint: dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # make predictions
    all_predictions = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = torch.FloatTensor(inputs).to('cpu')  # Ensure input data is on the correct device
            predictions = model(inputs)
            all_predictions.append(predictions)
    
    all_predictions = torch.cat(all_predictions, dim=0)
    predicted_labels = torch.argmax(all_predictions, dim=1).numpy()

    # unwindow
    unwindowed_predictions, _ = unwindow_inertial_data(test_data, test_dataset.ids, predicted_labels, window_size, window_overlap)

    # error if length mismatch after unwindowing
    if len(unwindowed_predictions) != len(test_data):
        raise ValueError('Mismatch between number of unwindowed predictions and original data rows.')

    # add the predicted labels to the original data
    test_data = test_data[:, :-1]  # remove zeros_column initially added
    test_data_with_predictions = np.hstack((test_data, unwindowed_predictions.reshape(-1, 1)))

    # create df
    df_predictions = pd.DataFrame(test_data_with_predictions, columns=column_names)
    df_predictions['label'] = df_predictions['label'].astype(int).map(labels_dict)
    df_predictions['sbj_id'] = df_predictions['sbj_id'].astype(int)


    # create directory for prediction files
    ts = datetime.datetime.fromtimestamp(int(time.time()))
    pred_logs_dir = os.path.join('logs_pred', model_name, str(ts))
    os.makedirs(pred_logs_dir, exist_ok=True)

    # save subject wise predictions
    for sbj in df_predictions['sbj_id'].unique():
        sbj_data = df_predictions[df_predictions['sbj_id'] == sbj]
        output_path = os.path.join(pred_logs_dir, f'sbj_{sbj}_2.csv' if sbj < 18 else f'sbj_{sbj}.csv')
        sbj_data.to_csv(output_path, index=False)    

    print(f'Predictions saved to {pred_logs_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='deepconvlstm', type=str, help='Name of the model used for training.')
    parser.add_argument('--ckpt_path', default='', type=str, help='Chekpoint to select for making predictions on test data.')
    parser.add_argument('--test_data', default='./test_data', type=str, help='Path for test data with csv files.')
    args = parser.parse_args()
    main(args)
