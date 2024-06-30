import warnings

warnings.filterwarnings('ignore')
import argparse
import datetime
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import torch
from torch.utils.data import DataLoader

from utils.torch_utils import InertialDataset
from utils.data_utils import unwindow_inertial_data
from inertial_baseline.AttendAndDiscriminate import AttendAndDiscriminate
from inertial_baseline.DeepConvLSTM import DeepConvLSTM

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
    is_seen_data: bool = args.seen_data

    if not model_name:
        raise ValueError('model_name cannot be empty.')
    if not ckpt_path:
        raise ValueError('checkpoint cannot be empty.')
    if not testdata_path:
        raise ValueError('test_data cannot be empty.')

    # get config
    config_path = '/'.join(ckpt_path.split('/')[:-2]) + '/cfg.txt'
    with open(config_path) as f:
        c = f.read().replace('\'', '"').replace('True', 'true').replace('False', 'false')
        config = json.loads(c)

    # load all csv files from testdata_path
    test_data = np.empty((0, config['dataset']['input_dim'] + (2 if is_seen_data else 1)))
    testdata_dir = os.fsencode(testdata_path)
    for file in os.listdir(testdata_dir):
        sbj_filename = os.fsdecode(file)
        if sbj_filename.endswith('.csv'):
            temp = pd.read_csv(os.path.join(testdata_path, sbj_filename), index_col=False).fillna(0).to_numpy()
            test_data = np.append(test_data, temp, axis=0)

    if is_seen_data:
        original_labels = test_data[:, -1]
        test_data = test_data[:, :-1]

    # match dimension
    zero_column = np.zeros((test_data.shape[0], 1))
    test_data = np.concatenate((test_data, zero_column), axis=1)

    # create dataset
    test_dataset = InertialDataset(test_data, config['dataset']['window_size'], config['dataset']['window_overlap'])
    test_loader = DataLoader(test_dataset, config['loader']['batch_size'], shuffle=False)

    # select model
    model = None
    if model_name == 'deepconvlstm':
        model = DeepConvLSTM(
            config['dataset']['input_dim'],  # should match the input dimension used during training
            19,  # num_classes should match the number of classes used during training
            config['dataset']['window_size'],  # should match the window size used during training
            config['model']['conv_kernels'],
            config['model']['conv_kernel_size'],
            config['model']['lstm_units'],
            config['model']['lstm_layers'],
            config['model']['dropout'],
        )
    elif model_name == 'attendanddiscriminate':
        model = AttendAndDiscriminate(
            config['dataset']['input_dim'],  # should match the input dimension used during training
            19,  # num_classes should match the number of classes used during training
            config['model']['hidden_dim'],
            config['model']['conv_kernels'],
            config['model']['conv_kernel_size'],
            config['model']['enc_layers'],
            config['model']['enc_is_bidirectional'],
            config['model']['dropout'],
            config['model']['dropout_rnn'],
            config['model']['dropout_cls'],
            config['model']['activation'],
            config['model']['sa_div'],
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
    unwindowed_predictions, _ = unwindow_inertial_data(
        test_data,
        test_dataset.ids,
        predicted_labels,
        config['dataset']['window_size'],
        config['dataset']['window_overlap'],
    )

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

    if is_seen_data:
        predictions_int = unwindowed_predictions.astype(int)
        predictions = np.array([labels_dict[pred] for pred in predictions_int])
        original_labels = np.array([str(label) for label in original_labels])
        unique_labels = list(labels_dict.values())

        conf_mat = confusion_matrix(original_labels, predictions, normalize='true', labels=unique_labels)
        v_acc = conf_mat.diagonal() / conf_mat.sum(axis=1)
        v_prec = precision_score(original_labels, predictions, average=None, zero_division=1, labels=unique_labels)
        v_rec = recall_score(original_labels, predictions, average=None, zero_division=1, labels=unique_labels)
        v_f1 = f1_score(original_labels, predictions, average=None, zero_division=1, labels=unique_labels)

        block5 = ''
        block5 += 'Acc {:>4.2f} (%)'.format(np.nanmean(v_acc) * 100)
        block5 += '\nPrec {:>4.2f} (%)'.format(np.nanmean(v_prec) * 100)
        block5 += '\nRec {:>4.2f} (%)'.format(np.nanmean(v_rec) * 100)
        block5 += '\nF1 {:>4.2f} (%)\n'.format(np.nanmean(v_f1) * 100)
        print(block5)

    print(f'Predictions saved to {pred_logs_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='deepconvlstm', type=str, help='Name of the model used for training.')
    parser.add_argument('--ckpt_path', default='', type=str, help='Chekpoint to select for making predictions on test data.')
    parser.add_argument('--test_data', default='./test_data', type=str, help='Path for test data with csv files.')
    parser.add_argument('--seen_data', default=False, type=bool, help='Whether the data is seen data.')
    args = parser.parse_args()
    main(args)
