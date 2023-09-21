import datetime
from pathlib import Path
import warnings

import click
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
import timm

from sparsam.dataset import ImageSet
from utils.utils import load_config, load_data
from utils.supervised import map_classes




@click.command()
@click.option(
    '--dataset_train',
    required=True,
    type=click.STRING,
    help='Dataset which was used to train model: blood_acevedo, blood_matek, blood_raabin, bm',
    default='bm'
)
@click.option(
    '--dataset_eval',
    required=True,
    type=click.STRING,
    help='Dataset which will be used for evaluation: blood_acevedo, blood_matek, blood_raabin, bm',
    default='blood_matek'
)
@click.option(
    '--device',
    type=click.STRING,
    required=True,
    default='cuda:0',
    help='GPU to use (e.g., cuda:0)'
)
@click.option(
    '--batch_size',
    type=click.INT,
    required=True,
    default=4,
    help='Batch size.'
)
@click.option(
    '--model_name',
    type=click.STRING,
    required=True,
    default='xcit_small_12_p8_224_dist',
    help='Choose a model from the timm library'
)
@click.option(
    '--save_path',
    type=click.STRING,
    required=True,
    help='Path to the folder where results should be stored'
)
@click.option(
    '--path_model_weights',
    type=click.STRING,
    required=False,
    help='Path to the model weights for evaluation'
)
def run(
    dataset_train, dataset_eval, device, batch_size, model_name, save_path,
     path_model_weights
):
    """
    Main function to run the evaluation and classification pipeline.

    Args:
        device (str): Device for evaluation (e.g., 'cuda:0').
        dataset_train (str): Dataset used for training.
        dataset_eval (str): Dataset used for evaluation.
        batch_size (int): Batch size for evaluation.
        model_name (str): Name of the model to use.
        save_path (str): Path to the folder where results should be stored.
        path_model_weights (str): Path to the model weights for evaluation.
    """
    evaluate(
        dataset_train, dataset_eval, device, batch_size, model_name, save_path,
        path_model_weights
    )
def evaluate(
    dataset_train, dataset_eval, device, batch_size, model_name, save_path,
     path_model_weights
):
    """
    Main function to run the evaluation and classification pipeline.

    Args:
        device (str): Device for evaluation (e.g., 'cuda:0').
        dataset_train (str): Dataset used for training.
        dataset_eval (str): Dataset used for evaluation.
        batch_size (int): Batch size for evaluation.
        model_name (str): Name of the model to use.
        save_path (str): Path to the folder where results should be stored.
        path_model_weights (str): Path to the model weights for evaluation.
    """
    print(f'Using {dataset_eval} dataset.')

    # Load data and configurations
    config, data_set_config, image_paths, image_labels, _ = load_data(dataset_eval)
    _, bm_config = load_config(dataset_train)

    train_classes = bm_config['class_names']
    eval_classes = data_set_config['class_names']

    if save_path is None:
        save_path = config['save_path']

    time = datetime.datetime.now()
    save_path = Path(save_path) / dataset_eval / Path(f'{time.year}_{time.month}_{time.day}-{time.hour}{time.minute}')

    if batch_size is None:
        config['batch_size'] = batch_size

    config['save_path'] = save_path
    class_names_bm = data_set_config['class_names']
    class_names_dataset = data_set_config['class_names']

    save_parameter = {
        'dataset': dataset_eval, 'device': device, 'batch_size': batch_size, 'model': model_name,
        'save_path': save_path,
        'class_names_dataset': class_names_dataset, 'train_images': image_paths['train']
    }

    df_save_parameter = pd.DataFrame([save_parameter])
    Path(save_path).mkdir(parents=True, exist_ok=True)
    df_save_parameter.to_csv(Path(save_path) / 'save_parameter.csv', index=False)


    image_paths['test'],image_labels['test'] = map_classes(image_paths['test'],image_labels['test'],train_classes)
    test_set = ImageSet(img_paths=image_paths['test'],
                       labels=image_labels['test'],
                       class_names=class_names_bm,
                       normalize=True)

    test_loader_parameter = config['data_loader_parameter']
    if batch_size is not None:
        test_loader_parameter['batch_size'] = batch_size
    test_loader_parameter['drop_last'] = False


    test_loader = DataLoader(test_set, **test_loader_parameter)

    model = timm.create_model(model_name, pretrained=True, num_classes=len(train_classes), in_chans=3)
    model.to(device)
    model.set_grad_checkpointing()
    model.to(torch.float32)
    path_to_weights = path_model_weights
    best_model = model
    weights = torch.load(path_to_weights, map_location=device)
    best_model.load_state_dict(weights)
    print(f'Loaded weight from {path_to_weights}')
    best_model.to(device)
    best_model.set_grad_checkpointing()
    best_model.eval()
    predictions = []
    true_labels = []

    considered_train_class = eval_classes.copy()
    if 'MYC' in eval_classes:
        considered_train_class.extend(['MMZ', 'PMO', 'MYB'])
    if 'NGB' not in eval_classes:
        considered_train_class.append('NGB')
    with torch.no_grad():
        for images, labels_eval in test_loader:
            images, labels_eval = images.to(device), labels_eval.to(device)
            outputs = best_model(images)

            # remove all pedictions for classes not in dataset
            mask_tensor = torch.tensor([1 if class_name in considered_train_class else 0 for class_name in train_classes])
            expanded_mask = mask_tensor.unsqueeze(0).expand_as(outputs).to(device)
            outputs = outputs * expanded_mask
            _, predicted = torch.max(outputs, 1)
            train_index_to_string = {idx:class_name for idx, class_name in enumerate(train_classes)}
            predicted = [train_index_to_string[idx.item()] for idx in predicted]
            if 'MYC' in eval_classes:
                predicted = ['MYC' if label in ['MMZ', 'PMO', 'MYB'] else label for label in predicted]
            if 'NGB' not in eval_classes:
                predicted = ['NGS' if label in ['NGB'] else label for label in predicted]

            labels_numpy = labels_eval.cpu().numpy()
            eval_index_to_class = {idx:class_name for idx, class_name in enumerate(eval_classes)}
            labels_eval = [eval_index_to_class[idx.item()] for idx in labels_numpy]
            predictions.extend(predicted)
            true_labels.extend(labels_eval)

    confusion_mat = confusion_matrix(true_labels, predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average=None)
    macro_precision = precision_score(true_labels, predictions, average='macro')
    macro_recall = recall_score(true_labels, predictions, average='macro')
    macro_f1 = f1_score(true_labels, predictions, average='macro')
    class_labels = np.unique(true_labels)
    metrics_df = pd.DataFrame({'Precision': precision, 'Recall': recall, 'F1 Score': f1}, index=class_labels)
    metrics_df.loc['All'] = [macro_precision, macro_recall, macro_f1]

    cm_df = pd.DataFrame(confusion_mat, index=class_labels, columns=class_labels)
    cm_df.to_csv(Path(config['save_path']) / Path(
        f'confusion_matrix.csv'), index=True, header=True)
    metrics_df.to_csv(Path(config['save_path']) / Path(
        'metrics.csv'), index=True, header=True)
    print(f'Results can be found at {save_path}')




if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    run()
