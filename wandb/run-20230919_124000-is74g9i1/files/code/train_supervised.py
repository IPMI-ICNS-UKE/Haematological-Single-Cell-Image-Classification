import click
import warnings
from copy import deepcopy
from pathlib import Path
import os
import datetime
from typing import Sequence, Callable
import numpy as np
import pandas as pd

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import train_test_split

import timm
from timm.scheduler import PlateauLRScheduler

import wandb

from sparsam.utils import EarlyStopper
from sparsam.dataset import ImageSet
from sparsam.helper import uniform_train_test_splitting

from utils.utils import load_data
from utils.supervised import GaussianBlur
from eval_supervised import evaluate


@click.command()
@click.option(
    '--dataset', required=True, type=click.STRING,
    help='Decide which data set should be used for training: blood_acevedo, blood_matek, blood_raabin, bm',
    default='blood_matek'
)
@click.option(
    '--device', type=click.STRING, required=True, default='cuda:0',
    help='Which GPU should be used: e.g. for the first GPU: cuda:0'
)
@click.option(
    '--batch_size', type=click.INT, required=True, default=4,
    help='Batch size.'
)
@click.option(
    '--model_name', type=click.STRING, required=True, default='xcit_small_12_p8_224_dist',
    help='Choose a model from the timm library'
)
@click.option(
    '--save_path', type=click.STRING, required=False, default='/home/lwenderoth/Documents/Doktorarbeit/Results/supervised',
    help='Path to the folder where results should be stored'
)
@click.option(
    '--max_train_image_number', type=click.INT, required=False,
    help='Specify the maximum number of training images'
)
@click.option(
    '--num_epochs', type=click.INT, required=True, default=1,
    help='Number of training epochs'
)
@click.option(
    '--train', type=click.BOOL, required=False, default=True,
    help='Specify whether to train the model'
)
@click.option(
    '--n_iter', type=click.INT, required=False, default=1,
    help='Number of iterations for classifier fitting using random images per class'
)
@click.option(
    '--resume', type=click.STRING, required=False,
    help='Path to weight with model to resume training'
)
def run(
    device, dataset, batch_size, model_name, save_path, max_train_image_number,
    num_epochs, train, n_iter, resume
):
    # Print dataset information
    print(f'Using {dataset} dataset.')

    # Load data and configuration
    config, data_set_config, image_paths, image_labels, not_used_images = load_data(dataset)

    # Set save path
    if save_path is None:
        save_path = config['save_path']
    time = datetime.datetime.now()
    save_path = Path(save_path) / dataset / Path(f'{time.year}_{time.month}_{time.day}-{time.hour}{time.minute}')

    # Update batch size in config
    if batch_size is not None:
        config['batch_size'] = batch_size
    config['save_path'] = save_path
    project_name = dataset / Path(f'{time.year}_{time.month}_{time.day}-{time.hour}{time.minute}')

    # Initialize WandB
    wandb.login()
    wandb.init(group='supervised_test', **config['wandb_parameter'], name=f'{project_name}')

    # Load class names
    class_names = data_set_config['class_names']
    for iter in range(n_iter):
        print(f'Start iteration {iter+1}/{n_iter}')
        if train:
            path_best_model_weights = train_supervised_model(
                device, config, dataset, image_paths, image_labels, not_used_images, save_path,
                model_name, max_train_image_number, num_epochs, class_names, batch_size, resume
            )
        evaluate(
            dataset_train=dataset, dataset_eval=dataset, device=device, batch_size=batch_size, model_name=model_name, save_path=save_path,
            path_model_weights=path_best_model_weights
        )

    wandb.finish()

def train_supervised_model(
    device: str,
    config: dict,
    dataset: str,
    image_paths: dict[Sequence],
    image_labels: dict[Sequence],
    not_used_images: list,
    save_path: str,
    model_name: str,
    max_train_image_number: int,
    num_epochs: int,
    class_names: Sequence[str],
    batch_size: int,
    resume: str
) -> str:
    """
    Train a supervised model.

    Args:
        device (str): Device for training (e.g., 'cuda:0').
        config (dict): Configuration parameters.
        dataset (str): Dataset name (' blood_acevedo, blood_matek, blood_raabin, bm').
        image_paths (dict[Sequence[Path]]): Dictionary of image paths.
        image_labels (dict[Sequence]): Dictionary of image labels.
        not_used_images (list): List of unused images.
        save_path (str): Path to the folder where results should be stored.
        model_name (str): Name of the model to use.
        max_train_image_number (int): Maximum number of training images to use.
        num_epochs (int): Number of training epochs.
        class_names (Sequence[str]): List of class names.
        batch_size (int): Batch size for training.
        resume (str): Path to weights for resuming training.

    Returns:
        str: Path to the best model weights.
    """

    image_paths['train'], image_paths['val'], image_labels['train'], image_labels['val'] = train_test_split(
        image_paths['train'], image_labels['train'], stratify=image_labels['train'], test_size=0.3, random_state=42
    )

    if max_train_image_number is not None:
        image_paths['train'], image_labels['train'], _, _ = uniform_train_test_splitting(
            image_paths['train'], image_labels['train'], n_samples_class=max_train_image_number
        )

    save_parameter = {
        'dataset': dataset, 'device': device, 'batch_size': batch_size,
        'model': model_name, 'save_path': save_path,
        'max_train_image_number': max_train_image_number, 'class_names': class_names,
        'train_images': image_paths['train']
    }

    # Create and save a DataFrame with save parameters
    df_save_parameter = pd.DataFrame([save_parameter])
    Path(save_path).mkdir(parents=True, exist_ok=True)
    df_save_parameter.to_csv(Path(save_path) / 'save_parameter.csv', index=False)

    # Define data augmentations
    flip_and_color_jitter = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
    ])

    data_augment = transforms.Compose([
        transforms.RandomPerspective(distortion_scale=0.5, p=0.75),
        transforms.RandomRotation(180),
        transforms.RandomResizedCrop(256, scale=(0.5, 1.)),
        flip_and_color_jitter,
        GaussianBlur(p=0.2, radius_min=0.1, radius_max=5.0),
    ])

    # Create datasets and data loaders
    train_set = ImageSet(
        img_paths=image_paths['train'], labels=image_labels['train'],
        class_names=class_names, normalize=True, data_augmentation=data_augment
    )

    val_set = ImageSet(
        img_paths=image_paths['val'], labels=image_labels['val'],
        class_names=class_names, normalize=True
    )

    test_set = ImageSet(
        img_paths=image_paths['test'], labels=image_labels['test'],
        class_names=class_names, normalize=True
    )

    data_loader_parameter = config['data_loader_parameter']
    if batch_size is not None:
        data_loader_parameter['batch_size'] = batch_size

    test_loader_parameter = deepcopy(data_loader_parameter)
    test_loader_parameter['drop_last'] = False

    labeled_train_loader = DataLoader(train_set, **data_loader_parameter)
    val_loader = DataLoader(val_set, **data_loader_parameter)
    test_loader = DataLoader(test_set, **data_loader_parameter)

    os.makedirs(Path(config['save_path']) / 'image_used_info', exist_ok=True)

    df_not_used_images = pd.DataFrame(not_used_images)
    df_not_used_images.to_csv(Path(config['save_path']) / Path('image_used_info')/f'iter_{iter}_not_used_images.csv')

    df_used_train_images = pd.DataFrame(image_paths['train'])
    df_used_train_images.to_csv(Path(config['save_path']) / Path('image_used_info')/f'iter_{iter}_used_train_images.csv')

    # Initialize the model
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = timm.create_model(model_name, pretrained=True, num_classes=len(class_names), in_chans=3)

    if resume is not None:
        weights = torch.load(resume, map_location=device)
        classifier_state_dict = {k: v for k, v in weights.items() if 'head' not in k}
        model.load_state_dict(classifier_state_dict, strict=False)
        print(f'Loaded weight to resume training from {resume}')

    path_to_weights = resume

    model.to(device)
    model.set_grad_checkpointing()
    model.to(torch.float32)

    criterion = torch.nn.CrossEntropyLoss(**config['loss_parameter'])
    optimizer = AdamW(model.parameters(), **config['optimizer_parameter'])
    lr_scheduler = PlateauLRScheduler(optimizer, **config['lr_scheduler_parameter'])
    early_stopper = EarlyStopper(**config['early_stopping_parameter'])

    # Training
    wandb.config.update(save_parameter, allow_val_change=True)
    model.to(device)
    best_balanced_acc = 0

    for epoch in range(num_epochs):
        model.train()
        for images, labels in labeled_train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        predictions = []
        true_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                predictions.extend(predicted.tolist())
                true_labels.extend(labels.tolist())

            balanced_acc = balanced_accuracy_score(true_labels, predictions)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Balanced Accuracy: {balanced_acc}')
        wandb.log({"epoch": epoch + 1, "loss": loss.item(), "balanced_accuracy_val": balanced_acc})

        # Adjust learning rate using lr_scheduler
        lr_scheduler.step(balanced_acc)
        torch.save(model.state_dict(), save_path / Path('weights') / Path(f'epoch_{epoch}_xcit_model.pt'))

        # Check for early stopping
        if early_stopper(balanced_acc):
            print('Early stopping triggered!')
            continue

        if balanced_acc > best_balanced_acc:
            best_balanced_acc = balanced_acc
            path_to_weights = save_path / Path('weights') / Path(f"iter_{iter}_best_xcit_model.pt")
            torch.save(model.state_dict(), path_to_weights)

    return path_to_weights



if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    run()