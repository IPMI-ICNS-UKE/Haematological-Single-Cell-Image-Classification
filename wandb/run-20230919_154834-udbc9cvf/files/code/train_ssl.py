import os
import datetime
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Callable, Tuple, Sequence

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.exceptions import UndefinedMetricWarning

import torchvision

import warnings

import torch
from torch.utils.data import DataLoader

import click
import wandb

from sparsam.train import create_dino_gym
from utils.utils import  load_data, load_model, load_weights
from utils.ssl import  create_dataset, edit_used_images
from eval_ssl import evaluate, summary_evaluation_results



#-----------------------------------------------------------------------------------------------------------------------
@click.command()
@click.option(
    '--dataset', required=True, type=click.STRING,
    help='Decide which data set should be used for training:  blood_acevedo, blood_matek, blood_raabin, bm',default='blood_matek'
 )
@click.option(
    '--device', type=click.STRING, required=True,default='cuda:0',
    help='Which GPU should be used: e.g. for first GPU: cuda:0'
)
@click.option(
    '--train',
    help='If you only want to evaluate a trained model choose False. If you want to train a model choose True.',
    type=click.BOOL, required=True,default=True,
)
@click.option(
    '--path_to_weights', type=click.STRING,required=False, default = '/home/lwenderoth/Documents/Promotion/Results/Weights/bm/pretrained_bm.pt',
    help='Need if train is False. Weights need to match the model structure.'
)
@click.option(
    '--batch_size', type=click.INT,required=True,default=4,
    help='Batchsize.'
)
@click.option(
    '--teacher_momentum', type=click.FLOAT,required=True, default=0.9995,
    help='Scale down for higher batchsizes.'
)
@click.option(
    '--model', type=click.STRING,required=True, default='xcit_small_12_p8_224_dist',
    help='choose model from timm libary'
)
@click.option(
    '--save_path', type=click.STRING,required=False, default='/home/lwenderoth/Documents/Doktorarbeit/Results',
    help='Path to folder where result should be stored'
)
@click.option(
    '--max_train_image_number', type=click.INT,required=False,
    help='To only use #number of images to train self-supervised'
)
@click.option(
    '--n_iter', type=click.INT,required=False,
    help='How often the classifier is fitted using random images per class'
)
@click.option(
    '--n_sample_eval', type=click.INT, required=False, default=50,
    help='choose n_sample, if 0 then all'
 )
@click.option(
    '--classifier', type=click.STRING, required=False, default='ALL',
    help='choose SVC, LR or KNN or ALL'
 )
@click.option(
    '--resume', type=click.STRING, required=False,
    help='Path to weight to resume training from there.'
 )
@click.option(
    '--data_augment', type=click.BOOL, required=False, default = True,
    help='True if one wants to do a seperate augmentation.'
 )
def run(device,train,path_to_weights,dataset,batch_size,teacher_momentum,model,save_path,max_train_image_number,n_iter,
        n_sample_eval,classifier,resume,data_augment):

    if data_augment:
        data_augmentation = torchvision.transforms.Compose([
            torchvision.transforms.RandomApply([
                torchvision.transforms.Grayscale(num_output_channels=3)
            ], p=0.05),
            torchvision.transforms.RandomHorizontalFlip()
        ])
    else:
        data_augmentation = None

    config, data_set_config, image_paths, image_labels,not_used_images = load_data(dataset)


    #make new folder in save_path directory
    if save_path is None:
        save_path = config['save_path']
    time = datetime.datetime.now()
    save_path = Path(save_path)/ dataset /Path(f'{time.year}_{time.month}_{time.day}-{time.hour}{time.minute}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    config['save_path'] = save_path
    save_parameter = {'dataset': dataset, 'device': device, 'if train': train, 'path_to_weights': path_to_weights,
                      'batch_size': batch_size,
                      'teacher_momentum': teacher_momentum, 'model': model, 'save_path': save_path,
                      'max_train_image_number': max_train_image_number, 'n_iter_eval': n_iter}
    df_save_parameter = pd.DataFrame([save_parameter])
    df_save_parameter.to_csv(Path(save_path)/'save_parameter.csv', index=False)
    if max_train_image_number is not None:
        image_paths, image_labels = edit_used_images(image_paths, image_labels, seed=0, max_train_size=max_train_image_number)
    unlabeled_train_set, labeled_train_set, labeled_val_set, test_set = create_dataset(data_set_config, image_paths, image_labels,data_augmentation=data_augmentation)

    df_not_used_images = pd.DataFrame(not_used_images)
    df_not_used_images.to_csv(Path(config['save_path']) / 'not_used_images.csv')
    df_used_train_images = pd.DataFrame(image_paths['train'])
    df_used_train_images.to_csv(Path(config['save_path']) / 'used_train_images.csv')

    # inizilize data_loader
    data_loader_parameter = config['data_loader_parameter']
    if batch_size is not None:
        data_loader_parameter['batch_size'] = batch_size
    test_loader_parameter = deepcopy(data_loader_parameter)
    test_loader_parameter['drop_last'] = False


    labeled_train_loader = DataLoader(labeled_train_set, **data_loader_parameter)
    val_loader = DataLoader(labeled_val_set, **data_loader_parameter)

    backbone = load_model(model=model,model_parameter=config['model_parameter'],device=device)

    classifier_pipeline = Pipeline([
        ('standardizer', PowerTransformer()),
        ('pca', PCA()),
        ('classifier', SVC(probability=True))
    ])

    if train:
        if resume is not None:
            try:
                backbone = load_weights(resume, backbone,device)
                print(f'Resumed with weigth from {resume}')
            except:
                print(f'Not a path to weights: {resume}')

        config['wandb_parameter']['group'] = dataset
        wandb.init(**config['wandb_parameter'])
        # There are further specific options like the scheduler, optimizer, data_augmentation etc. which may be chosen by the
        # user, but are not covered in this example. To see the options please inspect the "create_dino_gym" function
        gym = create_dino_gym(
            unalabeled_train_set=unlabeled_train_set,
            labeled_train_loader=labeled_train_loader,
            val_loader=val_loader,
            backbone_model=backbone,
            classifier=classifier_pipeline,
            logger=wandb,
            unlabeled_train_loader_parameters=data_loader_parameter,
            save_path=config['save_path'],
            device=device,
            metrics=[
                partial(classification_report, output_dict=True, zero_division=0),
            ],
            metrics_requires_probability=[False],
            resume_training_from_checkpoint=False,
            # Note: should be scaled down for larger batch sizes (0.996 for a batch size of 512) and up for smaller ones (see
            # https://arxiv.org/abs/2104.14294 for details)
            teacher_momentum=teacher_momentum,
            **config['gym']
        )

        student, teacher = gym.train()
        teacher.to(device)
        teacher.to(torch.float32)
        path_to_weights = Path(config['save_path']) / 'teacher.pt'
        torch.save(teacher.state_dict(), path_to_weights)

    report_path = evaluate(device, path_to_weights,fit_dataset=dataset, eval_dataset=dataset, batch_size=batch_size, model=model, n_iter=n_iter,
                           save_dir=Path(config['save_path']) / 'evaluation')
    # update save parameter
    save_parameter = {'dataset': f'fit_{dataset}_eval_{dataset}', 'device': device, 'if train': train,
                      'path_to_weights': path_to_weights,
                      'batch_size': batch_size,
                      'teacher_momentum': teacher_momentum, 'model': model, 'save_path': save_path,
                      'max_train_image_number': max_train_image_number, 'n_iter_eval': n_iter}
    df_save_parameter = pd.DataFrame([save_parameter])
    df_save_parameter.to_csv(Path(save_path) / 'save_parameter.csv', index=False)

    if n_sample_eval is not None:
        config['n_iterations'] = n_sample_eval

    summary_evaluation_results(report_path, n_sample=config['n_iterations'], classifier=classifier)

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning, module='sklearn.metrics')
    warnings.filterwarnings("ignore", category=UserWarning)
    run()
