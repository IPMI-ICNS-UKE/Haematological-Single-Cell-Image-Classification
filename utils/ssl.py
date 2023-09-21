import git
from copy import deepcopy
from pathlib import Path
from typing import Callable, Tuple, Sequence
import tqdm
from PIL import Image
import torch
import pandas as pd
import numpy as np
import dill
from sklearn.metrics import classification_report

from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

from sparsam.utils import model_inference, ModelMode
from sparsam.dataset import ImageSet
from sparsam.helper import uniform_train_test_splitting
from utils.utils import load_config

#-----------------------------------------------------------------------------------------------------------------------
def load_data_eval(data_set,batch_size=None):
    repo = git.Repo('./', search_parent_directories=True)
    current_path = Path(repo.working_tree_dir)
    config, data_set_config = load_config(data_set)

    splits = ['train','test']
    image_paths = dict.fromkeys(splits)
    image_labels = dict.fromkeys(splits)
    for split in splits:
        if data_set_config['split_root_path'] is not None:
            data_split_path = data_set_config['split_root_path'] / Path(split + '_split.csv')
            try:
                images_data = pd.read_csv(data_split_path,index_col=False)
                print(f'{split}_split loaded from {data_split_path}')
            except FileNotFoundError:
                print(f'No file named {Path(split + "_split.csv")} in directory {data_set_config["split_root_path"]} ')
                data_split_path = current_path / Path(f'splits/{data_set}') / Path(split + '_split.csv')
                images_data = pd.read_csv(data_split_path,
                                          index_col=False)
                print(f'{split}_split loaded from {data_split_path}')
        else:
            data_split_path = current_path / Path(f'splits/{data_set}') / Path(split + '_split.csv')
            images_data = pd.read_csv(data_split_path, index_col=False)
            print(f'{split}_split loaded from {data_split_path}')

        image_paths[split] = images_data['path'].tolist()
        for index,path in enumerate(image_paths[split]):
            path = Path(path)
            # path = Path(*path.parts[1:])
            image_paths[split][index] = data_set_config['image_root_path'] / path
        image_labels[split] = images_data['label'].tolist()

    train_set = ImageSet(img_paths=image_paths['train'],
                        labels=image_labels['train'],
                        class_names=data_set_config['class_names'],
                        normalize=True)

    # This is the independent test set
    test_set = ImageSet(img_paths=image_paths['test'],
                        labels=image_labels['test'],
                        class_names=data_set_config['class_names'],
                        normalize=True)

    # inizilize data_loader
    data_loader_parameter = config['data_loader_parameter']
    test_loader_parameter = deepcopy(data_loader_parameter)
    test_loader_parameter['drop_last'] = False

    if batch_size:
        data_loader_parameter['batch_size'] = batch_size

    train_loader = DataLoader(train_set, **data_loader_parameter)
    test_loader = DataLoader(test_set, **test_loader_parameter)
    for key in data_set_config.keys():
        config[key] = data_set_config[key]

    return config, train_loader, test_loader

#-----------------------------------------------------------------------------------------------------------------------
def edit_used_images(image_paths,image_labels,seed=0, max_train_size=None):
    if max_train_size is not None:
        image_paths['train'],_, image_labels['train'], _ = train_test_split(image_paths['train'], image_labels['train'], train_size=max_train_size, random_state=seed)
    return image_paths, image_labels

#-----------------------------------------------------------------------------------------------------------------------
def create_dataset(data_set_config,image_paths,image_labels,data_augmentation=None):
    class_names = data_set_config['class_names']
    unlabeled_train_set = ImageSet(img_paths=image_paths['train'],
                                   class_names=class_names,
                                   normalize=True,data_augmentation=data_augmentation)

    # These Datasets are optional: the labeled_train_set and val_set may be used to track the process during SSL
    # These Datasets are required, if a classifier is fitted after SSL to a specific task.
    X_train, X_validation, y_train, y_validation = train_test_split(image_paths['train'], image_labels['train'],
                                                                    test_size=0.33, random_state=42, shuffle=True)
    X_train, y_train, _, _ = uniform_train_test_splitting(X_train, y_train, n_samples_class=100)
    X_validation, y_validation, _, _ = uniform_train_test_splitting(X_validation, y_validation, n_samples_class=500)
    labeled_train_set = ImageSet(img_paths=X_train,
                                 labels=y_train,
                                 class_names=class_names,
                                 normalize=True)
    labeled_val_set = ImageSet(img_paths=X_validation,
                               labels=y_validation,
                               class_names=class_names,
                               normalize=True)

    # This is the independent test set
    test_set = ImageSet(img_paths=image_paths['test'],
                        labels=image_labels['test'],
                        class_names=class_names,
                        normalize=True)
    return unlabeled_train_set, labeled_train_set, labeled_val_set, test_set


#-----------------------------------------------------------------------------------------------------------------------
def extract_features(device, config, model, loader, splits=['train', 'test']):
    """
    Extract features from a neural network model for the specified data splits and save them to disk.

    Args:
        device (str): Device for inference (e.g., 'cuda:0').
        config (dict): Configuration parameters.
        model: The neural network model to use for inference.
        loader (dict): Dictionary of data loaders for different splits (e.g., 'train', 'test').
        splits (list, optional): List of data splits to process (default is ['train', 'test']).

    Returns:
        dict: A dictionary containing extracted features and labels for each data split.
    """
    features = dict.fromkeys(splits)
    labels = dict.fromkeys(splits)
    for split in splits:
        features[split], labels[split] = model_inference(
            loader[split],
            model=model,
            mode=ModelMode.EXTRACT_FEATURES,
            device=device
        )

        with open(config['save_path'] / Path(split + '_feature.pt'), 'wb') as h:
            dill.dump((features[split], labels[split]), h)
        print(f'Saved features to: {config["save_path"] / Path(split + "_feature.pt")}')
    for split in splits:
        features[split] = np.array(features[split])
        labels[split] = np.array(labels[split])
    return features, labels

#-----------------------------------------------------------------------------------------------------------------------

def calculate_predictions(classifier_pipeline,features,labels):
    # fit classifier, predict test features, generate classification report
    classifier_pipeline.fit(features['train'], labels['train'])
    preds = classifier_pipeline.predict(features['test'])
    report = classification_report(labels['test'], preds, output_dict=True, zero_division=0)
    return report
#-----------------------------------------------------------------------------------------------------------------------
def load_paths(data_set_config,split,current_path,data_set):

    data_split_path = current_path / Path(f'splits/{data_set}') / Path(split + '_split.csv')
    images_data = pd.read_csv(data_split_path, index_col=False)
    print(f'{split}_split loaded from {data_split_path}')

    paths = images_data['path'].tolist()
    for index, path in enumerate(paths):
        path = Path(path)
        # path = Path(*path.parts[1:])
        paths[index] = data_set_config['image_root_path'] / path
    labels = images_data['label'].tolist()
    return paths, labels

def intersect_classes(paths,labels,class_names_train,class_names_eval):
    class_names = set(class_names_train).intersection(class_names_eval)
    class_names = list(class_names)
    if ('MYC' in class_names_eval) or ('MYC' in class_names_train):
        print('MYC added')
        class_names.extend(['MYC','MMZ','PMO','MYB'])
    if 'NGB' not in class_names:
        print('NGB added')
        class_names.append('NGB')

    paths_keep = []
    labels_keep = []
    labels_not_keep = []
    for path,label in zip(paths,labels):
        if label in class_names:
            paths_keep.append(path)
            labels_keep.append(label)
        else:
            labels_not_keep.append(label)
    print(f'Kept classes {set(labels_keep)}')
    print(f'Dont kept {set(labels_not_keep)}')
    return paths_keep, labels_keep, list(set(labels_keep))

def load_data_eval(data_set_fit,dataset_eval,batch_size=None):
    repo = git.Repo('./', search_parent_directories=True)
    current_path = Path(repo.working_tree_dir)
    config_fit, data_set_config_fit = load_config(data_set_fit)
    config_eval, data_set_config_eval = load_config(dataset_eval)

    splits = ['train','test']
    image_paths = dict.fromkeys(splits)
    image_labels = dict.fromkeys(splits)

    image_paths['train'],image_labels['train'] = load_paths(data_set_config_fit, 'train', current_path, data_set_fit)
    image_paths['test'], image_labels['test'] = load_paths(data_set_config_eval, 'test', current_path, dataset_eval)

    image_paths['train'], image_labels['train'], data_set_config_fit['class_names'] = intersect_classes(paths=image_paths['train'],
                                                                  labels=image_labels['train'],
                                                                  class_names_train=data_set_config_fit['class_names'],
                                                                  class_names_eval=data_set_config_eval['class_names'])

    image_paths['test'], image_labels['test'],data_set_config_eval['class_names']  = intersect_classes(paths=image_paths['test'],labels=image_labels['test'],class_names_train=data_set_config_fit['class_names'],class_names_eval=data_set_config_eval['class_names'])

    train_set = ImageSet(img_paths=image_paths['train'],
                        labels=image_labels['train'],
                        class_names=data_set_config_fit['class_names'],
                        normalize=True)
    test_set = ImageSet(img_paths=image_paths['test'],
                        labels=image_labels['test'],
                        class_names=data_set_config_eval['class_names'],
                        normalize=True)

    # inizilize data_loader
    data_loader_parameter = config_fit['data_loader_parameter']
    data_loader_parameter['drop_last'] = False
    test_loader_parameter = config_eval['data_loader_parameter']
    test_loader_parameter['drop_last'] = False

    if batch_size:
        data_loader_parameter['batch_size'] = batch_size
        test_loader_parameter['batch_size'] = batch_size

    train_loader = DataLoader(train_set, **data_loader_parameter)
    test_loader = DataLoader(test_set, **test_loader_parameter)
    for key in data_set_config_fit.keys():
        config_fit[key] = data_set_config_fit[key]
    for key in data_set_config_eval.keys():
        config_eval[key] = data_set_config_eval[key]

    return config_fit, config_eval, train_loader, test_loader
#-----------------------------------------------------------------------------------------------------------------------
def load_image_as_np(image_path, target_size=(128,128)):
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        if target_size is not None:
            img = img.resize(target_size)
        img_np = np.array(img)
    return img_np




#-----------------------------------------------------------------------------------------------------------------------
def map_results(test_probas,labels_test,class_names_train,class_names_eval):
    labels_eval = labels_test
    outputs = torch.from_numpy(test_probas)

    _, predicted = torch.max(outputs, 1)
    train_index_to_string = {idx: class_name for idx, class_name in enumerate(class_names_train)}
    predicted = [train_index_to_string[idx.item()] for idx in predicted]
    if 'MYC' in class_names_eval:
        predicted = ['MYC' if label in ['MMZ', 'PMO', 'MYB'] else label for label in predicted]
    if 'NGB' not in class_names_eval:
        predicted = ['NGS' if label in ['NGB'] else label for label in predicted]

    eval_index_to_class = {idx: class_name for idx, class_name in enumerate(class_names_eval)}
    labels_eval = [eval_index_to_class[idx.item()] for idx in labels_eval]

    return predicted, labels_eval

#-----------------------------------------------------------------------------------------------------------------------




