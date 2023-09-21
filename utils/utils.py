import pandas as pd
import timm
import git
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import torch
import yaml
from pathlib import Path




#-----------------------------------------------------------------------------------------------------------------------
def load_data(data_set):
    config, data_set_config = load_config(data_set)

    repo = git.Repo('./', search_parent_directories=True)
    current_path = Path(repo.working_tree_dir)


    class_names = data_set_config['class_names']

    splits = ['train','test']
    image_paths = {key:[] for key in (splits)}
    image_labels = {key:[] for key in (splits)}

    not_used_images = []

    for split in splits:
        data_split_path = current_path / Path(f'splits/{data_set}') / Path(split + '_split.csv')
        images_data = pd.read_csv(data_split_path, index_col=False)
        print(f'{split}_split loaded from {data_split_path}')

        img_paths = images_data['path'].tolist()
        img_labels = images_data['label'].tolist()
        for path,label in zip(img_paths,img_labels):
            if label in class_names:
                path = Path(path)
                # path = Path(*path.parts[1:])
                image_paths[split].append(data_set_config['image_root_path'] / path)
                image_labels[split].append(label)
            else:
                not_used_images.append(path)
                print(f'Label not in class_names: {label}; image path: {path}')
    return config,data_set_config,image_paths,image_labels,not_used_images

#-----------------------------------------------------------------------------------------------------------------------
def load_config(data_set):
    repo = git.Repo('./', search_parent_directories=True)
    current_path = Path(repo.working_tree_dir)
    with open(current_path / 'configs/dino_config.yml') as f:
        config = yaml.load(f, yaml.Loader)
    cfg_path = Path(f'configs/{data_set}.yml')
    with open(current_path / cfg_path) as f:
        data_set_config = yaml.safe_load(f)
    return config, data_set_config
#-----------------------------------------------------------------------------------------------------------------------
def load_model(device,model,model_parameter):
    try:
        backbone = eval(f'timm.models.{model}(**model_parameter)')
        print(f'Model: {model} \n')
    except:
        backbone = timm.models.xcit_small_12_p8_224_dist(**model_parameter)
        print(f'Could not load model {model}. Loaded: xcit_small_12_p8_224_dist')

    backbone.to(device)
    backbone.set_grad_checkpointing()
    backbone.to(torch.float32)

    return backbone

def load_weights(path_to_weights,teacher,device):
    weights = torch.load(Path(path_to_weights),map_location=device)
    weights = {key.replace('backbone.', ''): weight for key, weight in weights.items() if 'backbone.' in key}
    try:
        teacher.load_state_dict(weights)
        print(f'Loaded weight from {path_to_weights}')
    except:
        print(f'WARNING: weights could not been loaded! Model and weights do not fit! Check path: {path_to_weights}')

    return teacher