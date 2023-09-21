import os
import json
import datetime
from copy import deepcopy
import dill
import numpy as np
import pandas as pd
from typing import Callable, Tuple, Sequence
from pathlib import Path

import warnings
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, precision_recall_fscore_support, confusion_matrix, balanced_accuracy_score

import tqdm

import click

from utils.utils import load_config, load_model, load_weights
from utils.ssl import extract_features, load_data_eval, map_results

from sparsam.helper import uniform_train_test_splitting, recursive_dict





#-----------------------------------------------------------------------------------------------------------------------
#%%
def summary_evaluation_results(report_path,n_sample,classifier):
    df_report = pd.read_csv(Path(report_path)/'report.csv')
    classifier_name = {'SVC':str("SVC(C=1, break_ties=True, cache_size=10000, class_weight='balanced',\n    probability=True)"),
                  'LR':"LogisticRegression(C=2.5, class_weight='balanced', max_iter=10000, n_jobs=-1)",
                  'KNN':"KNeighborsClassifier(n_jobs=-1, n_neighbors=3, weights='distance')"
                  }
    if n_sample == 0:
        n_sample = [1,5, 10, 25, 50, 100, 250, 500,750,1000,1500,2000]
    else:
        n_sample = [n_sample]

    if classifier == 'ALL':
        classifier = ['SVC','LR','KNN']
    else:
        classifier = list(classifier)

    df_results  = pd.DataFrame(columns=['classifier', 'n_sample', 'mean', 'std'])

    for sample in n_sample:
        df_report_sample = df_report.loc[df_report['n_sample']==sample]
        for one_classifier in classifier:
            df_report_class = df_report_sample.loc[df_report_sample['classifier']==classifier_name[one_classifier]]
            filtered_data = df_report_class['bal_acc'].dropna()
            mean = filtered_data.mean()
            std_dev = filtered_data.std()

            new_row = {
                'classifier': one_classifier,
                'n_sample': sample,
                'mean': mean,
                'std': std_dev
            }
            df_results = pd.concat([df_results, pd.DataFrame([new_row])], ignore_index=True)
    df_results.to_csv(Path(report_path)/'results_mean_std.csv')


def evaluate(device,path_to_weights,fit_dataset,eval_dataset,batch_size,model,n_iter,save_dir=None):
    config_fit, config_eval, train_loader, test_loader =  load_data_eval(fit_dataset,eval_dataset,batch_size=batch_size)
    time = datetime.datetime.now()
    if save_dir is None:
        save_dir = Path(config_fit['save_path'])/ Path('Evaluation')/ f'fit_{fit_dataset}_eval_{eval_dataset}' /Path(f'{time.year}_{time.month}_{time.day}-{time.hour}{time.minute}')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    config_fit['save_path'] = save_dir
    config_eval['save_path'] = save_dir

    # load weights
    if path_to_weights is not None:
        config_eval['weights_path'] = path_to_weights
    teacher = load_model(model=model,device=device, model_parameter=config_fit['model_parameter'])
    teacher = load_weights(path_to_weights,teacher,device=device)

    loader = {'train': train_loader, 'test': test_loader}
    features, labels = extract_features(device, config_fit, teacher, loader)
    features_eval, labels_eval = extract_features(device, config_eval, teacher, loader)

    features['test'], labels['test'] = features_eval['test'], labels_eval['test']



    results = recursive_dict()
    results_dict = recursive_dict()

    results_overview = {'n_sample': [], 'iteration': [], 'classifier': [], 'bal_acc': []}
    if n_iter is not None:
        config_fit['n_iterations'] = n_iter


    for eval_class_size in tqdm.tqdm(config_fit['eval_class_sizes']):
        for classifier_name in config_fit['classifier']:
            list_df_metics = []
            cfms = []
            for iteration in range(config_fit['n_iterations']):
                train_features, train_labels, _, _ = uniform_train_test_splitting(
                    features['train'], labels['train'], n_samples_class=eval_class_size, seed=int(iteration)
                )
                train_features = np.array(train_features)
                train_labels = np.array(train_labels)

                classifier_parameter = config_fit['classifier'][classifier_name]
                classifier = eval(classifier_name + f'(**{classifier_parameter})')

                pca = PCA()
                standardizer = PowerTransformer()
                classifier_pipeline = Pipeline([('standardizer', standardizer), ('pca', pca), ('classifier', classifier)])
                classifier_pipeline.fit(X=train_features, y=train_labels)
                test_probas = classifier_pipeline.predict_proba(features['test'])
                test_preds, eval_labels = map_results(test_probas,labels['test'],config_fit['class_names'],config_eval['class_names'])


                report = classification_report(eval_labels, test_preds,labels=config_fit['class_names'], target_names=config_fit['class_names'], output_dict=True)
                results[eval_class_size][classifier.__class__.__name__][iteration]['report'] = report

                test_bal_acc = balanced_accuracy_score(eval_labels, test_preds)
                os.makedirs(Path(config_fit['save_path']) / 'features_and_preds', exist_ok=True)
                with open(Path(config_fit['save_path']) /'features_and_preds'/ Path(f'test_features_{classifier_name}_{eval_class_size}_iter_{iteration}.pt'), 'wb') as h:
                    dill.dump((features['test'], labels['test'],test_preds), h)

                #save_confusionsmatrix
                cm = confusion_matrix(eval_labels, test_preds, labels = config_fit['class_names'])
                cfms.append(np.array(cm))
                cm_df = pd.DataFrame(cm, index=config_fit['class_names'], columns=config_fit['class_names'])
                os.makedirs(Path(config_fit['save_path'] )/'cmfs', exist_ok=True)
                cm_df.to_csv(Path(config_fit['save_path'] )/'cmfs'/ Path(f'confusion_matrix_{classifier_name}_{eval_class_size}_iter_{iteration}.csv'), index=True, header=True)

                precision, recall, f1, _ = precision_recall_fscore_support(eval_labels, test_preds, average=None)
                macro_precision = precision_score(eval_labels, test_preds, average='macro')
                macro_recall = recall_score(eval_labels, test_preds, average='macro')
                macro_f1 = f1_score(eval_labels, test_preds, average='macro')
                class_labels = np.unique(eval_labels)
                metrics_df = pd.DataFrame({'Precision': precision, 'Recall': recall, 'F1 Score': f1},
                                          index=class_labels)
                metrics_df.loc['All'] = [macro_precision, macro_recall, macro_f1]

                os.makedirs(Path(config_fit['save_path']) / 'metrics', exist_ok=True)
                metrics_df.to_csv(Path(config_fit['save_path']) / 'metrics' / Path(
                    f'metrics_{classifier_name}_{eval_class_size}_iter_{iteration}.csv'), index=True,
                             header=True)
                list_df_metics.append(metrics_df)
                results_overview['n_sample'].append(eval_class_size)
                results_overview['iteration'].append(iteration)
                results_overview['classifier'].append(classifier)
                results_overview['bal_acc'].append(test_bal_acc)

            mean_array = np.mean(cfms, axis=0)
            std_array = np.std(cfms, axis=0)
            os.makedirs(Path(config_fit['save_path']) / 'cmfs_overview', exist_ok=True)
            cm_mean_df = pd.DataFrame(mean_array, index=config_fit['class_names'], columns=config_fit['class_names'])
            cm_mean_df.to_csv(Path(config_fit['save_path']) / 'cmfs_overview' / Path(
                f'confusion_matrix_{classifier_name}_{eval_class_size}_mean.csv'), index=True, header=True)
            cm_std_df = pd.DataFrame(std_array, index=config_fit['class_names'], columns=config_fit['class_names'])
            cm_std_df.to_csv(Path(config_fit['save_path']) / 'cmfs_overview' / Path(
                f'confusion_matrix_{classifier_name}_{eval_class_size}_std.csv'), index=True, header=True)

            mean_df = pd.concat(list_df_metics).groupby(level=0).mean()
            std_df = pd.concat(list_df_metics).groupby(level=0).std()
            os.makedirs(Path(config_fit['save_path']) / 'metric_overview', exist_ok=True)
            mean_df.to_csv(Path(config_fit['save_path']) / 'metric_overview' / Path(
                f'metric_{classifier_name}_{eval_class_size}_mean.csv'), index=True, header=True)
            std_df.to_csv(Path(config_fit['save_path']) / 'metric_overview' / Path(
                f'metric_{classifier_name}_{eval_class_size}_std.csv'), index=True, header=True)

    results_df = pd.DataFrame({key: pd.Series(value) for key, value in report.items()})
    results_df.to_csv()


    results_dict['classifier'] = results
    results_df = pd.DataFrame({key: pd.Series(value) for key, value in results_overview.items()})
    results_df.to_csv(Path(config_fit['save_path']) / 'report.csv')
    with open(Path(config_fit['save_path'])/ 'results.json', 'w') as h:
        json.dump(results_dict, h)


    return Path(config_fit['save_path'])

#-----------------------------------------------------------------------------------------------------------------------
@click.command()
@click.option(
    '--dataset_fit', required=True, type=click.STRING,
    help='Decide which data set should be used for training: blood_acevedo, blood_matek, blood_raabin, bm',default='blood_matek'
 )
@click.option(
    '--dataset_eval', required=True, type=click.STRING,
    help='Decide which data set should be used for training: blood_acevedo, blood_matek, blood_raabin, bm',default='blood_matek'
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
    '--path_to_weights', type=click.STRING,required=False,
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
    '--save_path', type=click.STRING,required=True,
    help='Path to folder where result should be stored'
)
@click.option(
    '--max_train_image_number', type=click.INT,required=False,
    help='To only use #number of images to train self-supervised'
)
@click.option(
    '--n_iter', type=click.INT,required=False, default = 1,
    help='How often the classifier is fitted using random images per class'
)
@click.option(
    '--n_sample_eval', type=click.INT, required=False,
    help='choose n_sample, if 0 then all'
 )
@click.option(
    '--classifier', type=click.STRING, required=False, default='ALL',
    help='choose SVC, LR, KNN or ALL'
 )

def run(device,train,path_to_weights,dataset_fit,dataset_eval,batch_size,teacher_momentum,model,save_path,max_train_image_number,n_iter,
        n_sample_eval,classifier,):


    config, data_set_config = load_config(dataset_fit)


    #make new folder in save_path directory
    if save_path is None:
        save_path = config['save_path']
    time = datetime.datetime.now()
    save_path = Path(save_path)/ f'fit_{dataset_fit}_eval_{dataset_eval}'  /Path(f'{time.year}_{time.month}_{time.day}-{time.hour}{time.minute}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    config['save_path'] = save_path
    save_parameter = {'dataset': f'fit_{dataset_fit}_eval_{dataset_eval}' , 'device': device, 'if train': train, 'path_to_weights': path_to_weights,
                      'batch_size': batch_size,
                      'teacher_momentum': teacher_momentum, 'model': model, 'save_path': save_path,
                      'max_train_image_number': max_train_image_number, 'n_iter_eval': n_iter}
    df_save_parameter = pd.DataFrame([save_parameter])
    df_save_parameter.to_csv(Path(save_path)/'save_parameter.csv', index=False)

    # inizilize data_loader
    data_loader_parameter = config['data_loader_parameter']
    if batch_size is not None:
        data_loader_parameter['batch_size'] = batch_size
    test_loader_parameter = deepcopy(data_loader_parameter)
    test_loader_parameter['drop_last'] = False
    data_loader_parameter['drop_last'] = False


    report_path = evaluate(device, path_to_weights, dataset_fit,dataset_eval, batch_size, model, n_iter=n_iter,save_dir=Path(config['save_path'])/'evaluation')
    #update save parameter
    save_parameter = {'dataset': f'fit_{dataset_fit}_eval_{dataset_eval}', 'device': device, 'if train': train, 'path_to_weights': path_to_weights,
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
