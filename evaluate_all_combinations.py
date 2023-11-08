import eval_ssl
import os
import warnings
from sklearn.exceptions import UndefinedMetricWarning

def evaluate_ssl():
    data = ["blood_acevedo", "blood_matek", "blood_raabin", "bm"]
    # data = ["blood_acevedo", "blood_matek"]

    for d_fit in data:
        for d_eval in data:
            os.system("python eval_ssl.py --dataset_fit " + d_fit + " --dataset_eval " + d_eval +
                      " --device cuda:0 --train false --path_to_weights "
                      "/home/crohling/Documents/cell_classification/Weights/pretrained_bm.pt --batch_size 256 "
                      "--teacher_momentum 0.9995 --model xcit_small_12_p8_224_dist --save_path "
                      "/home/crohling/Documents/cell_classification/unsupervised_dino_bm_25 --n_sample_eval 100 --n_iter "
                      "25")

def evaluate_supervised():
    data = ["blood_acevedo", "blood_matek", "blood_raabin", "bm"]
    for d_train in ["blood_matek"]:
        for d_eval in data:
            for i in range(3):
                i = i+1
                os.system("python eval_supervised.py --dataset_train " + d_train + " --dataset_eval " + d_eval +
                          " --device cuda:0 --batch_size 4 "
                          "--model_name xcit_small_12_p8_224_dist --save_path "
                          "/home/crohling/Documents/cell_classification/supervised_second --path_model_weights /home/"
                          "crohling/Documents/cell_classification/Weights/supervised_clemens/final/"+ d_train +"_"+str(i)+"_best_xcit_model.pt")

def train_supervised():
    data = ["blood_matek"]
    for d in data:
        os.system("python train_supervised.py --dataset "+ d +" --device cuda:0 --batch_size 4 --model_name "
        "xcit_small_12_p8_224_dist --save_path /home/crohling/Documents/cell_classification/Weights/supervised_clemens "
                  "--num_epochs 250 --resume /home/crohling/Documents/cell_classification/Weights/supervised_clemens/blood_matek/2023_10_19-2038/weights/epoch_99_xcit_model.pt")



def visualize_ssl():
    for file in os.listdir("/home/crohling/Documents/cell_classification/unsupervised_dino_bm_25"):
        print(os.listdir(file))



if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning, module='sklearn.metrics')
    warnings.filterwarnings("ignore", category=UserWarning)
    evaluate_supervised()

