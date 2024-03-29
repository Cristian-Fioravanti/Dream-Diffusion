import time
import tensorflow as tf
import maxvit.models.hparams as hparams
import maxvit.models.maxvit as layers
import maxvit.models.eval_ckpt as eval_ckpt
import json

# Checkpoints location
CKPTS_DIRS = {
    'MaxViTTiny_i1k_224': 'ckpts/maxvittiny/i1k/224',
    'MaxViTTiny_i1k_384': 'ckpts/maxvittiny/i1k/384',
    'MaxViTTiny_i1k_512': 'ckpts/maxvittiny/i1k/512',
    'MaxViTSmall_i1k_224': 'ckpts/maxvitsmall/i1k/224',
    'MaxViTSmall_i1k_384': 'ckpts/maxvitsmall/i1k/384',
    'MaxViTSmall_i1k_512': 'ckpts/maxvitsmall/i1k/512',
    'MaxViTBase_i1k_224': 'ckpts/maxvitbase/i1k/224',
    'MaxViTBase_i1k_384': 'ckpts/maxvitbase/i1k/384',
    'MaxViTBase_i1k_512': 'ckpts/maxvitbase/i1k/512',
    'MaxViTLarge_i1k_224': 'ckpts/maxvitlarge/i1k/224',
    'MaxViTLarge_i1k_384': 'ckpts/maxvitlarge/i1k/384',
    'MaxViTLarge_i1k_512': 'ckpts/maxvitlarge/i1k/512',
}

DATASET_MAP = {
    'ImageNet-1K': 'i1k'
}


MODEL_NAME = "MaxViTSmall" #@param ["MaxViTTiny", "MaxViTSmall", "MaxViTBase", "MaxViTLarge"] {type:"string"}
TRAIN_SET = "ImageNet-1K"
TRAIN_IMAGE_SIZE = "512" #@param [224, 384, 512] {type:"string"}

CKPT_DIR = CKPTS_DIRS[f'{MODEL_NAME}_{DATASET_MAP[TRAIN_SET]}_{TRAIN_IMAGE_SIZE}']

import maxvit.models.eval_ckpt as eval_ckpt
import os

#@markdown ### Enter a file path:
# file_path = "https://upload.wikimedia.org/wikipedia/commons/f/fe/Giant_Panda_in_Beijing_Zoo_1.JPG" #@param {type:"string"}
dir_path = "photos"
INFER_IMAGE_SIZE = "1024" #@param [224, 384, 448, 512, 672, 768, 896, 1024] {type:"string"}

# Download label map file and image
labels_map_file = 'labels_map.json'
set_path = "testset" #@param ["testset", "trainset"] {type:"string"}
dir_path = dir_path + "/" + set_path
test_set = dir_path + "/test"
real_set = dir_path + "/real"
files_jpg = os.listdir(test_set)
numero_di_file = len(files_jpg)

pred_dict = {}
results_file_json = f"results/results_{MODEL_NAME}_{DATASET_MAP[TRAIN_SET]}_{TRAIN_IMAGE_SIZE}_{INFER_IMAGE_SIZE}.json"
if os.path.exists(results_file_json):
    with open(results_file_json, 'r') as f:
        pred_dict = json.load(f)

start_time = time.time()
# for index in range(numero_di_file):
eval_driver = eval_ckpt.MaxViTDriver(
            model_name=MODEL_NAME,
            model_input_size=TRAIN_IMAGE_SIZE,
            batch_size=1,
            image_size=int(INFER_IMAGE_SIZE),
            include_background_label=False,
            advprop_preprocessing=False,)
for file in files_jpg:
    mini_start_time = time.time()
    index = file.split(".")[0]
    # test_image_path = test_set + "/" + str(index) + ".jpg"
    # real_image_path = real_set + "/" + str(index) + ".jpg"
    if(not index in pred_dict):
        test_image_path = test_set + "/" + file
        real_image_path = real_set + "/" + file
        image_files = [test_image_path, real_image_path]

        print(f"MaxViT prediction {file}:")
        pred_idx, pred_prob, pred_classes = eval_driver.eval_example_images(CKPT_DIR, image_files, labels_map_file)
        
        pred_dict[index] = {}
        pred_dict[index]["Index"] = [pred.tolist() for pred in pred_idx]
        pred_dict[index]["Prob"] = pred_prob
        pred_dict[index]["Class"] = pred_classes
        mini_end_time = time.time()
        execution_time = mini_end_time - mini_start_time
        print(f"Tempo di esecuzione: {execution_time} secondi")

end_time = time.time()
execution_time = end_time - start_time
print(f"Tempo di esecuzione TOTALE: {execution_time} secondi")

top0_perc = 0
perc = 0
num = numero_di_file
copy_pred_dict = pred_dict
top0_perc_index = []
perc_index = []
for counter, data in pred_dict.items():
    if isinstance(data, dict):
        pred_indexes = data["Index"]
        pred_probs = data["Prob"]
        pred_classes = data["Class"]
        top0_test = pred_indexes[0][0]
        top0_real = pred_indexes[1][0]
        # print(top0_test, top0_real)
        if top0_test == top0_real:
            top0_perc_index.append(counter)
            top0_perc += 1
        if top0_real in pred_indexes[0]:
            perc_index.append(counter)
            perc += 1
top0_perc = top0_perc * 100
perc = perc * 100
pred_dict["top0_perc"] = top0_perc/num
pred_dict["perc"] = perc/num
pred_dict["top0_perc_index"] = top0_perc_index
pred_dict["perc_index"] = perc_index
print(num, top0_perc/num, perc/num, top0_perc_index, perc_index)
with open(results_file_json, "w") as file:
    file.write(str(pred_dict))
    print("Finito")
    file.close()