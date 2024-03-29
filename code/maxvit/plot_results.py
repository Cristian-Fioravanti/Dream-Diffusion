import os
import json
import pandas as pd
import matplotlib.pyplot as plt

data = {}
for file in os.listdir('D:\\NN\\OurDreamDiffusion\\code\\maxvit\\results'):
    if file.endswith('.json'):
        file_path = os.path.join('D:\\NN\\OurDreamDiffusion\\code\\maxvit\\results', file)
        file_name = os.path.splitext(file)[0]
        parts = file_name.split('_')
        if parts[0] == 'formatted':
            model = parts[1]
            par1 = parts[2]
            par2 = parts[3]
            key = f"{model}_{par1}_{par2}"
            with open(file_path, 'r') as f:
                content = json.load(f)
            data[key] = {}
            data[key]['top0_perc'] = content['top0_perc']
            data[key]['perc'] = content['perc']

models = ["MaxViTTiny", "MaxViTSmall", "MaxViTBase"]
for amodel in models:
    par1_list =[]
    par2_list =[]
    top_list =[]
    perc_list =[]
    for key, values in data.items():
        splitted = key.split('_')
        model = splitted[0]
        if(model == amodel):
            par1_list.append(splitted[1])
            par2_list.append(splitted[2])
            top_list.append(f"{round(values['top0_perc'], 2)}%")
            perc_list.append(f"{round(values['perc'], 2)}%")
    df = pd.DataFrame({'Trained resolution': par1_list, 'Infer image size': par2_list, 'Top-0': top_list, 'Top-5': perc_list})
    plt.figure(figsize=(6, 2))
    plt.title(amodel)
    plt.table(cellText=df.values, colLabels=df.columns, loc='center')
    plt.axis('off')
    plt.tight_layout()
    plt.show()