import os

for file in os.listdir('results'):
    if file.endswith('.json'):
        percorso_file = os.path.join('results', file)
        nome_file = os.path.splitext(file)[0]
        parti_nome = nome_file.split('_')
        if(parti_nome[0]=='results'):
            modello = parti_nome[1]
            par1 = parti_nome[3]
            par2 = parti_nome[4]
            
            chiave = f"{modello}_{par1}_{par2}"
            
            with open(percorso_file, 'r') as f:
                contenuto = f.read()
            
            contenuto = contenuto.replace("j's", 'js')
            contenuto = contenuto.replace("r's", 'rs')
            contenuto = contenuto.replace("y's", 'ys')
            contenuto = contenuto.replace("e's", 'es')
            contenuto = contenuto.replace("l's", 'ls')
            contenuto = contenuto.replace("'-", '-')
            contenuto = contenuto.replace("'", '"')
            with open(f"results/formatted_{chiave}.json", 'w') as f:
                f.write(contenuto)