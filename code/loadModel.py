import torch
from pathlib import Path


def print_model_info(model_path):
    # Carica il modello dalla cartella specificata
    model = torch.load(model_path)

    # Stampa il contenuto del modello
    print("Contenuto del Modello:")
    print(model)

    # Puoi anche esplorare ulteriormente i parametri del modello se necessario
    # Ad esempio, puoi stampare i parametri di ciascun layer con:
    # for name, param in model.named_parameters():
    #     print(f"Nome del parametro: {name}, Dimensione: {param.size()}")


if __name__ == "__main__":
    # Specifica il percorso del modello
    folder_path = "../dreamdiffusion/exps/results/generation/23-01-2024-19-39-15"
    model_name = "checkpoint.pth"
    model_path = Path(folder_path) / model_name

    # Chiama la funzione per stampare il contenuto del modello
    print_model_info(model_path)
