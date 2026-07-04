import os
import numpy as np
from ase.io import read, write

def converti_file_extxyz(file_input, file_output):
    """
    Legge un file extxyz e sostituisce i simboli chimici (stringhe) 
    con i rispettivi numeri atomici (interi), generando un file compatibile con gap_fit.
    """
    if not os.path.exists(file_input):
        print(f"Errore: Il file '{file_input}' non esiste nella cartella corrente!")
        return

    print(f"Lettura del file '{file_input}' in corso...")
    # Carichiamo la configurazione (o le configurazioni, nel caso di traiettorie)
    configs = read(file_input, index=":")
    
    print("Conversione delle specie chimiche in numeri atomici interi...")
    for atoms in configs:
        # Estraiamo i numeri atomici (es. Ta -> 73, O -> 8)
        numeri_atomici = atoms.get_atomic_numbers()
        
        # Sovrascriviamo l'array 'species' con numeri interi a 32 bit.
        # Questo costringerà ASE a scrivere "species:I:1" nella testata del file.
        atoms.arrays['species'] = np.array(numeri_atomici, dtype=np.int32)
        
    print(f"Scrittura del nuovo file convertito in '{file_output}'...")
    # Salviamo nel formato extxyz standard
    write(file_output, configs, format="extxyz")
    
    # Verifica rapida del file appena scritto
    with open(file_output, "r") as f:
        print("\n--- Anteprima del file generato (prime 3 righe) ---")
        for _ in range(3):
            print(f.readline().strip())
    print("-" * 50)
    print("Conversione completata con successo!")

if __name__ == "__main__":
    # Sostituisci con il nome del tuo file di input e il nome desiderato per l'output
    file_originale = "train_data.extxyz"
    file_convertito = "train_data_natom2.extxyz"
    
    converti_file_extxyz(file_originale, file_convertito)
