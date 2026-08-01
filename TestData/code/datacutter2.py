from ase.io import read, write

def dividi_trajectory(file_ingresso, file_uscita_1="parte_1.traj", file_uscita_2="parte_2.traj"):
    """
    Divide un file .traj in due file distinti in base a un indice di frame specificato.

    Parametri:
    ----------
    file_ingresso : str
        Il percorso del file .traj da dividere.
    indice_taglio : int
        Il numero del frame su cui effettuare il taglio. 
        I frame da 0 a (indice_taglio - 1) andranno nel primo file.
        I frame da indice_taglio alla fine andranno nel secondo file.
    file_uscita_1 : str, opzionale
        Nome del primo file generato.
    file_uscita_2 : str, opzionale
        Nome del secondo file generato.
    """
    print(f"Caricamento di '{file_ingresso}'...")
    
    # 1. Legge tutti i frame presenti nel file .traj
    frames = read(file_ingresso, index=":")
    totale_frame = len(frames)
    indice_taglio = totale_frame//2
    
    # 2. Controllo che l'indice di taglio sia valido
    if indice_taglio <= 0 or indice_taglio >= totale_frame:
        raise ValueError(
            f"L'indice di taglio ({indice_taglio}) non è valido. "
            f"Deve essere compreso tra 1 e {totale_frame - 1} (totale frame presenti: {totale_frame})."
        )
    
    # 3. Separazione dei frame usando lo slicing delle liste Python
    parte_1 = frames[:indice_taglio]
    parte_2 = frames[indice_taglio:]
    
    # 4. Scrittura dei due nuovi file .traj
    write(file_uscita_1, parte_1)
    write(file_uscita_2, parte_2)
    
    print("\n--- Operazione Completata ---")
    print(f"Totale frame nel file originale: {totale_frame}")
    print(f"-> File 1 ('{file_uscita_1}'): contiene {len(parte_1)} frame (da 0 a {indice_taglio - 1})")
    print(f"-> File 2 ('{file_uscita_2}'): contiene {len(parte_2)} frame (da {indice_taglio} a {totale_frame - 1})")


# ==========================================
# Esempio d'uso:
# ==========================================
if __name__ == "__main__":
    # Esempio: tagliamo il file "simulazione.traj" al frame numero 500
    dividi_trajectory(
        file_ingresso="traiettoria_unita.traj", 
        file_uscita_1="train_data.traj", 
        file_uscita_2="test_data.traj"
    )
