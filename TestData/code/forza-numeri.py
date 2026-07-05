def trasforma_extxyz_in_numeri(file_input, file_output):
    print(f"Elaborazione del file '{file_input}' in corso...")
    
    linee_scritte = 0
    with open(file_input, 'r') as fin, open(file_output, 'w') as fout:
        for line in fin:
            # 1. Se è la riga dell'header, cambiamo il formato da Stringa (S) a Intero (I)
            if "Properties=" in line:
                line = line.replace("species:S:1", "species:I:1")
                fout.write(line)
                continue
            
            # 2. Se è una riga di dati atomici, sostituiamo il simbolo con il numero atomico
            parts = line.split()
            if len(parts) >= 4:  # È una riga con i dati dell'atomo (Specie, X, Y, Z...)
                if parts[0] == "Ta":
                    parts[0] = "73"
                    line = "   ".join(parts) + "\n"
                elif parts[0] == "O":
                    parts[0] = "8"
                    line = "   ".join(parts) + "\n"
            
            fout.write(line)
            linee_scritte += 1

    print(f"Fatto! Generato il file '{file_output}' con numeri atomici puri.")

if __name__ == "__main__":
    # Inserisci qui il nome del tuo file originale (es. xx00.txt)
    file_originale = "train_data.extxyz" 
    file_new = "train_numeri_puri.extxyz"
    
    trasforma_extxyz_in_numeri(file_originale, file_new)
