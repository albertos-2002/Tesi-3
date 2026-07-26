from pathlib import Path

def carica_parametri_globali(filepath):
    """
    Legge un file di configurazione e imposta i parametri come variabili globali.
    Gestisce automaticamente stringhe, interi, float e booleani.
    """
    # Insiemi per riconoscere varie forme di booleani (case-insensitive)
    valori_veri = {'t', 'true', '1', 'y', 'yes', 'on'}
    valori_falsi = {'f', 'false', '0', 'n', 'no', 'off'}

    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            for riga in file:
                # 1. Rimuoviamo i commenti e gli spazi bianchi finali/iniziali
                riga_pulita = riga.split('#')[0].strip()
                
                # 2. Ignoriamo le righe vuote o senza il carattere ':'
                if not riga_pulita or ':' not in riga_pulita:
                    continue
                
                # 3. Separiamo la chiave dal valore
                chiave, valore = riga_pulita.split(':', 1)
                chiave = chiave.strip()
                valore = valore.strip()
                
                # Se il valore è vuoto dopo i due punti, lo impostiamo a None (o puoi scegliere di ignorarlo)
                if not valore:
                    globals()[chiave] = None
                    continue

                # 4. Parsing del valore
                valore_lower = valore.lower()
                
                # Controllo booleani
                if valore_lower in valori_veri:
                    valore_parsato = True
                elif valore_lower in valori_falsi:
                    valore_parsato = False
                else:
                    # Controllo numerici (int o float)
                    try:
                        if '.' in valore:
                            valore_parsato = float(valore)
                        else:
                            valore_parsato = int(valore)
                    except ValueError:
                        # Se non è né booleano né numero, rimane una stringa (es. i PATH)
                        valore_parsato = valore
                
                # 5. Assegnazione globale
                globals()[chiave] = valore_parsato
                
    except FileNotFoundError:
        print(f"Errore: Il file '{filepath}' non esiste.")
    except Exception as e:
        print(f"Si è verificato un errore inaspettato: {e}")

    #Controllo e creazione delle cartelle necessarie
    #Crea la cartella (e tutte le sottocartelle necessarie) se non esiste
    Path(PATH_OUT_FILE).mkdir(parents=True, exist_ok=True)
    print("Creata la cartella per i file di out")
    Path(PATH_OUT_GRAPH).mkdir(parents=True, exist_ok=True)
    print("Creata la cartella per i graph di out")    

# =========================================================================
#Dump automatico dei parametri registrati nella cartella di output
# =========================================================================
    try:
        path_dump = Path(PATH_OUT_FILE) / "param.log"
        with open(path_dump, 'w', encoding='utf-8') as file_dump:
            file_dump.write("# ==========================================\n")
            file_dump.write("# Dump dei parametri di configurazione caricati\n")
            file_dump.write("# ==========================================\n\n")
            
            for chiave, valore in globals().items():
                # Filtra per salvare solo le variabili in MAIUSCOLO
                if chiave.isupper() and not chiave.startswith('__'):
                    if isinstance(valore, bool):
                        val_str = "True" if valore else "False"
                    elif valore is None:
                        val_str = ""
                    else:
                        val_str = str(valore)
                        
                    file_dump.write(f"{chiave}: {val_str}\n")
                    
        print(f"Dump dei parametri salvato in: {path_dump}")
    except Exception as e:
        print(f"Errore durante il dump dei parametri: {e}")


# ==========================================
# Esempio di utilizzo
# ==========================================

# Supponiamo che il tuo file si chiami "param.config"
# carica_parametri_globali('param.config')

# Ora puoi usare le variabili direttamente nel tuo codice globale, ad esempio:
# print(IS_ZBL_ON) 
# print(TEMPERATURE * 2)
# print(PATH_POTENZIALE)




