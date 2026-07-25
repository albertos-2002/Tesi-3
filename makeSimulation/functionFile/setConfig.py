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

# ==========================================
# Esempio di utilizzo
# ==========================================

# Supponiamo che il tuo file si chiami "param.config"
# carica_parametri_globali('param.config')

# Ora puoi usare le variabili direttamente nel tuo codice globale, ad esempio:
# print(IS_ZBL_ON) 
# print(TEMPERATURE * 2)
# print(PATH_POTENZIALE)
