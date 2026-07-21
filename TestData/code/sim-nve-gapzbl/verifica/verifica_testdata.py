import numpy as np
from ase.io import iread
from quippy.potential import Potential

# =========================================================================
# CONFIGURAZIONE PERCORSI (MODIFICA QUI)
# =========================================================================
PATH_POTENZIALE = "../../training-gap-zbl/out_potenziale_gap_500_n2_l2_zbl.xml"
PATH_TRAIETTORIA = "../../test_data.extxyz"  # Il file che contiene i frame da controllare
PATH_OUTPUT_ENERGIE = "energie_predette_gap-testdata.txt"  # Il file dove salveremo i risultati

# =========================================================================
# 1. CARICAMENTO DEL POTENZIALE (IL CALCOLATORE)
# =========================================================================
print("=" * 60)
print(" AVVIO CALCOLO ENERGIE STATICHE CON POTENZIALE GAP")
print("=" * 60)

print(f"Caricamento del potenziale da: {PATH_POTENZIALE}...")
# Nota: usa la sintassi corretta per la tua versione (con o senza 'IP GAP')
calc = Potential(param_filename=PATH_POTENZIALE)
calc.name_ = "GAP"

# =========================================================================
# 2. LETTURA DEI FRAME E CALCOLO DELL'ENERGIA
# =========================================================================
print(f"Lettura della traiettoria da: {PATH_TRAIETTORIA}...")

# Lista vuota in cui salveremo l'energia di ogni frame
energie_calcolate = []

# Usiamo iread per non intasare la memoria RAM se il file è molto grande
for indice, atomi in enumerate(iread(PATH_TRAIETTORIA)):
    
    # Colleghiamo il potenziale GAP a questo specifico frame della traiettoria
    atomi.calc = calc
    
    # Calcoliamo l'energia potenziale totale di questa configurazione statico
    energia_potenziale = atomi.get_potential_energy()
    
    # Salviamo l'energia nella nostra lista
    energie_calcolate.append(energia_potenziale)
    
    # Stampiamo un feedback a schermo ogni 10 frame per vedere che il codice stia lavorando
    if indice % 10 == 0:
        print(f"Elaborazione Frame {indice:4d} | Energia GAP: {energia_potenziale:12.4f} eV")

print(f"\nElaborazione completata! Frame totali analizzati: {len(energie_calcolate)}")

# =========================================================================
# 3. SALVATAGGIO DEI DATI PER IL PARITY PLOT
# =========================================================================
print(f"Salvataggio delle energie in corso nel file: {PATH_OUTPUT_ENERGIE}...")

# Usiamo numpy per salvare la lista come una singola colonna di numeri in un file di testo.
# È il modo più pulito e facile da rileggere dopo per fare i grafici.
np.savetxt(PATH_OUTPUT_ENERGIE, energie_calcolate, fmt="%.6f", header="Energia_GAP_(eV)")

print("Operazione conclusa con successo!")
print("=" * 60)
