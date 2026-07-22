import numpy as np
import matplotlib.pyplot as plt
from ase.io import iread
from quippy.potential import Potential

# =========================================================================
# 1. CONFIGURAZIONE PERCORSI
# =========================================================================
PATH_POTENZIALE = "../../training-gap-zbl/out_potenziale_gap_500_n2_l2_zbl.xml"
PATH_TRAIETTORIA = "../../train_data.extxyz"  # Usa l'extxyz, contiene già le forze QE!
PATH_OUTPUT_FORZE = "forze_predette_gap-traindata.txt"

print("=" * 60)
print(" AVVIO CALCOLO E CONFRONTO FORZE: GAP vs QE")
print("=" * 60)

# =========================================================================
# 2. CARICAMENTO DEL POTENZIALE
# =========================================================================
print(f"Caricamento del potenziale GAP...")
calc = Potential(param_filename=PATH_POTENZIALE)
# calc.name_ = "GAP" # Usa il nome corretto per la tua versione, se necessario

# =========================================================================
# 3. LETTURA FRAME, ESTRAZIONE FORZE QE E CALCOLO FORZE GAP
# =========================================================================
print(f"Analisi della traiettoria: {PATH_TRAIETTORIA}...")

# Liste per accumulare le matrici delle forze
tutte_forze_qe = []
tutte_forze_gap = []

for indice, atomi in enumerate(iread(PATH_TRAIETTORIA)):
    
    # A. Estraiamo le forze vere (QE) dal file PRIMA di cambiare il calcolatore
    forze_vere_qe = atomi.get_forces()
    tutte_forze_qe.append(forze_vere_qe)
    
    # B. Attacchiamo il nostro GAP
    atomi.calc = calc
    
    # C. Calcoliamo le nuove forze predette dal GAP
    forze_predette_gap = atomi.get_forces()
    tutte_forze_gap.append(forze_predette_gap)
    
    if indice % 10 == 0:
        print(f"Elaborazione Frame {indice:4d} completata...")

# =========================================================================
# 4. ORGANIZZAZIONE E SALVATAGGIO DATI
# =========================================================================
# Le forze estratte sono liste di matrici (NumeroAtomi x 3).
# Usiamo np.vstack per "schiacciarle" in un'unica enorme matrice lunga (TotaleAtomi x 3)
forze_qe_array = np.vstack(tutte_forze_qe)
forze_gap_array = np.vstack(tutte_forze_gap)

print(f"\nEstrazione completata! Totale componenti di forza analizzate: {len(forze_qe_array)}")

# Salviamo le forze GAP su file come richiesto (formato: Fx, Fy, Fz per ogni riga)
print(f"Salvataggio delle forze predette in: {PATH_OUTPUT_FORZE}")
np.savetxt(PATH_OUTPUT_FORZE, forze_gap_array, fmt="%.6f", header="Fx_GAP Fy_GAP Fz_GAP (eV/Angstrom)")

# =========================================================================
# 5. CREAZIONE DEL PARITY PLOT A 3 PANNELLI (X, Y, Z)
# =========================================================================
print("Generazione del Parity Plot...")

# Creiamo una figura con 1 riga e 3 colonne (i 3 pannelli)
# figsize=(15, 5) rende l'immagine bella larga
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Estraiamo le singole componenti X, Y, Z per comodità
# Indice 0 = X, 1 = Y, 2 = Z
labels = ['Forza X', 'Forza Y', 'Forza Z']
colori = ['red', 'green', 'blue']

# Calcoliamo i limiti globali per tracciare una linea di parità perfetta e coerente
min_val = min(np.min(forze_qe_array), np.min(forze_gap_array))
max_val = max(np.max(forze_qe_array), np.max(forze_gap_array))

# Cicliamo sui 3 assi per creare i 3 grafici
for i in range(3):
    ax = axs[i]
    
    # Scatter plot della componente i-esima
    ax.scatter(forze_qe_array[:, i], forze_gap_array[:, i], 
               color=colori[i], alpha=0.4, s=5, edgecolors='none', label=f'GAP vs QE ({labels[i]})')
    
    # Linea di parità (y = x)
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5, label='Ideale (y = x)')
    
    # Estetica del singolo pannello
    ax.set_title(f'Parity Plot: {labels[i]} (traindata)', fontsize=12)
    ax.set_xlabel('Forza QE (eV/Å)', fontsize=10)
    ax.set_ylabel('Forza GAP (eV/Å)', fontsize=10)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(loc='lower right')

# Aggiustiamo gli spazi in automatico per evitare sovrapposizioni di scritte
plt.tight_layout()

# Salviamo e mostriamo
plt.savefig('parity_plot_forze-traindata.png', dpi=300, bbox_inches='tight')
plt.show()

print("Operazione conclusa con successo!")
print("=" * 60)
