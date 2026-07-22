import numpy as np
import matplotlib.pyplot as plt
from ase.io import iread

# =========================================================================
# CONFIGURAZIONE PERCORSI
# =========================================================================
PATH_ENERGIE_GAP = "energie_predette_gap-traindata.txt"
PATH_TRAIETTORIA = "../../train_data.extxyz"  # Il file con le energie vere (QE)

print("Caricamento dei dati in corso...")

# 1. CARICAMENTO DELLE ENERGIE PREDETTE (GAP)
# =========================================================================
# Leggiamo il file di testo generato dallo script precedente. 
# np.loadtxt ignora in automatico l'header iniziale.
energie_gap = np.loadtxt(PATH_ENERGIE_GAP)

# 2. ESTRAZIONE DELLE ENERGIE VERE (Quantum ESPRESSO)
# =========================================================================
energie_qe = []

# Leggiamo il file extxyz per estrarre l'energia vera di ogni frame
for atomi in iread(PATH_TRAIETTORIA):
    # Estraiamo l'energia totale salvata da QE nel file extxyz
    energia_vera = atomi.get_potential_energy()
    energie_qe.append(energia_vera)

energie_qe = np.array(energie_qe)

# Controllo di sicurezza: verifichiamo di avere lo stesso numero di dati
if len(energie_gap) != len(energie_qe):
    raise ValueError(f"Errore! I frame non combaciano: {len(energie_gap)} GAP vs {len(energie_qe)} QE.")

print(f"Analisi completata su {len(energie_gap)} frame.")

# 3. CREAZIONE DEL PARITY PLOT
# =========================================================================
plt.figure(figsize=(7, 6))

# Creiamo lo scatter plot con le energie totali
plt.scatter(energie_qe, energie_gap, 
            color='blue', alpha=0.6, s=15, edgecolors='k', linewidth=0.5, label='Dati GAP')

# Creiamo la linea di parità ideale (y = x)
# Troviamo il minimo e il massimo per tracciare la linea in modo perfetto
min_val = min(np.min(energie_qe), np.min(energie_gap))
max_val = max(np.max(energie_qe), np.max(energie_gap))

# Disegniamo la linea bisettrice tratteggiata rossa
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideale (y = x)')

# Estetica del grafico
plt.title('Parity Plot: Energie GAP vs Quantum ESPRESSO (traindata)', fontsize=14)
plt.xlabel('Energia QE (eV)', fontsize=12)
plt.ylabel('Energia GAP (eV)', fontsize=12)

plt.legend(loc='lower right')
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()

# Salva il grafico in formato PNG ad alta risoluzione (300 dpi)
plt.savefig('parity_plot_gap_vs_qe-traindata.png', dpi=300, bbox_inches='tight')

# Mostriamo il grafico a schermo
plt.show()
