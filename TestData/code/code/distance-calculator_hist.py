import numpy as np
import matplotlib.pyplot as plt
import os
import gc 
from ase.io import read

# =========================================================================
# 1. PARAMETRI E PERCORSI (MODIFICA QUI)
# =========================================================================
# Punta al file .traj prodotto dalla tua simulazione di Dinamica Molecolare
PATH_TRAIETTORIA = "../train_data.extxyz"

# Grafici finali
PATH_GRAFICO = "andamento_distanze_minime.png"
PATH_GRAFICO_ISTOGRAMMA = "istogramma_distanze_minime.png"

print("=" * 60)
print(f" AVVIO ANALISI DISTANZE (LETTURA TRAIETTORIA {PATH_TRAIETTORIA})")
print("=" * 60)

# =========================================================================
# 2. CARICAMENTO DELLA TRAIETTORIA
# =========================================================================
# index=':' permette di caricare in memoria tutti i frame salvati nel file .traj
print(f"Leggendo i frame della traiettoria dal file: {PATH_TRAIETTORIA} ...")
try:
    frames = read(PATH_TRAIETTORIA, index=':')
    num_frames = len(frames)
    print(f"-> Sono stati caricati correttamente {num_frames} frame dalla dinamica.")
except Exception as e:
    print(f"Errore durante la lettura della traiettoria: {e}")
    exit()

if num_frames == 0:
    print("Errore: Il file non contiene frame leggibili.")
    exit()

# Liste per raccogliere i dati da plottare
distanze_minime = []
frame_indices = list(range(num_frames))


# =========================================================================
# 3. CICLO SU TUTTI I FRAME DELLA DINAMICA
# =========================================================================
print("Inizio calcolo delle distanze minime per ogni frame...")

for step, atoms in enumerate(frames):
    
    # 1. Calcolo della matrice delle distanze applicando le PBC (mic=True)
    dist_matrix = atoms.get_all_distances(mic=True)
    
    # 2. Escludiamo la diagonale (distanza di un atomo da se stesso) impostandola a infinito
    np.fill_diagonal(dist_matrix, np.inf)
    
    # 3. Troviamo la distanza minima assoluta in questo specifico frame
    min_dist_frame = np.min(dist_matrix)
    
    # Salvataggio del dato per il grafico
    distanze_minime.append(min_dist_frame)
    
    # Stampa a schermo ogni 10 frame per non intasare l'output
    if step % 10 == 0:
        print(f"Frame {step:03d}/{num_frames} -> Distanza minima: {min_dist_frame:.4f} Å")

print("Analisi su tutti i frame terminata.")

# =========================================================================
# 4. GENERAZIONE GRAFICI FINALI
# =========================================================================
print("Generazione dei grafici...")

# 1. Grafico Lineare: Andamento della distanza minima nel tempo (o frame)
plt.figure(figsize=(10, 5))
plt.plot(frame_indices, distanze_minime, color='crimson', lw=1.5, marker='o', markersize=3)

plt.title("Evoluzione della Distanza Interatomica Minima")
plt.xlabel("Indice Frame della Dinamica")
plt.ylabel("Distanza Minima Assoluta (Å)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(PATH_GRAFICO, dpi=300)
plt.close()


# 2. Istogramma: Distribuzione delle distanze minime
plt.figure(figsize=(8, 5))
plt.hist(distanze_minime, bins=30, color='royalblue', alpha=0.7, edgecolor='black')
plt.title("Distribuzione delle Distanze Minime nella Traiettoria")
plt.xlabel("Distanza Minima (Å)")
plt.ylabel("Frequenza (Numero di Frame)")
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(PATH_GRAFICO_ISTOGRAMMA, dpi=300)
plt.close()

print(f"Tutto completato! I grafici sono stati salvati come {PATH_GRAFICO} e {PATH_GRAFICO_ISTOGRAMMA}.")
