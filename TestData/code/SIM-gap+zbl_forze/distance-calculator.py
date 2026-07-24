import numpy as np
import matplotlib.pyplot as plt
from ase.io import iread

# ==========================================
# 1. PARAMETRI DI CONFIGURAZIONE
# ==========================================
FILE_INPUT = "md_gap_zbl_nve.traj"                 # Il file di traiettoria generato dalla dinamica
FILE_OUTPUT_DATI = "statistiche_distanze_traj.csv"  # Dove salvare i risultati numerici
FILE_OUTPUT_PLOT = "plot_distanze_traj.png"        # Dove salvare il grafico finale
STAMPA_OGNI = 10                                   # Stampa a schermo ogni X frame

PATH_GRAFICO = "andamento_distanze_minime.png"
PATH_GRAFICO_ISTOGRAMMA = "istogramma_distanze_minime.png"

# ==========================================
# 2. INIZIALIZZAZIONE STRUTTURE DATI
# ==========================================
frame_indices = []
min_distances = []
max_distances = []
avg_distances = []

print(f"Inizio l'analisi del file: {FILE_INPUT}")
print(f"{'Frame':>8} | {'Min (Å)':>10} | {'Max (Å)':>10} | {'Media (Å)':>10}")
print("-" * 47)

# ==========================================
# 3. CICLO DI LETTURA E CALCOLO (OTTIMIZZATO PER LA RAM)
# ==========================================
# iread legge la traiettoria frame per frame senza caricare l'intero file in memoria
with open(FILE_OUTPUT_DATI, "w") as f_out:
    f_out.write("Frame,Min_Angstrom,Max_Angstrom,Media_Angstrom\n")
    
    for i, atoms in enumerate(iread(FILE_INPUT)):
        
        # 1. Calcolo della matrice delle distanze con condizioni al contorno periodiche (mic=True)
        dist_matrix = atoms.get_all_distances(mic=True)
        
        # 2. Escludiamo l'auto-interazione ponendo la diagonale a infinito
        np.fill_diagonal(dist_matrix, np.inf)
        
        # 3. Calcolo delle statistiche sul frame
        d_min = np.min(dist_matrix)
        valid_distances = dist_matrix[np.isfinite(dist_matrix)]
        d_max = np.max(valid_distances)
        d_avg = np.mean(valid_distances)
        
        # Salvataggio in memoria
        frame_indices.append(i)
        min_distances.append(d_min)
        max_distances.append(d_max)
        avg_distances.append(d_avg)
        
        # Salvataggio su file CSV
        f_out.write(f"{i},{d_min:.6f},{d_max:.6f},{d_avg:.6f}\n")
        
        # Stampa ogni tot frame
        if i % STAMPA_OGNI == 0:
            print(f"{i:8d} | {d_min:10.4f} | {d_max:10.4f} | {d_avg:10.4f}")

print("-" * 47)
print(f"Analisi completata su {len(frame_indices)} frame.")
print(f"Dati salvati in: {FILE_OUTPUT_DATI}")

# ==========================================
# 4. GENERAZIONE DEL GRAFICO
# ==========================================
print("Generazione del grafico in corso...")

plt.figure(figsize=(10, 6))

plt.plot(frame_indices, max_distances, label="Distanza Massima", color="red", linewidth=1.5, alpha=0.8)
plt.plot(frame_indices, avg_distances, label="Distanza Media", color="green", linewidth=2.0)
plt.plot(frame_indices, min_distances, label="Distanza Minima", color="blue", linewidth=1.5, alpha=0.8)

plt.title("Statistiche delle Distanze Interatomiche per Frame (Traiettoria MD)")
plt.xlabel("Indice Frame")
plt.ylabel("Distanza (Å)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

plt.tight_layout()
plt.savefig(FILE_OUTPUT_PLOT, dpi=300)
plt.close()

print(f"Grafico salvato in: {FILE_OUTPUT_PLOT}")


# 1. Grafico Lineare: Andamento della distanza minima nel tempo (o frame)
plt.figure(figsize=(10, 5))
plt.scatter(frame_indices, min_distances, color='crimson', marker='o', s=4)

plt.title("Evoluzione della Distanza Interatomica Minima")
plt.xlabel("Indice Frame della Dinamica")
plt.ylabel("Distanza Minima Assoluta (Å)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(PATH_GRAFICO, dpi=300)
plt.close()


# 2. Istogramma: Distribuzione delle distanze minime
plt.figure(figsize=(8, 5))
plt.hist(min_distances, bins=30, color='royalblue', alpha=0.7, edgecolor='black')
plt.title("Distribuzione delle Distanze Minime nella Traiettoria")
plt.xlabel("Distanza Minima (Å)")
plt.ylabel("Frequenza (Numero di Frame)")
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig(PATH_GRAFICO_ISTOGRAMMA, dpi=300)
plt.close()

print(f"Tutto completato! I grafici sono stati salvati come {PATH_GRAFICO} e {PATH_GRAFICO_ISTOGRAMMA}.")

