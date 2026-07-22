import numpy as np
import matplotlib.pyplot as plt
from ase.io import iread

# ==========================================
# 1. PARAMETRI DI CONFIGURAZIONE
# ==========================================
FILE_INPUT = "../test_data.extxyz"          # Il tuo file contenente i frame di QE
FILE_OUTPUT_DATI = "statistiche_distanze_test.csv" # Dove salvare i risultati numerici
FILE_OUTPUT_PLOT = "plot_distanze_test.png"        # Dove salvare il grafico finale
STAMPA_OGNI = 100                    # Stampa a schermo ogni X frame per monitorare l'avanzamento

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

# Apriamo il file di output per scriverci dentro man mano (salva RAM e previene perdite di dati)
with open(FILE_OUTPUT_DATI, "w") as f_out:
    f_out.write("Frame,Min_Angstrom,Max_Angstrom,Media_Angstrom\n")
    
    # ==========================================
# 3. CICLO DI LETTURA E CALCOLO (OTTIMIZZATO PER LA RAM)
# ==========================================
    # iread carica solo un frame alla volta in memoria
    for i, atoms in enumerate(iread(FILE_INPUT)):
        
        # Calcola la matrice di tutte le distanze (con condizioni periodiche se presenti)
        # Restituisce una matrice NxN (126x126)
        dist_matrix = atoms.get_all_distances(mic=True)
        
        # Estraiamo solo il triangolo superiore della matrice (senza la diagonale)
        # per non contare la distanza di un atomo con se stesso (che è 0) e non avere doppioni
        upper_triangle_indices = np.triu_indices(len(atoms), k=1)
        real_distances = dist_matrix[upper_triangle_indices]
        
        # Calcolo statistiche
        d_min = np.min(real_distances)
        d_max = np.max(real_distances)
        d_avg = np.mean(real_distances)
        
        # Salvataggio in memoria per il grafico finale
        frame_indices.append(i)
        min_distances.append(d_min)
        max_distances.append(d_max)
        avg_distances.append(d_avg)
        
        # Salvataggio su file
        f_out.write(f"{i},{d_min:.6f},{d_max:.6f},{d_avg:.6f}\n")
        
        # Stampa a schermo ogni tot frame
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

# Tracciamo le tre linee
plt.plot(frame_indices, max_distances, label="Distanza Massima", color="red", linewidth=1.5, alpha=0.8)
plt.plot(frame_indices, avg_distances, label="Distanza Media", color="green", linewidth=2.0)
plt.plot(frame_indices, min_distances, label="Distanza Minima", color="blue", linewidth=1.5, alpha=0.8)

# Formattazione del grafico
plt.title("Statistiche delle Distanze Interatomiche per Frame (Dinamica QE) (testdata)", fontsize=14, pad=15)
plt.xlabel("Indice del Frame", fontsize=12)
plt.ylabel("Distanza Interatomica (Å)", fontsize=12)

# Griglia e legenda
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc="best", fontsize=11, shadow=True)

# Salvataggio e chiusura
plt.tight_layout()
plt.savefig(FILE_OUTPUT_PLOT, dpi=300)
plt.show()

print(f"Grafico salvato in: {FILE_OUTPUT_PLOT}")
