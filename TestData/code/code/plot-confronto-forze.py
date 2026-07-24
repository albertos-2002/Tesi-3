import os
import numpy as np
import matplotlib.pyplot as plt

# =========================================================================
# 1. PARAMETRI E PERCORSI
# =========================================================================
PATH_CSV_CONFRONTO = "dati_confronto_forze.csv"

# Parametri dei raggi di cutoff (stessi della simulazione)
RAGGIO_INNER = 1.5
RAGGIO_OUTER = 2.5

# Check esistenza file
if not os.path.exists(PATH_CSV_CONFRONTO):
    raise FileNotFoundError(f"Il file '{PATH_CSV_CONFRONTO}' non è stato trovato. Assicurati che sia nella stessa cartella.")

print(f"Caricamento dati da '{PATH_CSV_CONFRONTO}' in corso...")

# =========================================================================
# 2. LETTURA E CARICAMENTO DATI
# =========================================================================
# Struttura CSV: Tempo_fs, Atomo_ID, Distanza_Minima_A, F_rad_tot, F_rad_gap, F_rad_zbl
dati_conf = np.genfromtxt(
    PATH_CSV_CONFRONTO, 
    delimiter=',', 
    skip_header=1, 
    dtype=None, 
    encoding='utf-8'
)

t_conf = dati_conf['f0']
dist_conf = dati_conf['f2']
f_tot_rad = dati_conf['f3']
f_gap_rad = dati_conf['f4']
f_zbl_rad = dati_conf['f5']

print("Dati caricati con successo!")

# =========================================================================
# 3. GRAFICO 1: CONFRONTO FORZE VS TEMPO
# =========================================================================
plt.figure(figsize=(10, 6))

plt.scatter(t_conf, f_tot_rad, color='black', s=8, alpha=0.5, label='Forza Totale (GAP + ZBL)')
plt.scatter(t_conf, f_gap_rad, color='royalblue', s=8, alpha=0.5, label='Forza GAP')
plt.scatter(t_conf, f_zbl_rad, color='crimson', s=8, alpha=0.5, label='Forza ZBL')

plt.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)

plt.title("Confronto Forze Radiali nel Tempo", fontsize=13, pad=12)
plt.xlabel("Tempo (fs)", fontsize=11)
plt.ylabel("Forza Radiale (eV/Å) [>0 Repulsiva, <0 Attrattiva]", fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(markerscale=2, fontsize=10, loc='upper right')

plt.tight_layout()
plt.savefig("grafico_confronto_forze_tempo.png", dpi=300)
print("Grafico 'grafico_confronto_forze_tempo.png' salvato.")
plt.show()  # Apre la finestra interattiva di Matplotlib

# =========================================================================
# 4. GRAFICO 2: CONFRONTO FORZE VS DISTANZA MINIMA
# =========================================================================
plt.figure(figsize=(10, 6))

plt.scatter(dist_conf, f_tot_rad, color='black', s=8, alpha=0.5, label='Forza Totale (GAP + ZBL)')
plt.scatter(dist_conf, f_gap_rad, color='royalblue', s=8, alpha=0.5, label='Forza GAP')
plt.scatter(dist_conf, f_zbl_rad, color='crimson', s=8, alpha=0.5, label='Forza ZBL')

# Linee verticali per i cutoff dello ZBL
plt.axvline(x=RAGGIO_INNER, color='green', linestyle='--', linewidth=1.5, label=f'r_inner ({RAGGIO_INNER} Å)')
plt.axvline(x=RAGGIO_OUTER, color='orange', linestyle=':', linewidth=1.5, label=f'r_outer ({RAGGIO_OUTER} Å)')
plt.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)

plt.title("Confronto Forze Radiali vs Distanza dal Primo Vicino", fontsize=13, pad=12)
plt.xlabel("Distanza dal Vicino più Prossimo (Å)", fontsize=11)
plt.ylabel("Forza Radiale (eV/Å) [>0 Repulsiva, <0 Attrattiva]", fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(markerscale=2, fontsize=10, loc='upper right')

plt.tight_layout()
plt.savefig("grafico_confronto_forze_distanza.png", dpi=300)
print("Grafico 'grafico_confronto_forze_distanza.png' salvato.")
plt.show()  # Apre la seconda finestra interattiva di Matplotlib
