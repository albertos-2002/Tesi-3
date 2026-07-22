import os
import numpy as np
import matplotlib.pyplot as plt
import gc # Garbage Collector per svuotare forzatamente la RAM
from ase.io import read
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from quippy.potential import Potential

# =========================================================================
# 1. PARAMETRI E PERCORSI (MODIFICA QUI)
# =========================================================================
PATH_POTENZIALE = "../training-gap-zbl/out_potenziale_gap_500_n2_l2_zbl.xml"
PATH_CONFIG_INIZIALE = "../test_data.extxyz"

# Cartelle per i frame spaziali
DIR_SALVATAGGIO_2D = "immagini_MD_NVE_2D"
DIR_SALVATAGGIO_3D = "immagini_MD_NVE_3D"

# File di Output Dati (Streaming su disco)
PATH_CSV_ENERGIE = "dati_energie.csv"
PATH_CSV_FORZE = "dati_forze.csv"

# Grafici finali
PATH_GRAFICO_ENERGIE = "andamento_energia_nve.png"
PATH_GRAFICO_FORZE = "andamento_forze_nve.png"

# Parametri della Dinamica
TIMESTEP = 1 * units.fs 
PASSI_TOTALI = 500 

# CRITICO PER LA RAM: Ogni quanti step disegnare l'immagine 2D/3D degli atomi?
# Impostalo a 1000 se fai 300mila step, altrimenti intaserai l'hard disk!
FREQUENZA_IMMAGINI = 10 

print("=" * 60)
print(" AVVIO SIMULAZIONE DINAMICA NVE (ALTA EFFICIENZA)")
print("=" * 60)

os.makedirs(DIR_SALVATAGGIO_2D, exist_ok=True)
os.makedirs(DIR_SALVATAGGIO_3D, exist_ok=True)

# =========================================================================
# 2. CARICAMENTO E PREPARAZIONE
# =========================================================================
calc = Potential(param_filename=PATH_POTENZIALE)
calc.name_ = "GAP"

atomi_md = read(PATH_CONFIG_INIZIALE, index=0)
atomi_md.calc = calc

# CORREZIONE: Riporta gli atomi dentro il box di simulazione [0, L]
atomi_md.wrap()

try:
    if atomi_md.get_velocities() is None or not np.any(atomi_md.get_velocities()):
        raise ValueError
except (ValueError, RuntimeError):
    MaxwellBoltzmannDistribution(atomi_md, temperature_K=3000)

dyn = VelocityVerlet(
    atomi_md, 
    timestep=TIMESTEP, 
    trajectory="md_gap_nve.traj", # Nota: anche questo file diventerà grande
    logfile="md_gap_nve.log",
    loginterval=100 # Alleggerisce il log a schermo
)

# =========================================================================
# 3. IMPOSTAZIONE FILE CSV (Svuotiamo vecchi file e scriviamo gli header)
# =========================================================================
with open(PATH_CSV_ENERGIE, 'w') as f_en:
    f_en.write("Tempo_fs,Temp_K,E_pot_eV,E_kin_eV,E_tot_eV\n")

with open(PATH_CSV_FORZE, 'w') as f_forze:
    f_forze.write("Tempo_fs,Forza_X,Forza_Y,Forza_Z\n")

# =========================================================================
# 4. IL MOTORE DI MONITORAGGIO (Zero Liste in RAM)
# =========================================================================
def monitoraggio_sistema():
    passo = dyn.get_number_of_steps()
    tempo_fs = passo * (TIMESTEP / units.fs)
    
    # --- 1. Scrittura Diretta Energie ---
    epot = atomi_md.get_potential_energy()
    ekin = atomi_md.get_kinetic_energy()
    etot = epot + ekin
    temp = atomi_md.get_temperature()
    
    with open(PATH_CSV_ENERGIE, 'a') as f_en:
        f_en.write(f"{tempo_fs:.3f},{temp:.2f},{epot:.5f},{ekin:.5f},{etot:.5f}\n")
    
    # --- 2. Scrittura Diretta Forze ---
    forze = atomi_md.get_forces()
    with open(PATH_CSV_FORZE, 'a') as f_forze:
        # Scriviamo le forze di tutti gli atomi per questo istante
        for fx, fy, fz in forze:
            f_forze.write(f"{tempo_fs:.3f},{fx:.5f},{fy:.5f},{fz:.5f}\n")
    
    # --- 3. Generazione Immagini (SOLO SE NECESSARIO) ---
    if passo % FREQUENZA_IMMAGINI == 0:
        posizioni = atomi_md.get_positions(wrap=True)
        simboli = np.array(atomi_md.get_chemical_symbols())
        
        pos_Ta = posizioni[simboli == 'Ta']
        pos_O = posizioni[simboli == 'O']
        nome_file = f"{passo:06d}_configurazione_spaziale.png"
        
        # 2D
        fig2d, ax2d = plt.subplots(figsize=(6, 6))
        ax2d.scatter(pos_Ta[:, 0], pos_Ta[:, 1], color='blue', s=80, label='Tantalo', edgecolors='black')
        ax2d.scatter(pos_O[:, 0], pos_O[:, 1], color='red', s=40, label='Ossigeno', edgecolors='black')
        ax2d.set_title(f"2D - Step {passo}")
        ax2d.axis('equal')
        fig2d.savefig(os.path.join(DIR_SALVATAGGIO_2D, nome_file), dpi=100)
        plt.close(fig2d)
        
        # 3D
        fig3d = plt.figure(figsize=(6, 6))
        ax3d = fig3d.add_subplot(111, projection='3d')
        ax3d.scatter(pos_Ta[:, 0], pos_Ta[:, 1], pos_Ta[:, 2], color='blue', s=80, edgecolors='black')
        ax3d.scatter(pos_O[:, 0], pos_O[:, 1], pos_O[:, 2], color='red', s=40, edgecolors='black')
        ax3d.set_title(f"3D - Step {passo}")
        fig3d.savefig(os.path.join(DIR_SALVATAGGIO_3D, nome_file), dpi=100)
        plt.close(fig3d)
        
        # CHIUSURA FORZATA DI TUTTI I GRAFICI E PULIZIA RAM (Cruciale!)
        plt.close('all')
        gc.collect()
        
        print(f"Step {passo:06d}/{PASSI_TOTALI} completato. Dati e Immagini salvati.")

# Esegue la funzione ad ogni step (i dati si salvano sempre, le immagini solo ogni tot)
dyn.attach(monitoraggio_sistema, interval=1)

# =========================================================================
# 5. ESECUZIONE SIMULAZIONE
# =========================================================================
print("\n--- INIZIO DINAMICA MOLECOLARE ---")
dyn.run(PASSI_TOTALI)
print("--- DINAMICA COMPLETATA ---\n")

# =========================================================================
# 6. POST-PROCESSING: GRAFICI FINALI
# =========================================================================
print("Caricamento dati dal disco per i grafici finali (potrebbe volerci un attimo)...")

# Leggiamo i file CSV generati. (Saltiamo la prima riga di header)
t_en, temp, ep, ek, et = np.loadtxt(PATH_CSV_ENERGIE, delimiter=',', skiprows=1, unpack=True)

# Nota: per 300.000 step, leggere tutte le forze per il plot potrebbe rallentare.
# Se il file è enorme, np.loadtxt lo gestisce comunque bene se hai un minimo di RAM per questa fase finale.
t_f, fx, fy, fz = np.loadtxt(PATH_CSV_FORZE, delimiter=',', skiprows=1, unpack=True)

print("Generazione grafico energie...")
fig_en, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Usiamo 'plot' invece di 'scatter' se i punti sono tantissimi, o uno scatter con s piccolissimo
ax1.plot(t_en, temp, color='orangered', lw=1)
ax1.set_title('Temperatura')
ax1.set_xlabel('Tempo (fs)')
ax1.set_ylabel('Temperatura (K)')
ax1.grid(True, linestyle=':', alpha=0.6)

ax2.plot(t_en, ep, color='royalblue', lw=1, label='E. Potenziale')
ax2.plot(t_en, ek, color='forestgreen', lw=1, label='E. Cinetica')
ax2.plot(t_en, et, color='black', lw=1.5, label='E. Totale')
ax2.set_title('Confronto Energie')
ax2.set_xlabel('Tempo (fs)')
ax2.set_ylabel('Energia (eV)')
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend()

ax3.plot(t_en, ep, color='royalblue', lw=1)
ax3.set_title('Dettaglio E. Potenziale')
ax3.set_xlabel('Tempo (fs)')
ax3.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig(PATH_GRAFICO_ENERGIE, dpi=300)
plt.close(fig_en)

print("Generazione grafico forze...")
fig_forze, (ax_fx, ax_fy, ax_fz) = plt.subplots(1, 3, figsize=(15, 5))

# Scatter per le forze (imposto un alpha basso così i punti si fondono senza coprirsi del tutto)
ax_fx.scatter(t_f, fx, color='red', s=1, alpha=0.1)
ax_fx.set_title('Forze Lungo X')
ax_fx.set_xlabel('Tempo (fs)')
ax_fx.set_ylabel('Forza (eV/Å)')
ax_fx.grid(True, linestyle=':', alpha=0.7)

ax_fy.scatter(t_f, fy, color='green', s=1, alpha=0.1)
ax_fy.set_title('Forze Lungo Y')
ax_fy.set_xlabel('Tempo (fs)')
ax_fy.grid(True, linestyle=':', alpha=0.7)

ax_fz.scatter(t_f, fz, color='blue', s=1, alpha=0.1)
ax_fz.set_title('Forze Lungo Z')
ax_fz.set_xlabel('Tempo (fs)')
ax_fz.grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
plt.savefig(PATH_GRAFICO_FORZE, dpi=300)
plt.close(fig_forze)

print("Tutto completato! RAM salvata.")
