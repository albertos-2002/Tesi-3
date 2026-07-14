import os
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from quippy.potential import Potential

# =========================================================================
# 1. PARAMETRI E PERCORSI
# =========================================================================
PATH_POTENZIALE = "../../results-test2/out_potenziale_gap-500-n2-l2.xml"
PATH_CONFIG_INIZIALE = "../../test_data.extxyz"

# Cartelle per i frame spaziali (2D e 3D)
DIR_SALVATAGGIO_2D = "2d"
DIR_SALVATAGGIO_3D = "3d"

# Output generali
PATH_TRAJ_OUTPUT = "md_gap_nve.traj"
PATH_LOG_OUTPUT = "md_gap_nve.log"
PATH_GRAFICO_ENERGIE = "andamento_energia_nve.png"
PATH_GRAFICO_FORZE = "andamento_forze_nve.png"

# Parametri della Dinamica
TIMESTEP = 0.1 * units.fs 
PASSI_TOTALI = 5000  

print("=" * 60)
print(" AVVIO SIMULAZIONE DINAMICA NVE")
print("=" * 60)

os.makedirs(DIR_SALVATAGGIO_2D, exist_ok=True)
os.makedirs(DIR_SALVATAGGIO_3D, exist_ok=True)

# =========================================================================
# 2. CARICAMENTO POTENZIALE E STRUTTURA
# =========================================================================
print(f"Caricamento del potenziale GAP e della struttura...")
calc = Potential(param_filename=PATH_POTENZIALE)
calc.name_ = "GAP"

atomi_md = read(PATH_CONFIG_INIZIALE, index=0)
atomi_md.calc = calc

try:
    if atomi_md.get_velocities() is None or not np.any(atomi_md.get_velocities()):
        raise ValueError
except (ValueError, RuntimeError):
    print("Velocità non trovate. Assegnazione Maxwell-Boltzmann (3000 K)...")
    MaxwellBoltzmannDistribution(atomi_md, temperature_K=3000)

# =========================================================================
# 3. IMPOSTAZIONE DINAMICA E MONITORAGGIO
# =========================================================================
dyn = VelocityVerlet(
    atomi_md, 
    timestep=TIMESTEP, 
    trajectory=PATH_TRAJ_OUTPUT,
    logfile=PATH_LOG_OUTPUT,
    loginterval=1
)

# Liste per salvare i dati in RAM ad ogni step
tempi_fs, temperature = [], []
e_potenziali, e_cinetiche, e_totali = [], [], []

# Liste per accumulare le forze di tutti gli atomi ad ogni step
elenco_forze_x = []
elenco_forze_y = []
elenco_forze_z = []
tempi_forze_ripetuti = []  # Serve per associare il tempo corretto a ciascun atomo nel grafico finale

def monitoraggio_sistema():
    passo = dyn.get_number_of_steps()
    tempo_fs = passo * (TIMESTEP / units.fs)
    
    # 1. Energie e Temperatura
    epot = atomi_md.get_potential_energy()
    ekin = atomi_md.get_kinetic_energy()
    etot = epot + ekin
    temp = atomi_md.get_temperature()
    
    tempi_fs.append(tempo_fs)
    temperature.append(temp)
    e_potenziali.append(epot)
    e_cinetiche.append(ekin)
    e_totali.append(etot)
    
    # 2. Forze di tutti gli atomi (senza medie o calcoli aggiuntivi)
    forze = atomi_md.get_forces()  # Array di shape (N_atomi, 3)
    elenco_forze_x.extend(forze[:, 0])
    elenco_forze_y.extend(forze[:, 1])
    elenco_forze_z.extend(forze[:, 2])
    # Ripetiamo il valore del tempo corrente per quanti sono gli atomi presenti
    tempi_forze_ripetuti.extend([tempo_fs] * len(atomi_md))
    
    # 3. Disposizione Spaziale degli Atomi
    posizioni = atomi_md.get_positions()
    simboli = np.array(atomi_md.get_chemical_symbols())
    
    pos_Ta = posizioni[simboli == 'Ta']
    pos_O = posizioni[simboli == 'O']
    
    nome_file = f"{passo:04d}_configurazione_spaziale.png"
    
    # --- Grafico 2D (Piano XY) ---
    plt.figure(figsize=(6, 6))
    plt.scatter(pos_Ta[:, 0], pos_Ta[:, 1], color='blue', s=80, label='Tantalo (Ta)', edgecolors='black')
    plt.scatter(pos_O[:, 0], pos_O[:, 1], color='red', s=40, label='Ossigeno (O)', edgecolors='black')
    plt.title(f"Configurazione Spaziale 2D - Step {passo:04d}")
    plt.xlabel("Coordinata X (Å)")
    plt.ylabel("Coordinata Y (Å)")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal') 
    plt.savefig(os.path.join(DIR_SALVATAGGIO_2D, nome_file), dpi=150)
    plt.close()
    
    # --- Grafico 3D (Spazio XYZ) ---
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pos_Ta[:, 0], pos_Ta[:, 1], pos_Ta[:, 2], color='blue', s=80, label='Tantalo (Ta)', edgecolors='black', depthshade=True)
    ax.scatter(pos_O[:, 0], pos_O[:, 1], pos_O[:, 2], color='red', s=40, label='Ossigeno (O)', edgecolors='black', depthshade=True)
    ax.set_title(f"Configurazione Spaziale 3D - Step {passo:04d}")
    ax.set_xlabel("Coordinata X (Å)")
    ax.set_ylabel("Coordinata Y (Å)")
    ax.set_zlabel("Coordinata Z (Å)")
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(DIR_SALVATAGGIO_3D, nome_file), dpi=150)
    plt.close()
    
    print(f"Step {passo:04d} completato. Immagini 2D e 3D salvate.")

dyn.attach(monitoraggio_sistema, interval=1)

# =========================================================================
# 4. ESECUZIONE SIMULAZIONE
# =========================================================================
print("\n--- INIZIO DINAMICA MOLECOLARE NVE ---")
dyn.run(PASSI_TOTALI)
print("--- DINAMICA COMPLETATA ---\n")

# =========================================================================
# 5. GENERAZIONE GRAFICI FINALI (Layout Orizzontale)
# =========================================================================
print("Generazione dei grafici finali...")

# --- GRAFICO 1: ENERGIE E TEMPERATURA (1 riga, 3 colonne) ---
fig_en, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

ax1.scatter(tempi_fs, temperature, color='orangered', s=10, label='Temperatura')
ax1.set_title('Temperatura')
ax1.set_xlabel('Tempo (fs)')
ax1.set_ylabel('Temperatura (K)')
ax1.grid(True, linestyle=':', alpha=0.6)

ax2.scatter(tempi_fs, e_potenziali, color='royalblue', s=10, label='E. Potenziale')
ax2.scatter(tempi_fs, e_cinetiche, color='forestgreen', s=10, label='E. Cinetica')
ax2.scatter(tempi_fs, e_totali, color='black', s=15, label='E. Totale')
ax2.set_title('Confronto Energie')
ax2.set_xlabel('Tempo (fs)')
ax2.set_ylabel('Energia (eV)')
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend()

ax3.scatter(tempi_fs, e_potenziali, color='royalblue', s=10, label='Solo E. Potenziale')
ax3.set_title('Dettaglio E. Potenziale')
ax3.set_xlabel('Tempo (fs)')
ax3.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig(PATH_GRAFICO_ENERGIE, dpi=300)
plt.close(fig_en)

# --- GRAFICO 2: ANDAMENTO FORZE (1 riga, 3 colonne - Scatter Plot Puro) ---
fig_forze, (ax_fx, ax_fy, ax_fz) = plt.subplots(1, 3, figsize=(15, 5))

# Plot dei punti individuali di forza per ogni componente
ax_fx.scatter(tempi_forze_ripetuti, elenco_forze_x, color='red', s=5, alpha=0.5)
ax_fx.set_title('Forze Lungo X')
ax_fx.set_xlabel('Tempo (fs)')
ax_fx.set_ylabel('Forza (eV/Å)')
ax_fx.grid(True, linestyle=':', alpha=0.7)

ax_fy.scatter(tempi_forze_ripetuti, elenco_forze_y, color='green', s=5, alpha=0.5)
ax_fy.set_title('Forze Lungo Y')
ax_fy.set_xlabel('Tempo (fs)')
ax_fy.set_ylabel('Forza (eV/Å)')
ax_fy.grid(True, linestyle=':', alpha=0.7)

ax_fz.scatter(tempi_forze_ripetuti, elenco_forze_z, color='blue', s=5, alpha=0.5)
ax_fz.set_title('Forze Lungo Z')
ax_fz.set_xlabel('Tempo (fs)')
ax_fz.set_ylabel('Forza (eV/Å)')
ax_fz.grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
plt.savefig(PATH_GRAFICO_FORZE, dpi=300)
plt.close(fig_forze)

print("Tutti i grafici e i frame spaziali (2D e 3D) sono pronti!")
print("=" * 60)
