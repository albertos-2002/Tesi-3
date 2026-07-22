import os
import numpy as np
import matplotlib.pyplot as plt
import gc # Garbage Collector per svuotare forzatamente la RAM
from ase.io import read
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.zbl import ZBL
from quippy.potential import Potential

# =========================================================================
# 0. DEFINIZIONE DEL CALCOLATORE ZBL + GAP (ZBLSwitchCalculator)
# =========================================================================
class ZBLSwitchCalculator(Calculator):
    """
    Calculator che combina un potenziale GAP con lo ZBL.
    Usa una funzione di switching morbida (smooth cutoff) basata sulla
    distanza minima tra le coppie atomiche.
    
    Regione 1: r <= r_inner -> 100% ZBL
    Regione 2: r_inner < r < r_outer -> Transizione morbida (Cosinusoidale)
    Regione 3: r >= r_outer -> 100% GAP
    """
    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, main_calc, r_inner=1.2, r_outer=1.8, **kwargs):
        super().__init__(**kwargs)
        self.main_calc = main_calc
        self.r_inner = r_inner
        self.r_outer = r_outer
        # Calcolatore ZBL nativo di ASE
        self.zbl_calc = ZBL(r_cut=r_outer)

    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        
        # 1. Calcolo del potenziale principale (GAP)
        atoms.calc = self.main_calc
        e_main = atoms.get_potential_energy()
        f_main = atoms.get_forces()
        
        # 2. Calcolo dello ZBL
        atoms.calc = self.zbl_calc
        e_zbl = atoms.get_potential_energy()
        f_zbl = atoms.get_forces()
        
        # Ripristiniamo self come calcolatore
        atoms.calc = self

        # 3. Calcolo delle distanze minime per ogni atomo (con PBC)
        dist_matrix = atoms.get_all_distances(mic=True)
        np.fill_diagonal(dist_matrix, np.inf)
        r_min = np.min(dist_matrix, axis=1)

        # 4. Calcolo del fattore di switching S(r) per ciascun atomo
        S = np.zeros(len(atoms))
        for i, r in enumerate(r_min):
            if r <= self.r_inner:
                S[i] = 1.0
            elif r >= self.r_outer:
                S[i] = 0.0
            else:
                x = (r - self.r_inner) / (self.r_outer - self.r_inner)
                S[i] = 0.5 * (1.0 + np.cos(np.pi * x))

        # 5. Combinazione di Energie e Forze
        S_mean = np.mean(S)
        e_total = S_mean * e_zbl + (1.0 - S_mean) * e_main

        f_total = np.zeros_like(f_main)
        for i in range(len(atoms)):
            f_total[i] = S[i] * f_zbl[i] + (1.0 - S[i]) * f_main[i]

        self.results['energy'] = e_total
        self.results['forces'] = f_total
        if 'stress' in properties:
            self.results['stress'] = (1 - S_mean) * atoms.get_stress()


# =========================================================================
# 1. PARAMETRI E PERCORSI
# =========================================================================
PATH_POTENZIALE = "../results-test2/out_potenziale_gap-500-n2-l2.xml"
PATH_CONFIG_INIZIALE = "../test_data.extxyz"

# Parametri della finestra di transizione ZBL (Å)
R_INNER = 1.5
R_OUTER = 2.5

# Cartelle per i frame spaziali
DIR_SALVATAGGIO_2D = "immagini_MD_NVE_2D"
DIR_SALVATAGGIO_3D = "immagini_MD_NVE_3D"

# File di Output Dati (Streaming su disco)
PATH_CSV_ENERGIE = "dati_energie.csv"
PATH_CSV_FORZE_DISTANZE = "dati_forze_distanze.csv"

# Grafici finali
PATH_GRAFICO_ENERGIE = "andamento_energia_nve.png"
PATH_GRAFICO_FORZE_TEMPO = "andamento_forze_tempo.png"
PATH_GRAFICO_FORZA_DISTANZA = "forza_vs_distanza_minima.png"
PATH_GRAFICO_3D = "forza_vs_distanza_vs_tempo_3D.png"

# Parametri della Dinamica
TIMESTEP = 1 * units.fs 
PASSI_TOTALI = 500 

# Frequenza salvataggio immagini 2D/3D (per non intasare il disco)
FREQUENZA_IMMAGINI = 10 

print("=" * 60)
print(" AVVIO SIMULAZIONE DINAMICA NVE (GAP + ZBL SWITCHING)")
print("=" * 60)

os.makedirs(DIR_SALVATAGGIO_2D, exist_ok=True)
os.makedirs(DIR_SALVATAGGIO_3D, exist_ok=True)

# =========================================================================
# 2. CARICAMENTO E INIZIALIZZAZIONE CALCOLATORE IBRIDO
# =========================================================================
gap_calc = Potential(param_filename=PATH_POTENZIALE)
gap_calc.name_ = "GAP"

# Inizializzazione della combinazione GAP + ZBL
calc_ibrido = ZBLSwitchCalculator(main_calc=gap_calc, r_inner=R_INNER, r_outer=R_OUTER)

atomi_md = read(PATH_CONFIG_INIZIALE, index=0)
atomi_md.calc = calc_ibrido

try:
    if atomi_md.get_velocities() is None or not np.any(atomi_md.get_velocities()):
        raise ValueError
except (ValueError, RuntimeError):
    MaxwellBoltzmannDistribution(atomi_md, temperature_K=3000)

dyn = VelocityVerlet(
    atomi_md, 
    timestep=TIMESTEP, 
    trajectory="md_gap_zbl_nve.traj",
    logfile="md_gap_zbl_nve.log",
    loginterval=100
)

# =========================================================================
# 3. IMPOSTAZIONE FILE CSV
# =========================================================================
with open(PATH_CSV_ENERGIE, 'w') as f_en:
    f_en.write("Tempo_fs,Temp_K,E_pot_eV,E_kin_eV,E_tot_eV\n")

with open(PATH_CSV_FORZE_DISTANZE, 'w') as f_fd:
    f_fd.write("Tempo_fs,Atomo_ID,Specie,Distanza_Minima_A,Modulo_Forza_eV_A,Fx,Fy,Fz\n")

# =========================================================================
# 4. IL MOTORE DI MONITORAGGIO
# =========================================================================
def monitoraggio_sistema():
    # Garantisce che gli atomi rimangano graficamente/fisicamente nella cella
    atomi_md.wrap()
    
    passo = dyn.get_number_of_steps()
    tempo_fs = passo * (TIMESTEP / units.fs)
    
    # --- 1. Scrittura Diretta Energie ---
    epot = atomi_md.get_potential_energy()
    ekin = atomi_md.get_kinetic_energy()
    etot = epot + ekin
    temp = atomi_md.get_temperature()
    
    with open(PATH_CSV_ENERGIE, 'a') as f_en:
        f_en.write(f"{tempo_fs:.3f},{temp:.2f},{epot:.5f},{ekin:.5f},{etot:.5f}\n")
    
    # --- 2. Calcolo Distanza Minima e Modulo/Componenti Forze ---
    dist_matrix = atomi_md.get_all_distances(mic=True)
    np.fill_diagonal(dist_matrix, np.inf)
    min_dists = np.min(dist_matrix, axis=1)
    
    forze = atomi_md.get_forces()
    f_mags = np.linalg.norm(forze, axis=1)
    simboli = atomi_md.get_chemical_symbols()
    
    with open(PATH_CSV_FORZE_DISTANZE, 'a') as f_fd:
        for idx in range(len(atomi_md)):
            fx, fy, fz = forze[idx]
            f_fd.write(f"{tempo_fs:.3f},{idx},{simboli[idx]},{min_dists[idx]:.6f},{f_mags[idx]:.6f},{fx:.6f},{fy:.6f},{fz:.6f}\n")
    
    # --- 3. Generazione Immagini (SOLO SE NECESSARIO) ---
    if passo % FREQUENZA_IMMAGINI == 0:
        posizioni = atomi_md.get_positions(wrap=True)
        simboli_arr = np.array(simboli)
        
        pos_Ta = posizioni[simboli_arr == 'Ta']
        pos_O = posizioni[simboli_arr == 'O']
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
        
        plt.close('all')
        gc.collect()
        
        print(f"Step {passo:06d}/{PASSI_TOTALI} completato. Dati e Immagini salvati.")

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
print("Caricamento dati dal disco per la generazione dei grafici finali...")

# --- GRAFICO 1: ENERGIE ---
t_en, temp, ep, ek, et = np.loadtxt(PATH_CSV_ENERGIE, delimiter=',', skiprows=1, unpack=True)

fig_en, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
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

# --- LETTURA DATI PER GRAFICI DI FORZA ---
dati_fd = np.genfromtxt(PATH_CSV_FORZE_DISTANZE, delimiter=',', skip_header=1, dtype=None, encoding='utf-8')

tempi = dati_fd['f0']
specie = dati_fd['f2']
distanze_min = dati_fd['f3']
moduli_forza = dati_fd['f4']
fx = dati_fd['f5']
fy = dati_fd['f6']
fz = dati_fd['f7']

mask_Ta = (specie == 'Ta')
mask_O = (specie == 'O')

# --- GRAFICO 2: FORZA NEL TEMPO ---
print("Generazione grafico Forze nel Tempo...")
fig_forze, (ax_fx, ax_fy, ax_fz) = plt.subplots(1, 3, figsize=(15, 5))

ax_fx.scatter(tempi, fx, color='red', s=1, alpha=0.1)
ax_fx.set_title('Forze Lungo X')
ax_fx.set_xlabel('Tempo (fs)')
ax_fx.set_ylabel('Forza (eV/Å)')
ax_fx.grid(True, linestyle=':', alpha=0.7)

ax_fy.scatter(tempi, fy, color='green', s=1, alpha=0.1)
ax_fy.set_title('Forze Lungo Y')
ax_fy.set_xlabel('Tempo (fs)')
ax_fy.grid(True, linestyle=':', alpha=0.7)

ax_fz.scatter(tempi, fz, color='blue', s=1, alpha=0.1)
ax_fz.set_title('Forze Lungo Z')
ax_fz.set_xlabel('Tempo (fs)')
ax_fz.grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
plt.savefig(PATH_GRAFICO_FORZE_TEMPO, dpi=300)
plt.close(fig_forze)

# --- GRAFICO 3: FORZA VS DISTANZA MINIMA 2D ---
print("Generazione grafico Forza vs Distanza Minima 2D...")
fig_fd, ax_fd = plt.subplots(figsize=(9, 6))

ax_fd.scatter(distanze_min[mask_Ta], moduli_forza[mask_Ta], color='royalblue', alpha=0.4, s=15, label='Tantalo (Ta)', edgecolors='none')
ax_fd.scatter(distanze_min[mask_O], moduli_forza[mask_O], color='crimson', alpha=0.4, s=15, label='Ossigeno (O)', edgecolors='none')

ax_fd.axvline(x=R_INNER, color='black', linestyle='--', alpha=0.7, label=f'r_inner ({R_INNER} Å)')
ax_fd.axvline(x=R_OUTER, color='gray', linestyle=':', alpha=0.7, label=f'r_outer ({R_OUTER} Å)')

ax_fd.set_title("Modulo della Forza vs Distanza dal Vicino più Prossimo (GAP + ZBL)", fontsize=13, pad=12)
ax_fd.set_xlabel("Distanza dal Vicino più Prossimo (Å)", fontsize=11)
ax_fd.set_ylabel("Modulo della Forza Netta (eV/Å)", fontsize=11)
ax_fd.grid(True, linestyle='--', alpha=0.6)
ax_fd.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig(PATH_GRAFICO_FORZA_DISTANZA, dpi=300)
plt.close(fig_fd)

# --- GRAFICO 4: SCANSIONE 3D (TEMPO VS DISTANZA VS FORZA) ---
print("Generazione grafico 3D (Tempo vs Distanza vs Forza)...")
fig_3d = plt.figure(figsize=(10, 8))
ax_3d = fig_3d.add_subplot(111, projection='3d')

ax_3d.scatter(tempi[mask_Ta], distanze_min[mask_Ta], moduli_forza[mask_Ta], color='royalblue', alpha=0.3, s=10, label='Tantalo (Ta)')
ax_3d.scatter(tempi[mask_O], distanze_min[mask_O], moduli_forza[mask_O], color='crimson', alpha=0.3, s=10, label='Ossigeno (O)')

ax_3d.set_title("Scansione 3D: Evoluzione nel Tempo di Distanza e Forza", fontsize=13, pad=15)
ax_3d.set_xlabel("Tempo (fs)", fontsize=10, labelpad=10)
ax_3d.set_ylabel("Distanza dal Vicino (Å)", fontsize=10, labelpad=10)
ax_3d.set_zlabel("Modulo della Forza (eV/Å)", fontsize=10, labelpad=10)

ax_3d.view_init(elev=25, azim=135)
ax_3d.legend(loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig(PATH_GRAFICO_3D, dpi=300)
plt.close(fig_3d)

print("Tutti i grafici e la simulazione sono stati completati con successo!")
