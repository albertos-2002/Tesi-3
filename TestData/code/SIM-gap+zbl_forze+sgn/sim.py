import os
import gc 
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from quippy.potential import Potential

# Import ASE per il potenziale ZBL esplicito e la combinazione dei calcolatori
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.mixing import SumCalculator
from ase.neighborlist import neighbor_list
from ase.units import Bohr, Hartree

# =========================================================================
# 1. PARAMETRI E PERCORSI
# =========================================================================
PATH_POTENZIALE = "../TRAINING-gap-2/out_potenziale_gap-500-n2-l2.xml"
PATH_CONFIG_INIZIALE = "../test_data.extxyz"

# Raggi di cutoff per lo spegnimento dello ZBL
RAGGIO_INNER = 1.5
RAGGIO_OUTER = 2.5

# Cartelle per i frame spaziali
DIR_SALVATAGGIO_2D = "immagini_MD_NVE_2D"
DIR_SALVATAGGIO_3D = "immagini_MD_NVE_3D"

# Output Dati (Streaming su disco)
PATH_CSV_ENERGIE = "dati_energie.csv"
PATH_CSV_FORZE_DISTANZE = "dati_forze_distanze.csv"
PATH_CSV_FORZE_ZBL = "dati_forze_zbl_pure.csv" # NUOVO FILE PER LE FORZE SOLO ZBL

# Grafici finali
PATH_GRAFICO_ENERGIE = "andamento_energia_nve.png"
PATH_GRAFICO_FORZE_TEMPO = "andamento_forze_tempo.png"
PATH_GRAFICO_FORZA_DISTANZA = "forza_vs_distanza_minima.png"
PATH_GRAFICO_3D = "forza_vs_distanza_vs_tempo_3D.png"
PATH_GRAFICO_DISTRIBUZIONE_ZBL = "distribuzione_forze_zbl.png" # NUOVO GRAFICO

# Parametri della Dinamica
TIMESTEP = 1.0 * units.fs 
PASSI_TOTALI = 500 

GENERA_IMMAGINI_SPAZIALI = True 
FREQUENZA_IMMAGINI = 10 

print("=" * 60)
print(" AVVIO SIMULAZIONE DINAMICA NVE (GAP + ZBL ESPLICITO DEL PROF.)")
print("=" * 60)

if GENERA_IMMAGINI_SPAZIALI:
    os.makedirs(DIR_SALVATAGGIO_2D, exist_ok=True)
    os.makedirs(DIR_SALVATAGGIO_3D, exist_ok=True)


# =========================================================================
# 2. DEFINIZIONE DEL POTENZIALE ZBL
# =========================================================================
class ZBLSwitchCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces"]

    _coefficients = np.array([0.1818, 0.5099, 0.2802, 0.02817], dtype=float)
    _exponents = np.array([3.2, 0.9423, 0.4029, 0.2016], dtype=float)

    def __init__(self, r_inner: float, r_outer: float, **kwargs):
        super().__init__(**kwargs)
        if r_inner <= 0.0 or r_outer <= r_inner:
            raise ValueError("Parametri r_inner/r_outer non validi.")
        self.r_inner = float(r_inner)
        self.r_outer = float(r_outer)

    def _switch(self, distances):
        distances = np.asarray(distances)
        switch = np.ones_like(distances)
        dswitch_dr = np.zeros_like(distances)

        transition = (distances > self.r_inner) & (distances < self.r_outer)
        t = (distances[transition] - self.r_inner) / (self.r_outer - self.r_inner)

        switch[transition] = 1.0 - 10.0 * t**3 + 15.0 * t**4 - 6.0 * t**5
        dswitch_dt = -30.0 * t**2 + 60.0 * t**3 - 30.0 * t**4
        dswitch_dr[transition] = dswitch_dt / (self.r_outer - self.r_inner)

        outside = distances >= self.r_outer
        switch[outside] = 0.0
        dswitch_dr[outside] = 0.0

        return switch, dswitch_dr

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        natoms = len(self.atoms)
        atomic_numbers = self.atoms.numbers.astype(float)

        i, j, r, displacement = neighbor_list("ijdD", self.atoms, self.r_outer, self_interaction=False)

        if len(r) == 0:
            energy = 0.0
            forces = np.zeros((natoms, 3))
            self.results = {"energy": energy, "free_energy": energy, "forces": forces}
            return

        if np.any(r < 1.0e-12):
            raise FloatingPointError("Due atomi sono praticamente sovrapposti.")

        Zi = atomic_numbers[i]
        Zj = atomic_numbers[j]

        aij = 0.8854 * Bohr / (Zi**0.23 + Zj**0.23)
        x = r / aij

        exponentials = np.exp(-x[:, None] * self._exponents[None, :])
        phi = np.sum(self._coefficients[None, :] * exponentials, axis=1)
        dphi_dx = -np.sum((self._coefficients * self._exponents)[None, :] * exponentials, axis=1)

        prefactor = Hartree * Bohr * Zi * Zj
        zbl_energy = prefactor * phi / r
        dzbl_dr = prefactor * (dphi_dx / (aij * r) - phi / r**2)

        switch, dswitch_dr = self._switch(r)

        pair_energy = switch * zbl_energy
        dpair_dr = switch * dzbl_dr + dswitch_dr * zbl_energy

        pair_forces = (dpair_dr / r)[:, None] * displacement

        forces = np.zeros((natoms, 3))
        for component in range(3):
            forces[:, component] = np.bincount(i, weights=pair_forces[:, component], minlength=natoms)

        energy = 0.5 * np.sum(pair_energy)
        self.results = {"energy": energy, "free_energy": energy, "forces": forces}


# =========================================================================
# 3. CARICAMENTO E FUSIONE POTENZIALI
# =========================================================================
calc_gap = Potential(param_filename=PATH_POTENZIALE)
calc_gap.name_ = "GAP"

calc_zbl = ZBLSwitchCalculator(r_inner=RAGGIO_INNER, r_outer=RAGGIO_OUTER)
calc_totale = SumCalculator([calc_gap, calc_zbl])

atomi_md = read(PATH_CONFIG_INIZIALE, index=0)
atomi_md.calc = calc_totale

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
# 4. PREPARAZIONE FILE DI SALVATAGGIO
# =========================================================================
with open(PATH_CSV_ENERGIE, 'w') as f_en:
    f_en.write("Tempo_fs,Temp_K,E_pot_eV,E_kin_eV,E_tot_eV\n")

with open(PATH_CSV_FORZE_DISTANZE, 'w') as f_fd:
    f_fd.write("Tempo_fs,Atomo_ID,Specie,Distanza_Minima_A,Modulo_Forza_eV_A,Fx,Fy,Fz\n")

# CSV dedicato alle sole forze ZBL
with open(PATH_CSV_FORZE_ZBL, 'w') as f_zbl:
    f_zbl.write("Tempo_fs,Atomo_ID,Distanza_Vicino_A,Forza_Radiale_ZBL,Tipo_Forza\n")

# =========================================================================
# 5. MOTORE DI MONITORAGGIO
# =========================================================================
def monitoraggio_sistema():
    passo = dyn.get_number_of_steps()
    tempo_fs = passo * (TIMESTEP / units.fs)
    
    # 1. Dati Energie Totali
    epot = atomi_md.get_potential_energy()
    ekin = atomi_md.get_kinetic_energy()
    etot = epot + ekin
    temp = atomi_md.get_temperature()
    
    with open(PATH_CSV_ENERGIE, 'a') as f_en:
        f_en.write(f"{tempo_fs:.3f},{temp:.2f},{epot:.5f},{ekin:.5f},{etot:.5f}\n")
    
    # 2. Forze Totali (GAP + ZBL)
    dist_matrix = atomi_md.get_all_distances(mic=True)
    np.fill_diagonal(dist_matrix, np.inf)
    min_dists = np.min(dist_matrix, axis=1)
    
    forze_tot = atomi_md.get_forces()
    f_mags = np.linalg.norm(forze_tot, axis=1)
    simboli = atomi_md.get_chemical_symbols()
    
    with open(PATH_CSV_FORZE_DISTANZE, 'a') as f_fd:
        for idx in range(len(atomi_md)):
            fx, fy, fz = forze_tot[idx]
            f_fd.write(f"{tempo_fs:.3f},{idx},{simboli[idx]},{min_dists[idx]:.6f},{f_mags[idx]:.6f},{fx:.6f},{fy:.6f},{fz:.6f}\n")
    
    # 3. REGISTRAZIONE ISOLATA DELLE FORZE ZBL
    calc_zbl.calculate(atomi_md)
    forze_zbl = calc_zbl.results["forces"]
    
    # Trova gli indici dei vicini più prossimi per proiettare la forza radiale
    nearest_idx = np.argmin(dist_matrix, axis=1)
    
    with open(PATH_CSV_FORZE_ZBL, 'a') as f_zbl:
        for idx in range(len(atomi_md)):
            r_min = min_dists[idx]
            if r_min < RAGGIO_OUTER: # Lo ZBL agisce solo sotto r_outer
                j_idx = nearest_idx[idx]
                # Vettore spostamento i -> j
                vec_ij = atomi_md.get_distance(idx, j_idx, vector=True, mic=True)
                unit_vec_ij = vec_ij / np.linalg.norm(vec_ij)
                
                # Proiezione della forza ZBL lungo la linea di congiunzione (Forza radiale)
                # F_rad = - F_zbl . u_ij (se punta via da j è repulsiva, cioè positiva)
                f_radiale = -np.dot(forze_zbl[idx], unit_vec_ij)
                tipo = "Repulsiva" if f_radiale > 0 else "Attrattiva"
                
                f_zbl.write(f"{tempo_fs:.3f},{idx},{r_min:.6f},{f_radiale:.6f},{tipo}\n")
            else:
                f_zbl.write(f"{tempo_fs:.3f},{idx},{r_min:.6f},0.000000,Nulla\n")

    # Immagini spaziali
    if GENERA_IMMAGINI_SPAZIALI and passo % FREQUENZA_IMMAGINI == 0:
        posizioni = atomi_md.get_positions(wrap=True)
        simboli_arr = np.array(simboli)
        pos_Ta = posizioni[simboli_arr == 'Ta']
        pos_O = posizioni[simboli_arr == 'O']
        nome_file = f"{passo:06d}_configurazione_spaziale.png"
        
        fig2d, ax2d = plt.subplots(figsize=(6, 6))
        ax2d.scatter(pos_Ta[:, 0], pos_Ta[:, 1], color='blue', s=80, label='Tantalo', edgecolors='black')
        ax2d.scatter(pos_O[:, 0], pos_O[:, 1], color='red', s=40, label='Ossigeno', edgecolors='black')
        ax2d.set_title(f"2D - Step {passo}")
        ax2d.axis('equal')
        fig2d.savefig(os.path.join(DIR_SALVATAGGIO_2D, nome_file), dpi=100)
        plt.close(fig2d)
        
        fig3d = plt.figure(figsize=(6, 6))
        ax3d = fig3d.add_subplot(111, projection='3d')
        ax3d.scatter(pos_Ta[:, 0], pos_Ta[:, 1], pos_Ta[:, 2], color='blue', s=80, edgecolors='black')
        ax3d.scatter(pos_O[:, 0], pos_O[:, 1], pos_O[:, 2], color='red', s=40, edgecolors='black')
        ax3d.set_title(f"3D - Step {passo}")
        fig3d.savefig(os.path.join(DIR_SALVATAGGIO_3D, nome_file), dpi=100)
        plt.close(fig3d)
        
        plt.close('all')
        gc.collect()

dyn.attach(monitoraggio_sistema, interval=1)

# =========================================================================
# 6. ESECUZIONE
# =========================================================================
print("\n--- INIZIO DINAMICA MOLECOLARE ---")
dyn.run(PASSI_TOTALI)
print("--- DINAMICA COMPLETATA ---\n")

# =========================================================================
# 7. GRAFICI FINALI
# =========================================================================
# --- GRAFICO DEDICATO ALLA DISTRIBUZIONE FORZE ZBL ---
print("Generazione grafico di distribuzione delle forze ZBL...")
dati_zbl = np.genfromtxt(PATH_CSV_FORZE_ZBL, delimiter=',', skip_header=1, dtype=None, encoding='utf-8')

dist_zbl = dati_zbl['f2']
f_rad = dati_zbl['f3']
mask_attive = (dist_zbl < RAGGIO_OUTER)

fig_zbl, (ax_hist, ax_scat) = plt.subplots(1, 2, figsize=(14, 6))

# Subplot A: Istogramma della natura delle forze
repulsive = f_rad[mask_attive & (f_rad > 0)]
attrattive = f_rad[mask_attive & (f_rad < 0)]

ax_hist.hist([repulsive, np.abs(attrattive)], bins=30, color=['firebrick', 'navy'], 
             label=['Forze Repulsive (F > 0)', 'Forze Attrattive (|F| se F < 0)'], alpha=0.7)
ax_hist.set_title("Distribuzione Frequenza Forze ZBL Pure")
ax_hist.set_xlabel("Magnitudine della Forza Radiale ZBL (eV/Å)")
ax_hist.set_ylabel("Conteggio Eventi")
ax_hist.legend()
ax_hist.grid(True, linestyle=':', alpha=0.6)

# Subplot B: Forza Radiale in funzione della Distanza
ax_scat.scatter(dist_zbl[mask_attive], f_rad[mask_attive], c=f_rad[mask_attive], 
                 cmap='coolwarm', s=20, alpha=0.6)
ax_scat.axhline(0, color='black', linestyle='--', linewidth=1)
ax_scat.axvline(RAGGIO_INNER, color='green', linestyle=':', label=f'r_inner ({RAGGIO_INNER} Å)')
ax_scat.axvline(RAGGIO_OUTER, color='orange', linestyle=':', label=f'r_outer ({RAGGIO_OUTER} Å)')

ax_scat.set_title("Forza Radiale ZBL vs Distanza (Segno Esplicito)")
ax_scat.set_xlabel("Distanza dal Vicino più Prossimo (Å)")
ax_scat.set_ylabel("Forza Radiale ZBL (eV/Å)\n[>0 Repulsiva | <0 Attrattiva]")
ax_scat.legend()
ax_scat.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig(PATH_GRAFICO_DISTRIBUZIONE_ZBL, dpi=300)
plt.close(fig_zbl)

print("Tutti i dati e i grafici sono stati generati con successo!")
