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

# Raggi di cutoff per lo spegnimento dello ZBL (Parametri del Professore)
RAGGIO_INNER = 1.5
RAGGIO_OUTER = 2.5

# Cartelle per i frame spaziali
DIR_SALVATAGGIO_2D = "immagini_MD_NVE_2D"
DIR_SALVATAGGIO_3D = "immagini_MD_NVE_3D"

# Output Dati (Streaming su disco)
PATH_CSV_ENERGIE = "dati_energie.csv"
PATH_CSV_FORZE_DISTANZE = "dati_forze_distanze.csv"
PATH_CSV_CONFRONTO = "dati_confronto_forze.csv" 
PATH_CSV_VELOCITA = "dati_velocita.csv" 

# Grafici finali
PATH_GRAFICO_ENERGIE = "andamento_energia_nve.png"
PATH_GRAFICO_FORZE_TEMPO = "andamento_forze_tempo.png"
PATH_GRAFICO_FORZA_DISTANZA = "forza_vs_distanza_minima.png"
PATH_GRAFICO_3D = "forza_vs_distanza_vs_tempo_3D.png"

# Grafici Confronto Forze (Radiali con Segno)
PATH_GRAFICO_CONFRONTO_TEMPO = "confronto_forze_tempo.png"
PATH_GRAFICO_CONFRONTO_DISTANZA = "confronto_forze_distanza.png"

# Grafici Velocità
PATH_GRAFICO_VELOCITA_TEMPO = "velocita_vs_tempo.png"
PATH_GRAFICO_VELOCITA_DISTANZA = "velocita_vs_distanza_minima.png"

# Parametri della Dinamica
TIMESTEP = 1.0 * units.fs 
PASSI_TOTALI = 500 

# Controllo generazione immagini spaziali degli atomi
GENERA_IMMAGINI_SPAZIALI = True 
FREQUENZA_IMMAGINI = 10 

print("=" * 60)
print(" AVVIO SIMULAZIONE DINAMICA NVE (GAP + ZBL + VELOCITÀ)")
print("=" * 60)

if GENERA_IMMAGINI_SPAZIALI:
    os.makedirs(DIR_SALVATAGGIO_2D, exist_ok=True)
    os.makedirs(DIR_SALVATAGGIO_3D, exist_ok=True)


# =========================================================================
# 2. DEFINIZIONE DEL POTENZIALE ZBL (Dal file sim.py del professore)
# =========================================================================
class ZBLSwitchCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces"]

    _coefficients = np.array([0.1818, 0.5099, 0.2802, 0.02817], dtype=float)
    _exponents = np.array([3.2, 0.9423, 0.4029, 0.2016], dtype=float)

    def __init__(self, r_inner: float, r_outer: float, **kwargs):
        super().__init__(**kwargs)

        if r_inner <= 0.0:
            raise ValueError("r_inner deve essere positivo.")
        if r_outer <= r_inner:
            raise ValueError("Deve essere r_outer > r_inner.")

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
# 3. CARICAMENTO E FUSIONE POTENZIALI (GAP + ZBL)
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
    loginterval=1
)

# =========================================================================
# 4. PREPARAZIONE FILE DI SALVATAGGIO
# =========================================================================
with open(PATH_CSV_ENERGIE, 'w') as f_en:
    f_en.write("Tempo_fs,Temp_K,E_pot_eV,E_kin_eV,E_tot_eV\n")

with open(PATH_CSV_FORZE_DISTANZE, 'w') as f_fd:
    f_fd.write("Tempo_fs,Atomo_ID,Specie,Distanza_Minima_A,Modulo_Forza_eV_A,Fx,Fy,Fz\n")

with open(PATH_CSV_CONFRONTO, 'w') as f_conf:
    f_conf.write("Tempo_fs,Atomo_ID,Distanza_Minima_A,F_rad_tot,F_rad_gap,F_rad_zbl\n")

with open(PATH_CSV_VELOCITA, 'w') as f_v:
    f_v.write("Tempo_fs,Atomo_ID,Specie,Distanza_Minima_A,Modulo_Velocita_A_fs,Vx,Vy,Vz\n")


# =========================================================================
# 5. MOTORE DI MONITORAGGIO
# =========================================================================
def monitoraggio_sistema():
    passo = dyn.get_number_of_steps()
    tempo_fs = passo * (TIMESTEP / units.fs)
    
    # 1. Scrittura Dati Energie
    epot = atomi_md.get_potential_energy()
    ekin = atomi_md.get_kinetic_energy()
    etot = epot + ekin
    temp = atomi_md.get_temperature()
    
    with open(PATH_CSV_ENERGIE, 'a') as f_en:
        f_en.write(f"{tempo_fs:.3f},{temp:.2f},{epot:.5f},{ekin:.5f},{etot:.5f}\n")
    
    # 2. Calcolo Distanze Minime e Forze Atomiche Totali
    dist_matrix = atomi_md.get_all_distances(mic=True)
    np.fill_diagonal(dist_matrix, np.inf)
    min_dists = np.min(dist_matrix, axis=1)
    nearest_idx = np.argmin(dist_matrix, axis=1)
    
    forze_tot = atomi_md.get_forces()
    f_mags = np.linalg.norm(forze_tot, axis=1)
    simboli = atomi_md.get_chemical_symbols()
    
    with open(PATH_CSV_FORZE_DISTANZE, 'a') as f_fd:
        for idx in range(len(atomi_md)):
            fx, fy, fz = forze_tot[idx]
            f_fd.write(f"{tempo_fs:.3f},{idx},{simboli[idx]},{min_dists[idx]:.6f},{f_mags[idx]:.6f},{fx:.6f},{fy:.6f},{fz:.6f}\n")
            
    # 3. Estrazione Forze Separate per Confronto e Proiezione Radiale (Modulo con segno)
    # I calcolatori hanno salvato i loro risultati durante atomi_md.get_forces()
    f_gap = calc_gap.results.get('forces', np.zeros_like(forze_tot))
    f_zbl = calc_zbl.results.get('forces', np.zeros_like(forze_tot))
    
    with open(PATH_CSV_CONFRONTO, 'a') as f_conf:
        for idx in range(len(atomi_md)):
            r_min = min_dists[idx]
            j_idx = nearest_idx[idx]
            
            vec_ij = atomi_md.get_distance(idx, j_idx, vector=True, mic=True)
            if r_min > 1e-12:
                unit_vec_ij = vec_ij / r_min
                frad_tot = -np.dot(forze_tot[idx], unit_vec_ij)
                frad_gap = -np.dot(f_gap[idx], unit_vec_ij)
                frad_zbl = -np.dot(f_zbl[idx], unit_vec_ij)
            else:
                frad_tot = frad_gap = frad_zbl = 0.0
                
            f_conf.write(f"{tempo_fs:.3f},{idx},{r_min:.6f},{frad_tot:.6f},{frad_gap:.6f},{frad_zbl:.6f}\n")
            
    # 4. Registrazione Velocità Atomiche (Convertite in Å/fs)
    velocita = atomi_md.get_velocities()
    if velocita is not None:
        v_fs = velocita * units.fs
        v_mags = np.linalg.norm(v_fs, axis=1)
        
        with open(PATH_CSV_VELOCITA, 'a') as f_v:
            for idx in range(len(atomi_md)):
                vx, vy, vz = v_fs[idx]
                f_v.write(f"{tempo_fs:.3f},{idx},{simboli[idx]},{min_dists[idx]:.6f},{v_mags[idx]:.6f},{vx:.6f},{vy:.6f},{vz:.6f}\n")

    # 5. Generazione Immagini Spaziali
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
        print(f"Step {passo:06d}/{PASSI_TOTALI} completato. Dati e Immagini salvati.")
        
    elif passo % 1000 == 0:
        print(f"Step {passo:06d}/{PASSI_TOTALI} completato.")

dyn.attach(monitoraggio_sistema, interval=1)


# =========================================================================
# 6. ESECUZIONE SIMULAZIONE
# =========================================================================
print("\n--- INIZIO DINAMICA MOLECOLARE ---")
dyn.run(PASSI_TOTALI)
print("--- DINAMICA COMPLETATA ---\n")


# =========================================================================
# 7. GENERAZIONE GRAFICI FINALI
# =========================================================================
print("Caricamento dati per la generazione dei grafici finali...")

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


# --- LETTURA DATI FORZE E DISTANZE TOTALI ---
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


# --- GRAFICO 2: COMPONENTI FORZA X, Y, Z (TEMPO) ---
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
fig_fd, ax_fd = plt.subplots(figsize=(9, 6))
ax_fd.scatter(distanze_min[mask_Ta], moduli_forza[mask_Ta], color='royalblue', alpha=0.4, s=15, label='Tantalo (Ta)', edgecolors='none')
ax_fd.scatter(distanze_min[mask_O], moduli_forza[mask_O], color='crimson', alpha=0.4, s=15, label='Ossigeno (O)', edgecolors='none')
ax_fd.axvline(x=RAGGIO_INNER, color='black', linestyle='--', alpha=0.7, label=f'r_inner ({RAGGIO_INNER} Å)')
ax_fd.axvline(x=RAGGIO_OUTER, color='gray', linestyle=':', alpha=0.7, label=f'r_outer ({RAGGIO_OUTER} Å)')

ax_fd.set_title("Modulo della Forza vs Distanza dal Vicino più Prossimo (GAP + ZBL)", fontsize=13, pad=12)
ax_fd.set_xlabel("Distanza dal Vicino più Prossimo (Å)", fontsize=11)
ax_fd.set_ylabel("Modulo della Forza Netta (eV/Å)", fontsize=11)
ax_fd.grid(True, linestyle='--', alpha=0.6)
ax_fd.legend(loc='upper right', fontsize=10)
plt.tight_layout()
plt.savefig(PATH_GRAFICO_FORZA_DISTANZA, dpi=300)
plt.close(fig_fd)


# --- GRAFICO 4: SCANSIONE 3D (TEMPO VS DISTANZA VS FORZA) ---
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


# --- GRAFICI 5 E 6: CONFRONTO FORZE (TOTALE, GAP, ZBL) ---
print("Generazione grafici di confronto forze (Totale vs GAP vs ZBL)...")
dati_conf = np.genfromtxt(PATH_CSV_CONFRONTO, delimiter=',', skip_header=1, dtype=None, encoding='utf-8')
t_conf = dati_conf['f0']
dist_conf = dati_conf['f2']
f_tot_rad = dati_conf['f3']
f_gap_rad = dati_conf['f4']
f_zbl_rad = dati_conf['f5']

fig_ct, ax_ct = plt.subplots(figsize=(10, 6))
ax_ct.scatter(t_conf, f_tot_rad, color='black', s=5, alpha=0.4, label='Forza Totale')
ax_ct.scatter(t_conf, f_gap_rad, color='royalblue', s=5, alpha=0.4, label='Forza GAP')
ax_ct.scatter(t_conf, f_zbl_rad, color='crimson', s=5, alpha=0.4, label='Forza ZBL')

ax_ct.set_title("Confronto Forze nel Tempo (Magnitudine con Segno)")
ax_ct.set_xlabel("Tempo (fs)")
ax_ct.set_ylabel("Forza Radiale (eV/Å) [>0 Repulsiva, <0 Attrattiva]")
ax_ct.grid(True, linestyle='--', alpha=0.6)
ax_ct.legend(markerscale=3)
plt.tight_layout()
plt.savefig(PATH_GRAFICO_CONFRONTO_TEMPO, dpi=300)
plt.close(fig_ct)

fig_cd, ax_cd = plt.subplots(figsize=(10, 6))
ax_cd.scatter(dist_conf, f_tot_rad, color='black', s=5, alpha=0.4, label='Forza Totale')
ax_cd.scatter(dist_conf, f_gap_rad, color='royalblue', s=5, alpha=0.4, label='Forza GAP')
ax_cd.scatter(dist_conf, f_zbl_rad, color='crimson', s=5, alpha=0.4, label='Forza ZBL')

ax_cd.axvline(x=RAGGIO_INNER, color='green', linestyle=':', label=f'r_inner ({RAGGIO_INNER} Å)')
ax_cd.axvline(x=RAGGIO_OUTER, color='orange', linestyle=':', label=f'r_outer ({RAGGIO_OUTER} Å)')
ax_cd.axhline(y=0, color='gray', linestyle='--', linewidth=1)

ax_cd.set_title("Confronto Forze vs Distanza dal Vicino più Prossimo")
ax_cd.set_xlabel("Distanza dal Vicino (Å)")
ax_cd.set_ylabel("Forza Radiale (eV/Å) [>0 Repulsiva, <0 Attrattiva]")
ax_cd.grid(True, linestyle='--', alpha=0.6)
ax_cd.legend(markerscale=3)
plt.tight_layout()
plt.savefig(PATH_GRAFICO_CONFRONTO_DISTANZA, dpi=300)
plt.close(fig_cd)


# --- GRAFICI 7 E 8: VELOCITÀ VS TEMPO E VELOCITÀ VS DISTANZA ---
print("Generazione grafici delle Velocità...")
dati_v = np.genfromtxt(PATH_CSV_VELOCITA, delimiter=',', skip_header=1, dtype=None, encoding='utf-8')
tempi_v = dati_v['f0']
specie_v = dati_v['f2']
distanze_min_v = dati_v['f3']
moduli_vel = dati_v['f4']

mask_Ta_v = (specie_v == 'Ta')
mask_O_v = (specie_v == 'O')

fig_vt, ax_vt = plt.subplots(figsize=(9, 6))
ax_vt.scatter(tempi_v[mask_Ta_v], moduli_vel[mask_Ta_v], color='royalblue', alpha=0.4, s=15, label='Tantalo (Ta)', edgecolors='none')
ax_vt.scatter(tempi_v[mask_O_v], moduli_vel[mask_O_v], color='crimson', alpha=0.4, s=15, label='Ossigeno (O)', edgecolors='none')

ax_vt.set_title("Modulo della Velocità vs Tempo", fontsize=13, pad=12)
ax_vt.set_xlabel("Tempo (fs)", fontsize=11)
ax_vt.set_ylabel("Modulo della Velocità (Å/fs)", fontsize=11)
ax_vt.grid(True, linestyle='--', alpha=0.6)
ax_vt.legend(loc='upper right', fontsize=10)
plt.tight_layout()
plt.savefig(PATH_GRAFICO_VELOCITA_TEMPO, dpi=300)
plt.close(fig_vt)

fig_vd, ax_vd = plt.subplots(figsize=(9, 6))
ax_vd.scatter(distanze_min_v[mask_Ta_v], moduli_vel[mask_Ta_v], color='royalblue', alpha=0.4, s=15, label='Tantalo (Ta)', edgecolors='none')
ax_vd.scatter(distanze_min_v[mask_O_v], moduli_vel[mask_O_v], color='crimson', alpha=0.4, s=15, label='Ossigeno (O)', edgecolors='none')

ax_vd.axvline(x=RAGGIO_INNER, color='black', linestyle='--', alpha=0.7, label=f'r_inner ({RAGGIO_INNER} Å)')
ax_vd.axvline(x=RAGGIO_OUTER, color='gray', linestyle=':', alpha=0.7, label=f'r_outer ({RAGGIO_OUTER} Å)')

ax_vd.set_title("Modulo della Velocità vs Distanza dal Vicino più Prossimo", fontsize=13, pad=12)
ax_vd.set_xlabel("Distanza dal Vicino più Prossimo (Å)", fontsize=11)
ax_vd.set_ylabel("Modulo della Velocità (Å/fs)", fontsize=11)
ax_vd.grid(True, linestyle='--', alpha=0.6)
ax_vd.legend(loc='upper right', fontsize=10)
plt.tight_layout()
plt.savefig(PATH_GRAFICO_VELOCITA_DISTANZA, dpi=300)
plt.close(fig_vd)

print("Tutto completato! Simulazione eseguita ed i grafici sono stati generati con successo.")