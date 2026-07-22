import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from quippy.potential import Potential

# =========================================================================
# 1. CONFIGURAZIONE
# =========================================================================
PATH_POTENZIALE = "../../results-test2/out_potenziale_gap-500-n2-l2.xml"
PATH_CONFIG_INIZIALE = "../../test_data.extxyz"

PATH_TRAJ_OUTPUT = "md_gap_nve.traj"
PATH_LOG_OUTPUT = "md_gap_nve.log"
PATH_DATI_CSV = "dati_termodinamici.csv" # NUOVO FILE PER LO STREAMING
PATH_GRAFICO_OUTPUT = "andamento_nve.png"

TIMESTEP = 0.00001 * units.fs 
PASSI_TOTALI = 250000 
INTERVALLO_SALVATAGGIO = 1000 # Salva i dati fisici e geometrici solo ogni 100 step

SIMULATION_TIME = PASSI_TOTALI * 0.00001
print("Tempo di simulazione totale [fs]:")
print(SIMULATION_TIME)

input("Enter to start")

# =========================================================================
# 2. CARICAMENTO (Identico a prima)
# =========================================================================
calc = Potential(param_filename=PATH_POTENZIALE)
calc.name_ = "GAP" 
atomi = read(PATH_CONFIG_INIZIALE, index=0)
atomi.calc = calc

try:
    velocita_iniziali = atomi.get_velocities()
    if velocita_iniziali is None or not np.any(velocita_iniziali):
        raise ValueError
except (ValueError, RuntimeError):
    MaxwellBoltzmannDistribution(atomi, temperature_K=3000)

# =========================================================================
# 3. IMPOSTAZIONE DINAMICA E STREAMING SU DISCO
# =========================================================================
# Riduciamo la frequenza di scrittura della traiettoria (risparmia GigaByte di disco!)
dyn = VelocityVerlet(
    atomi, 
    timestep=TIMESTEP, 
    trajectory=PATH_TRAJ_OUTPUT,
    logfile=PATH_LOG_OUTPUT,
    loginterval=INTERVALLO_SALVATAGGIO
)

# Creiamo il file CSV vuoto e scriviamo solo l'intestazione (header)
with open(PATH_DATI_CSV, 'w') as f:
    f.write("Tempo_fs,Temp_K,E_pot_eV,E_kin_eV,E_tot_eV\n")

# Questa funzione ora NON usa più le liste, scrive direttamente nel file!
def monitoraggio_sistema():
    passo_attuale = dyn.get_number_of_steps()
    tempo_attuale_fs = passo_attuale * (TIMESTEP / units.fs)
    
    epot = atomi.get_potential_energy()
    ekin = atomi.get_kinetic_energy()
    etot = epot + ekin
    temp = atomi.get_temperature()
    
    # Scrittura in modalità "append" (a) sul file. La RAM non si riempie mai.
    with open(PATH_DATI_CSV, 'a') as f:
        f.write(f"{tempo_attuale_fs:.2f},{temp:.2f},{epot:.4f},{ekin:.4f},{etot:.4f}\n")
    
    # Stampa a schermo per farti capire che il codice è vivo
    if passo_attuale % (INTERVALLO_SALVATAGGIO ) == 0:
        print(f"Step: {passo_attuale:>6d}/{PASSI_TOTALI} | Temp: {temp:>6.1f} K | Etot: {etot:>10.3f} eV")

# Attacchiamo la funzione con l'intervallo scelto
dyn.attach(monitoraggio_sistema, interval=INTERVALLO_SALVATAGGIO)

# =========================================================================
# 4. ESECUZIONE SIMULAZIONE
# =========================================================================
print("\n--- INIZIO DINAMICA MOLECOLARE NVE ---")
dyn.run(PASSI_TOTALI)
print("--- DINAMICA COMPLETATA ---\n")

# =========================================================================
# 5. LETTURA DATI E GRAFICO (Post-Processing)
# =========================================================================
print("Generazione dei grafici di controllo...")

# np.loadtxt è ultra-efficiente per leggere file grandi. 
# unpack=True scompatta automaticamente le colonne nelle nostre 5 variabili.
tempi_fs, temperature, e_potenziali, e_cinetiche, e_totali = np.loadtxt(
    PATH_DATI_CSV, delimiter=',', skiprows=1, unpack=True
)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

# Scatter plot usando i dati appena riletti dal disco
ax1.scatter(tempi_fs, temperature, color='orangered', s=5, alpha=0.7, label='Temperatura')
ax1.set_ylabel('Temperatura (K)', fontsize=11)
ax1.title.set_text('Dinamica NVE: Efficienza RAM')
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='upper right')

ax2.scatter(tempi_fs, e_potenziali, color='royalblue', s=5, alpha=0.6, label='E. Potenziale (V)')
ax2.scatter(tempi_fs, e_cinetiche, color='forestgreen', s=5, alpha=0.6, label='E. Cinetica (K)')
ax2.scatter(tempi_fs, e_totali, color='black', s=10, alpha=0.8, label='E. Totale (E = V + K)')
ax2.set_ylabel('Energia (eV)', fontsize=11)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='center right')

ax3.scatter(tempi_fs, e_potenziali, color='royalblue', s=5, alpha=0.8, label='Solo E. Potenziale')
ax3.set_xlabel('Tempo (fs)', fontsize=11)
ax3.set_ylabel('Energia Pot. (eV)', fontsize=11)
ax3.grid(True, linestyle=':', alpha=0.6)
ax3.legend(loc='upper right')

plt.tight_layout()
plt.savefig(PATH_GRAFICO_OUTPUT, dpi=300)
plt.show()
