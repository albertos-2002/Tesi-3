import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from quippy.potential import Potential

# =========================================================================
# 1. CONFIGURAZIONE PERCORSI E PARAMETRI
# =========================================================================
PATH_POTENZIALE = "../results-test2/out_potenziale_gap-500-n2-l2.xml"
PATH_CONFIG_INIZIALE = "../train_data.extxyz"

PATH_TRAJ_OUTPUT = "md_gap_nve-traindata.traj"
PATH_LOG_OUTPUT = "md_gap_nve-traindata.log"
PATH_GRAFICO_OUTPUT = "andamento_nve-traindata.png"

#ricavato dai file di out di qe
TIMESTEP = 4.8 * units.fs 

print("=" * 60)
print(" AVVIO SIMULAZIONE DINAMICA MOLECOLARE NVE (GAP)")
print("=" * 60)

# =========================================================================
# 2. CARICAMENTO DEL POTENZIALE E DELLA STRUTTURA
# =========================================================================
print(f"Caricamento del potenziale GAP da: {PATH_POTENZIALE}...")
calc = Potential(param_filename=PATH_POTENZIALE)
calc.name_ = "GAP" 

print(f"Lettura della configurazione iniziale da: {PATH_CONFIG_INIZIALE}...")
atomi = read(PATH_CONFIG_INIZIALE, index=0)

frame_totali = len(read(PATH_CONFIG_INIZIALE, index=":"))
passi_totali = frame_totali  
print(f"Struttura caricata ({len(atomi)} atomi). Numero passi impostato: {passi_totali}")

atomi.calc = calc

# =========================================================================
# 3. GESTIONE VELOCITÀ INIZIALI
# =========================================================================
try:
    velocita_iniziali = atomi.get_velocities()
    if velocita_iniziali is not None and np.any(velocita_iniziali):
        print("Velocità iniziali rilevate nel file extxyz e caricate con successo!")
    else:
        raise ValueError
except (ValueError, RuntimeError):
    print("ATTENZIONE: Velocità non trovate nel file extxyz.")
    print("Inizializzazione tramite distribuzione di Maxwell-Boltzmann a 3000 K...")
    MaxwellBoltzmannDistribution(atomi, temperature_K=3000)

# =========================================================================
# 4. IMPOSTAZIONE DELLA DINAMICA E RACCOLTA DATI
# =========================================================================
dyn = VelocityVerlet(
    atomi, 
    timestep=TIMESTEP, 
    trajectory=PATH_TRAJ_OUTPUT,
    logfile=PATH_LOG_OUTPUT,
    loginterval=1 # Il log standard di ASE scriverà nel file .log ogni 1 step
)

# Liste per salvare TUTTI i dati ad ogni singolo step
passi_salvati = []
tempi_fs = []
energie_potenziali = []
energie_cinetiche = []
energie_totali = []
temperature = []

def monitoraggio_sistema():
    passo_attuale = dyn.get_number_of_steps()
    tempo_attuale_fs = passo_attuale * (TIMESTEP / units.fs)
    
    epot = atomi.get_potential_energy()
    ekin = atomi.get_kinetic_energy()
    etot = epot + ekin
    temp = atomi.get_temperature() # Metodo nativo di ASE
    
    # 1. Salviamo i dati per il grafico (verrà eseguito ad ogni singolo step)
    passi_salvati.append(passo_attuale)
    tempi_fs.append(tempo_attuale_fs)
    energie_potenziali.append(epot)
    energie_cinetiche.append(ekin)
    energie_totali.append(etot)
    temperature.append(temp)
    
    # 2. Print a schermo SOLO ogni 10 step per non intasare il terminale
    if passo_attuale % 10 == 0:
        print(f"Step: {passo_attuale:>4d} | Tempo: {tempo_attuale_fs:>6.1f} fs | Temp: {temp:>6.1f} K | Epot: {epot:>10.3f} eV | Etot: {etot:>10.3f} eV")

# Attacchiamo la funzione con interval=1 (cattura tutti i dati!)
dyn.attach(monitoraggio_sistema, interval=1)

# =========================================================================
# 5. ESECUZIONE SIMULAZIONE
# =========================================================================
print("\n--- INIZIO DINAMICA MOLECOLARE NVE ---")
monitoraggio_sistema() # Catturiamo anche lo stato iniziale esatto
dyn.run(passi_totali)

print("--- DINAMICA COMPLETATA ---")
print(f"Traiettoria binaria salvata in: {PATH_TRAJ_OUTPUT}")
print(f"Log standard di ASE salvato in: {PATH_LOG_OUTPUT}\n")

# =========================================================================
# 6. GENERAZIONE DEL GRAFICO (SCATTER PLOT)
# =========================================================================
print("Generazione dei grafici di controllo...")

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

# PANNELLO 1: Temperatura (Scatter plot)
ax1.scatter(tempi_fs, temperature, color='orangered', s=10, alpha=0.7, label='Temperatura')
ax1.set_ylabel('Temperatura (K)', fontsize=11)
ax1.title.set_text('Dinamica NVE: Verifica della Conservazione - traindata')
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='upper right')

# PANNELLO 2: Energie (Scatter plot)
ax2.scatter(tempi_fs, energie_potenziali, color='royalblue', s=10, alpha=0.6, label='E. Potenziale (V)')
ax2.scatter(tempi_fs, energie_cinetiche, color='forestgreen', s=10, alpha=0.6, label='E. Cinetica (K)')
ax2.scatter(tempi_fs, energie_totali, color='black', s=15, alpha=0.8, label='E. Totale (E = V + K)')

ax2.set_xlabel('Tempo (fs)', fontsize=11)
ax2.set_ylabel('Energia (eV)', fontsize=11)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='center right')

# PANNELLO 3: Solo Energia Potenziale (Focus Dettagliato)
ax3.scatter(tempi_fs, energie_potenziali, color='royalblue', s=10, alpha=0.8, label='Solo E. Potenziale')
ax3.set_xlabel('Tempo (fs)', fontsize=11)
ax3.set_ylabel('Energia Pot. (eV)', fontsize=11)
ax3.grid(True, linestyle=':', alpha=0.6)
ax3.legend(loc='upper right')


plt.tight_layout()
plt.savefig(PATH_GRAFICO_OUTPUT, dpi=300)
plt.show()

print(f"Grafico salvato con successo in: {PATH_GRAFICO_OUTPUT}")
print("=" * 60)
