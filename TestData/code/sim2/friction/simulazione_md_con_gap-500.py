import numpy as np
from ase.io import read, iread
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
# Importiamo il calcolatore Quippy per leggere il potenziale GAP
from quippy.potential import Potential

def run_md_test():
    print("=" * 60)
    print(" AVVIO SIMULAZIONE DINAMICA MOLECOLARE CON POTENZIALE GAP")
    print("=" * 60)

    # 1. CARICAMENTO DEL POTENZIALE
    # ---------------------------------------------------------
    # Assicurati che il nome corrisponda esattamente al file .xml generato!
    file_potenziale = "../../results-test2/out_potenziale_gap-500-n2-l2.xml"
    print(f"Caricamento del potenziale da: {file_potenziale}...")
    calc = Potential('IP GAP', param_filename=file_potenziale)
    calc.name = "GAP"

    # 2. CARICAMENTO DELLA CONFIGURAZIONE INIZIALE
    # ---------------------------------------------------------
    # NOTA BENE: Qui usiamo il file ORIGINALE (quello con Ta e O).
    # ASE è intelligente: leggerà 'Ta' e 'O', capirà che sono Z=73 e Z=8, 
    # e passerà i numeri corretti al potenziale GAP in automatico!
    file_iniziale = "../../test_data.extxyz"
    print(f"Lettura del frame iniziale da: {file_iniziale}...")
    
    # Leggiamo solo il primo frame (index=0)
    atomi = read(file_iniziale, index=0)
    
    # Assegniamo il calcolatore GAP agli atomi
    #atomi.calc = calc
    atomi.set_calculator(calc)
    
    print(f"Sistema caricato: {len(atomi)} atomi nella cella.")
    
    print(f"Analisi del file {file_iniziale}...")
    # Contiamo i frame totali in modo leggero senza intasare la RAM
    totale_frame_originali = sum(1 for _ in iread(file_iniziale))
    print(f"-> Il file originale contiene esattamente {totale_frame_originali} frame.")

    # 3. IMPOSTAZIONE DELLA TEMPERATURA E VELOCITÀ
    # ---------------------------------------------------------
    #ricavata da QE
    temperatura_K = 3000.0
    print(f"Inizializzazione delle velocità per T = {temperatura_K} K...")
    MaxwellBoltzmannDistribution(atomi, temperature_K=temperatura_K)

    # 4. CONFIGURAZIONE DEL MOTORE DI DINAMICA MOLECOLARE
    # ---------------------------------------------------------
    # Usiamo Langevin per mantenere la temperatura costante (NVT)
    # Time step di 4.8fs, ricavato come differenza di due step consecutivi di QE
    timestep = 0.48 * units.fs
    
    dyn = Langevin(
        atomi, 
        timestep, 
        temperature_K=temperatura_K, 
        friction=1 / units.fs, # Coefficiente di attrito per il termostato
        trajectory='md_gap_test-500-fric.traj' # Il file dove salveremo i risultati
    )

    # Funzione per stampare lo stato durante la simulazione
    def print_status():
        epot = atomi.get_potential_energy()
        ekin = atomi.get_kinetic_energy()
        temp = ekin / (1.5 * units.kB * len(atomi))
        print(f"Step: {dyn.get_number_of_steps():>4} | Temp: {temp:>6.1f} K | Epot: {epot:>10.3f} eV | Ekin: {ekin:>8.3f} eV")

    # Colleghiamo la funzione di stampa per farla eseguire ogni 10 step
    dyn.attach(print_status, interval=10)

    # 5. ESECUZIONE DELLA SIMULAZIONE
    # ---------------------------------------------------------
    passi_totali = totale_frame_originali - 1 #Facciamo la dinamica per intero
    
    print("\n--- INIZIO DINAMICA MOLECOLARE ---")
    print_status() # Stampa lo stato al passo 0
    
    # Avviamo il motore
    dyn.run(passi_totali)
    
    print("--- SIMULAZIONE COMPLETATA ---")
    print("\nI risultati (posizioni, energie, forze) sono stati salvati in 'md_gap_test.traj'")
    print("Puoi visualizzarli da terminale scrivendo: ase gui md_gap_test-fric.traj")

if __name__ == "__main__":
    run_md_test()
