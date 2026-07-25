from functionFile import setConfig
from functionFile import classZBL

import numpy as np
import os

from ase import units
from ase.io import read
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.calculators.mixing import SumCalculator

from quippy.potential import Potential

class class_makeSimulation:

    #carichiamo direttamente il potenziale GAP
    #calc_gap = Potential(param_filename=setConfig.PATH_POTENZIALE)
    #calc_gap.name_ = "GAP"

    #facciamo un setting dei parametri generali che sevono per ogni motore di simulazione
    def __init__(self): #------------------------------------------------------------------------------------

        #Data la struttura del codice di lettura dei parametri
        #il potenziale GAP deve essere caricato dentro questa funzione
        self.calc_gap = Potential(param_filename=setConfig.PATH_POTENZIALE)
        self.calc_gap.name_ = "GAP"

        #Generazione calcolatore ZBL
        if setConfig.IS_ZBL_ON:
            self.calc_zbl = classZBL.class_zbl()
            self.calc_totale = SumCalculator([self.calc_gap, self.calc_zbl])
        else:
            self.calc_totale = self.calc_gap

        #estrazione della configurazione di partenza
        self.atomi_md = read(setConfig.PATH_CONFIG_INIZIALE, index=0)
        #assegna il calcolare determinato sopra
        self.atomi_md.calc = self.calc_totale

        #lettura e conversione dei parametri letti dal files
        if setConfig.TIMESTEP >= 0:
            self.timestep = setConfig.TIMESTEP * units.fs
        if setConfig.PASSI_TOTALI > 0:
            self.passi_totali = setConfig.PASSI_TOTALI
        if setConfig.TEMPERATURE >= 0:
            self.temperature = setConfig.TEMPERATURE

        
        if setConfig.DEBUG:
            print("Class makeSimulation created")

    def runSimulation(self): #-------------------------------------------------------------------------------
        self.checkVelocities()
        
        self.setSimulation()

        if setConfig.TERMINAL_LOG:
            self.dyn.attach(self.monitoraggio_sistema, interval=10)
        
        print("\n--- INIZIO DINAMICA MOLECOLARE ---")
        self.dyn.run(self.passi_totali)
        print("--- DINAMICA COMPLETATA ---\n")

    def checkVelocities(self): #-----------------------------------------------------------------------------
        velocities = self.atomi_md.get_velocities()
        # Se le velocità non esistono o sono tutte nulle, le inizializziamo
        if velocities is None or not np.any(velocities):
            print("ATTENZIONE: Velocità non trovate nel file extxyz.")
            print("Inizializzazione tramite distribuzione di Maxwell-Boltzmann ...")
            MaxwellBoltzmannDistribution(self.atomi_md, temperature_K=self.temperature)
        else:
            print("Velocità iniziali rilevate nel file extxyz e caricate con successo!")

    #prepara il motore della simulazione come un NVE
    def setSimulation(self): #-------------------------------------------------------------------------------
       self.dyn = VelocityVerlet(
            self.atomi_md, 
            timestep=self.timestep, 
            trajectory=os.path.join(setConfig.PATH_OUT_FILE, "md_gap+zbl_nve.traj"), 
            logfile=os.path.join(setConfig.PATH_OUT_FILE, "md_gap+zbl_nve.log"),
            loginterval=1
        )

    #funzione per fare il log dei risultati su terminale
    def monitoraggio_sistema(self): #------------------------------------------------------------------------
        passo_attuale = self.dyn.get_number_of_steps()
        tempo_attuale_fs = passo_attuale * (self.timestep / units.fs)
        epot = self.atomi_md.get_potential_energy()
        ekin = self.atomi_md.get_kinetic_energy()
        etot = epot + ekin
        temp = self.atomi_md.get_temperature()
        
        print(f"Step: {passo_attuale:>4d} | Tempo: {tempo_attuale_fs:>6.1f} fs | Temp: {temp:>6.1f} K | Epot: {epot:>10.3f} eV | Etot: {etot:>10.3f} eV")




