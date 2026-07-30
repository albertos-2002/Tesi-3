from functionFile import setConfig
from functionFile import classZBL

import numpy as np
import os
import sys

from ase import units
from ase.units import kB
from ase.io import read
from ase.io import Trajectory
from ase.calculators.mixing import SumCalculator
from ase.calculators.singlepoint import SinglePointCalculator

from quippy.potential import Potential

class class_makeMontecarlo:

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

        #Grafico polinomio di switch
        if setConfig.IS_ZBL_ON:
            self.calc_zbl.plot_switch()

        #estrazione della configurazione di partenza
        self.atomi_md = read(setConfig.PATH_CONFIG_INIZIALE, index=0)
        #assegna il calcolare determinato sopra
        self.atomi_md.calc = self.calc_totale
        
        
        #Il numero di step viene definito in Monte Carlo Sweep
        #i passi da svolgere saranno quindi setConfig.PASSI_TOTALI * numero atomi
        
        self.MCsteps = setConfig.PASSI_TOTALI * len(self.atomi_md) 
        #self.MCsteps: Numero totale di tentativi Monte Carlo
        
        if setConfig.DEBUG: print("Class makeSimulation created")


    #Esegue una simulazione Monte Carlo di Metropolis nell'insieme NVT.
    def runMetropolisMC(self):

        path_traj = os.path.join(setConfig.PATH_OUT_FILE, "data_frames.traj")
        traj = Trajectory(path_traj, 'w')

        # Costante di Boltzmann in eV/K (da ase.units.kB)
        kbT = kB * setConfig.TEMPERATURE

        # Inizializzazione dello stato 0
        #le posizioni originali sono già state lette e settate
        #self.atomi_md.set_positions(self.posizioni_originali)
        e_old = self.atomi_md.get_potential_energy()
        pos_old = self.atomi_md.get_positions().copy()

        # Salva lo stato iniziale
        traj.write(self.atomi_md)

        accettate = 0
        rifiutate = 0

        print(f"\n--- INIZIO METROPOLIS MONTE CARLO (NVT) ---")
        print(f"Temperatura:  {setConfig.TEMPERATURE:.1f} K  (k_B*T = {kbT:.4f} eV)")
        print(f"Step size:    {setConfig.MCSTEP_SIZE} Å")
        print(f"Modalità:     {setConfig.MOVE_TYPE.upper()}")
        print(f"Total Steps:  {self.MCsteps}\n")

        for step in range(1, self.MCsteps + 1):
            pos_proposte = pos_old.copy()

            if setConfig.MOVE_TYPE == 'single':
                # Scegliamo un singolo atomo casuale
                idx = np.random.randint(0, len(self.atomi_md))
                # Spostamento gaussiano o uniforme centrato in 0
                delta = np.random.uniform(-setConfig.MCSTEP_SIZE, setConfig.MCSTEP_SIZE, size=3)
                pos_proposte[idx] += delta
            elif setConfig.MOVE_TYPE == 'all':
                # Spostamento casuale per tutti gli atomi
                delta = np.random.uniform(-setConfig.MCSTEP_SIZE, setConfig.MCSTEP_SIZE, size=pos_old.shape)
                pos_proposte += delta
            else:
                print("Errore nel tipo di movimento")
                sys.exit()

            # Applichiamo la configurazione proposta
            self.atomi_md.set_positions(pos_proposte)
            e_new = self.atomi_md.get_potential_energy()

            # Calcolo della differenza di energia
            delta_E = e_new - e_old

            # CRITERIO DI METROPOLIS -------------------------------------------------
            if delta_E <= 0:
                accettata = True
            else:
                p_acc = np.exp(-delta_E / kbT)
                accettata = (np.random.rand() < p_acc) #boolean

            # AGGIORNAMENTO DELLO STATO ----------------------------------------------
            if accettata:
                accettate += 1
                e_old = e_new
                pos_old = pos_proposte.copy()
                f_old = self.atomi_md.get_forces()  # Salviamo anche le forze correnti
                stato = "ACCETTATA"
            else:
                rifiutate += 1
                # Ripristiniamo la geometria precedente
                self.atomi_md.set_positions(pos_old)
                stato = "RIFIUTATA"
                
                dump = self.atomi_md.get_potential_energy()
                dump = self.atomi_md.get_forces()
                dump = None

            # SCRITTURA E LOGGING ----------------------------------------------------
            if step % setConfig.MCLOG_INTERVAL == 0:
                # Scriviamo SEMPRE lo stato corrente (accettato o ripristinato)
                traj.write(self.atomi_md)
                
                f_max = np.max(np.linalg.norm(self.atomi_md.get_forces(), axis=1))
                rate = (accettate / step) * 100
                
                print(f"MC Step: {step:>6d} | [{stato:^9s}] | Epot: {e_old:>12.3f} eV | "
                      f"ΔE: {delta_E:>8.3f} eV | F_max: {f_max:>7.2f} eV/Å | Acc.Rate: {rate:>5.1f}%")

        traj.close()
        print(f"\n--- SIMULAZIONE COMPLETATA ---")
        print(f"Accettazione Finale: {accettate / self.MCsteps * 100:.2f}% ({accettate}/{self.MCsteps})")

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
