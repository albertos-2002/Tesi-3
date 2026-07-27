from functionFile import setConfig
from functionFile import classZBL

import numpy as np
import os

from ase.io import read, Trajectory
from ase.calculators.mixing import SumCalculator
from quippy.potential import Potential

class class_makePerturbation:

    def __init__(self): #----------------------------------------------------------------------------------------------------------------------------------------------------
        # Caricamento del potenziale GAP
        self.calc_gap = Potential(param_filename=setConfig.PATH_POTENZIALE)
        self.calc_gap.name_ = "GAP"

        # Generazione calcolatore ZBL (se attivo)
        if setConfig.IS_ZBL_ON:
            self.calc_zbl = classZBL.class_zbl()
            self.calc_totale = SumCalculator([self.calc_gap, self.calc_zbl])
            self.calc_zbl.plot_switch()
        else:
            self.calc_totale = self.calc_gap

        # Estrazione della configurazione di partenza
        self.atomi_md = read(setConfig.PATH_CONFIG_INIZIALE, index=0)
        self.atomi_md.calc = self.calc_totale
        
        # Salviamo le posizioni originali non perturbate
        self.posizioni_originali = self.atomi_md.get_positions().copy()

        if setConfig.DEBUG:
            print("Class makeSimulation created (Modalità Perturbazione Statica)")


    def runPerturbation(self): #---------------------------------------------------------------------------------------------------------------------------------------------
        
        # Fattori di conversione dalla distribuzione Chi (3 DOF) alla Gaussiana 1D
        if setConfig.SIGMA_LEVEL == 1:
            k = 1.879  # Il 68.27% dei moduli 3D cade entro 1.879 * sigma_1D
        elif setConfig.SIGMA_LEVEL == 2:
            k = 2.795  # Il 95.45% dei moduli 3D cade entro 2.795 * sigma_1D
        elif setConfig.SIGMA_LEVEL == 3:
            k = 3.921  # Il 99.73% dei moduli 3D cade entro 3.921 * sigma_1D
        else:
            raise ValueError("setConfig.SIGMA_LEVEL deve essere 1, 2 o 3.")

        # Calcolo del sigma della gaussiana 1D (per componenti x, y, z)
        sigma_comp = setConfig.TARGET_DISPLACEMENT / k

        # Creazione del file Trajectory di output
        path_traj = os.path.join(setConfig.PATH_OUT_FILE, "data_frames.traj")
        traj = Trajectory(path_traj, 'w')

        print("\n--- INIZIO GENERAZIONE FRAME PERTURBATI ---")
        print(f"Target displacement:  {setConfig.TARGET_DISPLACEMENT} Å")
        print(f"Livello statistico:   {setConfig.SIGMA_LEVEL} sigma")
        print(f"Sigma per componente: {sigma_comp:.5f} Å")

        # Ciclo di generazione
        for step in range(setConfig.PASSI_TOTALI + 1):
            if step == 0:
                # Frame 0: Configurazione originale
                self.atomi_md.set_positions(self.posizioni_originali)
                stato = "ORIGINALE"
            else:
                # Generazione rumore Gaussiano per tutti gli atomi (Nx3)
                rumore = np.random.normal(loc=0.0, scale=sigma_comp, size=self.posizioni_originali.shape)
                
                # Applicazione del rumore alla geometria originale
                self.atomi_md.set_positions(self.posizioni_originali + rumore)
                stato = "PERTURBATO"

            # Trigger del calcolo di Energia e Forze
            epot = self.atomi_md.get_potential_energy()
            forces = self.atomi_md.get_forces()

            # Scrittura sul file di traiettoria
            traj.write(self.atomi_md)

            # Monitoraggio
            self.monitoraggio_sistema(step, epot, forces, stato)

        traj.close()
        print("--- GENERAZIONE COMPLETATA ---\n")


    def runPerturbationPath(self): #------------------------------------------------------------------------------------------------------------------------

        # Fattori di conversione dalla distribuzione Chi (3 DOF) alla Gaussiana 1D
        if setConfig.SIGMA_LEVEL == 1:
            k = 1.879  
        elif setConfig.SIGMA_LEVEL == 2:
            k = 2.795  
        elif setConfig.SIGMA_LEVEL == 3:
            k = 3.921  
        else:
            raise ValueError("setConfig.SIGMA_LEVEL deve essere 1, 2 o 3.")

        # Calcolo del sigma della gaussiana 1D
        sigma_comp = setConfig.TARGET_DISPLACEMENT / k

        # Creazione del file Trajectory di output (nome diverso per non sovrascrivere)
        path_traj = os.path.join(setConfig.PATH_OUT_FILE, "data_frames.traj")
        traj = Trajectory(path_traj, 'w')

        print("\n--- INIZIO GENERAZIONE PATH (RANDOM WALK) ---")
        print(f"Displacement per step: {setConfig.TARGET_DISPLACEMENT} Å")
        print(f"Livello statistico:    {setConfig.SIGMA_LEVEL} sigma")
        print(f"Sigma per componente:  {sigma_comp:.5f} Å")

        for step in range(setConfig.PASSI_TOTALI + 1):
            if step == 0:
                # Frame 0: Configurazione iniziale di partenza
                self.atomi_md.set_positions(self.posizioni_originali)
                stato = "START"
            else:
                # Estraiamo le posizioni ATTUALI
                posizioni_attuali = self.atomi_md.get_positions()
                
                # Generiamo il rumore
                rumore = np.random.normal(loc=0.0, scale=sigma_comp, size=posizioni_attuali.shape)
                
                # Applichiamo il rumore sommandolo a quello che c'era nel frame precedente
                self.atomi_md.set_positions(posizioni_attuali + rumore)
                stato = f"STEP {step}"

            # Calcolo di Energia e Forze tramite i calcolatori ASE
            epot = self.atomi_md.get_potential_energy()
            forces = self.atomi_md.get_forces()

            # Salvataggio e monitoraggio
            traj.write(self.atomi_md)
            self.monitoraggio_sistema(step, epot, forces, stato)

        traj.close()
        print("--- GENERAZIONE PATH COMPLETATA ---\n")


    def monitoraggio_sistema(self, passo, epot, forces, stato): #-------------------------------------------------------------------------------------------------------------------
        """
        Adattato per la perturbazione statica: stampa l'energia e la forza massima.
        L'energia cinetica, totale e la temperatura non hanno senso qui poiché non c'è dinamica.
        """
        # Calcoliamo la forza massima (modulo vettoriale) presente nel sistema
        f_max = np.max(np.linalg.norm(forces, axis=1))
        
        print(f"Frame: {passo:>4d} [{stato:^10s}] | Epot: {epot:>12.3f} eV | Forza Max: {f_max:>8.3f} eV/Å")
