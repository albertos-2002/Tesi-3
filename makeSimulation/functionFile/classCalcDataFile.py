from functionFile import setConfig
from functionFile import classSimulation
from functionFile import classZBL

import os
import csv
import numpy as np
import pandas as pd

from ase.io import iread
from ase import units

class class_calcDataFile:

    def __init__(self): #------------------------------------------------------------------------------------
        
        #Usiamo dei dizionari per la gestione dei file
        self.file_aperti = {}
        self.writers = {}

        if setConfig.IS_ZBL_ON:
           self.simClass = classSimulation.class_makeSimulation()

        if setConfig.GENERA_FORZA_MODULO_DISTANZA or setConfig.GENERA_FORZA_MODULO_DISTANZA_3D or setConfig.GENERA_VELOCITA_DISTANZA or setConfig.GENERA_CONFRONTO_FORZE_MODULO_DISTANZE or setConfig.HISTO_DISTANZE:
            self.calcDistance = True        

        if setConfig.GENERA_FORZA_COMPONENTI_TEMPO or setConfig.GENERA_FORZA_MODULO_TEMPO or setConfig.GENERA_FORZA_MODULO_DISTANZA or setConfig.GENERA_FORZA_MODULO_DISTANZA_3D:
            self.calcForce = True

        if setConfig.GENERA_VELOCITA_TEMPO or setConfig.GENERA_VELOCITA_DISTANZA:
            self.calcVelocity = True

    def calcData(self): #--------------------------------------------------------------------------------------------------------------------------------------

        self.setFile()

        #estrazione di un singolo frame alla volta
        #la scrittura è affidata a singole funzioni
        for frame_idx, atomi in enumerate(iread(os.path.join(setConfig.PATH_OUT_FILE, "md_gap+zbl_nve.traj"))):

            time = frame_idx * setConfig.TIMESTEP

            #calcolo temperatura ed energia
            if setConfig.GENERA_ENERGIA_TEMPERATURA:
                self.writeTempEnerg(atomi, frame_idx, time)

            #calcolo delle forze totali
            if self.calcForce:
                self.forceCalculator(atomi, frame_idx)

            #estrazione delle forze dei calcolatori singoli se presenti
            if setConfig.IS_ZBL_ON:
                self.energCalculatorParts(atomi, frame_idx)
                self.forceCalculatorParts(atomi, frame_idx)

            #calcolo e scrittura delle distanze
            if self.calcDistance:
                self.distanceCalculator(atomi, frame_idx)
                
     
            #calcolo e scrittura delle velocità
            if self.calcVelocity:
                self.velocityCalculator(atomi, frame_idx)

        if setConfig.DEBUG: print("Concluso il ciclo for su tutti i frame")

        #chiusura di tutti i file apeti
        for f in self.file_aperti.values():
            f.close()

        if self.calcForce and self.calcDistance and setConfig.IS_ZBL_ON:
            self.projectionCalculator()
            
                
    def setFile(self): #-------------------------------------------------------------------------------------------------------------

        if setConfig.GENERA_ENERGIA_TEMPERATURA:
            f = open(os.path.join(setConfig.PATH_OUT_FILE, "temperatura-energia.csv"), "w", newline="")
            w = csv.writer(f)
            w.writerow(["Frame", "Tempo_fs", "Temperatura_K", "E_pot_eV", "E_kin_eV", "E_tot_eV"]) # Header
            self.file_aperti['temp-energ'] = f
            self.writers['temp-energ'] = w

            if setConfig.DEBUG: print("File energia-temperatura created and collected")

        if self.calcDistance:
            f = open(os.path.join(setConfig.PATH_OUT_FILE, "distanze.csv"), "w", newline="")
            w = csv.writer(f)
            w.writerow(["Frame_ID", "Atomo_ID", "Specie", "Dist_Min_A", "Dist_Media_A", "Dist_Max_A", "ID_primo_vicino", "ux_rad", "uy_rad", "uz_rad"])
            self.file_aperti['dist'] = f
            self.writers['dist'] = w

            if setConfig.DEBUG: print("File distanze created and collected")

        if self.calcForce:
            f = open(os.path.join(setConfig.PATH_OUT_FILE, "forze_tot.csv"), "w", newline="")
            w = csv.writer(f)
            w.writerow(["Frame_ID", "Atomo_ID", "Specie", "Modulo_Forza_eV_A", "Forza_X_eV_A", "Forza_Y_eV_A", "Forza_Z_eV_A"])
            self.file_aperti['force'] = f
            self.writers['force'] = w

            if setConfig.DEBUG: print("File forze created and collected")
            

        if setConfig.IS_ZBL_ON:
            f = open(os.path.join(setConfig.PATH_OUT_FILE, "energ_gap+zbl.csv"), "w", newline="")
            w = csv.writer(f)
            w.writerow(["Frame_ID", "epot_GAP_eV_A", "epot_ZBL_eV_A"])
            self.file_aperti['energ-part'] = f
            self.writers['energ-part'] = w

            if setConfig.DEBUG: print("File energ-part created and collected")

        if setConfig.IS_ZBL_ON:
            f = open(os.path.join(setConfig.PATH_OUT_FILE, "forze_gap+zbl.csv"), "w", newline="")
            w = csv.writer(f)
            w.writerow(["Frame_ID", "Atomo_ID", "Specie", "Modulo_GAP_eV_A", "Modulo_ZBL_eV_A", "F_gap_x", "F_gap_y", "F_gap_z", "F_zbl_x", "F_zbl_y", "F_zbl_z"])
            self.file_aperti['force-part'] = f
            self.writers['force-part'] = w

            if setConfig.DEBUG: print("File forze-part created and collected")


        if self.calcVelocity:
            f = open(os.path.join(setConfig.PATH_OUT_FILE, "velocita.csv"), "w", newline="")
            w = csv.writer(f)
            # Nota: le unità di misura interne di ASE per la velocità sono Angstrom / (unità di tempo interna)
            w.writerow(["Frame_ID", "Atomo_ID", "Specie", "Modulo_Velocita", "Velocita_X", "Velocita_Y", "Velocita_Z"])
            self.file_aperti['velocita'] = f
            self.writers['velocita'] = w
        
            if setConfig.DEBUG: print("File velocita created and collected")
                        

    def writeTempEnerg(self, atomObj, index, time): #---------------------------------------------------------------------------------------
        epot = atomObj.get_potential_energy()
        ekin = atomObj.get_kinetic_energy()
        etot = epot + ekin
        temp = atomObj.get_temperature()

        self.writers['temp-energ'].writerow([index,
                                              f"{time:.3f}", 
                                              f"{temp:.2f}", 
                                              f"{epot:.6f}", 
                                              f"{ekin:.6f}", 
                                              f"{etot:.6f}"
                                              ])

        if setConfig.DEBUG: print("Row - temperature and energy written")
            

    def distanceCalculator(self, atomObj, frame_idx): #-------------------------------------------------------------------------------------------------------------------------------------------            
        
        n_atomi = len(atomObj)
        simboli = atomObj.get_chemical_symbols()  # Restituisce una lista tipo ['Ta', 'Ta', ..., 'O', 'O']
        
        # 1. Matrice delle distanze completa (N x N)
        dist_matrix = atomObj.get_all_distances(mic=True)
        
        # 2. Calcolo della Distanza Media e Massima dagli altri atomi
        # (La somma della riga divisa per N-1 dà la media escludendo la diagonale che è 0)
        mean_dists = np.sum(dist_matrix, axis=1) / (n_atomi - 1)
        max_dists = np.max(dist_matrix, axis=1)
        
        # 3. Primo Vicino (minima distanza e relativo indice dell'atomo vicino)
        np.fill_diagonal(dist_matrix, np.inf)
        
        min_dists = np.min(dist_matrix, axis=1)
        nearest_idx = np.argmin(dist_matrix, axis=1)
        
        # 4. Scrittura sul file CSV di tutti i 126 atomi per il frame corrente
        for atom_id in range(n_atomi):

            #--- Calcolo del versore radiale ---
            vicino_id = nearest_idx[atom_id]
            r_min = min_dists[atom_id]

            #Estraiamo il vettore 3D specifico verso il primo vicino (vector=True)
            vec_ij = atomObj.get_distance(atom_id, vicino_id, mic=True, vector=True)

            #Normalizziamo per ottenere il versore (x, y, z)
            if r_min > 1e-12:
                unit_vec = vec_ij / r_min
            else:
                unit_vec = np.array([0.0, 0.0, 0.0])
                            
            ux_rad, uy_rad, uz_rad = unit_vec[0], unit_vec[1], unit_vec[2]

        
            self.writers['dist'].writerow([
                frame_idx,                          # Indice del frame
                atom_id,                            # ID dell'atomo (0 .. 125)
                simboli[atom_id],                   # Specie chimica ("Ta" o "O")
                f"{min_dists[atom_id]:.4f}",        # Distanza primo vicino
                f"{mean_dists[atom_id]:.4f}",       # Distanza media da tutti gli altri
                f"{max_dists[atom_id]:.4f}",        # Distanza massima da tutti gli altri
                nearest_idx[atom_id],               # ID dell'atomo primo vicino
                f"{ux_rad:.6f}",                    # ux_rad
                f"{uy_rad:.6f}",                    # uy_rad
                f"{uz_rad:.6f}"                     # uz_rad
            ])

        if setConfig.DEBUG: print("Row - distances written")


    def forceCalculator(self, atomObj, frame_idx): #----------------------------------------------------------------------------------------------------------------------------------------------

        forze = atomObj.get_forces()             # Matrice (N, 3) con le componenti Fx, Fy, Fz
        simboli = atomObj.get_chemical_symbols() # Lista dei simboli chimici ['Ta', 'O', ...]
    
        for atom_id in range(len(atomObj)):
            fx, fy, fz = forze[atom_id]
            modulo_f = np.linalg.norm(forze[atom_id])
            #modulo_f = np.sqrt(fx**2 + fy**2 + fz**2) # oppure np.linalg.norm(forze[atom_id])
        
            self.writers['force'].writerow([
                frame_idx,                      # Indice del frame
                atom_id,                        # ID dell'atomo (0 .. N-1)
                simboli[atom_id],               # Specie chimica ("Ta" o "O")
                f"{modulo_f:.6f}",              # Modulo della forza total
                f"{fx:.6f}",                    # Componente X
                f"{fy:.6f}",                    # Componente Y
                f"{fz:.6f}"                     # Componente Z
            ])

            if setConfig.DEBUG: print("Row - forza totale stampata")

        if setConfig.DEBUG: print("File forze totali stampato")


    def energCalculatorParts(self, atomObj, frame_idx): #------------------------------------------------------------------------------------------------------------------------------------------

        #calcoliamo e salviamo la sola energia GAP
        atomObj.calc = self.simClass.calc_gap
        e_pot_gap = atomObj.get_potential_energy()

        #calcoliamo e salviamo la sola energia ZBL
        atomObj.calc = self.simClass.calc_zbl
        e_pot_zbl = atomObj.get_potential_energy()

        self.writers['energ-part'].writerow([
            frame_idx,
            f"{e_pot_gap:.6f}",
            f"{e_pot_zbl:.6f}"
        ])

        if setConfig.DEBUG: print("Energie stampate correttamente")

    def forceCalculatorParts(self, atomObj, frame_idx): #-------------------------------------------------------------------------------------------------------------------------------------

        #calcoliamo e salviamo le forze GAP
        atomObj.calc = self.simClass.calc_gap   #Forze con GAP (matrice N x 3)
        forze_gap = atomObj.get_forces()
        mod_gap = np.linalg.norm(forze_gap, axis=1)  # Vettore di lunghezza N

        #calcoliamo e salviamo le forze ZBL
        atomObj.calc = self.simClass.calc_zbl   #Forze con ZBL (matrice N x 3)
        forze_zbl = atomObj.get_forces()
        mod_zbl = np.linalg.norm(forze_zbl, axis=1)  # Vettore di lunghezza N

        simboli = atomObj.get_chemical_symbols()

        for atom_id in range(len(atomObj)):

            #Estraiamo le componenti (x, y, z) dall'array dell'atomo
            fg_x, fg_y, fg_z = forze_gap[atom_id][0], forze_gap[atom_id][1], forze_gap[atom_id][2]
            fz_x, fz_y, fz_z = forze_zbl[atom_id][0], forze_zbl[atom_id][1], forze_zbl[atom_id][2]
        
            self.writers['force-part'].writerow([
                frame_idx,
                atom_id,
                simboli[atom_id],
                f"{mod_gap[atom_id]:.6f}",
                f"{mod_zbl[atom_id]:.6f}",   
                f"{fg_x:.6f}",
                f"{fg_y:.6f}",
                f"{fg_z:.6f}",   
                f"{fz_x:.6f}",
                f"{fz_y:.6f}",
                f"{fz_z:.6f}"                                                
            ])

            if setConfig.DEBUG: print("Row forze part stampata")

        if setConfig.DEBUG: print("File forze parts stampato")

    def velocityCalculator(self, atomObj, frame_idx): #-----------------------------------------------------------------------------------------------------------------------------------

        #Matrice (N, 3) con le componenti vx, vy, vz
        #Uità di misura in angstrom / fs
        velocita = atomObj.get_velocities() / units.fs      
        
        # Controllo di sicurezza: se per qualche motivo il frame non ha velocità
        if velocita is None:
            if setConfig.DEBUG: print(f"Nessuna velocità trovata nel frame {frame_idx}")
            return

        simboli = atomObj.get_chemical_symbols() # Lista dei simboli chimici

        for atom_id in range(len(atomObj)):
            vx, vy, vz = velocita[atom_id]
            
            # Calcolo del modulo usando numpy per massima efficienza
            modulo_v = np.linalg.norm(velocita[atom_id])
            
            self.writers['velocita'].writerow([
                frame_idx,                      # Indice del frame
                atom_id,                        # ID dell'atomo (0 .. N-1)
                simboli[atom_id],               # Specie chimica ("Ta" o "O")
                f"{modulo_v:.6f}",              # Modulo della velocità
                f"{vx:.6f}",                    # Componente X
                f"{vy:.6f}",                    # Componente Y
                f"{vz:.6f}"                     # Componente Z
            ])

        if setConfig.DEBUG: print("File velocità totali stampato")


    def projectionCalculator(self): #------------------------------------------------------------------------------------------------------------------

        colonne_da_caricare = ["F_gap_x", "F_gap_y", "F_gap_z", "F_zbl_x", "F_zbl_y", "F_zbl_z"]
        file_path = os.path.join(setConfig.PATH_OUT_FILE, "forze_gap+zbl.csv")
        df_f = pd.read_csv(file_path, usecols=colonne_da_caricare)
        if setConfig.DEBUG: print(f"Righe lette da forze_gap+zbl.csv: {len(df_f)}")

        colonne_da_caricare = ["Specie", "ux_rad", "uy_rad", "uz_rad", "Frame_ID", "Atomo_ID"]
        file_path = os.path.join(setConfig.PATH_OUT_FILE, "distanze.csv")
        df_d = pd.read_csv(file_path, usecols=colonne_da_caricare)
        if setConfig.DEBUG: print(f"Righe lette da distanze.csv: {len(df_d)}")

        # Calcolo della Proiezione Radiale ZBL (Prodotto Scalare puro F_zbl · u)
        # Risultato: < 0 per Repulsiva | > 0 per Attrattiva
        frad_zbl = ( df_f["F_zbl_x"] * df_d["ux_rad"]
                   + df_f["F_zbl_y"] * df_d["uy_rad"]
                   + df_f["F_zbl_z"] * df_d["uz_rad"]
                   )
        # Calcolo della Proiezione Radiale GAP (Prodotto Scalare puro F_gap · u)
        frad_gap = ( df_f["F_gap_x"] * df_d["ux_rad"]
                   + df_f["F_gap_y"] * df_d["uy_rad"]
                   + df_f["F_gap_z"] * df_d["uz_rad"]
                   )
        # Calcolo della Forza Radiale TOTALE (GAP + ZBL)
        frad_tot = frad_zbl + frad_gap

        # Creazione del NUOVO DataFrame
        df_risultati = pd.DataFrame({
            "Frame_ID": df_d["Frame_ID"].astype(int),
            "Atomo_ID": df_d["Atomo_ID"].astype(int),
            "Specie": df_d["Specie"],
            "GAP_Prj": frad_gap,
            "ZBL_Prj": frad_zbl,
            "TOT_Prj": frad_tot
            })

        # Salvataggio del nuovo DataFrame su un nuovo file CSV (senza l'indice di Pandas)
        pathf = os.path.join(setConfig.PATH_OUT_FILE,"forze_proiettate_radiali.csv")
        df_risultati.to_csv(pathf, index=False)
        print("Nuovo DataFrame creato e salvato con successo!")

        if setConfig.DEBUG: print(f"Righe finali unite e salvate: {len(df_risultati)}")

#----------------------------------------------------------------------------------------------------------------------------------------
