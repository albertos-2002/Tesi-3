from functionFile import setConfig

import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import gc
import pandas as pd

from ase.io import iread

class class_makeGraph:

    def __init__(self): #--------------------------------------------------------------------------------------------------------------------

        if setConfig.DEBUG: print("Classe graph creata correttamente")

    def makeGraph(self): #--------------------------------------------------------------------------------------------------------------------
        
        if setConfig.GENERA_ENERGIA_TEMPERATURA:
            self.makeEnergiaTemperatura()

        if setConfig.GENERA_IMMAGINI_SPAZIALI:
            self.makeSpaceConfiguration()

        if setConfig.GENERA_FORZA_COMPONENTI_TEMPO or setConfig.GENERA_FORZA_MODULO_TEMPO:
            self.makeForzeTotali()

        if setConfig.GENERA_FORZA_MODULO_DISTANZA or setConfig.GENERA_FORZA_MODULO_DISTANZA_3D:
            self.makeForceDistance()

        if setConfig.HISTO_DISTANZE:
            self.makeHistoDistanze()

        if setConfig.GENERA_VELOCITA_TEMPO or setConfig.GENERA_VELOCITA_DISTANZA:
            self.makeVelocity()

        if setConfig.GENERA_CONFRONTO_FORZE_MODULO_TEMPO or setConfig.GENERA_CONFRONTO_FORZE_MODULO_DISTANZE: 
            if setConfig.IS_ZBL_ON:
                self.makeConfrontoForze()
                self.makeForceProjection()

        

    def makeEnergiaTemperatura(self): #--------------------------------------------------------------------------------------------------------------------

        if setConfig.DEBUG:print(f"makeEnergiaTemperatura")
        #carichiamo solo alcune colonne di dati
        colonne_da_caricare = ["Tempo_fs","Temperatura_K","E_pot_eV","E_kin_eV","E_tot_eV"]
        file_path = os.path.join(setConfig.PATH_OUT_FILE, "temperatura-energia.csv")
        df = pd.read_csv(file_path, usecols=colonne_da_caricare)
        if setConfig.DEBUG:print(f"Righe lette da temperatura-energia.csv: {len(df)}")
        path_fig = os.path.join(setConfig.PATH_OUT_GRAPH, "andamento_energia.png")

        fig_en, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        ax1.scatter(df["Tempo_fs"], df["Temperatura_K"], color='orangered', s=5, alpha=0.7)
        ax1.set_title('Temperatura')
        ax1.set_xlabel('Tempo (fs)')
        ax1.set_ylabel('Temperatura (K)')
        ax1.grid(True, linestyle=':', alpha=0.6)

        ax2.scatter(df["Tempo_fs"], df["E_pot_eV"], color='royalblue', s=5, label='E. Potenziale', alpha=0.7)
        ax2.scatter(df["Tempo_fs"], df["E_kin_eV"], color='forestgreen', s=5, label='E. Cinetica', alpha=0.7)
        ax2.scatter(df["Tempo_fs"], df["E_tot_eV"], color='black', s=5, label='E. Totale', alpha=0.7)
        ax2.set_title('Confronto Energie')
        ax2.set_xlabel('Tempo (fs)')
        ax2.set_ylabel('Energia (eV)')
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.legend(markerscale=5)

        ax3.scatter(df["Tempo_fs"], df["E_pot_eV"], color='royalblue', s=5, alpha=0.7)
        ax3.set_title('Dettaglio E. Potenziale')
        ax3.set_xlabel('Tempo (fs)')
        ax3.grid(True, linestyle=':', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig(path_fig, dpi=300)
        if not setConfig.KEEP_OPEN:
            plt.close(fig_en)
        else:
            plt.show(block=False)
            plt.pause(0.1)
            input("Press any key to continue.........")
            plt.close(fig_en)

        if setConfig.DEBUG: print("Grafico temperatura-energia stampato")

    def makeSpaceConfiguration(self): #------------------------------------------------------------------------------------------------------------------

        if setConfig.DEBUG:print(f"makeSpaceConfiguration")
        #Gestione dei path di salvataggio
        path2d = os.path.join(setConfig.PATH_OUT_GRAPH , setConfig.DIR_SALVATAGGIO_2D)
        path3d = os.path.join(setConfig.PATH_OUT_GRAPH , setConfig.DIR_SALVATAGGIO_3D)
        os.makedirs(path2d, exist_ok=True)
        os.makedirs(path3d, exist_ok=True)

        for frame_idx, atomi in enumerate(iread(os.path.join(setConfig.PATH_OUT_FILE, "data_frames.traj"))):

            if frame_idx % setConfig.FREQUENZA_IMMAGINI == 0:

                posizioni = atomi.get_positions(wrap=True)
                simboli = atomi.get_chemical_symbols()
                simboli_arr = np.array(simboli)
                                
                pos_Ta = posizioni[simboli_arr == 'Ta']
                pos_O = posizioni[simboli_arr == 'O']
                nome_file = f"{frame_idx:06d}_configurazione_spaziale.png"
                                
                fig2d, ax2d = plt.subplots(figsize=(6, 6))
                ax2d.scatter(pos_Ta[:, 0], pos_Ta[:, 1], color='blue', s=80, label='Ta', edgecolors='black')
                ax2d.scatter(pos_O[:, 0], pos_O[:, 1], color='red', s=40, label='O', edgecolors='black')
                ax2d.set_title(f"2D - Step {frame_idx}")
                ax2d.axis('equal')
                ax2d.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
                fig2d.savefig(os.path.join(path2d, nome_file), dpi=100)
                plt.close(fig2d)
                                
                fig3d = plt.figure(figsize=(6, 6))
                ax3d = fig3d.add_subplot(111, projection='3d')
                ax3d.scatter(pos_Ta[:, 0], pos_Ta[:, 1], pos_Ta[:, 2], color='blue', s=80, edgecolors='black', label="Ta")
                ax3d.scatter(pos_O[:, 0], pos_O[:, 1], pos_O[:, 2], color='red', s=40, edgecolors='black', label='O')
                ax3d.set_title(f"3D - Step {frame_idx}")
                ax3d.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
                fig3d.savefig(os.path.join(path3d, nome_file), dpi=100)
                plt.close(fig3d)
                                                
                plt.close('all')
                gc.collect()
                print(f"Step {frame_idx:06d}/{setConfig.PASSI_TOTALI} completato. Dati e Immagini salvati.")        
        

    def makeForzeTotali(self): #----------------------------------------------------------------------------------------------------------------------

        if setConfig.DEBUG:print(f"makeForzeTotali")
        #carichiamo solo alcune colonne di dati
        colonne_da_caricare = ["Frame_ID", "Modulo_Forza_eV_A","Forza_X_eV_A","Forza_Y_eV_A","Forza_Z_eV_A"]
        file_path = os.path.join(setConfig.PATH_OUT_FILE, "forze_tot.csv")
        df = pd.read_csv(file_path, usecols=colonne_da_caricare)
        if setConfig.DEBUG:print(f"Righe lette da forze_tot.csv: {len(df)}")
        path_fig = os.path.join(setConfig.PATH_OUT_GRAPH, "andamento_forze_totali.png")
        path_figm = os.path.join(setConfig.PATH_OUT_GRAPH, "andamento_forze_totali_blocchi.png")

        #prima costruiamo un grafico in cui sono printate tutte le forze
        fig1, ax1  = plt.subplots(figsize=(10, 5))    

        ax1.scatter(df["Frame_ID"], df["Modulo_Forza_eV_A"], color='purple', s=2, alpha=0.5, label="Modulo Forza")
        ax1.scatter(df["Frame_ID"], df["Forza_X_eV_A"], color='red', s=1, alpha=0.2, label="Forza X")
        ax1.scatter(df["Frame_ID"], df["Forza_Y_eV_A"], color='green', s=1, alpha=0.2, label="Forza Y")
        ax1.scatter(df["Frame_ID"], df["Forza_Z_eV_A"], color='blue', s=1, alpha=0.2, label="Forza Z")
        
        ax1.set_title('Forze totali')
        ax1.set_xlabel('Frame ID')
        ax1.set_ylabel('Forza (eV/Å)')
        ax1.grid(True, linestyle=':', alpha=0.7)
        ax1.legend(markerscale=5)

        fig1.savefig(path_fig, dpi=300)
        if not setConfig.KEEP_OPEN:
            plt.close(fig1)
        else:
            plt.show(block=False)
            plt.pause(0.1)
            input("Press any key to continue.........")
            plt.close(fig1)


        fig1m, axsm = plt.subplots(2, 2, figsize=(15, 10))
        ax1m = axsm[0, 0]
        ax2m = axsm[0, 1]
        ax3m = axsm[1, 0]
        ax4m = axsm[1, 1]

        ax1m.scatter(df["Frame_ID"], df["Modulo_Forza_eV_A"], color='purple', s=2, alpha=0.5, label="Modulo Forza")
        ax1m.set_title('Modulo forza')
        ax1m.set_xlabel('Frame ID')
        ax1m.set_ylabel('Forza (eV/Å)')        
        ax1m.grid(True, linestyle=':', alpha=0.7)
        
        ax2m.scatter(df["Frame_ID"], df["Forza_X_eV_A"], color='red', s=1, alpha=0.2, label="Forza X")
        ax2m.set_title('Forza X')
        ax2m.set_xlabel('Frame ID')
        ax2m.set_ylabel('Forza (eV/Å)')        
        ax2m.grid(True, linestyle=':', alpha=0.7)
        
        ax3m.scatter(df["Frame_ID"], df["Forza_Y_eV_A"], color='green', s=1, alpha=0.2, label="Forza Y")
        ax3m.set_title('Forza Y')
        ax3m.set_xlabel('Frame ID')
        ax3m.set_ylabel('Forza (eV/Å)')        
        ax3m.grid(True, linestyle=':', alpha=0.7)
        
        ax4m.scatter(df["Frame_ID"], df["Forza_Z_eV_A"], color='blue', s=1, alpha=0.2, label="Forza Z") 
        ax4m.set_title('Forza Z')
        ax4m.set_xlabel('Frame ID')
        ax4m.set_ylabel('Forza (eV/Å)')               
        ax4m.grid(True, linestyle=':', alpha=0.7)

        plt.tight_layout()
        fig1m.savefig(path_figm, dpi=300)
        if not setConfig.KEEP_OPEN:
            plt.close(fig1m)
        else:
            plt.show(block=False) 
            plt.pause(0.1)
            input("Press any key to continue.........")
            plt.close(fig1m)

        gc.collect()

    def makeForceDistance(self): #------------------------------------------------------------------------------------------------------------------

        if setConfig.DEBUG:print(f"makeForceDistance")
        #Modulo della forza totale in funzione del valore della distanza minima
        colonne_da_caricare = ["Specie", "Modulo_Forza_eV_A", "Frame_ID"]
        file_path = os.path.join(setConfig.PATH_OUT_FILE, "forze_tot.csv")
        df_f = pd.read_csv(file_path, usecols=colonne_da_caricare)
        if setConfig.DEBUG:print(f"Righe lette da forze_tot.csv: {len(df_f)}")

        colonne_da_caricare = ["Specie", "Dist_Min_A", "Frame_ID"]
        file_path = os.path.join(setConfig.PATH_OUT_FILE, "distanze.csv")
        df_d = pd.read_csv(file_path, usecols=colonne_da_caricare)
        if setConfig.DEBUG:print(f"Righe lette da distanze.csv: {len(df_d)}")
        
        path_fig = os.path.join(setConfig.PATH_OUT_GRAPH, "andamento_forze_totali_distanza.png")
        path_fig3d = os.path.join(setConfig.PATH_OUT_GRAPH, "andamento_forze_totali_distanza3d.png")
        
        fig_fd, ax_fd = plt.subplots(figsize=(9, 6))

        ax_fd.scatter(df_d[df_d["Specie"]=="Ta"]["Dist_Min_A"], df_f[df_f["Specie"]=="Ta"]["Modulo_Forza_eV_A"], color='royalblue', alpha=0.3, s=5, label='Tantalo (Ta)', edgecolors='none')
        ax_fd.scatter(df_d[df_d["Specie"]=="O"]["Dist_Min_A"], df_f[df_f["Specie"]=="O"]["Modulo_Forza_eV_A"], color='crimson', alpha=0.3, s=5, label='Ossigeno (O)', edgecolors='none')

        if setConfig.IS_ZBL_ON:
            ax_fd.axvline(x=setConfig.RAGGIO_INNER, color='black', linestyle='--', alpha=0.7, label=f'r_inner ({setConfig.RAGGIO_INNER} Å)')
            ax_fd.axvline(x=setConfig.RAGGIO_OUTER, color='gray', linestyle=':', alpha=1, label=f'r_outer ({setConfig.RAGGIO_OUTER} Å)')

        if setConfig.IS_ZBL_ON:
            ax_fd.set_title("Modulo della Forza vs Distanza dal Vicino più Prossimo (GAP + ZBL)", fontsize=13, pad=12)
        else:
            ax_fd.set_title("Modulo della Forza vs Distanza dal Vicino più Prossimo (GAP)", fontsize=13, pad=12)
        
        ax_fd.set_xlabel("Distanza dal Vicino più Prossimo (Å)", fontsize=11)
        ax_fd.set_ylabel("Modulo della Forza Netta (eV/Å)", fontsize=11)
        ax_fd.grid(True, linestyle='--', alpha=0.6)
        ax_fd.legend(loc='upper right', fontsize=10, markerscale=5)
        plt.tight_layout()
        plt.savefig(path_fig, dpi=300)
        if not setConfig.KEEP_OPEN:
            plt.close(fig_fd)
        else:
            plt.show(block=False)
            plt.pause(0.1)
            input("Press any key to continue.........")
            plt.close(fig_fd)

        #Modulo della forza totale in funzione della distanza e del frame
        fig_3d = plt.figure(figsize=(10, 8))
        ax_3d = fig_3d.add_subplot(111, projection='3d')

        ax_3d.scatter(df_f[df_f["Specie"]=="Ta"]["Frame_ID"], 
                      df_d[df_d["Specie"]=="Ta"]["Dist_Min_A"], 
                      df_f[df_f["Specie"]=="Ta"]["Modulo_Forza_eV_A"], 
                      color='royalblue', alpha=0.3, s=10, label='Tantalo (Ta)')
        ax_3d.scatter(df_f[df_f["Specie"]=="O"]["Frame_ID"], 
                      df_d[df_d["Specie"]=="O"]["Dist_Min_A"], 
                      df_f[df_f["Specie"]=="O"]["Modulo_Forza_eV_A"], 
                      color='crimson', alpha=0.3, s=10, label='Ossigeno (O)')

        ax_3d.set_title("Scansione 3D: Evoluzione nel Tempo di Distanza e Forza", fontsize=13, pad=15)
        ax_3d.set_xlabel("Frame ID", fontsize=10, labelpad=10)
        ax_3d.set_ylabel("Distanza dal Vicino (Å)", fontsize=10, labelpad=10)
        ax_3d.set_zlabel("Modulo della Forza (eV/Å)", fontsize=10, labelpad=10)
        ax_3d.view_init(elev=25, azim=135)
        ax_3d.legend(loc='upper left', fontsize=10, markerscale=3)
        plt.tight_layout()
        plt.savefig(path_fig3d, dpi=300)
        if not setConfig.KEEP_OPEN:
            plt.close(fig_3d)
        else:
            plt.show(block=False)
            plt.pause(0.1)
            input("Press any key to continue.........")
            plt.close(fig_3d)

    def makeHistoDistanze(self): #-------------------------------------------------------------------------------------------------------------------

        if setConfig.DEBUG:print(f"makeHistoDistanze")
        colonne_da_caricare = ["Dist_Min_A"]
        file_path = os.path.join(setConfig.PATH_OUT_FILE, "distanze.csv")
        df = pd.read_csv(file_path, usecols=colonne_da_caricare)
        if setConfig.DEBUG:print(f"Righe lette da distanze.csv: {len(df)}")
        path_fig = os.path.join(setConfig.PATH_OUT_GRAPH, "histo_distanze.png")

        fig_hist, ax_hist = plt.subplots(figsize=(8, 5))

        ax_hist.hist(
            df["Dist_Min_A"], 
            bins=30, 
            color='royalblue', 
            alpha=0.5, 
            label='Distanza minima', 
            edgecolor='royalblue', 
            linewidth=0.3
        )

        ax_hist.set_xlabel("Distanza Minima ($\AA$)")
        ax_hist.set_ylabel("Conteggio")
        ax_hist.set_title("Distribuzione delle Distanze Minime dal Vicino più Prossimo")
        ax_hist.legend(markerscale=5)
        ax_hist.grid(True, linestyle='--', alpha=0.6)

        fig_hist.savefig(path_fig, dpi=300)
        if not setConfig.KEEP_OPEN:
            plt.close(fig_hist)
        else:
            plt.show(block=False)
            plt.pause(0.1)
            input("Press any key to continue.........")
            plt.close(fig_hist)
        
    def makeVelocity(self): #----------------------------------------------------------------------------------------------------------------------
    
        if setConfig.DEBUG:print(f"makeVelocity")
        #Modulo della velocità in funzione del valore della distanza minima
        colonne_da_caricare = ["Specie", "Modulo_Velocita", "Frame_ID"]
        file_path = os.path.join(setConfig.PATH_OUT_FILE, "velocita.csv")
        df_v = pd.read_csv(file_path, usecols=colonne_da_caricare)
        if setConfig.DEBUG:print(f"Righe lette da velocita.csv: {len(df_v)}")

        colonne_da_caricare = ["Specie", "Dist_Min_A"]
        file_path = os.path.join(setConfig.PATH_OUT_FILE, "distanze.csv")
        df_d = pd.read_csv(file_path, usecols=colonne_da_caricare)
        if setConfig.DEBUG:print(f"Righe lette da distanze.csv: {len(df_d)}")

        path_figd = os.path.join(setConfig.PATH_OUT_GRAPH, "velocita_vs_distanza_minima.png")
        path_figt = os.path.join(setConfig.PATH_OUT_GRAPH, "velocita_vs_tempo.png")


        fig_vt, ax_vt = plt.subplots(figsize=(9, 6))

        ax_vt.scatter(df_v[df_v["Specie"]=="Ta"]["Frame_ID"], 
                      df_v[df_v["Specie"]=="Ta"]["Modulo_Velocita"], 
                      color='royalblue', alpha=0.4, s=15, label='Tantalo (Ta)', edgecolors='none')
        ax_vt.scatter(df_v[df_v["Specie"]=="O"]["Frame_ID"], 
                      df_v[df_v["Specie"]=="O"]["Modulo_Velocita"], 
                      color='crimson', alpha=0.4, s=15, label='Ossigeno (O)', edgecolors='none')

        ax_vt.set_title("Modulo della Velocità vs Tempo", fontsize=13, pad=12)
        ax_vt.set_xlabel("Frame ID", fontsize=11)
        ax_vt.set_ylabel("Modulo della Velocità (Å/fs)", fontsize=11)
        
        ax_vt.grid(True, linestyle='--', alpha=0.6)
        ax_vt.legend(loc='upper right', fontsize=10, markerscale=3)
        plt.tight_layout()
        plt.savefig(path_figt, dpi=300)
        if not setConfig.KEEP_OPEN:
            plt.close(fig_vt)
        else:
            plt.show(block=False)
            plt.pause(0.1)
            input("Press any key to continue.........")
            plt.close(fig_vt)


        fig_vd, ax_vd = plt.subplots(figsize=(9, 6))
        ax_vd.scatter(df_d[df_d["Specie"]=="Ta"]["Dist_Min_A"], 
                      df_v[df_v["Specie"]=="Ta"]["Modulo_Velocita"], 
                      color='royalblue', alpha=0.4, s=15, label='Tantalo (Ta)', edgecolors='none')
        ax_vd.scatter(df_d[df_d["Specie"]=="O"]["Dist_Min_A"], 
                      df_v[df_v["Specie"]=="O"]["Modulo_Velocita"], 
                      color='crimson', alpha=0.4, s=15, label='Ossigeno (O)', edgecolors='none')

        if setConfig.IS_ZBL_ON:
            ax_vd.axvline(x=setConfig.RAGGIO_INNER, color='black', linestyle='--', alpha=0.7, label=f'r_inner ({setConfig.RAGGIO_INNER} Å)')
            ax_vd.axvline(x=setConfig.RAGGIO_OUTER, color='gray', linestyle=':', alpha=1, label=f'r_outer ({setConfig.RAGGIO_OUTER} Å)')

        ax_vd.set_title("Modulo della Velocità vs Distanza dal Vicino più Prossimo", fontsize=13, pad=12)
        ax_vd.set_xlabel("Distanza dal Vicino più Prossimo (Å)", fontsize=11)
        ax_vd.set_ylabel("Modulo della Velocità (Å/fs)", fontsize=11)
        ax_vd.grid(True, linestyle='--', alpha=0.6)
        ax_vd.legend(loc='upper right', fontsize=10, markerscale=3)

        plt.tight_layout()
        plt.savefig(path_figd, dpi=300)
        if not setConfig.KEEP_OPEN:
            plt.close(fig_vd)  
        else:
            plt.show(block=False)
            plt.pause(0.1)
            input("Press any key to continue.........")
            plt.close(fig_vd)

    def makeConfrontoForze(self): #---------------------------------------------------------------------------------------------------------------------

        if setConfig.DEBUG:print(f"makeConfrontoForze")
        #Confronto dei moduli del GAP e dello ZBL
        colonne_da_caricare = ["Specie", "Modulo_GAP_eV_A", "Modulo_ZBL_eV_A"]
        file_path = os.path.join(setConfig.PATH_OUT_FILE, "forze_gap+zbl.csv")
        df_f = pd.read_csv(file_path, usecols=colonne_da_caricare)
        if setConfig.DEBUG:print(f"Righe lette da forze_gap+zbl.csv: {len(df_f)}")

        colonne_da_caricare = ["Specie", "Dist_Min_A", "Frame_ID"]
        file_path = os.path.join(setConfig.PATH_OUT_FILE, "distanze.csv")
        df_d = pd.read_csv(file_path, usecols=colonne_da_caricare)
        if setConfig.DEBUG:print(f"Righe lette da distanze.csv: {len(df_d)}")

        colonne_da_caricare = ["Specie", "Modulo_Forza_eV_A"]
        file_path = os.path.join(setConfig.PATH_OUT_FILE, "forze_tot.csv")
        df_r = pd.read_csv(file_path, usecols=colonne_da_caricare)
        if setConfig.DEBUG:print(f"Righe lette da forze_tot.csv: {len(df_r)}")

        path_figd = os.path.join(setConfig.PATH_OUT_GRAPH, "confronto_forze_vs_distanza_minima.png")
        path_figt = os.path.join(setConfig.PATH_OUT_GRAPH, "confronto_forze_vs_tempo.png")        


        fig_ft, ax_ft = plt.subplots(figsize=(9, 6))

        ax_ft.scatter(df_d[df_d["Specie"]=="Ta"]["Frame_ID"], df_f[df_f["Specie"]=="Ta"]["Modulo_GAP_eV_A"], color='royalblue', alpha=0.3, s=5, label='Modulo forza GAP (Ta)', edgecolors='none')
        ax_ft.scatter(df_d[df_d["Specie"]=="O"]["Frame_ID"], df_f[df_f["Specie"]=="O"]["Modulo_GAP_eV_A"], color='crimson', alpha=0.3, s=5, label='Modulo forza GAP (O)', edgecolors='none')
        ax_ft.scatter(df_d[df_d["Specie"]=="Ta"]["Frame_ID"], -df_f[df_f["Specie"]=="Ta"]["Modulo_ZBL_eV_A"], color='purple', alpha=0.3, s=5, label='- Modulo forza ZBL (Ta)', edgecolors='none')
        ax_ft.scatter(df_d[df_d["Specie"]=="O"]["Frame_ID"], -df_f[df_f["Specie"]=="O"]["Modulo_ZBL_eV_A"], color='orange', alpha=0.3, s=5, label='- Modulo forza ZBL (O)', edgecolors='none')
        ax_ft.scatter(df_d[df_d["Specie"]=="Ta"]["Frame_ID"], df_r[df_r["Specie"]=="Ta"]["Modulo_Forza_eV_A"], color='olive', alpha=0.3, s=5, label='Modulo forza risultante (Ta)', edgecolors='none')
        ax_ft.scatter(df_d[df_d["Specie"]=="O"]["Frame_ID"], df_r[df_r["Specie"]=="O"]["Modulo_Forza_eV_A"], color='green', alpha=0.3, s=5, label='Modulo forza risultante (O)', edgecolors='none')

        ax_ft.set_title("Modulo della Forza vs Tempo (GAP vs ZBL)", fontsize=13, pad=12)
        ax_ft.set_xlabel("Frame Id", fontsize=11)
        ax_ft.set_ylabel("Modulo della Forza Netta (eV/Å)", fontsize=11)
        ax_ft.grid(True, linestyle='--', alpha=0.6)
        ax_ft.legend(loc='upper right', fontsize=10, markerscale=5)
        plt.tight_layout()
        plt.savefig(path_figt, dpi=300)
        if not setConfig.KEEP_OPEN:
            plt.close(fig_ft)
        else:
            plt.show(block=False)
            plt.pause(0.1)
            input("Press any key to continue.........")
            plt.close(fig_ft)


        fig_fd, ax_fd = plt.subplots(figsize=(9, 6))

        ax_fd.scatter(df_d[df_d["Specie"]=="Ta"]["Dist_Min_A"], df_f[df_f["Specie"]=="Ta"]["Modulo_GAP_eV_A"], color='royalblue', alpha=0.3, s=5, label='Modulo forza GAP (Ta)', edgecolors='none')
        ax_fd.scatter(df_d[df_d["Specie"]=="O"]["Dist_Min_A"], df_f[df_f["Specie"]=="O"]["Modulo_GAP_eV_A"], color='crimson', alpha=0.3, s=5, label='Modulo forza GAP (O)', edgecolors='none')
        ax_fd.scatter(df_d[df_d["Specie"]=="Ta"]["Dist_Min_A"], -df_f[df_f["Specie"]=="Ta"]["Modulo_ZBL_eV_A"], color='purple', alpha=0.3, s=5, label='- Modulo forza ZBL (Ta)', edgecolors='none')
        ax_fd.scatter(df_d[df_d["Specie"]=="O"]["Dist_Min_A"], -df_f[df_f["Specie"]=="O"]["Modulo_ZBL_eV_A"], color='orange', alpha=0.3, s=5, label='- Modulo forza ZBL (O)', edgecolors='none') 
        ax_fd.scatter(df_d[df_d["Specie"]=="Ta"]["Dist_Min_A"], df_r[df_r["Specie"]=="Ta"]["Modulo_Forza_eV_A"], color='olive', alpha=0.3, s=5, label='Modulo forza risultante (Ta)', edgecolors='none')
        ax_fd.scatter(df_d[df_d["Specie"]=="O"]["Dist_Min_A"], df_r[df_r["Specie"]=="O"]["Modulo_Forza_eV_A"], color='green', alpha=0.3, s=5, label='Modulo forza risultante (O)', edgecolors='none')

        ax_fd.axvline(x=setConfig.RAGGIO_INNER, color='black', linestyle='--', alpha=0.7, label=f'r_inner ({setConfig.RAGGIO_INNER} Å)')
        ax_fd.axvline(x=setConfig.RAGGIO_OUTER, color='gray', linestyle=':', alpha=0.7, label=f'r_outer ({setConfig.RAGGIO_OUTER} Å)')

        ax_fd.set_title("Modulo della Forza vs Distanza dal Vicino più Prossimo (GAP vs ZBL)", fontsize=13, pad=12)
        ax_fd.set_xlabel("Distanza dal Vicino più Prossimo (Å)", fontsize=11)
        ax_fd.set_ylabel("Modulo della Forza Netta (eV/Å)", fontsize=11)
        ax_fd.grid(True, linestyle='--', alpha=0.6)
        ax_fd.legend(loc='upper right', fontsize=10, markerscale=5)
        plt.tight_layout()
        plt.savefig(path_figd, dpi=300)
        if not setConfig.KEEP_OPEN:
            plt.close(fig_fd)
        else:
            plt.show(block=False)
            plt.pause(0.1)
            input("Press any key to continue.........")
            plt.close(fig_fd)

    def makeForceProjection(self): #------------------------------------------------------------------------------------------------------------------------------------------

        if setConfig.DEBUG:print(f"makeForceProjection")
        #Proiezione radiale di GAP e ZBL
        colonne_da_caricare = ["Specie", "GAP_Prj", "ZBL_Prj", "TOT_Prj", "Frame_ID"]
        file_path = os.path.join(setConfig.PATH_OUT_FILE, "forze_proiettate_radiali.csv")
        df = pd.read_csv(file_path, usecols=colonne_da_caricare)
        if setConfig.DEBUG:print(f"Righe lette da forze_proiettate_radiali.csv: {len(df)}")

        path_fig = os.path.join(setConfig.PATH_OUT_GRAPH, "Proiezione_radiale_forze.png")

        fig_ft, ax_ft = plt.subplots(figsize=(9, 6))

        ax_ft.scatter(df["Frame_ID"], df["GAP_Prj"], color='royalblue', alpha=0.4, s=5, label='Proiezione forza GAP', edgecolors='none')
        #ax_ft.scatter(df[df["Specie"]=="Ta"]["Frame_ID"], df[df["Specie"]=="Ta"]["GAP_Prj"], color='royalblue', alpha=0.3, s=5, label='Proiezione forza GAP (Ta)', edgecolors='none')
        #ax_ft.scatter(df[df["Specie"]=="O"]["Frame_ID"],  df[df["Specie"]=="O"]["GAP_Prj"], color='crimson', alpha=0.3, s=5, label='Proiezione forza GAP (O)', edgecolors='none')
        ax_ft.scatter(df["Frame_ID"], df["ZBL_Prj"], color='purple', alpha=0.4, s=5, label='Proiezione forza ZBL', edgecolors='none')
        #ax_ft.scatter(df[df["Specie"]=="Ta"]["Frame_ID"], df[df["Specie"]=="Ta"]["ZBL_Prj"], color='purple', alpha=0.3, s=5, label='Proiezione forza ZBL (Ta)', edgecolors='none')
        #ax_ft.scatter(df[df["Specie"]=="O"]["Frame_ID"],  df[df["Specie"]=="O"]["ZBL_Prj"], color='orange', alpha=0.3, s=5, label='Proiezione forza ZBL (O)', edgecolors='none')
        ax_ft.scatter(df["Frame_ID"], df["TOT_Prj"], color='olive', alpha=0.4, s=5, label='Proiezione forza risultante', edgecolors='none')
        #ax_ft.scatter(df[df["Specie"]=="Ta"]["Frame_ID"], df[df["Specie"]=="Ta"]["TOT_Prj"], color='olive', alpha=0.3, s=5, label='Proiezione forza risultante (Ta)', edgecolors='none')
        #ax_ft.scatter(df[df["Specie"]=="O"]["Frame_ID"],  df[df["Specie"]=="O"]["TOT_Prj"], color='green', alpha=0.3, s=5, label='Proiezione forza risultante (O)', edgecolors='none')

        ax_ft.set_title("Proiezione della Forza sul raggio ij", fontsize=13, pad=12)
        ax_ft.set_xlabel("Frame Id", fontsize=11)
        ax_ft.set_ylabel("Proiezione della Forza Netta (eV/Å) \n >0 Attrattiva | <0 Repulsiva", fontsize=11, multialignment='center')
        ax_ft.grid(True, linestyle='--', alpha=0.6)
        ax_ft.legend(loc="best", fontsize=10, markerscale=5)
        plt.tight_layout()
        plt.savefig(path_fig, dpi=300)
        if not setConfig.KEEP_OPEN:
            plt.close(fig_ft)
        else:
            plt.show(block=False)
            plt.pause(0.1)
            input("Press any key to continue.........")
            plt.close(fig_ft)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
