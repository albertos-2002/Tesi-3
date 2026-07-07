import numpy as np
import matplotlib.pyplot as plt
from ase.io import Trajectory, read

def genera_grafico_confronto():
    print("=" * 60)
    print(" CONFRONTO ENERGIE: TimeStep 4.8fs vs TimeStep 0.48fs")
    print("=" * 60)

    # 1. CARICAMENTO DELLE TRAIETTORIE
    # ---------------------------------------------------------
    # Inserisci qui i percorsi corretti dei tuoi file .traj
    file_48 = "../md_gap_test-500.traj" 
    file_048 = "md_gap_test-500-fric.traj"
    file_qe = "../../test_data.extxyz"

    print(f"Caricamento traiettoria 0.01friction da: {file_48}...")
    traj_48 = Trajectory(file_48)  
    print(f"Caricamento traiettoria 1friction da: {file_048}...")
    traj_048 = Trajectory(file_048)
    print(f"Caricamento traiettoria QE da: {file_qe}...")
    traj_qe = read(file_qe, index=":")

    len_traj_48 = len(traj_48)
    print(f"Frame traj_48: {len(traj_48)}")
    len_traj_048 = len(traj_048)
    print(f"Frame traj_048: {len(traj_048)}")
    
    # 2. ESTRAZIONE DELLE ENERGIE
    # ---------------------------------------------------------
    energie_48 = []
    energie_048 = []
    energie_qe = []

    print("\nEstrazione delle energie in corso...")
    for i in range(len_traj_48):
        # Estraiamo l'energia potenziale totale del frame i
        e_48 = traj_48[i].get_potential_energy()
        energie_48.append(e_48)
    
    for i in range(len_traj_048):
        e_048 = traj_048[i].get_potential_energy()
        energie_048.append(e_048)
        
    for i in range(len(traj_qe)):
        e_qe = traj_qe[i].get_potential_energy()
        energie_qe.append(e_qe)        
        
    # Convertiamo in array NumPy per comodità matematica
    energie_48 = np.array(energie_48)
    energie_048 = np.array(energie_048)
    energie_qe = np.array(energie_qe)

    unita_misura = "Energia (eV)"
    
    index_48 = np.arange(len_traj_48)
    index_048 = np.arange(len_traj_048)
    index_qe = np.arange(len(traj_qe))
    
    # 3. GENERAZIONE DEL GRAFICO
    # ---------------------------------------------------------
    plt.figure(figsize=(7, 6))
    
    # Disegniamo i punti di confronto
    plt.scatter(index_48, energie_48, color='blue', alpha=0.5, s=20, edgecolors='k', label='Friction 0.01')
    plt.scatter(index_048, energie_048, color='red', alpha=0.5, s=20, edgecolors='k', label='Friction 1')
    plt.scatter(index_qe, energie_qe, color='green', alpha=0.5, s=20, edgecolors='k', label='QE reference')
    
    # Calcoliamo i limiti ideali per la retta a 45°
    min_val = min(energie_48.min(), energie_048.min(), energie_qe.min())
    max_val = max(energie_48.max(), energie_048.max(), energie_qe.max())
    padding = (max_val - min_val) * 0.05
    limiti = [min_val - padding, max_val + padding]
    
    # Impostiamo limiti, etichette e dettagli grafici
    #plt.xlim(limiti)
    plt.ylim(limiti)
    plt.xlabel(f"Indice temporale", fontsize=12)
    plt.ylabel(f"Energia Potenziale - {unita_misura}", fontsize=12)
    plt.title("Grafico di confronto", fontsize=14, fontweight='bold')  
    
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    # Salviamo il grafico in un file immagine
    nome_grafico = "confronto_energie_friction-500.png"
    plt.savefig(nome_grafico, dpi=300)
    print(f"\nGrafico salvato con successo in: '{nome_grafico}'")
    
    # Mostra a schermo (se sei in un ambiente grafico, es. locale o Jupyter)
    plt.show()

if __name__ == "__main__":
    genera_grafico_confronto()
