import numpy as np
import matplotlib.pyplot as plt
from ase.io import Trajectory, read

def genera_grafico_confronto():
    print("=" * 60)
    print(" CONFRONTO ENERGIE: QUANTUM ESPRESSO vs GAP")
    print("=" * 60)

    # 1. CARICAMENTO DELLE TRAIETTORIE
    # ---------------------------------------------------------
    # Inserisci qui i percorsi corretti dei tuoi file .traj
    # Se un file si trova in un'altra cartella, usa i percorsi relativi (es. "../cartella/file.traj")
    file_qe = "../test_data.extxyz" 
    file_gap = "md_gap_test-500.traj"

    print(f"Caricamento traiettoria Quantum Espresso da: {file_qe}...")
    traj_qe = read(file_qe, index=":")
    
    print(f"Caricamento traiettoria simulata GAP da: {file_gap}...")
    traj_gap = Trajectory(file_gap)

    # Verifichiamo che abbiano lo stesso numero di frame per fare un confronto 1:1
    print(f"Frame QE rilevati: {len(traj_qe)} | Frame GAP rilevati: {len(traj_gap)}")
    
    # Se i calcoli si sono interrotti prima, prendiamo il minimo comune per evitare errori
    num_frame = min(len(traj_qe), len(traj_gap))
    if len(traj_qe) != len(traj_gap):
        print(f"[ATTENZIONE] Il numero di frame differisce. Il confronto avverrà sui primi {num_frame} frame.")

    # 2. ESTRAZIONE DELLE ENERGIE
    # ---------------------------------------------------------
    energie_qe = []
    energie_gap = []

    print("\nEstrazione delle energie in corso...")
    for i in range(num_frame):
        # Estraiamo l'energia potenziale totale del frame i
        e_qe = traj_qe[i].get_potential_energy()
        e_gap = traj_gap[i].get_potential_energy()
        
        energie_qe.append(e_qe)
        energie_gap.append(e_gap)

    # Convertiamo in array NumPy per comodità matematica
    energie_qe = np.array(energie_qe)
    energie_gap = np.array(energie_gap)

    # OPZIONALE: Spesso è utile sottrarre l'energia del primo frame (passo 0) 
    # per confrontare le *energie relative* anziché i valori assoluti totali.
    # Se vuoi l'energia assoluta commenta le due righe sotto:
    #energie_qe = energie_qe - energie_qe[0]
    #energie_gap = energie_gap - energie_gap[0]
    unita_misura = "Energia Relativa (eV)"
    # unita_misura = "Energia Assoluta (eV)"  # Scommenta se usi l'assoluta

    # CALCOLO METRICHE DI ERRORE
#    rmse = np.sqrt(np.mean((energie_gap - energie_qe) ** 2))
    # Dividiamo per il numero di atomi per avere l'errore per atomo (es. su 100 atomi)
#    num_atomi = len(traj_qe[0])
#    rmse_per_atomo = (rmse / num_atomi) * 1000 # Convertito in meV/atomo
#    print(f"-> Errore RMSE Totale: {rmse:.3f} eV")
#    print(f"-> Errore RMSE per atomo: {rmse_per_atomo:.2f} meV/atomo")

    # 3. GENERAZIONE DEL GRAFICO DI PARITÀ
    # ---------------------------------------------------------
    plt.figure(figsize=(7, 6))
    
    # Disegniamo i punti di confronto
    plt.scatter(energie_qe, energie_gap, color='blue', alpha=0.6, edgecolors='k', label='Dati di test')
    
    # Calcoliamo i limiti ideali per la retta a 45°
    min_val = min(energie_qe.min(), energie_gap.min())
    max_val = max(energie_qe.max(), energie_gap.max())
    padding = (max_val - min_val) * 0.05
    limiti = [min_val - padding, max_val + padding]
    
    # Disegniamo la retta di parità perfetta (Y = X)
    plt.plot(limiti, limiti, color='red', linestyle='--', linewidth=2, label='Parità Perfetta (Y=X)')
    
    # Impostiamo limiti, etichette e dettagli grafici
    plt.xlim(limiti)
    plt.ylim(limiti)
    plt.xlabel(f"Quantum Espresso - {unita_misura}", fontsize=12)
    plt.ylabel(f"GAP Potenziale - {unita_misura}", fontsize=12)
    plt.title("Grafico di Parità: Energie QE vs GAP", fontsize=14, fontweight='bold')
    
    # Box di testo interno con l'errore RMSE
#    testo_errore = f"RMSE: {rmse_per_atomo:.1f} meV/atomo"
#    plt.gca().text(0.05, 0.95, testo_errore, transform=plt.gca().transAxes, 
#                   fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    # Salviamo il grafico in un file immagine
    nome_grafico = "confronto_energie_qe_gap-500.png"
    plt.savefig(nome_grafico, dpi=300)
    print(f"\nGrafico salvato con successo in: '{nome_grafico}'")
    
    # Mostra a schermo (se sei in un ambiente grafico, es. locale o Jupyter)
    plt.show()

if __name__ == "__main__":
    genera_grafico_confronto()
