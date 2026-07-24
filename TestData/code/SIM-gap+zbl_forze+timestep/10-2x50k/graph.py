import numpy as np
import matplotlib.pyplot as plt

# =========================================================================
# 1. PARAMETRI E PERCORSI FILE
# =========================================================================
# Raggi di cutoff (necessari per tracciare le linee di demarcazione nei grafici)
RAGGIO_INNER = 1.5
RAGGIO_OUTER = 2.5

# Input: i file CSV generati dalla dinamica
PATH_CSV_ENERGIE = "dati_energie.csv"
PATH_CSV_FORZE_DISTANZE = "dati_forze_distanze.csv"
PATH_CSV_FORZE_ZBL = "dati_forze_zbl_pure.csv"

# Output: i nomi dei grafici che verranno salvati
PATH_GRAFICO_ENERGIE = "andamento_energia_nve.png"
PATH_GRAFICO_FORZE_TEMPO = "andamento_forze_tempo.png"
PATH_GRAFICO_FORZA_DISTANZA = "forza_vs_distanza_minima.png"
PATH_GRAFICO_3D = "forza_vs_distanza_vs_tempo_3D.png"
PATH_GRAFICO_DISTRIBUZIONE_ZBL = "distribuzione_forze_zbl.png"

print("=" * 50)
print(" AVVIO POST-PROCESSING: GENERAZIONE GRAFICI")
print("=" * 50)

# =========================================================================
# 2. GRAFICO 1: ENERGIE E TEMPERATURA
# =========================================================================
print("-> Generazione grafico Energie...")
try:
    t_en, temp, ep, ek, et = np.loadtxt(PATH_CSV_ENERGIE, delimiter=',', skiprows=1, unpack=True)

    fig_en, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.plot(t_en, temp, color='orangered', lw=1)
    ax1.set_title('Temperatura')
    ax1.set_xlabel('Tempo (fs)')
    ax1.set_ylabel('Temperatura (K)')
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2.plot(t_en, ep, color='royalblue', lw=1, label='E. Potenziale')
    ax2.plot(t_en, ek, color='forestgreen', lw=1, label='E. Cinetica')
    ax2.plot(t_en, et, color='black', lw=1.5, label='E. Totale')
    ax2.set_title('Confronto Energie')
    ax2.set_xlabel('Tempo (fs)')
    ax2.set_ylabel('Energia (eV)')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend()

    ax3.plot(t_en, ep, color='royalblue', lw=1)
    ax3.set_title('Dettaglio E. Potenziale')
    ax3.set_xlabel('Tempo (fs)')
    ax3.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(PATH_GRAFICO_ENERGIE, dpi=300)
    plt.close(fig_en)
except Exception as e:
    print(f"Errore nella lettura di {PATH_CSV_ENERGIE}: {e}")


# =========================================================================
# 3. LETTURA DATI FORZE TOTALI (GAP + ZBL)
# =========================================================================
print("-> Caricamento dati forze e distanze (può richiedere qualche secondo)...")
try:
    dati_fd = np.genfromtxt(PATH_CSV_FORZE_DISTANZE, delimiter=',', skip_header=1, dtype=None, encoding='utf-8')

    tempi = dati_fd['f0']
    specie = dati_fd['f2']
    distanze_min = dati_fd['f3']
    moduli_forza = dati_fd['f4']
    fx = dati_fd['f5']
    fy = dati_fd['f6']
    fz = dati_fd['f7']

    mask_Ta = (specie == 'Ta')
    mask_O = (specie == 'O')

    # --- GRAFICO 2: FORZA NEL TEMPO ---
    print("-> Generazione grafico Forze nel Tempo...")
    fig_forze, (ax_fx, ax_fy, ax_fz) = plt.subplots(1, 3, figsize=(15, 5))

    ax_fx.scatter(tempi, fx, color='red', s=1, alpha=0.1)
    ax_fx.set_title('Forze Lungo X')
    ax_fx.set_xlabel('Tempo (fs)')
    ax_fx.set_ylabel('Forza (eV/Å)')
    ax_fx.grid(True, linestyle=':', alpha=0.7)

    ax_fy.scatter(tempi, fy, color='green', s=1, alpha=0.1)
    ax_fy.set_title('Forze Lungo Y')
    ax_fy.set_xlabel('Tempo (fs)')
    ax_fy.grid(True, linestyle=':', alpha=0.7)

    ax_fz.scatter(tempi, fz, color='blue', s=1, alpha=0.1)
    ax_fz.set_title('Forze Lungo Z')
    ax_fz.set_xlabel('Tempo (fs)')
    ax_fz.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.savefig(PATH_GRAFICO_FORZE_TEMPO, dpi=300)
    plt.close(fig_forze)

    # --- GRAFICO 3: FORZA VS DISTANZA MINIMA 2D ---
    print("-> Generazione grafico Forza vs Distanza Minima 2D...")
    fig_fd, ax_fd = plt.subplots(figsize=(9, 6))

    ax_fd.scatter(distanze_min[mask_Ta], moduli_forza[mask_Ta], color='royalblue', alpha=0.4, s=15, label='Tantalo (Ta)', edgecolors='none')
    ax_fd.scatter(distanze_min[mask_O], moduli_forza[mask_O], color='crimson', alpha=0.4, s=15, label='Ossigeno (O)', edgecolors='none')

    ax_fd.axvline(x=RAGGIO_INNER, color='black', linestyle='--', alpha=0.7, label=f'r_inner ({RAGGIO_INNER} Å)')
    ax_fd.axvline(x=RAGGIO_OUTER, color='gray', linestyle=':', alpha=0.7, label=f'r_outer ({RAGGIO_OUTER} Å)')

    ax_fd.set_title("Modulo della Forza vs Distanza dal Vicino più Prossimo", fontsize=13, pad=12)
    ax_fd.set_xlabel("Distanza dal Vicino più Prossimo (Å)", fontsize=11)
    ax_fd.set_ylabel("Modulo della Forza Netta (eV/Å)", fontsize=11)
    ax_fd.grid(True, linestyle='--', alpha=0.6)
    ax_fd.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(PATH_GRAFICO_FORZA_DISTANZA, dpi=300)
    plt.close(fig_fd)

    # --- GRAFICO 4: SCANSIONE 3D ---
    print("-> Generazione grafico 3D...")
    fig_3d = plt.figure(figsize=(10, 8))
    ax_3d = fig_3d.add_subplot(111, projection='3d')

    ax_3d.scatter(tempi[mask_Ta], distanze_min[mask_Ta], moduli_forza[mask_Ta], color='royalblue', alpha=0.3, s=10, label='Tantalo (Ta)')
    ax_3d.scatter(tempi[mask_O], distanze_min[mask_O], moduli_forza[mask_O], color='crimson', alpha=0.3, s=10, label='Ossigeno (O)')

    ax_3d.set_title("Scansione 3D: Evoluzione nel Tempo di Distanza e Forza", fontsize=13, pad=15)
    ax_3d.set_xlabel("Tempo (fs)", fontsize=10, labelpad=10)
    ax_3d.set_ylabel("Distanza dal Vicino (Å)", fontsize=10, labelpad=10)
    ax_3d.set_zlabel("Modulo della Forza (eV/Å)", fontsize=10, labelpad=10)

    ax_3d.view_init(elev=25, azim=135)
    ax_3d.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig(PATH_GRAFICO_3D, dpi=300)
    plt.close(fig_3d)

except Exception as e:
    print(f"Errore nell'elaborazione di {PATH_CSV_FORZE_DISTANZE}: {e}")


# =========================================================================
# 4. GRAFICO 5: DISTRIBUZIONE FORZE ZBL (REPULSIVE/ATTRATTIVE)
# =========================================================================
print("-> Generazione grafico distribuzione forze ZBL...")
try:
    dati_zbl = np.genfromtxt(PATH_CSV_FORZE_ZBL, delimiter=',', skip_header=1, dtype=None, encoding='utf-8')

    dist_zbl = dati_zbl['f2']
    f_rad = dati_zbl['f3']
    
    # Maschera per considerare solo i punti in cui lo ZBL è effettivamente acceso
    mask_attive = (dist_zbl < RAGGIO_OUTER)

    fig_zbl, (ax_hist, ax_scat) = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot A: Istogramma
    repulsive = f_rad[mask_attive & (f_rad > 0)]
    attrattive = f_rad[mask_attive & (f_rad < 0)]

    ax_hist.hist([repulsive, np.abs(attrattive)], bins=30, color=['firebrick', 'navy'], 
                 label=['Forze Repulsive (F > 0)', 'Forze Attrattive (|F| se F < 0)'], alpha=0.7)
    ax_hist.set_title("Distribuzione Frequenza Forze ZBL Pure")
    ax_hist.set_xlabel("Magnitudine della Forza Radiale ZBL (eV/Å)")
    ax_hist.set_ylabel("Conteggio Eventi")
    ax_hist.legend()
    ax_hist.grid(True, linestyle=':', alpha=0.6)

    # Subplot B: Scatter Plot Forza vs Distanza
    ax_scat.scatter(dist_zbl[mask_attive], f_rad[mask_attive], c=f_rad[mask_attive], 
                     cmap='coolwarm', s=20, alpha=0.6)
    
    ax_scat.axhline(0, color='black', linestyle='--', linewidth=1)
    ax_scat.axvline(RAGGIO_INNER, color='green', linestyle=':', label=f'r_inner ({RAGGIO_INNER} Å)')
    ax_scat.axvline(RAGGIO_OUTER, color='orange', linestyle=':', label=f'r_outer ({RAGGIO_OUTER} Å)')

    ax_scat.set_title("Forza Radiale ZBL vs Distanza (Segno Esplicito)")
    ax_scat.set_xlabel("Distanza dal Vicino più Prossimo (Å)")
    ax_scat.set_ylabel("Forza Radiale ZBL (eV/Å)\n[>0 Repulsiva | <0 Attrattiva]")
    ax_scat.legend()
    ax_scat.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(PATH_GRAFICO_DISTRIBUZIONE_ZBL, dpi=300)
    plt.close(fig_zbl)
except Exception as e:
    print(f"Errore nell'elaborazione di {PATH_CSV_FORZE_ZBL}: {e}")

print("=" * 50)
print(" PROCESSO COMPLETATO! Tutti i grafici sono stati generati.")
print("=" * 50)