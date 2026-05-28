import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Impostazioni Griglia (Spazio Locale Centrato)
# ==========================================
estensione = 4.0
r_cut = 3.5                  # Il nostro raggio di cutoff
risoluzione = 300
x = np.linspace(-estensione, estensione, risoluzione)
y = np.linspace(-estensione, estensione, risoluzione)
X, Y = np.meshgrid(x, y)

# Per le armoniche, ci servono le coordinate polari (Raggio e Angolo)
R = np.sqrt(X**2 + Y**2)      # Distanza dal centro
Theta = np.arctan2(Y, X)      # Direzione (angolo)

# Maschera: tutto ciò che è fuori dal raggio cutoff viene azzerato
maschera = R <= r_cut

# ==========================================
# 2. Creazione delle Funzioni di Base (I "mattoni" di SOAP)
# ==========================================
# Funzione Radiale (R_n): Definisce come l'intensità cambia con la distanza.
# Usiamo una formula semplice che sfuma dolcemente a zero al bordo (r_cut).
R_1 = (r_cut - R)**2 * np.exp(-R) * maschera

# Funzioni Angolari (Y_l): Definiscono la direzione.
# l=0: Simmetria circolare (tipo orbitale s)
Base_s = R_1 * 1.0 

# l=1: Simmetria dipolare (tipo orbitale p, orientato orizzontalmente)
Base_px = R_1 * np.cos(Theta)

# l=2: Simmetria quadripolare (tipo orbitale d, 4 lobi)
Base_d = R_1 * np.cos(2 * Theta)

# ==========================================
# 3. Creazione del Grafico
# ==========================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# Usiamo una mappa DIVERGENTE (seismic) perché queste funzioni hanno valori negativi!
# Calcoliamo il massimo assoluto per centrare il bianco esattamente sullo zero
limite_colore = max(np.max(Base_s), np.max(np.abs(Base_px)))

opzioni_plot = {
    'cmap': 'seismic',
    'origin': 'lower',
    'extent': [-estensione, estensione, -estensione, estensione],
    'vmin': -limite_colore,
    'vmax': limite_colore
}

# --- Pannello 1: l = 0 (Simmetria Totale) ---
ax1.imshow(Base_s, **opzioni_plot)
ax1.set_title("Componente l=0 (Tipo 's')\nCattura la massa totale", fontsize=14, fontweight='bold', pad=10)

# --- Pannello 2: l = 1 (Asimmetria) ---
ax2.imshow(Base_px, **opzioni_plot)
ax2.set_title("Componente l=1 (Tipo 'p')\nCattura lo sbilanciamento Destra/Sinistra", fontsize=14, fontweight='bold', pad=10)

# --- Pannello 3: l = 2 (Dettagli complessi) ---
im = ax3.imshow(Base_d, **opzioni_plot)
ax3.set_title("Componente l=2 (Tipo 'd')\nCattura le direzioni incrociate", fontsize=14, fontweight='bold', pad=10)

# Formattazione comune
for ax in [ax1, ax2, ax3]:
    cerchio = plt.Circle((0, 0), r_cut, color='black', fill=False, linestyle='--', linewidth=1.5, alpha=0.5)
    ax.add_patch(cerchio)
    ax.set_aspect('equal')
    #ax.axis('off')
    ax.grid('on', color="gray", linestyle="dashed")
    # Remove numbers from the axes
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Barra dei colori globale
cbar = fig.colorbar(im, ax=[ax1, ax2, ax3], fraction=0.02, pad=0.04)
cbar.set_label('Valore della Funzione (Rosso=Positivo, Blu=Negativo)', rotation=270, labelpad=20, fontsize=12)

plt.show()
