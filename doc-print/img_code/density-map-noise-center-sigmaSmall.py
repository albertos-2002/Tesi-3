import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Impostazioni Iniziali e Struttura Globale
# ==========================================
np.random.seed(42)
lato_scatola = 10.0
sigma = 0.2                  # Ridotto leggermente per evidenziare i picchi locali
r_cut = 3.5                  # Raggio di cutoff tipico di SOAP

# Creiamo la struttura perturbata di partenza (come prima)
punti_reticolo = np.linspace(2.0, 8.0, 4)
X_atomi, Y_atomi = np.meshgrid(punti_reticolo, punti_reticolo)
posizioni_regolari = np.c_[X_atomi.ravel(), Y_atomi.ravel()]
intensita_rumore = 0.3
rumore = np.random.normal(0, intensita_rumore, posizioni_regolari.shape)
posizioni_globali = posizioni_regolari + rumore

# ==========================================
# 2. TRASLAZIONE SOAP: Scegliamo il centro del mondo
# ==========================================
# Troviamo l'atomo più vicino al centro geometrico della scatola (5.0, 5.0)
centro_scatola = np.array([5.0, 5.0])
distanze_dal_centro = np.linalg.norm(posizioni_globali - centro_scatola, axis=1)
indice_atomo_centrale = np.argmin(distanze_dal_centro)

# Questo sarà il nostro atomo di riferimento (il "centro del mondo")
atomo_centro_coordinate = posizioni_globali[indice_atomo_centrale]

# TRASLAZIONE: Sottraiamo le sue coordinate a tutti gli atomi.
# Ora l'atomo centrale si troverà esattamente in (0, 0)
posizioni_locali = posizioni_globali - atomo_centro_coordinate

# ==========================================
# 3. Creazione della nuova griglia spaziale centrata in (0,0)
# ==========================================
# Lo spazio ora si estende da -lato/2 a +lato/2, con lo zero al centro
estensione = 5.0
risoluzione = 300
x = np.linspace(-estensione, estensione, risoluzione)
y = np.linspace(-estensione, estensione, risoluzione)
X, Y = np.meshgrid(x, y)

# ==========================================
# 4. Calcolo della Densità Locale (All'interno del Cutoff)
# ==========================================
densita_locale = np.zeros_like(X)

for atomo in posizioni_locali:
    # Calcoliamo la distanza dell'atomo dall'origine (0,0)
    distanza_da_origine = np.linalg.norm(atomo)
    
    # SOAP considera solo gli atomi entro il raggio di cutoff (rcut)
    if distanza_da_origine <= r_cut:
        x_i, y_i = atomo
        distanza_quadrata = (X - x_i)**2 + (Y - y_i)**2
        densita_locale += np.exp(-distanza_quadrata / (2 * sigma**2))

# ==========================================
# 5. Creazione del Grafico
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Elemento grafico per il disegno del cerchio di cutoff su entrambi i pannelli
for ax in [ax1, ax2]:
    cerchio = plt.Circle((0, 0), r_cut, color='red', fill=False, linestyle='--', linewidth=2, alpha=0.6, label='Raggio Cutoff ($r_{cut}$)')
    ax.add_patch(cerchio)

# --- Pannello Sinistro: Atomi Discreti nel sistema locale ---
# Disegnamo tutti gli atomi vicini in grigio
ax1.scatter(posizioni_locali[:, 0], posizioni_locali[:, 1], s=300, color='slategray', edgecolor='black', zorder=2)
# Evidenziamo l'atomo centrale (l'origine) in oro
ax1.scatter(0, 0, s=350, color='gold', edgecolor='black', zorder=3, label='Atomo di Riferimento (0,0)')

ax1.set_title("A. Intorno Atomico Locale (Centrato)", fontsize=14, fontweight='bold', pad=15)
ax1.set_xlim(-estensione, estensione)
ax1.set_ylim(-estensione, estensione)
ax1.set_aspect('equal')
ax1.grid(True, linestyle=':', alpha=0.6) # Mostriamo la griglia per vedere lo zero al centro
ax1.legend(loc='upper right')
# Remove numbers from the axes
ax1.set_xticklabels([])
ax1.set_yticklabels([])



# --- Pannello Destro: Mappa di Densità Locale SOAP ---
mappa_colori = ax2.imshow(densita_locale, origin='lower', 
                          extent=[-estensione, estensione, -estensione, estensione], 
                          cmap='plasma')
# Evidenziamo il punto centrale (0,0) anche sulla mappa di densità
ax2.scatter(0, 0, marker='+', color='white', s=200, linewidth=2)

ax2.set_title(r"B. Mappa di Densità Locale SOAP $\rho_{i}(\mathbf{r})$", fontsize=14, fontweight='bold', pad=15)
ax2.set_aspect('equal')
ax2.grid(True, linestyle=':', alpha=0.3, color='white')
# Remove numbers from the axes
ax2.set_xticklabels([])
ax2.set_yticklabels([])



# Barra dei colori
cbar = fig.colorbar(mappa_colori, ax=ax2, fraction=0.046, pad=0.04)
cbar.set_label('Intensità della Densità Locale [u.a.]', rotation=270, labelpad=20, fontsize=12)

plt.tight_layout()
plt.show()
