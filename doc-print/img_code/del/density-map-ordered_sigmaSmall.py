import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Impostazioni Iniziali (Struttura Ordinata)
# ==========================================
lato_scatola = 10.0          # Dimensione dell'area 2D
sigma = 0.1                  # Parametro di smoothing (larghezza della gaussiana)

# Creiamo una disposizione ORDINATA (un reticolo regolare 4x4 = 16 atomi)
punti_reticolo = np.linspace(2.0, 8.0, 4) # 4 punti equidistanti tra 2 e 8
X_atomi, Y_atomi = np.meshgrid(punti_reticolo, punti_reticolo)

# Trasformiamo la griglia di atomi in una lista di coordinate (x, y)
posizioni_atomi = np.c_[X_atomi.ravel(), Y_atomi.ravel()]
num_atomi = len(posizioni_atomi)

# ==========================================
# 2. Creazione della griglia spaziale
# ==========================================
risoluzione = 300
x = np.linspace(0, lato_scatola, risoluzione)
y = np.linspace(0, lato_scatola, risoluzione)
X, Y = np.meshgrid(x, y)

# ==========================================
# 3. Calcolo della Densità Atomica Liscia (SOAP Step 1)
# ==========================================
densita = np.zeros_like(X)

for atomo in posizioni_atomi:
    x_i, y_i = atomo
    distanza_quadrata = (X - x_i)**2 + (Y - y_i)**2
    densita += np.exp(-distanza_quadrata / (2 * sigma**2))

# ==========================================
# 4. Creazione del Grafico (Plotting)
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- Pannello Sinistro: Atomi Discreti ---
ax1.scatter(posizioni_atomi[:, 0], posizioni_atomi[:, 1], 
            s=400, color='slategray', edgecolor='black', zorder=2)
ax1.set_title("A. Struttura Atomica Ordinata", fontsize=16, fontweight='bold', pad=15)
ax1.set_xlim(0, lato_scatola)
ax1.set_ylim(0, lato_scatola)
ax1.set_aspect('equal')
#ax1.axis('off')
ax1.axis('on')
ax1.grid('on')
# Remove numbers from the axes
ax1.set_xticklabels([])
ax1.set_yticklabels([])

# --- Pannello Destro: Densità Continua ------------------------------------------------------
mappa_colori = ax2.imshow(densita, origin='lower', 
                          extent=[0, lato_scatola, 0, lato_scatola], 
                          cmap='viridis')
ax2.set_title(r"B. Mappa di Densità $\rho(\mathbf{r})$", fontsize=16, fontweight='bold', pad=15)
ax2.set_aspect('equal')
#ax2.axis('off')
ax2.axis('on')
ax2.grid('on', color="black")
# Remove numbers from the axes
ax2.set_xticklabels([])
ax2.set_yticklabels([])

# Barra dei colori
cbar = fig.colorbar(mappa_colori, ax=ax2, fraction=0.046, pad=0.04)
cbar.set_label('Intensità (Alta/Bassa Densità) [u.a.]', rotation=270, labelpad=20, fontsize=12)

plt.tight_layout()
plt.show()
