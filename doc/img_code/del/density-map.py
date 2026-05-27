import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Impostazioni Iniziali
# ==========================================
np.random.seed(84986)           # Fissiamo il seed per avere sempre la stessa disposizione
num_atomi = 60               # Numero di atomi nel piano
lato_scatola = 7.0          # Dimensione dell'area 2D
sigma = 0.8                  # Parametro di smoothing (larghezza della gaussiana)

# Generiamo coordinate casuali x, y per i nostri atomi
posizioni_atomi = np.random.rand(num_atomi, 2) * lato_scatola

# ==========================================
# 2. Creazione della griglia spaziale
# ==========================================
# Creiamo una fitta griglia di punti per calcolare la densità continua
risoluzione = 300
x = np.linspace(0, lato_scatola, risoluzione)
y = np.linspace(0, lato_scatola, risoluzione)
X, Y = np.meshgrid(x, y)

# ==========================================
# 3. Calcolo della Densità Atomica Liscia (SOAP Step 1)
# ==========================================
# Inizializziamo il campo di densità a zero
densita = np.zeros_like(X)

# Sommiamo la gaussiana per ogni singolo atomo
for atomo in posizioni_atomi:
    x_i, y_i = atomo
    # Distanza al quadrato tra ogni punto della griglia e l'atomo i
    distanza_quadrata = (X - x_i)**2 + (Y - y_i)**2
    # Equazione della gaussiana
    densita += np.exp(-distanza_quadrata / (2 * sigma**2))

# ==========================================
# 4. Creazione del Grafico (Plotting)
# ==========================================
# Creiamo una figura con due pannelli affiancati
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- Pannello Sinistro: Atomi Discreti ---
ax1.scatter(posizioni_atomi[:, 0], posizioni_atomi[:, 1], 
            s=400, color='slategray', edgecolor='black', zorder=2)
ax1.set_title("A. Struttura Atomica Discreta", fontsize=16, fontweight='bold', pad=15)
ax1.set_xlim(0, lato_scatola)
ax1.set_ylim(0, lato_scatola)
ax1.set_aspect('equal')
#ax1.axis('off') # Nascondiamo gli assi per un look più pulito
ax1.axis('on')
ax1.grid('on')
# Remove numbers from the axes
ax1.set_xticklabels([])
ax1.set_yticklabels([])

# --- Pannello Destro: Densità Continua ---------------------------------------------------------------
# Usiamo imshow per generare la mappa di calore (heatmap)
mappa_colori = ax2.imshow(densita, origin='lower', 
                          extent=[0, lato_scatola, 0, lato_scatola], 
                          cmap='viridis') # 'viridis' è un'ottima mappa colori standard
#                          cmap='cividis') #
ax2.set_title(r"B. Mappa di Densità $\rho(\mathbf{r})$", fontsize=16, fontweight='bold', pad=15)
ax2.set_aspect('equal')
#ax2.axis('off')
ax2.axis('on')
ax2.grid('on', color="black")
# Remove numbers from the axes
ax2.set_xticklabels([])
ax2.set_yticklabels([])

# Aggiungiamo la barra dei colori (Legenda)
cbar = fig.colorbar(mappa_colori, ax=ax2, fraction=0.046, pad=0.04)
cbar.set_label('Intensità (Alta/Bassa Densità) [u.a.]', rotation=270, labelpad=20, fontsize=12)

# Mostriamo il risultato
plt.tight_layout()
plt.show()
