import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Generazione della Densità Locale
# ==========================================
np.random.seed(42)
sigma = 0.6
r_cut = 3.5

punti_reticolo = np.linspace(2.0, 8.0, 4)
X_atomi, Y_atomi = np.meshgrid(punti_reticolo, punti_reticolo)
posizioni_regolari = np.c_[X_atomi.ravel(), Y_atomi.ravel()]
posizioni_globali = posizioni_regolari + np.random.normal(0, 0.25, posizioni_regolari.shape)

# Centriamo il sistema
centro_scatola = np.array([5.0, 5.0])
atomo_centro = posizioni_globali[np.argmin(np.linalg.norm(posizioni_globali - centro_scatola, axis=1))]
posizioni_locali = posizioni_globali - atomo_centro

# Griglia e Coordinate Polari
estensione = 4.0
risoluzione = 200
x = np.linspace(-estensione, estensione, risoluzione)
y = np.linspace(-estensione, estensione, risoluzione)
X, Y = np.meshgrid(x, y)

R = np.sqrt(X**2 + Y**2)
Theta = np.arctan2(Y, X)
maschera = R <= r_cut

# Calcolo Densità (solo atomi entro il cutoff)
densita_locale = np.zeros_like(X)
for atomo in posizioni_locali:
    if np.linalg.norm(atomo) <= r_cut:
        densita_locale += np.exp(-((X - atomo[0])**2 + (Y - atomo[1])**2) / (2 * sigma**2))

# ==========================================
# 2. Definizione dei Filtri (Inclusione dei vari 'm' ed 'l')
# ==========================================
R_1 = (r_cut - R)**2 * np.exp(-R) * maschera

Filtro_s  = R_1 * 1.0                  # l=0, m=0  (Simmetria circolare)
Filtro_px = R_1 * np.cos(Theta)        # l=1, m=+1 (Dipolo, asse X)
Filtro_py = R_1 * np.sin(Theta)        # l=1, m=-1 (Dipolo, asse Y)
Filtro_d  = R_1 * np.cos(2 * Theta)    # l=2, m=+2 (Quadripolo, 4 lobi a croce)

# ==========================================
# 3. Integrazione (Proiezione Matematica Rigorosa)
# ==========================================
area_pixel = (2 * estensione / risoluzione)**2

c_s  = np.sum(densita_locale * Filtro_s) * area_pixel
c_px = np.sum(densita_locale * Filtro_px) * area_pixel
c_py = np.sum(densita_locale * Filtro_py) * area_pixel
c_d  = np.sum(densita_locale * Filtro_d) * area_pixel

# ==========================================
# 4. Visualizzazione
# ==========================================
filtri = [
    (Filtro_s, c_s, "l=0, m=0\n(Sfera)"),
    (Filtro_px, c_px, "l=1, m=+1\n(Dipolo X)"),
    (Filtro_py, c_py, "l=1, m=-1\n(Dipolo Y)"),
    (Filtro_d, c_d, "l=2, m=+2\n(Quadripolo)")
]

fig, axs = plt.subplots(4, 3, figsize=(12, 16))

for i, (filtro, coeff, titolo) in enumerate(filtri):
    # Pannello Sinistro: Densità Locale
    axs[i, 0].imshow(densita_locale, cmap='plasma', origin='lower', extent=[-estensione, estensione, -estensione, estensione])
    axs[i, 0].set_title("Densità Locale $\\rho(r)$")
    
    # Pannello Centrale: Filtro
    vmax = np.max(np.abs(filtro)) if np.max(np.abs(filtro)) > 0 else 1
    axs[i, 1].imshow(filtro, cmap='seismic', origin='lower', extent=[-estensione, estensione, -estensione, estensione], vmin=-vmax, vmax=vmax)
    axs[i, 1].set_title(f"Filtro Base:\n{titolo}")
    
    # Pannello Destro: Sovrapposizione
    intersezione = densita_locale * filtro
    imax = np.max(np.abs(intersezione)) if np.max(np.abs(intersezione)) > 0 else 1
    axs[i, 2].imshow(intersezione, cmap='seismic', origin='lower', extent=[-estensione, estensione, -estensione, estensione], vmin=-imax, vmax=imax)
    axs[i, 2].set_title(f"Sovrapposizione\nCoeff. estratto = {coeff:.3f}")
    
    # Pulizia layout per ogni asse
    for ax in axs[i]:
        ax.add_patch(plt.Circle((0, 0), r_cut, color='black', fill=False, linestyle='--', alpha=0.3))
        ax.axis('off')
        ax.set_aspect('equal')

plt.tight_layout()
plt.show()
