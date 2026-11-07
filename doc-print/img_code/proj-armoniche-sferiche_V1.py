import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Generazione Densità Locale (Dal codice precedente)
# ==========================================
np.random.seed(42)
lato_scatola = 10.0
sigma = 0.6
r_cut = 3.5

punti_reticolo = np.linspace(2.0, 8.0, 4)
X_atomi, Y_atomi = np.meshgrid(punti_reticolo, punti_reticolo)
posizioni_regolari = np.c_[X_atomi.ravel(), Y_atomi.ravel()]
posizioni_globali = posizioni_regolari + np.random.normal(0, 0.3, posizioni_regolari.shape)

# Centriamo il sistema sull'atomo centrale
centro_scatola = np.array([5.0, 5.0])
atomo_centro_coordinate = posizioni_globali[np.argmin(np.linalg.norm(posizioni_globali - centro_scatola, axis=1))]
posizioni_locali = posizioni_globali - atomo_centro_coordinate

# Griglia spaziale centrata in (0,0)
estensione = 4.0
risoluzione = 300
x = np.linspace(-estensione, estensione, risoluzione)
y = np.linspace(-estensione, estensione, risoluzione)
X, Y = np.meshgrid(x, y)

R = np.sqrt(X**2 + Y**2)
Theta = np.arctan2(Y, X)
maschera = R <= r_cut

# Calcolo della DENSITÀ LOCALE REALE
densita_locale = np.zeros_like(X)
for atomo in posizioni_locali:
    if np.linalg.norm(atomo) <= r_cut:
        x_i, y_i = atomo
        densita_locale += np.exp(-((X - x_i)**2 + (Y - y_i)**2) / (2 * sigma**2))

# ==========================================
# 2. Definizione dei Filtri (Funzioni di Base)
# ==========================================
R_1 = (r_cut - R)**2 * np.exp(-R) * maschera
Filtro_s = R_1 * 1.0                   # l = 0
Filtro_p = R_1 * np.cos(Theta)         # l = 1

# ==========================================
# 3. IL VERO STEP SOAP: Proiezione (Moltiplicazione e Somma)
# ==========================================
# Moltiplichiamo la densità degli atomi per il filtro, pixel per pixel
sovrapposizione_s = densita_locale * Filtro_s
sovrapposizione_p = densita_locale * Filtro_p

# Calcoliamo il coefficiente finale (l'integrale approssimato come somma dei pixel)
# Normalizziamo per l'area del pixel per avere un valore numerico pulito
area_pixel = (2 * estensione / risoluzione)**2
c_10 = np.sum(sovrapposizione_s) * area_pixel  # n=1, l=0
c_11 = np.sum(sovrapposizione_p) * area_pixel  # n=1, l=1

# ==========================================
# 4. Visualizzazione del Processo
# ==========================================
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# Opzioni grafiche
F_cut = plt.Circle((0, 0), r_cut, color='black', fill=False, linestyle='--', alpha=0.5)

# RIGA 1: Analisi con la Sfera (l=0)
axs[0, 0].imshow(densita_locale, cmap='plasma', origin='lower', extent=[-estensione, estensione, -estensione, estensione])
axs[0, 0].set_title("1. Densità Locale Atomi $\\rho(r)$")

axs[0, 1].imshow(Filtro_s, cmap='seismic', origin='lower', extent=[-estensione, estensione, -estensione, estensione], vmin=-1, vmax=1)
axs[0, 1].set_title("2. Filtro di Base ($l=0$)")

axs[0, 2].imshow(sovrapposizione_s, cmap='inferno', origin='lower', extent=[-estensione, estensione, -estensione, estensione])
axs[0, 2].set_title(f"3. Intersezione Risultante\nCOEFFICIENTE $c_{{10}}$ = {c_10:.3f}")

# RIGA 2: Analisi con il Dipolo (l=1)
axs[1, 0].imshow(densita_locale, cmap='plasma', origin='lower', extent=[-estensione, estensione, -estensione, estensione])
axs[1, 1].imshow(Filtro_p, cmap='seismic', origin='lower', extent=[-estensione, estensione, -estensione, estensione], vmin=-1, vmax=1)
axs[1, 2].imshow(sovrapposizione_p, cmap='seismic', origin='lower', extent=[-estensione, estensione, -estensione, estensione], vmin=-1, vmax=1)
axs[1, 2].set_title(f"3. Intersezione Risultante\nCOEFFICIENTE $c_{{11}}$ = {c_11:.3f}")

# Pulizia assi
for ax in axs.ravel():
    ax.add_patch(plt.Circle((0, 0), r_cut, color='black', fill=False, linestyle='--', alpha=0.3))
    ax.axis('off')
    ax.set_aspect('equal')

plt.tight_layout()
plt.show()
