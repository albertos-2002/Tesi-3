import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# 1. Definizione dei parametri della Gaussiana Bivariata
media = [0, 0]          # Centrata in (0,0)
# Matrice di covarianza (varianza di X = 1, varianza di Y = 1, nessuna correlazione)
covarianza = [[1, 0], 
              [0, 1]]

# Valore di X dove vogliamo effettuare il taglio
x_taglio = 1.0

# 2. Creazione della griglia 3D (Meshgrid)
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)

# Calcolo della densità di probabilità (Z) per ogni punto della griglia
pos = np.dstack((X, Y))
rv = multivariate_normal(media, covarianza)
Z = rv.pdf(pos)

# 3. Calcolo della curva di taglio (Gaussiana condizionata su X = x_taglio)
# Fissiamo X al valore di taglio e facciamo variare Y
Y_taglio = np.linspace(-3, 3, 100)
X_taglio = np.full_like(Y_taglio, x_taglio)
# Calcoliamo la Z lungo questa specifica linea retta
pos_taglio = np.dstack((X_taglio, Y_taglio))
Z_taglio = rv.pdf(pos_taglio).reshape(-1)

# 4. Configurazione del grafico 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Disegna la superficie 3D (la cupola) con una mappa di colori sfumata
superficie = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9, linewidth=0, antialiased=True)

# Disegna la linea di taglio (evidenziata in rosso spesso)
ax.plot(X_taglio, Y_taglio, Z_taglio, color='red', linewidth=4, label=f'Taglio a X = {x_taglio}')

# Opzionale: Riempi l'area sotto il taglio per renderlo ancora più visibile
# Per farlo in 3D "inganniamo" fill_between proiettando poligoni, oppure usiamo un trucco visivo con i punti
ax.add_collection3d(plt.fill_between(Y_taglio, 0, Z_taglio, color='red', alpha=0.2), zs=x_taglio, zdir='x')

# 5. Personalizzazione e visualizzazione
ax.set_title('Gaussiana bivariata')
ax.set_xlabel('Asse X')
ax.set_ylabel('Asse X*')
ax.set_zlabel('Densità di Probabilità')
#ax.legend()

# Orientazione iniziale della telecamera per vedere bene il taglio
ax.view_init(elev=25, azim=-45)

plt.show()
