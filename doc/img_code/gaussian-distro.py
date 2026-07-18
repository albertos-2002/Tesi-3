import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 1. Impostazione dei parametri della distribuzione
media = 0          # Media (μ) - il centro della curva
dev_std = 1        # Deviazione standard (σ) - la larghezza della curva

# 2. Creazione dei valori per l'asse X
# Per avere un bel grafico completo, creiamo un range che va da -4 a +4 deviazioni standard
x = np.linspace(media - 4*dev_std, media + 4*dev_std, 1000)

# 3. Calcolo dei valori per l'asse Y (Funzione di Densità di Probabilità)
y = norm.pdf(x, media, dev_std)

# 4. Creazione del grafico
# Disegniamo la linea di contorno della curva
plt.plot(x, y, color='darkblue', linewidth=2, label=f'Media={media}, Dev. Std={dev_std}')

# Riempiamo l'area sotto la curva con un colore
plt.fill_between(x, y, color='skyblue', alpha=0.5)

# 5. Personalizzazione del grafico
plt.title('Curva Gaussiana')
plt.xlabel('Valori')
plt.ylabel('Densità di Probabilità')
#plt.legend()
plt.grid(alpha=0.4, linestyle='--')

# Mostra il risultato a schermo
plt.show()
