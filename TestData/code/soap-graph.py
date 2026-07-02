import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# 1. Caricamento Dati
# =====================================================================
# Sostituisci questo blocco con il caricamento della tua matrice reale:
# soap_data = np.load("tuo_file_soap.npy") 
# Assicurati che la shape sia (4597, 126, 952)

# Generazione di dati fittizi per l'esempio (uso 500 frame per non saturare la RAM)
# Ho aggiunto un po' di "rumore" e un "salto strutturale" a metà simulazione
n_frames_tot = 500
n_atoms = 126
n_features = 952
soap_data = np.random.rand(n_frames_tot, n_atoms, n_features)

# Simulo un salto strutturale al frame 250
soap_data[250:] += np.random.rand(1, 1, n_features) * 0.5 

# =====================================================================
# 2. Selezione dell'Atomo e Preparazione Dati
# =====================================================================
# Scegliamo un atomo casuale (da 0 a 125)
atomo_selezionato = np.random.randint(0, n_atoms)
print(f"Atomo selezionato casualmente per l'analisi: Indice {atomo_selezionato}")

# Estraiamo la storia temporale solo per questo atomo. 
# La shape diventerà (N_frames, 952)
storia_atomo = soap_data[:, atomo_selezionato, :]

# Separiamo l'array in due blocchi sfalsati di un frame per il confronto p1 vs p2
# p_corrente: dal frame 0 al penultimo
# p_successivo: dal frame 1 all'ultimo
p_corrente = storia_atomo[:-1, :]
p_successivo = storia_atomo[1:, :]

# =====================================================================
# 3. Calcolo del Kernel (Similarità Coseno vettorializzata)
# =====================================================================
# Formula: K(p1, p2) = (p1 * p2) / (|p1| * |p2|)

# Prodotto scalare (p1 * p2) riga per riga
prodotto_scalare = np.sum(p_corrente * p_successivo, axis=1)

# Norma (magnitudo) dei vettori |p1| e |p2|
norma_corrente = np.linalg.norm(p_corrente, axis=1)
norma_successivo = np.linalg.norm(p_successivo, axis=1)

# Calcolo finale del kernel per tutte le transizioni contemporaneamente
valori_kernel = prodotto_scalare / (norma_corrente * norma_successivo)

# =====================================================================
# 4. Visualizzazione Grafica
# =====================================================================
# Creiamo l'asse X (gli indici delle transizioni: 1->2, 2->3, ecc.)
asse_x = np.arange(1, len(valori_kernel) + 1)

fig, ax = plt.subplots(figsize=(12, 6))

# Plottiamo i dati grezzi come punti
ax.scatter(asse_x, valori_kernel, color='royalblue', alpha=0.6, s=10, label='Transizione Frame (i $\\rightarrow$ i+1)')

# Opzionale: Calcoliamo una media mobile per vedere meglio il trend senza il rumore termico
finestra_media = 20
if len(valori_kernel) > finestra_media:
    media_mobile = np.convolve(valori_kernel, np.ones(finestra_media)/finestra_media, mode='valid')
    ax.plot(asse_x[finestra_media-1:], media_mobile, color='red', linewidth=2, label=f'Media mobile ({finestra_media} frame)')

# Formattazione del grafico
ax.set_title(f"Evoluzione Strutturale Locale (Atomo {atomo_selezionato})\nMisurata tramite Kernel SOAP tra Frame Successivi", fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("Indice di Transizione (Frame)", fontsize=12)
ax.set_ylabel(r"Valore del Kernel $K(p_i, p_{i+1})$", fontsize=12)

# La similarità del coseno va da -1 a 1, ma per descrittori postivi è solitamente tra 0 e 1.
# Mettiamo un limite superiore a 1.01 per questioni estetiche.
ax.set_ylim(bottom=np.min(valori_kernel)*0.98, top=1.005) 
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='lower right')

plt.tight_layout()
plt.show()
