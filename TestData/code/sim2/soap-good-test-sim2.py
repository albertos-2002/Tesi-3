import numpy as np
import matplotlib.pyplot as plt
import sys
import ase.io
from dscribe.descriptors import SOAP
import time

#=======================================
#ASE
#=======================================
traiettoria_completa = []

# Lettura della traiettoria =========================================================================================================
traiettoria_caricata = ase.io.read('../traiettoria_unita.traj', index=':')
print(f"check len: {len(traiettoria_caricata)}")
traiettoria_completa = traiettoria_caricata


#=======================================
#SOAP
#=======================================
#Usiamo adesso i dati processati da ASE per produrre il SOAP

# 3. Ora puoi passare la lista unica a DScribe
start_time_soap = time.time()

# Configurare il Descrittore SOAP
# i parametri qui inseriti che si riferiscono alla "risoluzione" del risultato sono stati inseriti da Gemini e possono essere facilmente modficati
soap = SOAP(
    species=["Ta", "O"], # Gli elementi chimici presenti nel tuo sistema
    periodic=True,       # Essenziale: è una cella periodica di bulk!
    r_cut=5.0,           # Raggio di taglio in Angstrom (5.0 è un buon punto di partenza)
    n_max=8,             # Numero di funzioni di base radiali (risoluzione radiale)
    l_max=6,             # Grado massimo delle armoniche sferiche (risoluzione angolare)
    sigma=0.5
    )

# Creiamo i descrittori per l'intera traiettoria.
# n_jobs=-1 usa tutti i core della tua CPU per parallelizzare il calcolo.
features = soap.create(traiettoria_completa, n_jobs=-1)
print(f"Matrice SOAP finale: {features.shape}")

end_time_soap = time.time()

#Diamo una stima del tempo impiegato per la lettura --------------------------------------------------
print(f"Calcolo terminato in {end_time_soap - start_time_soap:.1f} secondi.")

# Convertiamo in un array NumPy per facilitare il lavoro con il Machine Learning ---------------------
soap_array = np.array(features)

# =====================================================================
# 1. Caricamento Dati
# =====================================================================
soap_data = soap_array 

# =====================================================================
# 2. Selezione dell'Atomo e Preparazione Dati
# =====================================================================
# Scegliamo un atomo casuale (da 0 a 125)
#atomo_selezionato = np.random.randint(0, n_atoms)
atomo_selezionato = 42 
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
valori_kernel_lineare = prodotto_scalare / (norma_corrente * norma_successivo)

valori_kernel = valori_kernel_lineare ** 4
  

# =====================================================================
# 4. Visualizzazione Grafica
# =====================================================================
# Creiamo l'asse X (gli indici delle transizioni: 1->2, 2->3, ecc.)
asse_x = np.arange(1, len(valori_kernel) + 1)

fig, ax = plt.subplots(figsize=(12, 6))

# Plottiamo i dati grezzi come punti
#ax.scatter(asse_x[1000:1101], valori_kernel[1000:1101], color='royalblue', alpha=0.6, s=10, label='Transizione Frame (i $\\rightarrow$ i+1)')
ax.scatter(asse_x, valori_kernel, color='royalblue', alpha=0.6, s=10, label='Transizione Frame (i $\\rightarrow$ i+1)')

# Formattazione del grafico
ax.set_title(f"Evoluzione Strutturale Locale (Atomo {atomo_selezionato})\nMisurata tramite Kernel SOAP tra Frame Successivi", fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("Indice di Transizione (Frame)", fontsize=12)
ax.set_ylabel(r"Valore del Kernel $K(p_i, p_{i+1})$", fontsize=12)

# La similarità del coseno va da -1 a 1, ma per descrittori postivi è solitamente tra 0 e 1.
# Mettiamo un limite superiore a 1.01 per questioni estetiche.
ax.set_ylim(bottom=np.min(valori_kernel)*0.995, top=1.005) 
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='lower right')

plt.tight_layout()

plt.savefig('kernel-4-qe-goodsoap22.png', dpi=200, bbox_inches='tight', transparent=True)

plt.show()
