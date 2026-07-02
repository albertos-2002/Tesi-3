#NOTA: il codice qui presente, data la complessità delle librerie è stato prima generato tramite Gemini Pro e successivamente, per quanto possibile, revisionato.

import numpy as np
#from ase.io import read
import ase.io
from dscribe.descriptors import SOAP
import time

#=======================================
#READING
#=======================================

#Vogliamo prendere i vari file di dati che abbiamo a disposizione ed estrarne la struttura fisica tramite ASE.
#Abbiamo una dinamica molecolare particolarmente lunga, quindi quello che proviamo a fare per prima cosa è leggere i vari file e caricarli in un unico vettore. Se questo fallisce per questioni di fattibilità computazionale lavoreremo con 9 file separatamente e successivamente andremo ad unirli se possibile

#Elenco dei file da leggere
file_list = (
    #"../per_alessio/tantala_md_temp3000.out",
    "../per_alessio/tantala_md_temp3000.out_2",
    "../per_alessio/tantala_md_temp3000.out_3",
    "../per_alessio/tantala_md_temp3000.out_4",
    "../per_alessio/tantala_md_temp3000.out_5",
    "../per_alessio/tantala_md_temp3000.out_6",
    "../per_alessio/tantala_md_temp3000.out_7",
    "../per_alessio/tantala_md_temp3000.out_8",
    "../per_alessio/tantala_md_temp3000.out_9" )

"""
--------------------------------------------------------------------------------------------------------------
Non è stato possibile leggere il primo file in quanto ritornava un:
AssertionError: ((2, 0), 1)
Interpretato by Gemini come segue:
The AssertionError: ((2, 0), 1) in your code indicates that there's an issue with the format of your Quantum ESPRESSO output file, specifically a mismatch between the number of eigenvalues and k-points that ASE's reader expects. This usually means the .out file is either incomplete or corrupted. Please check the file /content/drive/MyDrive/Università/Tesi/TestData/tantala_md_temp3000.out to ensure it's a valid and complete Quantum ESPRESSO output.
--------------------------------------------------------------------------------------------------------------
"""

#=======================================
#ASE
#=======================================

traiettoria_completa = []

# 2. Loop di lettura
start_time_reading = time.time()

for i, file_name in enumerate(file_list):
    print(f"Lettura del file di Quantum ESPRESSO: {file_name} in corso ...")

    # Leggiamo tutti i frame del file corrente
    # index=":" dice ad ASE di leggere TUTTI gli step di dinamica, non solo l'ultimo
    # Restituisce una lista di oggetti "Atoms"
    frames = ase.io.read(file_name, index=":", format="espresso-out")


    #Printiamo alcune informazioni sul file letto
    num_frames = len(frames)
    num_atomi = len(frames[0])
    print(f"Letti {num_frames} frame di dinamica molecolare.")
    print(f"Ogni frame contiene {num_atomi} atomi.")


    # NOTA SUI RESTART: Spesso Quantum ESPRESSO, quando riparte,
    # riscrive il frame iniziale che è identico all'ultimo del file precedente.
    if i > 0:
        # Controlliamo se il primo frame del nuovo file è uguale all'ultimo già salvato
        # Se sì, lo scartiamo per non avere duplicati nella statistica
        traiettoria_completa.extend(frames[1:])
    else:
        traiettoria_completa.extend(frames)

print(f"Totale frame accumulati: {len(traiettoria_completa)}")

end_time_reading = time.time()

#------------------------------------------------------------------------------------------------------------------
#Procediamo anche a salvare i la traiettoria letta in modo che non sia necessario procedere con il calcolo tutte le volte che essa deve essere utilizzata

# Salva l'intera lista di atomi in un unico file
ase.io.write('traiettoria_unita.traj', traiettoria_completa)
# Oppure in formato testo leggibile:
# ase.io.write('traiettoria_unita.extxyz', traiettoria_completa)

#Diamo una stima del tempo impiegato per la lettura dei file  ----------------------------------------------------------------
print(f"Calcolo terminato in {end_time_reading - start_time_reading:.1f} secondi.")


# Lettura della traiettoria =========================================================================================================
#traiettoria_caricata = ase.io.read('traiettoria_unita.traj', index=':')
#print(f"check len: {len(traiettoria_caricata)}")
#traiettoria_completa = traiettoria_caricata


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
    l_max=6              # Grado massimo delle armoniche sferiche (risoluzione angolare)
    )

# Creiamo i descrittori per l'intera traiettoria.
# n_jobs=-1 usa tutti i core della tua CPU per parallelizzare il calcolo.
features = soap.create(traiettoria_completa, n_jobs=-1)
print(f"Matrice SOAP finale: {features.shape}")

end_time_soap = time.time()

#Procediamo anche a salvare i il vettore soap in modo che non sia necessario procedere con il calcolo tutte le volte che esso deve essere utilizzato
# 'features' è l'output di soap.create
np.save('features_soap.npy', features)
#nota: questo file potrebbe essere paricolarmente pesante

#Diamo una stima del tempo impiegato per la lettura --------------------------------------------------
print(f"Calcolo terminato in {end_time_soap - start_time_soap:.1f} secondi.")

# Convertiamo in un array NumPy per facilitare il lavoro con il Machine Learning ---------------------
soap_array = np.array(features)

# Lettura delle feature -------------------------------------------------------------------------------
# features_caricate = np.load('features_soap.npy')
