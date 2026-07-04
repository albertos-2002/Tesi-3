import numpy as np
from ase.io import read, write

print("CARICAMENTO E DIVISIONE DEL DATASET (TRAIN / TEST)")

traj_completa = read("traiettoria_unita.traj", index=":")

n_totale = len(traj_completa)
meta = n_totale // 2

# Divisione: prima metà per addestrare, seconda per testare
train_set = traj_completa[:meta]
test_set = traj_completa[meta:]

print(f"Totale frames: {n_totale}")
print(f"Frames per il Training: {len(train_set)}")
print(f"Frames per il Test: {len(test_set)}")

# Salviamo i file nel formato .extxyz, che è il formato nativo richiesto da GAP
write("train_data.extxyz", train_set)
write("test_data.extxyz", test_set)


#Divisione della matrice di soap
soap_complete = np.load("features_soap.npy")
soap_train, soap_test = soap_complete[:meta], soap_complete[meta:]

print(f"Shape blocco Train: {soap_train.shape}")
print(f"Shape blocco Test: {soap_test.shape}")

np.save("soap_train.npy", soap_train)
np.save("soap_test.npy", soap_test)
print("File SOAP salvati con successo!")


#Scrittura delle traiettorie con numeri atomici

for atoms in train_set:
    # Creiamo un array esplicito di stringhe con i simboli chimici
    # e lo salviamo come array di informazioni aggiuntive dell'atomo.
    # Questo assicura che ASE crei la formattazione corretta "species:S:1" nel file XYZ.
    atoms.arrays['species'] = atoms.get_chemical_symbols()

# 2. Salviamo in formato extxyz forzando la sintassi compatibile con QUIP
write("train_data_natom.extxyz", train_set, format="extxyz")
print("File salvato correttamente con i simboli chimici formattati per QUIP!")


for atoms in test_set:
    # Creiamo un array esplicito di stringhe con i simboli chimici
    # e lo salviamo come array di informazioni aggiuntive dell'atomo.
    # Questo assicura che ASE crei la formattazione corretta "species:S:1" nel file XYZ.
    atoms.arrays['species'] = atoms.get_chemical_symbols()

# 2. Salviamo in formato extxyz forzando la sintassi compatibile con QUIP
write("train_test_natom.extxyz", test_set, format="extxyz")
print("File salvato correttamente con i simboli chimici formattati per QUIP!")

