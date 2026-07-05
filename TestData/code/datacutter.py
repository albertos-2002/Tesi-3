import numpy as np
from ase.io import read, write
from ase import Atoms

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

configs = train_set
nuove_configs = []

for atoms in configs:
    # Crea un nuovo oggetto usando i numeri atomici puri (Ta -> 73, O -> 8)
    nuovo = Atoms(
        numbers=atoms.get_atomic_numbers(),
        positions=atoms.get_positions(),
        cell=atoms.get_cell(),
        pbc=atoms.get_pbc()
    )
    if 'forces' in atoms.arrays:
        nuovo.arrays['forces'] = atoms.arrays['forces']
    nuovo.info.update(atoms.info)
    # Forza la colonna delle specie come numeri interi (Integer)
    nuovo.arrays['species'] = np.array(atoms.get_atomic_numbers(), dtype=np.int32)
    nuove_configs.append(nuovo)

# Salva il nuovo file
write("train_numeri.extxyz", nuove_configs, format="extxyz")
print("File 'train_numeri.extxyz' generato con successo!")

