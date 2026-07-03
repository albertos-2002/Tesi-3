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
