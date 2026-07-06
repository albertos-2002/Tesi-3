from ase.io import read

# Leggiamo il frame incriminato
atomi = read("../test_data.extxyz", index=0)

print("=" * 50)
print("   DIAGNOSTICA FILE EXTXYZ")
print("=" * 50)
print(f"Numero di atomi letti: {len(atomi)}")
print(f"Cella geometrica (Unit Cell):\n{atomi.get_cell()}")
print("\nPosizioni dei primi 10 atomi:")
print(atomi.get_positions()[:10])
print("\nElementi chimici dei primi 10 atomi:")
print(atomi.get_chemical_symbols()[:10])
print("=" * 50)
