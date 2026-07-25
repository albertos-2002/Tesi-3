from functionFile import setConfig

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import neighbor_list
from ase.units import Bohr, Hartree

class class_zbl(Calculator):

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, **kwargs): #------------------------------------------------------------------------------------
        super().__init__(**kwargs)

        self.coefficients = np.array([0.1818, 0.5099, 0.2802, 0.02817], dtype=float)
        self.exponents = np.array([3.2, 0.9423, 0.4029, 0.2016], dtype=float)

        self.raggio_inner = 0.0
        self.raggio_outer = 0.0

        self.checkandsetParameters()
    
        if setConfig.DEBUG:
            print("Class ZBL created")

    #controlla che i parametri siano definiti positivi
    #copia i parametri in variabili locali
    def checkandsetParameters(self): #-----------------------------------------------------------------------
        if setConfig.RAGGIO_INNER <= 0.0:
            raise ValueError("r_inner deve essere positivo.")
        else:
            self.raggio_inner = setConfig.RAGGIO_INNER
            
        if setConfig.RAGGIO_OUTER <= self.raggio_inner:
            raise ValueError("Deve essere r_outer > r_inner.")
        else:
            self.raggio_outer = setConfig.RAGGIO_OUTER

    def modParameters(self): #-------------------------------------------------------------------------------
        print("Not implemented")

    #implementa la funzione di switch per il potenziale
    #implementa la derivata della funzione di switch
    def switch(self, distances): #---------------------------------------------------------------------------
        #lista delle distanze entro un cutoff
        distances = np.asarray(distances)
        
        #questo pone il potenziale al velore massimo dentro il raggio_inner
        switch = np.ones_like(distances)
        dswitch_dr = np.zeros_like(distances) #derivata della funzione di switch
        
        #determina quali sono entro lo switch
        transition = (distances > self.raggio_inner) & (distances < self.raggio_outer)
        t = (distances[transition] - self.raggio_inner) / (self.raggio_outer - self.raggio_inner)

        #nella zona di switch viene scalato con il polinomio
        switch[transition] = 1.0 - 10.0 * t**3 + 15.0 * t**4 - 6.0 * t**5
        dswitch_dt = -30.0 * t**2 + 60.0 * t**3 - 30.0 * t**4
        dswitch_dr[transition] = dswitch_dt / (self.raggio_outer - self.raggio_inner)
    
        #fuori dal raggio_outer non ci deve essere potenziale
        outside = distances >= self.raggio_outer
        switch[outside] = 0.0
        dswitch_dr[outside] = 0.0

        return switch, dswitch_dr

    #implementa il vero e proprio calcolatore

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        natoms = len(self.atoms)
        atomic_numbers = self.atoms.numbers.astype(float) #estrae il numero atomico
        
        #Calcolo delle distanze | gestito da ASE
        i, j, r, displacement = neighbor_list("ijdD", self.atoms, self.raggio_outer, self_interaction=False)

        #Se non ci sono atomi dentro raggio_outer conclude il calcolo
        if len(r) == 0:
            energy = 0.0
            forces = np.zeros((natoms, 3))
            self.results = {"energy": energy, "free_energy": energy, "forces": forces}
            return
        
        #Controllo che non ci sia sovrapposizione
        if np.any(r < 1.0e-12):
            raise FloatingPointError("Due atomi sono praticamente sovrapposti.")

        Zi = atomic_numbers[i]
        Zj = atomic_numbers[j]
        # Raggio di schermatura a_ij (in unità atomiche Bohr)
        aij = 0.8854 * Bohr / (Zi**0.23 + Zj**0.23)
        x = r / aij

        # Funzione di schermatura phi(x) = sum_k (c_k * exp(-d_k * x))
        exponentials = np.exp(-x[:, None] * self.exponents[None, :])
        phi = np.sum(self.coefficients[None, :] * exponentials, axis=1)
        # Derivata della funzione di schermatura dphi/dx
        dphi_dx = -np.sum((self.coefficients * self.exponents)[None, :] * exponentials, axis=1)
        
        # Prefattore coulombiano (convertito nelle unità di misura di ASE)
        prefactor = Hartree * Bohr * Zi * Zj
        # Energia ZBL non smorzata per ciascuna coppia
        zbl_energy = prefactor * phi / r
        # Derivata analitica dell'energia ZBL pura rispetto a r
        dzbl_dr = prefactor * (dphi_dx / (aij * r) - phi / r**2)

        switch, dswitch_dr = self.switch(r)

        # Energia effettiva smorzata per ogni coppia: E_pair = S(r) * U_ZBL(r)
        pair_energy = switch * zbl_energy
        # Derivata dell'energia smorzata: dE_pair/dr = S(r) * dU_ZBL/dr + dS/dr * U_ZBL(r)
        dpair_dr = switch * dzbl_dr + dswitch_dr * zbl_energy

        pair_forces = (dpair_dr / r)[:, None] * displacement

        forces = np.zeros((natoms, 3))
        for component in range(3):
            forces[:, component] = np.bincount(i, weights=pair_forces[:, component], minlength=natoms)

        energy = 0.5 * np.sum(pair_energy)
        # Salvataggio nel dizionario interno di ASE
        self.results = {
            "energy": energy, 
            "free_energy": energy, 
            "forces": forces
        }
