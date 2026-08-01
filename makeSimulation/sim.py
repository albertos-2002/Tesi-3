from functionFile import setConfig
from functionFile import classSimulation
from functionFile import classPerturbation
from functionFile import classCalcDataFile
from functionFile import classGraph
from functionFile import classMontecarlo
import gc

def main():

    print("Setting dei parametri...")
    setConfig.carica_parametri_globali("param.config")

    print("Run della simulazione...")
    #i parametri decidono che tipo di simulazione deve essere svolta
    if setConfig.NVE_SIMULATION:
        sim = classSimulation.class_makeSimulation()
        sim.runSimulation()
        print("Eliminazione classe in corso..........")
        del sim
        gc.collect()
    
    if setConfig.RANDOMWALK_FIXED:
        sim = classPerturbation.class_makePerturbation()
        sim.runPerturbation()
        print("Eliminazione classe in corso..........")
        del sim
        gc.collect()
    
    if setConfig.RANDOMWALK:
        sim = classPerturbation.class_makePerturbation()
        sim.runPerturbationPath()
        print("Eliminazione classe in corso..........")
        del sim
        gc.collect()
    
    if setConfig.MONTECARLO_METROPOLIS:
        sim = classMontecarlo.class_makeMontecarlo()
        sim.runMetropolisMC()
        print("Eliminazione classe in corso..........")
        del sim
        gc.collect()


    print("Caricamento dati...")
    if not setConfig.SKIP_CALC:
        cdf = classCalcDataFile.class_calcDataFile()
        cdf.calcData()
        print("Eliminazione classe in corso..........")
        del cdf
        gc.collect()

    print("Generazione dei grafici finali...")
    graph = classGraph.class_makeGraph()
    graph.makeGraph()

    if setConfig.KEEP_OPEN:
        input("Press any key to continue.........")

    print("Tutto completato! Simulazione eseguita ed i grafici sono stati generati con successo.")
    
if __name__ == "__main__":
    main()
