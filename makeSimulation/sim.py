from functionFile import setConfig
from functionFile import classSimulation
from functionFile import classPerturbation
from functionFile import classCalcDataFile
from functionFile import classGraph
from functionFile import classMontecarlo

def main():

    print("Setting dei parametri...")
    setConfig.carica_parametri_globali("param.config")

    print("Run della simulazione...")
    #i parametri decidono che tipo di simulazione deve essere svolta
    if setConfig.NVE_SIMULATION:
        sim = classSimulation.class_makeSimulation()
        sim.runSimulation()        
    
    if setConfig.RANDOMWALK_FIXED:
        sim = classPerturbation.class_makePerturbation()
        sim.runPerturbation()
    
    if setConfig.RANDOMWALK:
        sim = classPerturbation.class_makePerturbation()
        sim.runPerturbationPath()
    
    if setConfig.MONTECARLO_METROPOLIS:
        sim = classMontecarlo.class_makeMontecarlo()
        sim.runMetropolisMC()


    print("Caricamento dati...")
    if not setConfig.SKIP_CALC:
        cdf = classCalcDataFile.class_calcDataFile()
        cdf.calcData()

    print("Generazione dei grafici finali...")
    graph = classGraph.class_makeGraph()
    graph.makeGraph()

    print("Tutto completato! Simulazione eseguita ed i grafici sono stati generati con successo.")
    
if __name__ == "__main__":
    main()
