from functionFile import setConfig
from functionFile import classSimulation
from functionFile import classCalcDataFile
from functionFile import classGraph


def main():

    print("Setting dei parametri...")
    setConfig.carica_parametri_globali("param.config")

    print("Run della simulazione...")
    sim = classSimulation.class_makeSimulation()
    sim.runSimulation()

    print("Caricamento dati...")
    cdf = classCalcDataFile.class_calcDataFile()
    cdf.calcData()

    print("Generazione dei grafici finali...")
    graph = classGraph.class_makeGraph()
    graph.makeGraph()

    print("Tutto completato! Simulazione eseguita ed i grafici sono stati generati con successo.")
    
if __name__ == "__main__":
    main()
