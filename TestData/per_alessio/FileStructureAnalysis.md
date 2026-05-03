# Data file structure analysis

(gemini aided)

Usiamo come file di riferimento "tantala_md_temp3000.out_3" (è il file di minore dimensione)

72344 righe

questo file è una simulazione -> prendiamo un calcolo e lo ripetiamo per n istanti di tempo ottenendo quindi il comportamento del matriale per x secondi

- righe 1 - 45

	informazioni preliminari sul programma

- righe 49 - 117 

	?

- righe 121 - 249

	definisce le coordinate iniziali del cristallo

- righe 251 - 278

	?

- righe 280 - 556 (tag: Self-consistent Calculation)

	procede iterativamente fino alla convergenza alla SCC, in questo caso sono state necessarie 25 iterazioni

- righe 558 - 646

	da i risultati del calcolo per la parte di energia

- righe 648 - 777

	da i risultati del calcolo per le parte delle forze

- righe 779 - 911

	vengono fornite le nuove posizioni degli atomi dopo che il tempo viene fatto scorre per un quantiativo t, praticamente viene eseguita la parte che da il nome di dinamica alla dinamica molecolare

- righe 914 - 927

	?

- righe 929 - 1466

	ricomincia con le SCC, il printig di energie e forze e il calcolo delle nuove posizioni 


la struttura a blocchi viene ripetuta per tutto il resto del file