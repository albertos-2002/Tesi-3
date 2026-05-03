# Brief explanation of what we want to achive

Il risultato che vogliamo ottenere è un modello di ML che sia in grado di fare simulazioni fisiche di materiali.

1) abbiamo definito la il materiale e la sua struttura cristallina

2) usiamo qunatum espresso per generare la struttura del materiale secondo la teoria del DFT e i relativi potenziali

> we start here

3) usiamo ASE per tradurre in python la struttura fisica

4) usiamo DSCRIBE per generare il SOAP

vogliamo generare un SOAP per ottenere un ogetto che non sia dipendente dalla rappresentazione, in questo modo invece di insegnare al modello di ML cosa succede in un SdR predefinito gli insegniamo la fisica del comportamento relativo tra gli oggetti che consideriamo