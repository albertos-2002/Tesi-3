# Template Presentazione Unipd (Modernizzato)

(Build by AI)

Questo template LaTeX per presentazioni (Beamer) mantiene intatto l'aspetto ufficiale del tema scuro dell'Università di Padova, ma ottimizza profondamente il codice sorgente, riunendolo in un unico file per facilitarne la gestione e prevenire errori di compilazione.

## Struttura della Cartella
Affinché il template compili correttamente, assicurati che la cartella sia strutturata così:

/ (Cartella Progetto)
├── main.tex                 # Il tuo file di presentazione principale
├── beamerthemeUnipd.sty     # Il file unico del tema (non va modificato)
├── README.md                # Questo file
└── images/                  # Cartella in cui inserire le immagini
    └── unipd_logo.png       # Logo obbligatorio per l'intestazione

## Come si usa
Nel file `main.tex`, il tema viene caricato tramite il comando:
`\usetheme[pageofpages=di]{Unipd}`

### Opzioni supportate:
* **pageofpages**: Sostituisce la parola tra il numero della slide corrente e il totale (es. `[pageofpages=di]` stampa "1 di 10").
* **logo**: Permette di affiancare un secondo logo (es. quello del dipartimento) al logo di ateneo nella prima pagina. Esempio d'uso: `\usetheme[logo=logo_dipartimento.png]{Unipd}`.

### Slide Speciali
È stato configurato un ambiente apposito per creare slide interamente rosse, perfette per le copertine o la slide finale di ringraziamento. Per usarlo:
```latex
\begin{emptyframe}
    Grazie per l'attenzione!
\end{emptyframe}
