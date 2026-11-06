# Template Presentazione Unipd (Beamer - Tema Scuro)

Un template moderno e pulito per diapositive in LaTeX/Beamer con la veste grafica ufficiale dell'**Università degli Studi di Padova**. 

Il tema è stato ottimizzato integrando tutti i sotto-moduli in un unico file di stile (`beamerthemeUnipd.sty`), risolvendo i problemi di allineamento e garantendo la piena compatibilità con il formato 16:9 e il tema scuro.

---

## 📁 Struttura dei File

Per il corretto funzionamento, organizza i file del progetto nella seguente struttura:

```text
/ (Directory principale del progetto)
├── main.tex                 # Il tuo file LaTeX principale
├── beamerthemeUnipd.sty     # File unico di stile del tema
├── README.md                # Questo file di documentazione
└── images/                  # Cartella contenente le immagini e i loghi
    └── unipd_logo.png       # Logo ufficiale dell'Ateneo
```

---

## 🚀 Guida Rapida all'Uso

### 1. Inizializzazione del Documento
Nel tuo file `main.tex`, imposta la classe `beamer` (consigliato in 16:9) e carica il tema `Unipd`:

```latex
\documentclass[aspectratio=169]{beamer}

% Specificare il percorso delle immagini
\graphicspath{{./images/}}

% Caricamento del Tema Unipd
\usetheme[pageofpages=di]{Unipd}

\title[Titolo Breve]{Titolo Esteso della Tesi o Presentazione}
\subtitle{Sottotitolo o Corso di Laurea}
\author[Cognome]{Relatore: Prof. Nome Cognome \and Laureando: Nome Cognome}
\date{Anno Accademico 2025/2026}

\begin{document}

% Prima pagina (Copertina)
\begin{frame}[plain, noframenumbering]
    \titlepage
\end{frame}

% Diapositiva di contenuto
\begin{frame}{Titolo della Slide}
    Contenuto della diapositiva...
\end{frame}

\end{document}
```

---

## ⚙️ Opzioni del Tema (`\usetheme[...]`)

Puoi personalizzare il comportamento del tema passando i seguenti parametri all'interno delle parentesi quadre di `\usetheme`:

| Opzione | Descrizione | Valore Predefinito | Esempio |
| :--- | :--- | :--- | :--- |
| **`pageofpages`** | Testo separatore per il conteggio pagine nel piè di pagina. | `of` | `pageofpages=di` |
| **`logo`** | Nome dell'immagine per un **secondo logo** (es. Dipartimento) da affiancare a quello Unipd nella copertina. | *(vuoto)* | `logo=logo_dipartimento.png` |
| **`logoxshift`** | Spostamento orizzontale del logo nell'intestazione delle slide interne. | `0pt` | `logoxshift=-2mm` *(a sx)* o `3mm` *(a dx)* |
| **`logoyshift`** | Spostamento verticale del logo nell'intestazione delle slide interne. | `0pt` | `logoyshift=-1mm` *(in basso)* o `2mm` *(in alto)* |

### Esempio con tutte le opzioni attive:
```latex
\usetheme[
    pageofpages=di,
    logo=logo_dipartimento.png,
    logoxshift=-1mm,
    logoyshift=-0.5mm
]{Unipd}
```

---

## 🎨 Caratteristiche Grafiche e Comandi Speciali

* **Sfondo Scuro (`nice_dark_bg`):** Sfondo ad alto contrasto ottimizzato per schermi e proiettori.
* **Rosso Unipd (`red_unipd` e `darker_red`):** Utilizzato per la prima pagina, la barra del titolo e gli elementi di rilievo.
* **Centratura Assoluta della Copertina:** Il logo e i titoli della prima pagina vengono posizionati esattamente al centro geometrico dello schermo tramite coordinate esterne (TikZ), ignorando i margini del testo.

### Slide di Separazione / Ringraziamenti (`emptyframe`)
È presente un ambiente speciale per creare diapositive con sfondo completamente rosso, ideali per stacchi di sezione, domande o ringraziamenti finali:

```latex
\begin{emptyframe}
    Domande?
\end{emptyframe}
```

---

## 🔧 Risoluzione Problemi Frequenti

* **`File 'unipd_logo.png' not found`:** Verificare che l'immagine del logo sia presente nella cartella `images/` e che il nome rispetti le maiuscole/minuscole.
* **`File 'beamerthemeUnipd.sty' not found`:** Assicurarsi che il file `.sty` si trovi nella stessa cartella del file `main.tex`.