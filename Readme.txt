## Italiano ##

Informazioni preliminari:
In caso di problemi di requisiti è provvisto il file 'requirements.txt', si consiglia di creare un conda/venv
ove installare tutti i pacchetti necessari

Conda:
conda create --name torch --file requirements.txt
conda activate torch

venv:
virtualenv torch
source torch/bin/activate
python -m pip install -r requirements.txt
(non testato, si consiglia conda)

Per visualizzare l'output dei modelli proposti è disponibile lo script 'make_demo.py'
i parametri richiesti sono:

* [--source, -s] il path dell'immagine o video da processare
* [--device, -d] il device da utilizzare per il calcolo, 'cpu' o 'cuda' (default: cpu)
* [--net, -n] la rete da testare, 'fcn8' oppure 'our' (default: our)
* [--weights, -w] il path dei pesi trainati per la rete (Fondamentale)
* [--out, -o] il file di destinazione SENZA ESTENSIONE (di default l' estensione sarà la stessa del file in ingresso,
  default: out.<estensione ingresso>

 Esempio:
    make_demo.py -s gitaalmuseo.png -w ../pesi/our.pth -d cuda -o /home/pippo/Scrivania/videofighissimo

 Nota:
    Per quanto riguarda il video verà prodotto unicamente un video analogo con la visualizzazione delle classi
    ottenute dalla rete, per le immagini invece verrà prodotta anche una sotto cartella /pictures/ contenente
    immagini delle regiorni contenenti (si spera) quadri