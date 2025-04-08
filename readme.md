# !!! DISCLAIMER !!!
Il repository è ancora in fase di lavorazione quindi in caso di errori è consigliato segnalarlo nella sezione issues sperando che un santo arrivi e faccia una pull request.
La parte sull'allenamento è attualmente molto grezza (esempio di una porcheria): alcuni parametri importanti per l'allenamento non vengono inseriti ad ogni allenamento nella riga di comando quando si avvia il file, ma vengono invece settati nel codice
in maniera statica.
Serimentate, se ci sono parametri che necessitano di essere settati prima di ogni training e quindi di essere definiti all'avvio dell'allenamnto da riga di comando, fate issue e pull request grazie

## Installa dipendenze:
```bash 
pip install -r requirements.txt
```
# !!! Questo repository permette l'avvio e il training di un modello usando la GPU, è necessario scaricare CUDA 12.6 !!!
Link per installare CUDA:
https://developer.nvidia.com/cuda-12-6-0-download-archive


## Installa il modello pre-trained:
!!! Discaimer !!!
Il modello può essere cambiato ma è sconsigliato poichè è testato solo con gemma-2-2b-it
```bash
huggingface-cli download google/gemma-2-2b-it  --local-dir ./models/base/gemma-2-2b-it
```

---------------------------------------------------------------

## Dataset:
Fornire un file in formato txt all'interno della cartella /dataset con il nome "raw_dataset.txt", all'interno deve contenere degli esempi di risposte (Dopo ogni risposta deve andare a capo) e avviare dataset_generator.py.
Verrà generato un file dentro la cartella /dataset chiamato "fine_tune_dataset.jsonl", che sarà quello da utilizzare per l'allenamento.

## Training
Il modello può essere allenato con il seguente comando:
```bash
python fine_tune.py <model_path> <fine_tuned_model_name> <dataset_name> <epochs>
```
1. < model_path >
ES: Se si vuole allenare il modello scaricato di default ((gemma-2-2b-it)) e sono stati eseguiti TUTTI i passaggi scritti in precedenza:
Come <model_path> va insierito: models/base/gemma-2-2b-it
2. < fine_tuned_model_name >
Si inserisca il nome da dare al modello dopo il fine-tunining, verrà salvato nel seguente path:
models/finetuned/*
3. < dataset_name >
Fornire il nome del dataset da utilizzare ovvero quello generato in precedenza (fine_tune_dataset.jsonl)
4. < epochs >
Numero di epoche da usare nel training (int)

### Dopo ogni training, nel path del modello fine-tunato (models/finetuned/{NOME_MODELLO}) troverete un file chiamato benchmarks.png, è un grafico che mostra l'andamento del loss in funzione delle interazioni.

## Come avviare un modello:
Per avviare un modello bisogna eseguire il seguente comando:
```bash
python run_model.py <model_path> [max_tokens]
```
1. < model_path >
Tutti i modelli scaricati si trovano in models/base/*.<br>
ES: Se si vuole avviare il modello scaricato di default pre-trained ((gemma-2-2b-it)) e sono stati eseguiti TUTTI i passaggi scritti in precedenza:
Come <model_path> va insierito: models/base/gemma-2-2b-it<br>
ES: Se si vuole avviare il modello fine-tunato e sono stati eseguiti TUTTI i passaggi scritti in precedenza:
Come <model_path> va inserito: models/finetuned/{NOME_MODELLO}
2. < max_tokens >
Parametro che deterimina la lunghezza massima di token che possono essere generati dalle risposte di un modello

### Il modello verrà avviato e risponderà ai prompt presenti nel seguente file "test/questions.txt".
### Le risposte vengono salvate rispettivamente
1. "test/base/%Y-%m-%d_%H-%M-%S.json" se si avvia un modello base.
2. "test/finetuned/%Y-%m-%d_%H-%M-%S.json" se si avvia un modello fine-tunato.