# Generierung pragmatischer Bildbeschreibungen auf Wortebene:
## Ein Balanceakt zwischen Qualität und Informativität

## Implementierung für die Modulprüfung in "Methoden der angewandten Computerlinguistik"


### Beschreibung  

In diesem Git-Repository befindet sich die Implementierung zur Hausarbeit "Generierung pragmatischer Bildbeschreibungen auf Wortebene: Ein Balanceakt zwischen Qualität und Informativität" im Modul "Methoden der angewandten Computerlinguistik" mitsamt generierter Bildbeschreibungen und Evaluationsergebnisse. 



### Ausführung des Codes

1. Klonen, Virtuelles Umgebung erstellen und Abhängikeiten installieren

```
git clone https://github.com/rmokady/CLIP_prefix_caption && cd CLIP_prefix_caption
conda env create -f environment.yml
conda activate clip_prefix_caption
```
2. Einen "data"-Ordner erstellen, das Abstract-Scenes Dataset von https://www.microsoft.com/en-ph/download/details.aspx?id=52035 herunterladen und in den Ordner entpacken 

3. Das Skript "preproc_dataset.py" ausführen, um das Dataset vorzuverarbeiten

4. Das Skript "hyperparameter_tuning.py" oder "test.py" ausführen 



### Annerkennung

Der Code ist eine in Teilen stark abgeänderte Version des Repositorys https://github.com/rmokady/CLIP_prefix_caption für das Paper "ClipCap: CLIP Prefix for Image Captioning" (Mokady, Hertz und Bermano 2021). Außerdem beinhaltet das Repository eine leicht abgeänderte Version des Repositorys https://github.com/ramavedantam/cider für die CIDEr-Metrik zur Messung der Qualität von Bildbeschreibungen. 



