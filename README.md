# Wie wird inference.ipynb gestartet?
Hierfür ist ein Rechner mit GPU und installierten CUDA Treibern notwendig.

Um die Jupyter Notebooks starten zu können müssen zudem einige Python packages installiert werden. 
Diese wurden in der m-p-python-req.txt Datei mit aktiviertem venv und 
```bash
pip freeze > requirements.txt
```
festgehalten.

Um diese zu laden muss ein neues virtuelles Envirmonment erstellt und der folgende Command ausgeführt werden:
```bash
pip install -r m-p-python-req.txt
```

Nun müssen noch die absoluten Pfade im Notebook angepasst werden. 
Anschließend sollte das Inference Notebook ausführbar sein. 

Fragen können gerne an armin.straller@hs-augsburg.de gestellt werden. 

## Weitere Infos
- Eine GPU und CUDA sind notwendig für die Ausführung der Notbooks und Python Skripte
- Die Dokumentation liegt als pdf im Ordner /documentation
- Die Inference verwendet bereits gerasterte Szenarien. Diese liegen im Ordner /data/validation
- Link zur HS Cloud https://cloud.hs-augsburg.de/s/ZTYk9wJZBa5Xmw7
