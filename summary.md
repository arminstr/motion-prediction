# Zusammenfassung

## Positiv:
* Ansatz zeigt erste vielversprechende Ergebnisse
* Convolutional LSTMs sind für die Bewegungsvorhersage auf basis von gerasterten Objektinformationen grundlegend geeignet

## Negativ:
* Durch inkonsequente einfärbung der Trainingsdaten ist es zum aktuellen Zeitpunkt schwer mehrfach Prediktionen durchzuführen.
* Die Kartendaten hinterlassen Fragmente bei der Vorhersage. Diese können vermutlich durch die Bearbeitung der Loss Funktion beeinflusst und reduziert werden.
* Aktuell ist die Nachverarbeitung der Prediktion noch nicht weit genug ausgereift um zuverlässig Prediktionen für weitere Schritte zu ermöglichen.
* Training auf einer Größeren GPU ermöglicht größere Fenster für die Convolutional Layers. Dies könnte einer Verbesserung bei Verschiedenen Bewegungsrichtungen von Fahrzeugen bewirken.

## Lessons Learned:
* ConvLSTMs benötigen viel Grafikspeicher
* Gradient Accumulation ermöglicht den Speicherbedarf für Zwischenergebnisse auf der GPU zu verringern
* Der Rechenaufwand für die Rasterisierung der Daten sollte nicht unterschätzt werden
* Die Inference Zeit ist deutlich schlechter als bei einer Bild-Klassifikation. Dies liegt an der größe des Netzwerks und der Menge der Eingangsdaten. Dies sollte für die praktische Verwendbarkeit des ConvLSTM+Raster Ansatzes berücksichtigt werden. 
* Durch höhere Auflösung des Rasters könnten Objekte besser unterschieden und unabhängig voneinander prediziert werden. Dies führt allerdings zu einer geringeren betrachtbaren Kartengröße oder zu einem größeren und langsameren Netzwerk.
