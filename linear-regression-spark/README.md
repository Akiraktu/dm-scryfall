## Kartenpreisvorhersage mit random Forest Regression

## Einleitung
In diesem Projekt versuchen wir, durch die spielrelevanten Merkmale einer Magic the Gathering Karte, den Preis der Karte vorherzusagen. 
Als Modell verwenden wir eine Random Forest Regression. Das Projekt demonstriert den kompletten Workflow eines Machine Learning Modells: von der Datenbeschaffung, über die Datenaufbereitung und das Feature Engineering, bis hin zur Modellerstellung und -evaluierung.

Unsere Datenquelle ist das Oracle Dataset von [Scryfall](https://scryfall.com/docs/api/bulk-data), das eine umfassende Sammlung von Magic the Gathering Karteninformationen in Form einer JSON-Datei bietet.

## Feature Engineering

### Verarbeitung der umgewandelten Manakosten (processCmc)
Die umgewandelten Manakosten (cmc) sind ein wichtiger Prädiktor, können aber bei Karten wie Ländern fehlen. Die Funktion processCmc löst dieses Problem, indem sie zuerst sicherstellt, dass die Spalte vom numerischen Typ Double ist, um Berechnungsfehler zu vermeiden. Anschließend berechnet sie den durchschnittlichen cmc-Wert des gesamten Datensatzes und füllt damit alle fehlenden Einträge auf.

### Verarbeitung von Stärke und Widerstandskraft (processPowerToughness)
Die Stärke (power) und Widerstandskraft (toughness) einer Karte stellen eine Herausforderung dar, da sie als Strings gespeichert sind und nicht-numerische Werte (z.B. * oder X) enthalten können. Unsere Funktion processPowerToughness bewältigt dies durch eine benutzerdefinierte Transformation, die rein numerische Strings in Ganzzahlen umwandelt und alle anderen Werte auf null setzt. Danach berechnet sie den Durchschnitt für jedes Attribut und ersetzt alle null-Einträge damit. Dies liefert eine saubere, numerische Repräsentation für jede Karte.

### Aufbereitung der Farbmerkmale (processColors)
Die Farbe einer Karte ist ein entscheidendes kategorisches Merkmal. In den Rohdaten wird sie jedoch in Arrays gespeichert, die für eine Regression ungeeignet sind. Die Funktion processColors löst dieses Problem durch Ersetzen der kategorischen Werte mit effektiv numerischen (booleans). Sie erzeugt für jede der fünf Manafarben (W, U, B, R, G) eine eigene boolesche Spalte, die sowohl aus dem Feld colors als auch aus color_identity abgeleitet wird. Um einen wichtigen Kartentyp explizit zu erfassen, fügt die Funktion außerdem eine is_colorless-Flag hinzu. Diese wird auf true gesetzt, wenn eine Karte keine Farben hat, wodurch sichergestellt wird, dass dieses kritische Attribut nicht verloren geht.

### Verarbeitung des Edhrec Rankings (processEdhrecRank)
Das Edhrec Ranking ist bereits ein Integer, da jedoch nicht alle Karten ein Ranking haben müssen wir uns noch um fehlende Werte kümmern. Die Funktion setzt bei diesen Reihen die Spalte auf 0, um eine Null-Value zu repräsentieren.

### Verarbeitung der Schlüsselwörter (processKeywords)
Die Schlüsselwörter (Keywords) werden als String Array gespeichert. Um diese richtig einordnen zu können werden zuerst alle verfügbaren Schlüsselwörter mit der Hilfsmethode (countValuesInColumn) aus dem Dataframe extrahiert und nach deren Vorkommen sortiert. Da es teilweise auch Schlüsselwörter gibt, welche nur sehr selten vorkommen werden nur die ersten 50 (getTopValues) in neue Spalten umgewandelt und anschließend auf true/false gesetzt, sofern eine Karte dises Schlüsselwort besitzt. Dies deckt ungefähr alle Schlüsselwörter ab, welche 50 Mal oder öfters vorkommen.

### Verarbeitung der Typenzeile (processTypeLine)
Die Typenzeile (type_line) ähnelt der Verarbeitung der Schlüsselwörter. Auch hier besitzt jede Karte eine Liste von Wörtern, welche relevant für die Art der Karte sind. Die Herausforderung hier besteht darin, dass diese Typen in einem String gespeichert sind und zuerst extrahiert und in einer neuen Spalte gespeichert werden müssen. 
Anschließend kann mit dieser neuen Spalte genau wie bei den Schlüsselwörtern weiter gemacht werden. Es werden wieder alle Typen, welche 50 Mal oder öfters vorkommen (dies entspricht hier den top 120 Typen) in eine eigene Spalte umgewandelt und mit true/false befüllt.

### Verarbeitung der Legalitäten (processLegalities)
Die Legalitäten (legalities) sind in einer Map gespeichert, welches das Format als Key und den Legalitätsstatus als Value besitzt. Um mit den Legalitäten arbeiten zu können, werden zuerst die einzelnen Schlüssel (Formate) extrahiert und anschließend für jedes Format eine eigene Spalte erstellt. Der Wert in der Map ist ein String "legal" oder "not_legal", welcher in einen Boolean umgewandelt und zum setzten der neuen Spalten genutzt wird.

### Verarbeitung der Raritäten (processRarity)
Die Raritäten (rarity) können nicht NULL sein. Daher muss hier nur der als String vorliegende Wert in eine für den Random Forest Regression geeignete Form gebracht werden.
Hierzu wird zuerst ein StringIndexer angewendet, welcher die Raritäten in numerische Werte umwandelt. Da es sich hier um ein nominales Attribut handelt, wird anschließend ein OneHotEncoder angewendet, welcher die numerischen Werte in eine binäre Vektor-Darstellung umwandelt.

## Fazit

Unsere vorhergesagten Preise weichen um ca. 2,93€ von den tatsächlichen Preisen ab. Die Genauigkeit unseres Modells bei simpleren Karten, mit wenig oder keinem Orakeltext, ist erstaunlich gut.
Für die meisten Karten des unteren Preispektrums ist der vorhergesagte Preis deutlich niedriger als der RMSE von 2,93 €. Bei höherpreisigen Karten, welche entsprechend auch meistens einen komplexeren Orakeltext vorweisen bzw. eine höhere Quantität an Keywords und Typen haben, ist die Vorhersage weniger akkurat, was wir zu Anfangs auch so prognostiziert haben.

Mögliche nächste Schritte, um die Vorhersagegenauigkeit zu verbessern, wären:
- Analyse der Orakeltexte, die wir in dieser Trainingsinstanz aufgrund extrem hoher Komplexität komplett ausgelassen haben
- Nutzen von anderen Modellen als Random Forest Regression
- Optimierung der Hyperparameter des Random Forest Modells
