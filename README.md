# Zastosowanie algorytmów rankingowych do wyboru najlepszego klasyfikatora w systemach wykrywania intruzji na podstawie zbioru danych NSL-KDD przy wykorzystaniu różnych metryk oceny.

Systemy wykrywania intruzji (IDS) odgrywają kluczową rolę w ochronie sieci przed atakami i nieautoryzowanym dostępem. Współczesne IDS muszą przetwarzać miliony pakietów danych, co stawia wysokie wymagania w zakresie precyzji i szybkości detekcji anomalii. W związku z tym kluczowe staje się opracowanie efektywnych metod oceny i wyboru klasyfikatorów, które najlepiej spełniają wymagania konkretnego systemu IDS.
W ramach tego projektu wykorzystamy zmodyfikowany zbiór danych NSL-KDD oraz algorytmy rankingowe, takie jak PROMETHEE, VIKOR i AHP. Celem jest ocena skuteczności różnych klasyfikatorów w systemach IDS na podstawie wybranych metryk, takich jak dokładność, F1-score, czy czas obliczeń, a także wyznaczenie najlepszego klasyfikatora do wykrywania intruzji w określonym środowisku.

## Przyjęta metodyka

* Wybór zbioru danych - W projekcie wykorzystamy zbiór NSL-KDD, który zawiera oznaczenia ataków i poziom trudności klasyfikacji.
* Klasyfikacja - Stosujemy różne rodzaje klasyfikatorów, aby uzyskać metryki, takie jak dokładność, F1-score, i czas obliczeń, które posłużą do dalszej analizy przy użyciu algorytmów rankingowych.
* Ranking klasyfikatorów - Wyniki klasyfikatorów zostaną ocenione za pomocą algorytmów rankingowych, takich jak PROMETHEE, VIKOR oraz AHP.
* Analiza wyników - Obliczenia algorytmów rankingowych zostaną przeprowadzone w środowisku Python, co pozwoli na rangowanie klasyfikatorów pod kątem ich efektywności i przydatności w systemach IDS.


## Pobranie repozytorium

```
git clone https://github.com/michalsciubisz/IDS_project.git
```

## Instalacja bibliotek

```
pip install -r requirements.txt
```

## Istotne linki:

[Artykuł referencyjny](https://www.researchgate.net/publication/269399129_TOPSIS_Based_Multi-Criteria_Decision_Making_of_Feature_Selection_Techniques_for_Network_Traffic_Dataset )

[Baza danych](https://www.kaggle.com/datasets/hassan06/nslkdd/data )
