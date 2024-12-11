# Zastosowanie algorytmu TOPSIS do selekcji cech w systemach wykrywania intruzji na bazie zbioru danych NSL-KDD

Podstawowym problemem, który staramy się rozwiązać, jest redukcja czasu obliczeń w procesie detekcji intruzji, przy jednoczesnym zachowaniu akceptowalnej dokładności klasyfikacji. Wielowymiarowość danych oraz duża liczba cech zwiększają złożoność obliczeniową, co wpływa negatywnie na efektywność systemów IDS.
W projekcie zastosujemy algorytm TOPSIS, który umożliwia ocenę i wybór najbardziej efektywnych metod selekcji cech spośród różnych alternatyw. Algorytm TOPSIS polega na wyborze rozwiązania najbliższego rozwiązaniu idealnemu, z uwzględnieniem wielu atrybutów.

## Przyjęta metodyka

* Wybór zbioru danych - W projekcie wykorzystamy zbiór NSL-KDD, zbiór ten zawiera oznaczenia ataków i poziom trudności klasyfikacji.
* Klasyfikacja - Różne rodzaje klasyfikatorów zostaną wykorzystane, do uzyskania metryk pozwalających na dalszą analizę przy użyciu algorytmu TOPSIS.
* Selekcja cech - Zastosujemy dziesięć różnych technik selekcji cech, a wyniki zostaną ocenione za pomocą algorytmu TOPSIS.
* Analiza wyników - Wyniki TOPSIS zostaną obliczone w środowisku Python, co pozwoli na rangowanie technik selekcji cech pod kątem efektywności.

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
