# Snake

## Intro
Przykładowo, gdy idzie w prawo, może poruszyć się w górę, w dół, lub w dalej w prawo.
W dalszej części będziemy o kierunkach mówić względem węża -> (gdy porusza się w prawo, nasza góra jest jego lewą stroną). I `bezwzględnych` -> względem nas :D
Dlatego wąż, może poruszać się prosto, w lewo lub w prawo względem węża, ale też w górę, dół, prawo, lewo bezwzględnie

## Model
Wykorzystaliśmy Reinforcement learning, który pozwala naszemu modelowi uczyć się w środowisku, jakim jest gra.

### Wejście
Model składa się z 11 neuronów wejściowych
3 neurony odpowiadają za jego `wzrok`
- Czy przed nim, znajduje się niebezpieczeństwo
- Czy po prawej znajduje się niebezpieczeństwo
- Czy po lewej znajduje się niebezpieczeństwo

Kolejne z bezwględnego kierunku
- W górę?
- W dół?
- W prawo?
- W lewo?

Kolejne, z lokalizacji jabłka
- Czy jabłko jest nad głową?
- Czy jabłko jest pod głową?
- Czy jabłko jest po prawej od głowy?
- Czy jabłko jest po lewej od głowy?

### Wyjście
Na wyjściu mamy 3 stany, które mapujemy na akcje -> prosto, w prawo, w lewo

### Warstwa ukryta
256 neuronów warstwy ukrytej.

Po przegranej grze snake dostaje ujemne punkty, po wygranej dodatnie.

## Wniosek
Snake gubi się gdy jest dłuższy, prawdopodobnie dlatego, że `widzi` tylko na jedną kratkę. W ramach udoskonalenia projektu moglibyśmy pomyśleć o innych danych wejściowych, które by pozwoliły snakeowi na lepsza grę.
