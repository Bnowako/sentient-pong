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

## Obsługa
Aplikację należy uruchomić poleceniem ``` python3 agent.py```. <br/><br/>
Po uruchomieniu aplikacji użytkownik może wybrać czy chce wyświetlić trening wyszkolonego przez nas modelu, zobaczyć trening od zera lub wyświetlić porównanie tych 2 opcji. Następnie nalezy wybrać czy chcemy wyświetlić GUI i widzieć zachowanie snakea oraz ile gier ma trwać trening.
<br/> <br/>
W zależności od wybranych ustawień użytkownik zobaczy wykresy oraz porównania treningów w pop-upach. 

W momencie, w którym wybierzemy trening istniejącego już modelu i rekord zostanie pobity, model zostanie nadpisany.

## Wniosek
Snake gubi się gdy jest dłuższy, prawdopodobnie dlatego, że `widzi` tylko na jedną kratkę. W ramach udoskonalenia projektu moglibyśmy pomyśleć o innych danych wejściowych, które by pozwoliły snakeowi na lepsza grę.

