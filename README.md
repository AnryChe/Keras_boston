1. Задача регрессии - предсказание стоимости недвижимости, поэтому и метрика accuracy нулевая

2. Можно было бы доставать имя модели тоже в цикле, если бы использовались разные имена, а именовать можно через параметр name при создании: 

keras.Sequential(name="my_sequential")

или же можно было бы доставать параметры слоев 
