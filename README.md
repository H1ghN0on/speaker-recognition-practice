Практика по направлению "Распознавание диктора"

Введение.

Режимы работы системы распознавания диктора:

1. Верификация (аутентификация) и идентификация;
2. Распознавание на закрытом и открытом множествах;
3. Текстозависимое и текстонезависимое распознавание.

Верификация - установление идентичности диктора с помощью оценки схожести между голосами в двух фонограммах, сравнение как «один к одному». Выход: ответ «да» или «нет».
Идентификация - определение неизвестного на основе одного из известных. Сравнение выполняется по принципу «один ко многим». Выход: метка известного диктора, к которому отнесен неизвестный.

Распознавание на закрытом множестве - все дикторы внутри заданного множества фонограмм известны.
Распознавание на открытом множестве - не все дикторы внутри заданного множества фонограмм известны.

Текстозависимое распознавание - диктору требуется произнести заранее определённую фразу, необходимую для принятия решения биометрической системой.
Текстонезависимое распознавание - альтернативно не требуется.

Конвейер голосовой биометрии:

1. Получение фонограммы
2. Предобработка - шумоочистка, определение областей речевой активности и т.д.
3. Диаризация - разделение речевого сигнала на сегменты, в каждом из которых присутствует голос только одного диктора. Все сегменты с одинаковым диктором помечаются определённым тегом.
4. Построение дикторской модели - формирование дескриптора, привязанного к сегментам речи конкретного диктора.
5. Сравнение моделей
6. Результат сравнения

**Часть 1. Информативные признаки речевых сигналов, извлечение признаков.**

  Библиотеки:
1. math - выполнение математических операций
2. scipy.fftpack - быстрое преобразование Фурье
3. numpy - мощные многомерные массивы и работа с ними
4. matplotlib.pyplot - создание графиков и визуализация данных
5. scikit-image - работа с изображениями
6. torchaudio - набор функций для глубокой работы с аудиоданными

Перед извлечением акустических признаков из фонограммы применяется процедура - преэмфазис - применение фильтра верхних частот к сигналу. Это требуется для того, чтобы снизить погрешность при определении диктора.

Наличие большого количества энергии на низких частотах может привести к искажению результатов анализа, так как это может маскировать или скрывать другие уникальные характеристики речи, которые могут быть важными для идентификации диктора.

Процедура преэмфазиса может быть описана выражение y(n) = x(n) + ax(n - 1), где y(n) - результирующий сигнал, x(n) - исходный сигнал, n - сэмпл. Если a < 0, то фильтр рассматривается как фильтр верхних частот. Для данной задачи он обычно равен -0.97.

![Screenshot from 2024-03-08 19-04-58](https://github.com/H1ghN0on/speaker-recognition-practice/assets/65870074/f70e115a-3a57-4319-8c90-38076d5684e6)


Акустические признаки - представление речевого сигнала в частотно-временной области.

Для вычисления акустических признаков можно воспользоваться следующей последовательностью:

1. Представить сигнал в виде набора фреймов.
   
   Частоты сигнала меняются со временем, поэтому в большинстве случаев не имеет смысла выполнять преобразование Фурье для всего сигнала, поскольку со временем мы потеряем контуры частоты сигнала. Чтобы избежать этого, мы можем с уверенностью предположить, что частоты сигнала постоянны в течение очень короткого периода времени. Следовательно, выполнив преобразование Фурье в этом коротком временном кадре, мы можем получить хорошее приближение частотных контуров сигнала путем объединения соседних кадров.

  Если сигнал длинный, становится крайне неэффективно делать FFT. Не хотелось бы пытаться выполнить FFT для аудиофайла длиной даже в короткую песню. В этом случае мы разбиваем сигнал на куски разумного размера, выполняем FFT для каждого и усредняем результаты.

   Размер фрейма = 25мс, шаг фрейма = 10мс.

  - Преобразование длины кадра и шага к количеству сэмплов (времени) в зависимости от частоты дискретизации.
  - Вычисление количества кадров, на которые нужно разделить сигнал.
  - Добавление "паддинга" к сигналу, чтобы каждый кадр имел одинаковое количество сэмплов.
  - Разделение усиленного сигнала на кадры с помощью индексации с учетом длины кадра и шага.
   
2. Каждый кадр пропустить через окно Хэмминга.

   Это требуется для сглаживания резких перепадов в сигнале (особенно по краям), уменьшить влияние этих артефактов на сигнал.
   
3. Вычислить одномерный спектр Фурье.

    Это требуется для преобразования сигнала из временной области в частотную область. Во временной области значения сигнала известно для всех действительных чисел в случае непрерывного времени, или в разные отдельные моменты в случае дискретного времени (иначе говоря, как сигнал изменяется во времени). Частотная область показывает, насколько много сигнала лежит в пределах каждой заданной частотной полосы в диапазоне частот.
   
   Значение NFFT = 512 в быстром преобразовании Фурье означает, что расчеты будут проводиться с использованием 512 точек данных.
   
   Размер FFT определяет количество ячеек, используемых для разделения окна на равные полосы или ячейки. Следовательно, интервал (bin) представляет собой выборку спектра и определяет частотное разрешение окна. По умолчанию: N(bin) = Размер NFFT/2.
   
5. На основе спектра Фурье рассчитать спектр мощности.
   
     Спектральная плотность мощности - это функция, которая отображает распределение мощности сигнала в зависимости от его частоты. Она позволяет увидеть, какая энергия сигнала сконцентрирована в различных частотных диапазонах. 
   
6. Рассчитать мел-банк фильтров.
  
     Мел — психофизическая единица высоты звука, учитывает особенности восприятия звуков разной высоты у человека. 1000 мел примерно равны 100 Гц, шкала логарифмическая.

     Мел-фильтры устроены таким образом, чтобы спектральная информация сигнала соответствовала способности человеческого слуха воспринимать различные частоты (более чувствителен к низким частотам, менее - к высоким частотам). Размер окна в мел-фильтре определяет ширину полосы пропускания фильтра и, таким образом, определяет частотные характеристики фильтра.

   Окно предназначено для сглаживания границ участков сигнала во временной области, чтобы избежать артефактов при дискретизации и преобразовании Фурье. Окна мел-шкалы, расположенные на мел-оси равномерно, в частотной шкале сужаются у низких частот и расширяются у высоких частот.

  ![Screenshot from 2024-03-08 19-05-34](https://github.com/H1ghN0on/speaker-recognition-practice/assets/65870074/5addd88f-4c3c-48b3-8466-2ef923a74d79)


7. Перемножить квадрат амплитудно-частотной характеристики (АЧХ) каждого фильтра со спектром мощности, который был вычислен для каждого фрейма, и просуммировать коэффициенты получившихся спектров, рассчитав энергии внутри соответствующих полос банка фильтров.
  
    Это делается для вычисления энергии сигнала в различных частотных полосах. Этот метод называется квантованием по мел-шкале. Подсчитав энергию внутри каждой полосы банка фильтров, можно получить более детальное представление о спектральных характеристиках сигнала и использовать эту информацию для дальнейшего анализа или обработки сигнала.

8. Вычислить логарифм от значений энергий.

    Необходимо для выделения основной информации из энергий в различных полосах частот. Логарифмирование позволяет сгладить различия между большими и маленькими значениями энергии и улучшить различение основных компонентов звукового сигнала.

9. Вычислить дискретное косинусное преобразование от логарифмов значений энергий.

     Позволяет представить полученные значения в виде мел-частотных кепстральных коэффициентов.

   ![Screenshot from 2024-03-08 19-06-02](https://github.com/H1ghN0on/speaker-recognition-practice/assets/65870074/93c70128-df10-42d3-ab6a-787a58c4840b)

При передаче сигнала через канал связи могут возникать искажения, такие как изменения в усилении, шумы и другие помехи, которые могут привести к неверному распознаванию или классификации акустических признаков.

Линейный инвариантный к сдвигу фильтр обладает свойством сохранения формы сигнала после его сдвига. Если на вход фильтра подается сигнал и затем он сдвигается (например, по оси времени), то на выходе фильтра будет получен такой же сигнал, но сдвинутый на ту же величину и в том же направлении. 

Моделирование канального эффекта с использованием ЛИС фильтра позволяет адаптировать акустические признаки к искажениям, вызванным каналом связи, и корректировать их значения для улучшения точности анализа. Данная процедура получила название процедуры нормализации

Идея выполнения процедуры нормализации состоит в вычислении среднего вектора наблюдаемых акустических признаков и центрирования всех векторов акустических признаков произнесения на это среднее.

Процедура масштабирования признаков отличается от процедуры нормализации тем, что при её выполнении необходимо вычислить вектор среднеквадратического отклонения, на который необходимо поэлементно разделить каждый нормализованный вектор акустических признаков некоторого произнесения.

  ![Screenshot from 2024-03-10 16-33-49](https://github.com/H1ghN0on/speaker-recognition-practice/assets/65870074/cf5a7f64-f9b0-4300-ad98-00fba7afa8e9)

Первая компонента логарифмов энергий на выходе мел-банка фильтров для 10 записей (5 мужских и 5 женских). Зеленая гистограмма - мужские записи, красная гистограмма - женские записи.

![Screenshot from 2024-03-10 18-40-45](https://github.com/H1ghN0on/speaker-recognition-practice/assets/65870074/99f62797-1a52-47f1-ab21-f806a4a3495d)

