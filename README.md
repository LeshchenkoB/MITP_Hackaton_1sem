# MITP_Hackathon_1sem
Проектный практикум 1 семестра магистратуры "Науки о данных" МФТИ

**Состав команды:**
1. Лещенко Борис (тимлид)
2. Оскина Надежда
3. Аглуллин Дмитрий
4. Сысуем Никита
5. Биткулов Марат
6. Аликин Александр

## Трек: Мониторинг экосистемы через IT-решения

Область проекта: Здоровье домашних птиц, с фокусом на автоматизацию диагностики заболеваний у кур с использованием методов глубокого обучения.

## Бизнес-постановка задачи

**Цель проекта:** Определение наличия или отсутствия заболеваний у куриц на основе анализа фотографий их помета.

**Область применения:** Сельское хозяйство, ветеринария, мониторинг здоровья птицы.

**Проблема:** Заболевания у кур могут привести к снижению продуктивности, увеличению затрат на лечение и убыткам для фермеров.

**Решение:** Использование компьютерного зрения для автоматизации диагностики на основе изображений куринного помета.
С помощью telegram-чат-бота осуществляется предсказание наличия у птицы одного из четырех состояний: "Здоровый", "Кокцидиоз", "Болезнь Нью-Касла", "Сальмонеллез" ("Healthy", "Coccidiosis", "New Castle Disease", "Salmonella").

Данный подход можно использовать и как "второе мнение", так и для ранней/самостоятельной диагностики заболеваний.

Предполагается, что данный инструмент позволит своевременно производить необходимые мероприятия, направленные на локализацию проблемы и ее решения.

## Область DS -> Computer Vision

Задача: Классификация изображений с множественными классами (multiclass classification).

Используемые модели: ResNet18, EfficientNetB3, EfficientNetB5.

Выбор моделей связан с их эффективностью в задаче классификации изображений и способностью хорошо справляться с более сложными задачами с несколькими классами.

## Обоснование выбора моделей
### ResNet18:

Эта модель позволяет эффективно обучаться даже на сложных наборах данных. Она отлично подходит для задач с небольшим числом классов. Компактность модели делает её полезной для начального анализа и сравнений.

### EfficientNetB3 и EfficientNetB5:

Эти модели оптимизированы для работы с высокоразрешёнными изображениями, что важно, если текстуры или мелкие особенности на изображениях играют роль в классификации.
EfficientNetB3 обеспечивает баланс между вычислительной сложностью и точностью, в то время как EfficientNetB5 предназначена для более точной классификации благодаря увеличенной глубине и ширине сети.

### Комбинация моделей:

Использование нескольких архитектур позволяет сравнить их производительность и определить лучшую модель для каждого класса, что особенно полезно, если классы различаются по сложности их распознавания.
Именно по этому, в своем решении мы не стали отдавать предпочтение какой-либо одной модели, а реализовали возможность выбора в чат-боте модель, на основании которой делается предсказание.

## Состав проекта:
**файлы моделей:**   
pytorch-resnet18.ipynb  
EffecientNetB3.ipynb  
EfficientNetB5.ipynb  

**файлы Telegram чат-бота:** 

**файлы окружения:**  
requirementsResNet18.txt  
requirementsENB3.txt  
requirementsENB5.txt  

**веса моделей (дообученные модели, используемые для предикта):**  
[ResNet18](https://drive.google.com/file/d/1osRY3Wv3TXKlhT9wbPtFQd9_cjWZOFK0/view?usp=drive_link)  
[EfficientNetB3](https://drive.google.com/file/d/1dxG_qHsnk81qe_oLsdSoBptDeo5ZEJfC/view?usp=drive_link)
[EfficientNetB5](https://drive.google.com/file/d/12mB8aARcSAJa4Q50lyAB-E1xULvfUfah/view?usp=drive_link)

**Датасет используемый для дообучения моделей**  
[Ссылка_1](https://www.kaggle.com/datasets/chandrashekarnatesh/poultry-diseases?select=data)  
[Ссылка_2](https://www.kaggle.com/datasets/gauravduttakiit/poultry-diseases-detection/data) 
