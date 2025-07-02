> [!IMPORTANT]   
> **Если вы публично делитесь переведенным результатом, и опытный переводчик-человек не участвовал в тщательном переводе или проверке, пожалуйста, отметьте его как машинный перевод в заметном месте. Если на нормальном русском. Если вы не способны сделать минимальную проверку качества, перед публикацией, или вы не умеете в тайп, то укажите в описании к загражуемой манге, что это "машинный" перевод. Большинство русских сервисов тупо забанит вам аккаунт после попытки залива "машинного" перевода. Уделите хоть каплю внимания редактуре.**

# BallonTranslator
[简体中文](/README.md) | [English](/README_EN.md) | [pt-BR](../doc/README_PT-BR.md) | Русский | [日本語](../doc/README_JA.md) | [Indonesia](../doc/README_ID.md) | [Tiếng Việt](../doc/README_VI.md) | [한국어](../doc/README_KO.md)

Еще один инструмент для компьютерного перевода комиксов/манги на основе глубокого обучения.

<img src="src/ui0.jpg" div align=center>

<p align=center>
предпросмотр
</p>

# Особенности
* Полностью автоматизированный перевод  
  - Поддерживает автоматическое обнаружение текста, распознавание, удаление и перевод. Общая производительность зависит от этих модулей.
  - Верстка основана на оценке форматирования оригинального текста.
  - Хорошо работает с мангой и комиксами.
  - Улучшенная верстка манга->английский, английский->китайский (на основе выделения областей баллонов).
  
* Редактирование изображений  
  - Поддерживает редактирование масок и ретушь (что-то вроде инструмента точечного восстановления в Photoshop) 
  - Адаптирован для изображений с экстремальным соотношением сторон, таких как веб-комиксы
  
* Редактирование текста  
  - Поддерживает богатое форматирование текста и [пресеты стилей текста](https://github.com/dmMaze/BallonsTranslator/pull/311), переведенные тексты можно редактировать интерактивно.
  - Поддерживает поиск и замену
  - Поддерживает экспорт/импорт в/из документов Word

# Установка

## На Windows
Если вы не хотите устанавливать Python и Git самостоятельно и у вас есть доступ к Интернету:  
Скачайте BallonsTranslator_dev_src_with_gitpython.7z с [MEGA](https://mega.nz/folder/gmhmACoD#dkVlZ2nphOkU5-2ACb5dKw) или [Google Drive](https://drive.google.com/drive/folders/1uElIYRLNakJj-YS0Kd3r3HE-wzeEvrWd?usp=sharing), распакуйте его и запустите launch_win.bat.   
Запустите scripts/local_gitpull.bat, чтобы получить последнее обновление.

## Запуск исходного кода

Установите [Python](https://www.python.org/downloads/release/python-31011) **<= 3.12** (не используйте версию, установленную из Microsoft Store) и [Git](https://git-scm.com/downloads).

```bash
# Клонируйте этот репозиторий
$ git clone https://github.com/dmMaze/BallonsTranslator.git ; cd BallonsTranslator

# Запустите приложение
$ python3 launch.py
```

Обратите внимание, что при первом запуске будут автоматически установлены необходимые библиотеки и загружены модели. Если загрузки не удались, вам нужно будет скачать папку **data** (или отсутствующие файлы, упомянутые в терминале) с [MEGA](https://mega.nz/folder/gmhmACoD#dkVlZ2nphOkU5-2ACb5dKw) или [Google Drive](https://drive.google.com/drive/folders/1uElIYRLNakJj-YS0Kd3r3HE-wzeEvrWd?usp=sharing) и сохранить ее в соответствующем пути в папке с исходным кодом.

## Сборка приложения для macOS (совместимо как с процессорами Intel, так и с Apple Silicon)
<i>Обратите внимание, что macOS также может запускать исходный код, если это не работает.</i>  

![запись экрана 2023-09-11 14 26 49](https://github.com/hyrulelinks/BallonsTranslator/assets/134026642/647c0fa0-ed37-49d6-bbf4-8a8697bc873e)

#### 1. Подготовка
-   Загрузите библиотеки и модели с [MEGA](https://mega.nz/folder/gmhmACoD#dkVlZ2nphOkU5-2ACb5dKw "MEGA") или [Google Drive](https://drive.google.com/drive/folders/1uElIYRLNakJj-YS0Kd3r3HE-wzeEvrWd?usp=sharing)


<img width="1268" alt="скриншот 2023-09-08 13 44 55_7g32SMgxIf" src="https://github.com/dmMaze/BallonsTranslator/assets/134026642/40fbb9b8-a788-4a6e-8e69-0248abaee21a">

-  Поместите все загруженные ресурсы в папку с названием data, конечная структура каталога должна выглядеть так:

```
data
├── libs
│   └── patchmatch_inpaint.dll
└── models
    ├── aot_inpainter.ckpt
    ├── comictextdetector.pt
    ├── comictextdetector.pt.onnx
    ├── lama_mpe.ckpt
    ├── manga-ocr-base
    │   ├── README.md
    │   ├── config.json
    │   ├── preprocessor_config.json
    │   ├── pytorch_model.bin
    │   ├── special_tokens_map.json
    │   ├── tokenizer_config.json
    │   └── vocab.txt
    ├── mit32px_ocr.ckpt
    ├── mit48pxctc_ocr.ckpt
    └── pkuseg
        ├── postag
        │   ├── features.pkl
        │   └── weights.npz
        ├── postag.zip
        └── spacy_ontonotes
            ├── features.msgpack
            └── weights.npz

7 директорий, 23 файла
```

-  Установите инструмент командной строки pyenv для управления версиями Python. Рекомендуется установка через Homebrew.
```
# Установка через Homebrew
brew install pyenv

# Установка через официальный скрипт
curl https://pyenv.run | bash

# Настройка окружения оболочки после установки
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
```


#### 2. Сборка приложения
```
# Перейдите в рабочую директорию `data`
cd data

# Клонируйте ветку `dev` репозитория
git clone -b dev https://github.com/dmMaze/BallonsTranslator.git

# Перейдите в рабочую директорию `BallonsTranslator`
cd BallonsTranslator

# Запустите скрипт сборки, на этапе pyinstaller потребуется ввести пароль, введите пароль и нажмите enter
sh scripts/build-macos-app.sh
```
> 📌Упакованное приложение находится в ./data/BallonsTranslator/dist/BallonsTranslator.app, перетащите приложение в папку приложений macOS для установки. Готово к использованию без дополнительной настройки Python.


</details> 

# Использование

**Рекомендуется запускать программу в терминале на случай, если она аварийно завершится и не оставит никакой информации, см. следующий gif.**
<img src="doc/src/run.gif">  
- При первом запуске приложения, пожалуйста, выберите переводчик и установите исходный и целевой языки, нажав на значок настроек.
- Откройте папку, содержащую изображения комикса (манги/маньхуа/манхвы), которые нуждаются в переводе, нажав на значок папки.
- Нажмите кнопку `Run` и дождитесь завершения процесса.

Форматы шрифта, такие как размер и цвет, определяются программой автоматически в этом процессе. Вы можете предопределить эти форматы, изменив соответствующие опции с "decide by program" на "use global setting" в панели конфигурации->Typesetting. (глобальные настройки - это те форматы, которые отображаются на правой панели форматирования шрифта, когда вы не редактируете ни один текстовый блок на сцене)

## Редактирование изображений

### Инструмент ретуши
<img src="src/imgedit_inpaint.gif">
<p align = "center">
Режим редактирования изображения, инструмент ретуши
</p>

### Инструмент прямоугольника
<img src="src/rect_tool.gif">
<p align = "center">
Инструмент прямоугольника
</p>

Чтобы 'стереть' нежелательные результаты ретуши, используйте инструмент ретуши или инструмент прямоугольника с нажатой **правой кнопкой** мыши.  
Результат зависит от того, насколько точно алгоритм ("метод 1" и "метод 2" на gif) извлекает маску текста. Он может работать хуже на сложном тексте и фоне.  

## Редактирование текста
<img src="src/textedit.gif">
<p align = "center">
Режим редактирования текста
</p>

<img src="src/multisel_autolayout.gif" div align=center>
<p align=center>
Пакетное форматирование текста и автоматическая компоновка
</p>

<img src="src/ocrselected.gif" div align=center>
<p align=center>
OCR и перевод выбранной области
</p>

## Горячие клавиши
* ```A```/```D``` или ```pageUp```/```Down``` для перелистывания страниц
* ```Ctrl+Z```, ```Ctrl+Shift+Z``` для отмены/повтора большинства операций. (обратите внимание, что стек отмены будет очищен после перелистывания страницы)
* ```T``` для режима редактирования текста (или кнопка "T" на нижней панели инструментов).
* ```W``` для активации режима создания текстового блока, затем перетащите мышь по холсту с нажатой правой кнопкой, чтобы добавить новый текстовый блок. (см. gif редактирования текста)
* ```P``` для режима редактирования изображения.  
* В режиме редактирования изображения используйте ползунок в правом нижнем углу для управления прозрачностью исходного изображения.
* Отключите или включите любые автоматические модули через строку заголовка->run, запуск со всеми отключенными модулями пересоздаст и перерисует весь текст в соответствии с соответствующими настройками.  
* Установите параметры автоматических модулей на панели конфигурации.  
* ```Ctrl++```/```Ctrl+-``` (Также ```Ctrl+Shift+=```) для изменения размера изображения.
* ```Ctrl+G```/```Ctrl+F``` для глобального поиска/поиска на текущей странице.
* ```0-9``` для настройки прозрачности текстового слоя
* Для редактирования текста: жирный - ```Ctrl+B```, подчеркнутый - ```Ctrl+U```, курсив - ```Ctrl+I``` 
* Установите тень текста и прозрачность на панели стиля текста -> Effect.  
* ```Alt+Стрелки``` или ```Alt+WASD``` (```pageDown``` или ```pageUp``` в режиме редактирования текста) для переключения между текстовыми блоками.
  
<img src="src/configpanel.png">

## Режим без графического интерфейса (Запуск без GUI)
``` python
python launch.py --headless --exec_dirs "[DIR_1],[DIR_2]..."
```
Обратите внимание, что конфигурация (исходный язык, целевой язык, модель ретуши и т.д.) будет загружена из config/config.json.  
Если отрисованный размер шрифта неправильный, укажите логическое DPI вручную через ```--ldpi ```, типичные значения - 96 и 72.


# Модули автоматизации
Этот проект сильно зависит от [manga-image-translator](https://github.com/zyddnys/manga-image-translator), онлайн-сервис и обучение моделей недешевы, пожалуйста, рассмотрите возможность пожертвования проекту:  
- Ko-fi: <https://ko-fi.com/voilelabs>
- Patreon: <https://www.patreon.com/voilelabs>
- 爱发电: <https://afdian.net/@voilelabs>  

[Sugoi translator](https://sugoitranslator.com/) создан [mingshiba](https://www.patreon.com/mingshiba).
  
## Обнаружение текста
 * Поддерживает обнаружение английского и японского текста, код обучения и подробности можно найти в [comic-text-detector](https://github.com/dmMaze/comic-text-detector)
* Поддерживает использование обнаружения текста из [Starriver Cloud (Tuanzi Manga OCR)](https://cloud.stariver.org.cn/). Необходимо заполнить имя пользователя и пароль, автоматический вход будет выполняться при каждом запуске программы.

   * Для подробных инструкций см. **Инструкции по использованию Tuanzi OCR**: (только на [китайском](doc/团子OCR说明.md) и [бразильском португальском](doc/Manual_TuanziOCR_pt-BR.md))
   
## OCR
 * Все модели mit* взяты из manga-image-translator, поддерживают распознавание английского, японского и корейского языков и извлечение цвета текста.
 * [manga_ocr](https://github.com/kha-white/manga-ocr) от [kha-white](https://github.com/kha-white), распознавание текста для японского языка, с основным фокусом на японской манге.
 * Поддерживает использование OCR из [Starriver Cloud (Tuanzi Manga OCR)](https://cloud.stariver.org.cn/). Необходимо заполнить имя пользователя и пароль, автоматический вход будет выполняться при каждом запуске программы.
   * Текущая реализация использует OCR для каждого текстового блока отдельно, что приводит к более медленной скорости и не дает значительного улучшения точности. Это не рекомендуется. При необходимости используйте вместо этого Tuanzi Detector.
   * При использовании Tuanzi Detector для обнаружения текста рекомендуется установить OCR в none_ocr для прямого чтения текста, экономя время и уменьшая количество запросов.
   * Для подробных инструкций см. **Инструкции по использованию Tuanzi OCR**: (только на [китайском](doc/团子OCR说明.md) и [бразильском португальском](doc/Manual_TuanziOCR_pt-BR.md))

## Ретушь
  * AOT взят из [manga-image-translator](https://github.com/zyddnys/manga-image-translator).
  * Все lama* дообучены с использованием [LaMa](https://github.com/advimman/lama)
  * PatchMatch - это алгоритм из [PyPatchMatch](https://github.com/vacancy/PyPatchMatch), эта программа использует [модифицированную версию](https://github.com/dmMaze/PyPatchMatchInpaint) от меня. 
  

## Переводчики
Доступные переводчики: Google, DeepL, ChatGPT, Sugoi, Caiyun, Baidu. Papago и Yandex.
 * Google закрыл сервис перевода в Китае, пожалуйста, установите соответствующий 'url' в панели конфигурации на *.com.
 * [Caiyun](https://dashboard.caiyunapp.com/), [ChatGPT](https://platform.openai.com/playground), [Yandex](https://yandex.com/dev/translate/), [Baidu](http://developers.baidu.com/) и [DeepL](https://www.deepl.com/docs-api/api-access) переводчики требуют токен или API-ключ.
 * DeepL & Sugoi переводчик (и его преобразование CT2 Translation) благодаря [Snowad14](https://github.com/Snowad14).
 * Sugoi переводит с японского на английский полностью оффлайн. Скачайте [оффлайн модель](https://drive.google.com/drive/folders/1KnDlfUM9zbnYFTo6iCbnBaBKabXfnVJm), переместите "sugoi_translator" в BallonsTranslator/ballontranslator/data/models. 
 * [Sakura-13B-Galgame](https://github.com/SakuraLLM/Sakura-13B-Galgame), отметьте ```low vram mode``` в панели конфигурации, если вы запускаете его локально на одном устройстве и столкнулись с сбоем из-за нехватки vram (включено по умолчанию).
 * DeepLX: Пожалуйста, обратитесь к [Vercel](https://github.com/bropines/Deeplx-vercel) или [deeplx](https://github.com/OwO-Network/DeepLX)
 * Добавлена библиотека [Translators](https://github.com/UlionTse/translators) которая поддерживает доступ к некоторым сервисам переводчиков без api ключей. О поддерживаемых сервисах можете узнать [тут](https://github.com/UlionTse/translators#supported-translation-services). 

Для других хороших оффлайн английских переводчиков, пожалуйста, обратитесь к этой [ветке обсуждения](https://github.com/dmMaze/BallonsTranslator/discussions/515).  
Чтобы добавить новый переводчик, пожалуйста, обратитесь к [how_to_add_new_translator](doc/how_to_add_new_translator.md), это просто как создание подкласса BaseClass и реализация двух интерфейсов, затем вы можете использовать его в приложении, вы можете внести свой вклад в проект.  


## FAQ и прочее
* Если ваш компьютер имеет GPU NVIDIA или Apple Silicon, программа включит аппаратное ускорение. 
* Добавлена поддержка [saladict](https://saladict.crimx.com) (*Универсальный профессиональный всплывающий словарь и переводчик страниц*) в мини-меню при выделении текста. [Руководство по установке](doc/saladict.md)
* Ускорение производительности, если у вас есть устройство [NVIDIA's CUDA](https://pytorch.org/docs/stable/notes/cuda.html) или [AMD's ROCm](https://pytorch.org/docs/stable/notes/hip.html), так как большинство модулей использует [PyTorch](https://pytorch.org/get-started/locally/).
* Шрифты берутся из системных шрифтов вашей системы.
* Благодарность [bropines](https://github.com/bropines) за русскую локализацию.
* Добавлен экспорт в скрипт JSX для Photoshop от [bropines](https://github.com/bropines). </br> Чтобы прочитать инструкции, улучшить код и просто покопаться, чтобы увидеть, как это работает, вы можете перейти в `scripts/export to photoshop` -> `install_manual.md`.
