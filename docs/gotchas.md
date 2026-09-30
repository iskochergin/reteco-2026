# Грабли

Что ломалось и почему. Каждая строка стоила времени.

- `pytrec_eval` с PyPI требует сборки и падает. Ставить `pytrec-eval-terrier`,
  модуль импортируется под тем же именем. В `requirements.txt` кита указан
  именно нерабочий `pytrec_eval>=0.5`.
- `pyserini` 2.4 требует Python **>=3.12** и тянет torch. Системный python на
  этой машине 3.14 — venv поднимается на 3.12 через `uv`.
- **`/usr/libexec/java_home -v 21` врёт.** На этой машине он возвращает JRE
  1.8 из плагина апплетов и выходит с кодом 0. Проверка «java существует»
  проходит, а pyserini падает позже и невнятно. `scripts/env.sh` поэтому
  проверяет кандидата запуском: мажорная версия 21, наличие `javac` (JRE не
  годится) и наличие самой libjvm.
- `JVM_PATH` на macOS — это `libjvm.dylib`, а не `libjvm.so`. В README кита
  указан `.so`, это путь для Linux.
- Архитектура JVM и питона должны совпадать: arm64-python с x86_64-JVM даёт
  `mach-o, but wrong architecture` уже внутри `dlopen`.
- `official_baseline.py` кэширует результаты по доменам в
  `<out>/<track>/<domain>/results.json` — прерванный прогон возобновляется.
- `--track2 all` не работает, домены надо перечислять.
- Реплики трека 2 (`turns[].query`) — чистый текст, HTML в них нет (0 % на
  dev). Сырой HTML со Stack Exchange может быть в `original_query`.
- В chat template OCC-RAG заголовок документа игнорируется, подставляется
  только `doc['text']`.
- OCC-RAG: только greedy-декодирование. `repetition_penalty` и
  `no_repeat_ngram_size` ломают структурный формат.
- Эндпоинт AIRI кладёт весь структурный след в поле `reasoning`, `content`
  приходит пустым. Нужен `skip_special_tokens: False`.

## Вызов OCC-RAG

```python
client.chat.completions.create(
    model="AIRI/OCC-1-1.7B",
    messages=[{"role": "user", "content": question}],
    temperature=0, max_tokens=1200,
    extra_body={
        "chat_template_kwargs": {
            "documents": [{"text": ctx}],
            "enable_thinking": False,
        },
        "skip_special_tokens": False,
    },
)
```

Скорость эндпоинта: 200 примеров за минуту при 5–6 параллельных запросах.

## Окружение на этой машине (M3 Pro, 18 ГБ, macOS)

Установка (один раз):

```bash
brew install openjdk@21 uv
uv venv --python 3.12 ~/.reteco-venv
uv pip install --python ~/.reteco-venv/bin/python \
    pyarrow huggingface_hub tqdm gensim pytrec-eval-terrier pyserini sentence-transformers
git clone --depth 1 https://github.com/DataScienceUIBK/RETECO.git vendor/RETECO
~/.reteco-venv/bin/hf download DataScience-UIBK/RETECO-SemEval2027 --repo-type dataset \
    --local-dir data/reteco --include "track2_recor/*" "split_manifest.json"
PY=~/.reteco-venv/bin/python
$PY scripts/score.py --check-baseline      # сверка с таблицей организаторов
```

- Системный python — 3.14, pyserini его не поддерживает. venv на 3.12 через `uv`,
  лежит в `~/.reteco-venv`, **вне Yandex.Disk** — синхронизация не переживает
  десятки тысяч файлов.
- `/usr/libexec/java_home -v 21` **врёт**: возвращает JRE 1.8 из плагина апплетов
  с кодом 0. Поэтому путь к JDK 21 от brew задан явно в `scripts/retrieve.py`.
- `JVM_PATH` на macOS — `libjvm.dylib`, а не `.so` из README кита.
- Модели ≥ 0.6B на MPS и эмбеддинги 507 тыс. документов локально не запускать:
  памяти не хватает, ноут лагает. Это в Colab.
- AIRI `inference.airi.net:46783`: TCP проходит, TLS-рукопожатие таймаутится —
  похоже, нужен VPN.

## Данные: topic_id не уникален между доменами (29 сентября 2026)

В шести доменах из BRIGHT диалоги пронумерованы `0`, `1`, `2`…, и один
`topic_id` вида `2_turn_1` встречается одновременно в biology, psychology,
robotics, sustainable_living. В dev 634 уникальных id на 858 реплик, в train
1287 на 2113. Организаторам это не мешает — каждый домен отдельный файл.

Нам мешает везде, где реплики разных доменов лежат в одном словаре.
Первая версия `rewrite.py` держала кэш по `topic_id`, и 371 реплика dev
получила переписывание из чужого домена; таблица вариантов запроса была
неверной. **Ключ реплики везде — пара (домен, topic_id).** Так сейчас в
`rewrite.py`, `retrieve.py --queries` и в Colab-ноутбуках.

## Claude CLI: таймаут subprocess.run не спасает от зависания

`subprocess.run(..., timeout=N)` при таймауте убивает только процесс `claude`,
а его дочерние процессы держат канал вывода открытым, и `communicate()` ждёт
вечно. Так шесть вызовов провисели 15 минут. Лечение в `rewrite.py`: запуск с
`start_new_session=True` и при таймауте `os.killpg(pid, SIGKILL)`.

Фильтр безопасности Claude иногда отказывает на пачке реплик (категория
`bio` — вопросы биологии и медицины): «Sonnet 5 can't help with this».
`rewrite.py` при отказе разбивает пачку по одной реплике, а если отказ
повторился — берёт исходный текст с пометкой `fallback`.

## HuggingFace: скачивание зависает на Xet

Новый транспорт `hf-xet` на этой машине встаёт намертво на первых мегабайтах
(модель висела 16 минут на 15 МБ). Лечится `HF_HUB_DISABLE_XET=1`. Без токена
HF к тому же режет скорость — при частых скачиваниях завести `HF_TOKEN`.

## BM25 кита ≠ BM25 Lucene

Официальный BM25 кита — это gensim `LuceneBM25Model`, а не сам Lucene. На
dev с историей: кит 0.4379, Lucene через pyserini 0.3967 (−0.041) при том же
анализаторе и k1/b. Вероятная причина: Lucene берёт повторы слова в запросе
линейно (слово из истории ×5 весит ×5), а gensim их насыщает. На длинных
запросах с историей это заметно; BRIGHT-репродукция (arXiv:2509.02558) по той
же причине ввела query-side BM25. Сравнивать RM3 надо с Lucene-BM25, а не с
китом.

## Промпты Diver: карточка модели против кода RECOR

`config_sentence_transformers.json` у `AQ-MedAI/Diver-Retriever-4B`: запрос
`Instruct: Given a web search query, retrieve relevant passages that answer
the query\nQuery:`, документ **без префикса**. Авторы RECOR в своём
`retrievers.py` добавляли документу `Represent this text:` — так получено их
0.545. Мы берём промпты из конфига модели (как её обучали); если результат
заметно ниже 0.545, первым делом проверить этот префикс.
