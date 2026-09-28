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
