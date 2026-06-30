# Методология подсчёта метрик аукциона

## Назначение
Документ описывает порядок расчёта метрик для сравнения влияния механизмов доверия (none, personal, system, personal+system) на переговоры агентов и итоговые показатели аукциона.

## Используемые инструменты
- Скрипт: `spade_llm/demo/platform/auction/count_metrics.py`
- Входные данные:
  - `dialogues/dialogue_start_*.txt` — логи диалогов между агентами.
  - `auction_logs/winning_bid_split_*.txt` — логи успешных итоговых ставок.
- Выходные данные:
  - `report_<label>.json` — отчёт с метриками.
  - PNG-графики в папке `output/<label>/`.

## Общий порядок действий

### 1. Подготовка
Запустить сценарий аукциона несколько раз подряд с одной настройкой доверия.
После каждого прогона убедиться, что появились:
- новые файлы в `dialogues/`,
- новые файлы в `auction_logs/` (только при успешных покупках).

### 2. Запуск подсчёта метрик для одного прогона
```bash
python3 spade_llm/demo/platform/auction/count_metrics.py \
  --dialogues dialogues \
  --auction-logs auction_logs \
  --output output/without \
  --label without
```
Параметры:
- `--dialogues` — папка с логами диалогов.
- `--auction-logs` — папка с логами победных ставок.
- `--output` — папка для отчётов и графиков.
- `--label` — метка механизма доверия.

Результат:
- `output/none/report_none.json`
- `output/none/dialogue_length_hist.png`
- `output/none/round_agreement_hist.png`
- `output/none/agent_margin_bar.png`
- `output/none/agent_success_rate_bar.png`
- `output/none/winning_bids_overview.png`

### 3. Агрегация результатов по всем механизмам доверия
После получения отчётов для всех настроек запустить:
```bash
python3 spade_llm/demo/platform/auction/count_metrics.py \
  --aggregate \
  output/none/report_none.json \
  output/without/report_without.json \
  output/all/report_all.json \
  --output output/aggregate
```
Результат:
- сравнительные графики в `output/aggregate/`.

## Расчёт метрик

### 1. Средняя длина диалогов
Для каждого файла `dialogue_start_*.txt` считается число строк вида:
```
Self [agent] (ActionType): ...
Opponent [agent] (ActionType): ...
```
Средняя длина = сумма длин всех диалогов / количество диалогов.

### 2. Распределение раундов договорённости
Для успешных диалогов (`outcome == success`) фиксируется номер раунда, на котором контрагент прислал `ShopList`.
**Один раунд — это завершённый обмен сообщениями: одно сообщение от каждого агента.**
Номер раунда равен порядковому номеру сообщения согласия, делённому на 2.

Отказ от переговоров (`refused`) выделяется отдельным золотым столбцом `refused` — это осознанное решение контрагента, которое экономит токены, а не неудача.

Неудачные исходы (`timeout`, `round_limit`, `error`, `incomplete`) группируются в столбец `fail`.
На графике указывается процент refused и процент неудачных диалогов.

### 3. Количество успешных покупок пользователя
Равно количеству файлов `winning_bid_split_*.txt`.
Каждый такой файл появляется только после согласия пользователя (`user_decision == True`).

### 4. Средняя маржа агентов в выигрышных сделках
Для каждого winning_bid_split:
- для каждого агента берётся `Margin` из лога,
- усредняется по агентам внутри сделки,
- затем усредняется по всем сделкам.

### 5. Индивидуальные метрики по агентам
Для каждого агента считаются:
- `winning_bid_count` — число победных ставок, в которых участвовал агент.
- `avg_margin_pct` — средняя маржа агента в победных сделках.
- `initiated_dialogues` — число диалогов, где агент был инициатором.
- `received_dialogues` — число диалогов, где агент был контрагентом.
- `successful_dialogues` — число диалогов с исходом `success`.
- `success_rate` — доля успешных диалогов (`successful_dialogues / (initiated + received)`).
- `avg_dialogue_length` — средняя длина диалогов с участием агента.
- `refused_negotiations` — число случаев, когда агент отказался от переговоров как контрагент.
- `refusal_rate` — доля отказов среди полученных предложений (`refused_negotiations / received_dialogues`).

## Формат отчёта JSON
```json
{
  "label": "none",
  "global": {
    "avg_dialogue_length": 5.2,
    "avg_length_success": 6.5,
    "avg_length_refused": 1.2,
    "avg_length_failed": 8.1,
    "total_dialogues": 42,
    "successful_dialogues": 24,
    "refused_dialogues": 8,
    "failed_dialogues": 10,
    "refused_rate": 0.19,
    "failure_rate": 0.238,
    "successful_user_purchases": 10,
    "avg_agent_margin_pct": 7.5,
    "avg_v": 850,
    "avg_p": 810,
    "avg_c": 760
  },
  "agents": {
    "first_merchant": {
      "winning_bid_count": 7,
      "avg_margin_pct": 6.2,
      "initiated_dialogues": 12,
      "received_dialogues": 10,
      "successful_dialogues": 15,
      "success_rate": 0.68,
      "avg_dialogue_length": 4.9,
      "refused_negotiations": 3,
      "refusal_rate": 0.3
    }
  }
}
```

## Примечания
- Диалоги и победные ставки не сопоставляются жёстко по времени: метрики считаются независимо по множествам логов.
- Если `auction_logs/` пуста, метрики 3 и 4 равны нулю, но метрики 1, 2 и 5 всё равно считаются.
- Для повторных аукционов (после отказа пользователя) создаётся новый набор диалогов, но winning_bid_split появляется только при финальном успехе.
- `refused` — это не неудача диалога, а решение контрагента не вступать в переговоры. В отчёте refused учитывается отдельно.
