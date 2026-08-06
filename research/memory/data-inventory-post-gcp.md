# Data inventory post-GCP (2026-08-06) — где какое сырьё лежит

Составлено после GCP-outage 2026-08-05 (биллинг закрыт, все 4 аккаунта обоих
Google-логинов OPEN=False). Полные леджер-записи:
`ops_evac-20260806_modal_to_hf_public` + `_AMEND1_completed_verified`.

## Слой 1 — GCS (ЗАМОРОЖЕН, не удалён; единственная копия сырья и деплоя)

Объекты отдают 403 «billing account … disabled in state closed», верхнеуровневый
листинг работает → данные физически живы, тикает grace-период. Вернуть = открыть
ЛЮБОЙ биллинг и прилинковать проект.

- `gs://market-data-0998ac51/` (project-0998ac51, virgin.ship03@gmail.com):
  `raw/` (book/trades parquet), `features_v1/` (61-фич снапшоты, источник
  годовых AxB/FIXQ датасетов), `feats_sub60/`, `hd2_cache_v1/`,
  `hd2_sub60_cache{,_z}/`, `research_runs/` (ВСЕ артефакты ранов И деплой-бандлы
  `research_runs/deploy_h150/{SYM}` — A0-3/Bg0-3 бустеры+refs+tau, которые
  live/axb_boot.py тянет при старте движка).
- `gs://market-data-eu-d39e90d0/` (project-d39e90d0…) — полный EU-бэкап того же.
- `gs://blackdigital-scalper-data/` (project-26a24ad0…, blackdigital.kz) —
  исходный бакет до миграции, тот же состав без sub60.
- `gs://recorder-data-asia-{0998ac51,d39e90d0}/` — записи рекордера.

**Значит:** годовые датасеты деплой-класса и задеплоенные модели существуют
ТОЛЬКО здесь. Без GCS лайв не перезапустить и не переобучить.

## Слой 2 — Modal (НОВЫЙ ДОМ КОМПЬЮТА, рабочие копии деривативов)

5 живых workspace (профили в ~/.modal.toml оператора: virginship05–08 +
virgin-ship03), в каждом volume `hd2-cache`:

- **virginship06 ≡ virginship08** (зеркала, самые полные): `hd2/` 8 символов —
  DOGE/XRP/BNB 545д, SOL 544д (с 2024-11-09), LTC 1243д (с 2022-12-08),
  LINK 788д (с 2023-09-22), BTC 363д/ETH 362д (с 2025-05-09), всё до
  2026-05-08; fp16 80-ch LOB стрим + 6 globals + first-passage лейблы
  H=180/600/1800 (замороженный hd1_seq_core).
- **ws06 дополнительно**: `midts/` все 8 символов ({ts,mid} событийный ряд —
  субстрат переразметки на произвольный горизонт); секреты `hf-token`,
  `evac-ws-tokens`.
- **ws08 дополнительно**: `sub60/` — 1s-грид кэш DOGE 362д (5.1G) / ETH 361д
  (12.9G) / LINK 244д (4.5G): 71 фича + LOB-стрим + rH60/y60/updn.
- ws05, ship03: SOL/LTC 500д (2024-12-25..2026-05-08); ship03 ещё `hd2-smoke`.
- `results/` во всех пяти — чекпойнты/предсказания hd2-ревизий.

Пайтфолы Modal (измерены): на Windows-клиенте обязателен `PYTHONUTF8=1` (cp1251
роняет CLI на «✓»); ВНУТРИ контейнера MODAL_TOKEN_ID/SECRET игнорируются —
кросс-workspace мост требует вычистить все MODAL_* из env сабпроцесса.

## Слой 3 — Hugging Face (ВЕЧНОЕ ПУБЛИЧНОЕ ЗЕРКАЛО, сверено по счётчикам)

Эвакуировано 2026-08-06 in-cloud (`scripts/modal_evac_hf.py`), счётчики файлов
совпали с volume по всем символам:

- `delmiron27/ml-research-hd2-streams` — 52.09 GiB, hd2 8 символов
- `delmiron27/ml-research-midts` — 5.87 GiB, midts 8 символов
- `delmiron27/ml-research-sub60` — 22.49 GiB, sub60 DOGE/ETH/LINK
- `delmiron27/ml-research-results` — ~4.2 GiB, подпапка на каждый workspace
- (старые приватные scalper-bot-hd2-{cache,results} = SOL/LTC 500д, поглощены
  новыми; scalper-bot-hd2-midts исторически ПУСТОЙ)

Гидрация HF→Modal: паттерн hydrate в `archive/scripts/hd2_to_hf.py`.
Пайтфолы HF: hf_hub>=0.34 (0.26 падает UnboundLocalError на стейл
snapshot_download `.cache` внутри гидрированных папок); квота 1000 API-req/5мин
на аккаунт — двух параллельных пушеров она режет (429), пушить последовательно.

## Слой 4 — локально (Windows-хост оператора)

Данных нет. Есть: исходники рекордера `C:\Dev\crypto-market-recorder` (лайв-код
жил на GCP VM), копия витрины `C:\Dev\_ta_tmp`, GitHub-токен
`C:\Разработки\ml_research_token.txt`, Modal-токены `C:\Пароли\`.

## Чего НЕ существует вне GCS (проверено, не предполагается)

Сырой CL parquet, features_v1, feats_sub60-исходники, funding/OI сырьё,
записи рекордера, деплой-бандлы моделей. Вендор Cryptolake — подписка истекла
(повторная покупка = запасной путь пересборки).
