---
name: entry-fill-model-fix
description: "2026-07-26: every maker cell before this date is an UPPER BOUND — the default entry-fill model fills unconditionally on a gap past our level; opt-in fix shipped (build_samples --emit-level-flow + grid_sim_exitdbg --strict-entry-fill), validated 6/6 vs live and on a full day (fill 0.72 -> 0.42), NOT yet re-measured at cell scale"
metadata:
  node_type: memory
  type: project
---

**Проблема (измерена):** `live_sim::simulate_maker_entry` (husdc-rev1) наливает
безусловно, как только тач ушёл за наш уровень:
`if b.bid < level_px - eps { return FILLED }` — без потока и без очереди. Сама
модель очереди честная (`--queue-mult 1.0` / `--exit-queue-mult 1.0` = всегда
последний), ветка её просто обходит. На живом якоре DOGEUSDC все 3 фантомных
филла прошли через неё; на полном дне модель наливает **0.72 против 0.42** у
корректного правила, и выжившие филлы хуже (netl −1.78 → −3.82bp) — выдуманные
были благоприятными (подпись adverse selection).

**Почему нельзя было починить на месте:** `flow_paths.npy` не различает цены
(суммарный тейкер-объём за тик). Промежуточный патч «требовать хоть какой-то
поток в тике разрыва» ИЗМЕРЕН И ОТВЕРГНУТ (rev15: на USDT не убрал ничего, на
USDC убил реальный филл).

**Фикс (внедрён, оба флага по умолчанию ВЫКЛ — историческая
воспроизводимость сохранена):**
`build_samples --emit-level-flow` → `flow_lvl_paths.npy [ns,h,2]` =
[продажи по цене ≤ entry_long, покупки по цене ≥ entry_short] за тик, цены
читаются отдельным проходом; `grid_sim_exitdbg --strict-entry-fill
--level-flow-paths ...` → учебниковое правило очереди без веток по книге.
Валидация: 6/6 на реальных live-событиях, сборка чистая, сквозной прогон дня.

**Не измерено:** влияние на τ-отобранные ячейки, на USDT-книге (где считались
все исторические ячейки) и в годовом масштабе. **До этого прогона ни одну
ячейку переоценивать нельзя.**

**Важно про площадки:** фичи и скоры считаются на USDT и в симе, и в live — это
верно и менять НЕ надо. На USDC происходят только филлы. Рекордер пишет
DOGEUSDC depth+aggTrade, поэтому поправка на реализм филлов измерима на уже
имеющихся данных и **не требует покупки истории** (покупка нужна только для
годового ПЕРЕОБУЧЕНИЯ, это отдельный вопрос).

Документировано там, где ищут: `research/runtime/README.md` §Entry-fill model +
`KNOWN_PITFALLS.md` §Maker fill model. Код: `scripts/build_samples_husdc.rs`,
`scripts/grid_sim_exitdbg.rs`. Скрипты: `subs60_price_resolved_fill_ref.py`
(референс против live), `subs60_smoke_usdc_fill.py` (сквозной прогон),
`subs60_strict_fill_check.py` (отвергнутый вариант). Леджер OPS-EXEC rev15-16.
Связано: [[exec-policy-lock-divergence]], [[scope-bound-claims]],
[[verify-mechanism-before-verdict]].
