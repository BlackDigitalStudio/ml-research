---
name: exec-policy-lock-divergence
description: "MEASURED 2026-07-26: deployed exec (v2, all 4 instances) = ONE position at a time, but every validation number (year cells, recorder gates, bpd, Sharpe) scored EVERY above-tau 3s decision as an independent overlapping trade — live keeps ~4% (DOGE) / 23% (XRP) of measured bpd; exec v3 slot pool written, rollout pending user M/capital decision"
metadata:
  node_type: memory
  type: project
---

**Находка (user-raised, подтверждена измерением; леджер OPS-EXEC rev1,
`exec_lock_audit-20260726`):** во всех валидационных ячейках (годовые PERFOLD
`sel_fixed`/`sel_dyn`→`trades()`, recorder-EV гейты `causal()`, bpd=EV×tpd,
Sharpe из дневных сумм) **каждое** решение 3с-сетки выше τ — независимая
сделка (вход 60с, холд 150с от филла), перекрытия свободны. Задеплоенный
`axb_exec.py` v2 — `busy`-замок: одна сделка за раз, сигналы в занятости
молча выбрасываются (очереди нет). Live-факт DOGE FIXQ t10 0716–0725:
104 сигнала → 15 сделок (**85.6% потеряно**); busy заполненной сделки
эмпирически 152–183с (мед. 155с), не-филл 60с.

**Экономика (lock-sim на `_recev_h150anch2_*`, задеплоенные frozen τ, 17д):**
DOGE: посчитано +75.3 bpd (9.47 сд/д, EV +7.95) → с замком **+3.1 bpd (4.1%)**,
EV +0.96 — замок берёт систематически ХУДШУЮ сделку кластера (first +2.93 vs
rest +10.27). XRP: +76.7 → +17.6 bpd (22.9%). Слоты (one-way, skip-opposite;
противоположные перекрытия DOGE 4.3%/XRP 0% ⇒ hedge-mode не нужен):
kept% DOGE 4/8/18/30/63/85 при M=1/2/3/4/6/8; XRP 23/39/54/69/86/98.
Полная точность ≈ **M=8 × ≥5.5 USDC/сделку ≈ 44 USDC нотионала на символ** —
при общем балансе 20.43 на 4 инстанса недостижимо; при фиксированном капитале
USD-выгода от M растёт только когда minNotional перестаёт связывать размер.

**Сделано:** `live/axb_exec.py` **v3** — слот-пул `MAX_CONC` (1 = легаси
бит-в-бит), стэкинг одной стороны, выход строго СВОИМ количеством
(reduce-only, не positionAmt), фикс-нотионал wallet×SIZE_FRAC×MULT/MAX_CONC,
`skip_opposite`, env `DAY_TRADES_HALT`/`LEVERAGE`/`NOTIONAL_MULT`; юниты
DOGE/XRP + README v3; мок-тест `live/test_axb_exec_slots.py` 4/4;
скрипты аудита `scripts/subs60_{live_drop_audit,exec_lock_quant,exec_slot_sens}.py`.
**НЕ задеплоено** — выбор M/капитала/лева за пользователем.

**ЗАДЕПЛОЕНО (финал 01:41:53 UTC, OPS-EXEC rev2→rev6):** DOGE — единственный
live-инстанс, весь депозит 20.43 USDC, exec v3 **M=50 / mult 14.5 / lev 25**
(этапы за утро: M=8→15→50; ГОД: пик конкурентности 50, M=15 держал лишь 47%
годового bpd; усечение ухудшает и худший день, и Шарп — rev5);
XRP/ETH/BTC ВЫВЕДЕНЫ из live, их VM ОСТАНОВЛЕНЫ (~00:35 UTC), контрфактуал —
recorder-реплей (префиксы `_recev_*_fw*` засеяны). Пер-трейд паритет 0716–0725:
16/16 live-филлов воспроизведены (corr 0.935, med |Δ| 2.3bp, bias +1.7bp), но
**3/19 live-промаха входа при sim-филле всех — филлы моделируются на USDT-книге,
исполнение на тонкой USDC: winner-skewed haircut ≈×0.6/попытку (n=19)**;
ожидание v3 ≈ 85% × 0.6–0.85 от оконных +75 bpd. Мониторинг: еженедельный
паритет с DOGE-якоря; shadow-решения — только по пессимизированной нижней
границе (run-out→taker-cross; штраф стэкнутой очереди) — вариант ещё не
реализован. GCP: остаток $6.5 ≈ 3–4 дня recorder-VM (на ней же DOGE live).

**Попутно:** прод-exec разошёлся 3 версиями (ETH-вариант с LEVERAGE/
NOTIONAL_MULT НЕ снапшочен в витрине → нарушение ритуала); юнит DOGE в репо
имел TRADE_BUDGET=5 при проде t10 (исправлено); halt 40 сд/день режет
горячие дни посчитанной политики (0723: 79 сигналов). XRP/ETH live
недобирают сигналы (0.3–0.4/д при бюджете 5) — τ-сиды бегут горячо,
отдельная known issue. Связано: [[h150-sim-live-parity]],
[[live-trading-deploy]], [[scope-bound-claims]].
