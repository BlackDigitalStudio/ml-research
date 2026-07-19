---
name: rev19-extension-symbols
description: "HD3 rev19 MEASURED (2026-07-18) — BNB/LTC/SOL honest-H5100 surface: TRUE funding argmax на всех трёх (якорь DOGE/XRP вне DOGE/XRP — неверная семантика); no-ToD символ-условен (BNB +2.9, LTC −5.3); argmax-ячейки +2.5/+1.2/−0.1 — в 5–10× слабее деплой-класса, деплой-кандидатов нет; VM stopped"
metadata:
  node_type: memory
  type: project
---

HD3 rev19 (ledger `tb3s-20260718_bnb_ltc_sol_honest_rebuild`, RESEARCH_LOG s30).
Цепочка BTC/ETH-улучшений применена к BNB/LTC/SOL (LINK исключён — 119д дыра):
честная пересборка H5100 × фандинг {true, anch} × no-ToD (DROP_COLS 67-70), протокол
v2.1, 4 сида, t5, батареи OPS-GATES rev2.

- **Поверхность (ENS t5)**: BNB anch −3.44 / true −0.34 / **true×no-ToD +2.51** (BOOT P77,
  jitter.02 P36); LTC anch −6.14 / **true +1.21** (P64, jit P5) / no-ToD −4.09; SOL anch
  −5.41 / **true −0.11** (P50) / no-ToD −0.98.
- **TRUE-фандинг — argmax на всех трёх** (Δ +3.1/+7.4/+5.3 vs anch); live funding col13 —
  топ-колонка Bg (8.3–8.9% gain) на всех трёх. Деплойная якорь-семантика символ-условна:
  работает ТОЛЬКО на DOGE/XRP.
- **no-ToD НЕ обобщается**: удвоение ETH не воспроизвелось (BNB +2.9, SOL ~0, LTC −5.3),
  хотя все три модели клоко-доминированы как ETH (ToD 24–27% Bg gain) — доля ToD в gain
  не предсказывает знак ответа на абляцию.
- **rev8-негативы — не артефакт H1800**: плотности ~1.5/с (покрытие ~1200s ≥ 510s);
  сдвиги дал фандинг, не горизонт.
- **Secondary**: биндинг-гейт (BOOT ≥90 / jitter.02 / сид-гейт) не проходит ни одна
  ячейка — деплой-кандидатов из тира нет. h150-класс на BNB/LTC/SOL собирает в 5–10×
  меньше деплой-класса (+12..+16).
- Артефакты: `maker_labels_tb3s_h150d{,anch}` BNB/LTC/SOL + `_v2notod/` (36 SEED json,
  216 PERFOLD, 576 model-дампов, 9 ENS, rev19_logs.tgz). VM xsym-32 stopped 2026-07-18.
- rev20 (BNB deep-dive, джиттер отменён юзером): сиды 4-7 draw-FAIL (магнитуда +2.51 была дро-чувствительна; 8-seed DYN +2.09), пул-сид-гейт -2.37 FAIL (per-seed отрицателен — эдж чисто ансамблевый); рабочая форма BNB = КОНСЕНСУС (монотонно: >=7/8 +5.91 bpd+9.79 P90 0 negmo; 8/8 +10.05 P94), ETH-флор переносится отрицательно; harmony подтверждён в обе стороны; потолок ~+1.5%/мес @0.5 — резервная ячейка, капитал не предложен (ledger bnb_levers-20260719, s31).
- Открытые ходы (решение юзера): rev14/15-рычаги на BNB-notod (8 сидов, tau-floor, FIXQ),
  hold-sweep, либо остановить расширение на 4 задеплоенных символах.
Related: [[xsym-cross-symbol-run]], [[btc-eth-honest-rebuild]], [[scope-bound-claims]].
