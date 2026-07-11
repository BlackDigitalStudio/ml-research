# Vol-aware experiment — final report (2026-04-28)

## TL;DR

**0 / N grid configs profitable across all 6 ансамбль variants.** Best n≥50 net = **−2.73%** на 2-arch pure volaware (transformer + patchtst). Это **хуже** original 14-arch baseline **−1.83%**. Vol-aware label engineering один (без SSL/multi-task) **не даёт +EV** на этом cache.

## Цель эксперимента

Тестировать гипотезу: **vol-aware TP/SL labels** (per-sample TP ∝ rv_120s, SL = TP/RR) + переобученный ансамбль улучшит signal-to-noise vs original ансамбль с фиксированными TP=0.20%/SL=0.10%.

## Что сделано

| Этап | Статус |
|------|--------|
| 1. Generate vol-aware labels (`scripts/gen_volaware_labels.py`) | DONE — UP=20%, DN=19%, FL=61% (vs original 7.5/7.7/84.7%) |
| 2. Retrain 6 архитектур volaware на TPU (CPCV 6 folds) | DONE — transformer, patchtst, mamba, chronos_bolt_small, hybrid_mamba_attn полные 6/6 |
| 3. timesfm_2p5_200m volaware | trained 6/6, **inference FAILED** (module API mismatch на scalper-tcn) |
| 4. SSL pretraining transformer (`scripts/pretrain_ssl_transformer.py`) | **FAILED** — NaN gradients после первой epoch |
| 5. Multi-task aux loss transformer | **NOT IMPLEMENTED** — требует переписать transformer model |
| 6. Bagging 5 trans с different seeds | STARTED → STOPPED (только seed 43 fold 0-2 done) |
| 7. Build pure-volaware stacker + grid | DONE на 2-arch и 5-arch вариантах |

## Vol-aware archs trained (CPCV 6 folds)

В `gs://scalper-bot-research-data/checkpoints_volaware/`:

| Arch | Params | Val F1[UP] | Val F1[DN] | Val prec_NF |
|------|--------|------------|------------|-------------|
| transformer | 2.78M | 0.413 | 0.211 | 0.307 |
| patchtst | 3.82M | 0.157 | 0.219 | 0.305 |
| mamba | 1.22M | 0.309 | 0.325 | 0.325 |
| chronos_bolt_small | 20.4M | 0.227 | 0.150 | 0.310 |
| hybrid_mamba_attn | 1.51M | 0.280 | 0.296 | 0.288 |
| timesfm_2p5_200m | 200M | trained but inference failed | | |

**Per-arch prec_NF на vol-aware target = 0.29-0.33** (vs 0.26-0.27 на original target — улучшение +3-7pp). Но ансамбль не транслирует это в +EV.

## Stacker variants — full results table

| Setup | val_acc | prec_NF | best n≥50 net | n_trades |
|-------|---------|---------|---------------|----------|
| **Original 14-arch baseline (узкий 1260 grid)** | 0.760 | 0.287 | **−1.83%** | 38 |
| Original 14-arch + 500K wide grid | — | — | −2.46% | 60 |
| Original 14-arch + vol-aware TP/SL static grid | — | — | −2.39% | 80 |
| Combined: volaware T + 14 originals | 0.558 | 0.366 | −50.28% | 1900 |
| Combined + Optuna best params | 0.552 | 0.344 | −30.28% | 1171 |
| Volaware T only | 0.555 | 0.354 | −25.61% | 703 |
| 2-arch pure volaware (T + patchtst, partial) | 0.549 | 0.341 | −3.36% | 64 |
| **2-arch pure volaware (T + patchtst, full coverage)** | **0.549** | **—** | **−2.73%** | 69 |
| 5-arch pure volaware (T + patchtst + mamba + chronos + hybrid) | 0.549 | **0.000** | **collapsed** | 0 |

**Ключевой парадокс:** добавление volaware-обученных архитектур к стэкеру не улучшает а **collapses** stacker на all-FLAT. С 5 archs prec_NF=0%, n_predictions_NF=0/37085. Это происходит независимо от XGBoost hparams (Optuna best vs defaults).

## Optuna stacker hparam sweep

50 trials на CPU96 (96 vCPU) с objective `prec_NF × sqrt(coverage)`:
- Best score: 0.1498
- Best params: `n_estimators=250, max_depth=3, learning_rate=0.057, gamma=4.36, reg_alpha=2.9, reg_lambda=0.89`

Optuna params не помогли — даже на combined ансамбле дали prec_NF=0.344 (vs 0.366 без Optuna).

## Confidence/margin analysis

Combined stacker max softmax confidence:
- median = 0.478, p75 = 0.523, **p95 = 0.580**
- Non-FLAT predictions confidence < 0.40 для всех 5394 (из 37k holdout)

Margin = P(top) − P(second):
- Median margin **< 0.05** для non-FLAT predictions
- То есть модель не различает UP vs DN decisively — argmax выбирает direction с почти случайным шумом

**Поэтому "поднять confidence threshold" физически невозможно — модель не уверена настолько.**

## Почему 5-arch collapses в FLAT-only

Гипотеза: stacker (трёхклассовый XGBoost с sqrt-inv-freq class weights) учится на validation set где **train val class распределение скоро смещено**. С больше archs (15 features) и одинаково шумным сигналом каждой — гради bow boost converges на dominant prior P(FL) ≈ 0.61. Добавление архитектур не привносит новой информации, а добавляет шум.

Validation walk-forward (last 25%) скорее всего имеет другой ratio classes vs training set → stacker overfits на training prior и predicts FLAT для всего validation.

## Что точно не работает (на этом cache)

1. **Vol-aware labels один (без SSL/multi-task)** — улучшает per-arch prec_NF на 4-9pp но не транслирует в +EV
2. **Mixing volaware с original archs** — distribution mismatch → −30 до −50% net
3. **Adding more archs** в pure volaware ансамбль — collapses stacker (5 archs хуже 2 archs)
4. **Optuna heavy regularization params** — убивает signal в small feature space
5. **SSL pretraining via future_ret regression** — NaN gradients, не сходится

## Что не было проверено (могло бы помочь)

1. **Multi-task aux loss transformer** (cls + regr + recon) — требует переписать transformer model
2. **Calibration** softmax (isotonic / Platt) перед stacker — могло бы fixed margin issue
3. **Margin-based entry** вместо max-prob threshold — stacker margin < 0.05, нужна другая metric
4. **5 transformer'ов с разными seeds (bagging)** — был stop'ed, только seed 43 fold 0-2
5. **Actual train walk-forward на recorder data** (post-cache) — нужны новые данные

## Финальный вывод

**Vol-aware label engineering** на текущем cache (148K working / 37K holdout) **не приводит к +EV ансамблю**. Best в этом эксперименте (−2.73%) хуже original baseline (−1.83%).

Per-arch precision улучшилась на vol-aware labels (+4-9pp prec_NF), но ансамбль не использует эту information productively — stacker либо collapses на FL, либо предсказывает direction с margin < 0.05 (random noise).

**Реалистичный путь к +EV:**
1. Накопить ≥30 days post-cache recorder data → новый CPCV walk-forward на свежих данных
2. ETH-BTC lead-lag и другие cross-asset features
3. Multi-task transformer (классификация + регрессия PnL + reconstruction)
4. Pivot strategy: longer holding zone, different asset, или RL action policy

## Артефакты

- `scripts/gen_volaware_labels.py` — vol-aware label generator
- `scripts/build_stacker_from_oof.py`, `build_stacker_opt.py` — stacker fitters
- `scripts/grid_volaware_static.py` — vol-aware grid sweep
- `scripts/optuna_stacker.py` — XGBoost hparam search
- `scripts/pretrain_ssl_transformer.py` — SSL (failed)
- `data/_cache/samples_v3_999h_1777219216_volaware_*` — vol-aware cache

В GCS:
- `gs://scalper-bot-research-data/checkpoints_volaware/{transformer,patchtst,mamba,chronos_bolt_small,timesfm_2p5_200m,hybrid_mamba_attn}/`
- `gs://scalper-bot-research-data/oof_volaware/{transformer,patchtst,mamba,chronos_bolt_small,hybrid_mamba_attn}/`

## Стоимость

- TPU v6e-1 spot × 4 VMs × ~10 hours = ~$8
- CPU96 (n2-standard-96) on-demand × ~5 часов = ~$17
- GCS egress + storage = <$2

Total: ~$25-30
