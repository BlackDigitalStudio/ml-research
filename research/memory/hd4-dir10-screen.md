# HD4 dir10 screen — 10s-направление предсказуемо model-free (stage 1 measured)

**Cell**: CL год (2025-05-09..2026-06-02), tb3s 3s-grid, F71 из maker_labels_tb3s_h150
dailies, forward mid log-ret H∈{5,10,15,20,30,60}s ([H−2,H], без future-slack), daily
rank-IC / hit / signed capture @|x|-cuts. Prereg+measured `dir10-20260712_cl_screen`
(HD4 rev1); RESEARCH_LOG s24.

**Результат**: COMP rank-IC@10s BTC **+0.239** / ETH +0.146 / XRP +0.143 / DOGE +0.119,
все месяцы положительные на всех 4 символах. Argmax = форма книги (OBI_L1/microprice,
imb_L5): BTC ric +0.266, hit@q90 0.666, capture +0.73bp/10s @q90. Порядок символов
ИНВЕРТИРОВАН к h150 maker-EV (BTC там слабейший). IC ~половинится 5s→15s; capture
монотонен по силе сигнала (гейт по «уверенности» имеет на что опираться).

**Почему это важно**: гипотеза-2 юзера (продление h150-холда 10s-квантами по сигналу
направления в fill+150s) прошла stage-1 feasibility: сырьё для гейта есть, сильнее всего
именно на BTC/ETH, куда расширяемся.

**How to apply / что дальше**:
- Stage 2 (нужен prereg): conditional cut на fill+150s h150-сделок + стоимость
  перестановки pegged-exit → EV продления. Daily npz `research_runs/h2_dir10/daily/`
  ({ts,mid,bid0,ask0,R,RV}) построены под это — не пересчитывать заново.
- Кросс-БИРЖЕВЫЕ колонки в CL мертвы → отдельный rev на днях рекордера.
- capture = валовый mid-ход, НЕ EV; hit<0.5 при cap>0 на DOGE/XRP OFI-ячейках =
  асимметричный payoff — читать capture, не hit.
- VM dir10-1 (e2-standard-8, europe-west1-b, gen-lang-client) ОСТАНОВЛЕНА, диск жив —
  рестартовать для stage 2. Скрипты: scripts/subs60_dir10_screen.py / _report.py.
- xsym-32 НЕ стартует пока жив axb-xrp-1 (квота CPUS_ALL_REGIONS=32, 2 vCPU заняты XRP).

Связанные: [[h150-deploy-candidate]], [[xsym-cross-symbol-run]], [[scope-bound-claims]].

**Chain-амендмент (measured 2026-07-12)**: правило юзера «держим ещё 10s пока сигнал жив»
работает при МЯГКОМ пороге продолжения (вход q90/q99, продолжение q50 того же знака):
+0.16..+0.28bp/эпизод сверху первого окна, added>0 в 95–100% дней; строгий порог
(=входному) убивает цепи. Шаги 2–4 положительны на всех 4 символах. Ledger
`dir10-20260712_cl_chain`, артефакты {SYM}_chain.npz.
