//! hd1_seq_build — HD1-seq heavy data path (Rust; Python only orchestrates).
//!
//! FROZEN under HD1 rev25 (research/hypotheses.jsonl, freeze d56344f).
//! Reads ONE symbol-day's raw 20-level cryptolake LOB parquet (flat
//! schema bid_{k}_price/size, ask_{k}_price/size, int64 ns `timestamp`),
//! plus the features_v1 decision-point indices (.npy i64), and emits the
//! packed per-tick L2 context windows + first-passage labels.
//!
//! The label/scope/window math is BIT-EXACT to the frozen Python
//! contract (scripts/hd1_seq_core.py == ha5_screen._first_passage):
//! decision points ok=(idx>0)&(idx<n-1)[::4]; per H jH=searchsorted(
//! ts,t0+H*1e9,"left") clamped; first-passage up=m0*(1+f)/dn=m0*(1-f),
//! scan mid[i0+1..=jH], first-hit, tie u<=d -> +1. Verified by
//! tests/test_hd1_parity_rust.py before any sweep (parity gate).
//!
//! Usage:
//!   hd1_seq_build --book A.parquet [B.parquet ...] \
//!     --indices idx.npy --out-dir DIR [--max-l 512]
//!
//! Outputs (.npy, C-order, little-endian) in DIR:
//!   X.npy        (n_dp, L, 46) f32  causal per-tick feature windows
//!   i.npy        (n_dp,)       i64  book row index of each decision pt
//!   t0.npy       (n_dp,)       i64  ns timestamp at decision pt
//!   y0_{H}.npy   (n_dp,)       i8   first-passage {+1,-1,0}  H in 180/300/600
//!   rH_{H}.npy   (n_dp,)       f32  forward log-return ln(mid[jH]/m0)

use std::fs::File;
use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use arrow::array::{Array, Float64Array, Int64Array};
use clap::Parser;
use ndarray::{Array1, Array3};
use ndarray_npy::{read_npy, write_npy};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

// ---- frozen constants (== scripts/hd1_seq_core.py) ---------------------
const STRIDE: usize = 4;
const NS: i64 = 1_000_000_000;
const F_T0: f64 = 0.0013;
const N_LEVELS: usize = 20;
const N_TICK_FEAT: usize = 2 * N_LEVELS + 6; // 46
const HS: [i64; 3] = [180, 300, 600];

#[derive(Parser, Debug)]
#[command(about = "HD1-seq packed L2 windows + first-passage labels")]
struct Args {
    /// raw/book parquet file(s) for ONE symbol-day, in read order
    #[arg(long, num_args = 1.., required = true)]
    book: Vec<PathBuf>,
    /// features_v1 indices.npy (i64) — decision-point grid
    #[arg(long, required = true)]
    indices: PathBuf,
    /// output directory
    #[arg(long, required = true)]
    out_dir: PathBuf,
    /// max context length (windows stored at this L; smaller L = slice)
    #[arg(long, default_value_t = 512)]
    max_l: usize,
}

/// Flat 20-level cryptolake book. Row order preserved; files concatenated
/// in argv order.
struct Book {
    ts: Vec<i64>,
    bid_p: Vec<f64>, // row-major (n, 20)
    bid_s: Vec<f64>,
    ask_p: Vec<f64>,
    ask_s: Vec<f64>,
}

fn read_books(paths: &[PathBuf]) -> Result<Book> {
    let mut ts = Vec::<i64>::new();
    let (mut bid_p, mut bid_s) = (Vec::<f64>::new(), Vec::<f64>::new());
    let (mut ask_p, mut ask_s) = (Vec::<f64>::new(), Vec::<f64>::new());

    let names: Vec<String> = {
        let mut v = vec!["timestamp".to_string()];
        for k in 0..N_LEVELS {
            v.push(format!("bid_{k}_price"));
        }
        for k in 0..N_LEVELS {
            v.push(format!("bid_{k}_size"));
        }
        for k in 0..N_LEVELS {
            v.push(format!("ask_{k}_price"));
        }
        for k in 0..N_LEVELS {
            v.push(format!("ask_{k}_size"));
        }
        v
    };

    for path in paths {
        let file = File::open(path).with_context(|| format!("open {path:?}"))?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let schema = builder.schema().clone();
        let idx_of = |n: &str| -> Result<usize> {
            schema
                .index_of(n)
                .with_context(|| format!("book parquet missing column `{n}`"))
        };
        let col_idx: Vec<usize> =
            names.iter().map(|n| idx_of(n)).collect::<Result<_>>()?;
        let reader = builder.build()?;
        for b in reader {
            let b = b?;
            let nrows = b.num_rows();
            let tcol = b
                .column(col_idx[0])
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| anyhow!("timestamp not Int64"))?;
            ts.extend_from_slice(tcol.values());

            // dst slices: levels 0..20 contiguous per group; we want
            // row-major (row, level) so push per row across levels.
            let f64col = |gi: usize, k: usize| -> Result<&Float64Array> {
                b.column(col_idx[1 + gi * N_LEVELS + k])
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| anyhow!("depth col not Float64"))
            };
            let grp = |gi: usize, dst: &mut Vec<f64>| -> Result<()> {
                let cols: Vec<&Float64Array> =
                    (0..N_LEVELS).map(|k| f64col(gi, k)).collect::<Result<_>>()?;
                for r in 0..nrows {
                    for c in &cols {
                        dst.push(c.value(r));
                    }
                }
                Ok(())
            };
            grp(0, &mut bid_p)?;
            grp(1, &mut bid_s)?;
            grp(2, &mut ask_p)?;
            grp(3, &mut ask_s)?;
        }
    }
    let n = ts.len();
    if bid_p.len() != n * N_LEVELS {
        return Err(anyhow!("book row/col mismatch: ts={n} bidp={}", bid_p.len()));
    }
    Ok(Book {
        ts,
        bid_p,
        bid_s,
        ask_p,
        ask_s,
    })
}

#[inline]
fn nz(x: f64) -> f32 {
    // numpy nan_to_num(nan=0, posinf=0, neginf=0)
    if x.is_finite() {
        x as f32
    } else {
        0.0
    }
}

/// Per-tick 46-feature transform — bit-for-bit hd1_seq_core.tick_features.
fn tick_features(bk: &Book) -> Vec<f32> {
    let n = bk.ts.len();
    let mut f = vec![0.0f32; n * N_TICK_FEAT];
    let at = |v: &Vec<f64>, r: usize, k: usize| v[r * N_LEVELS + k];

    // mid / prev mid
    let mut mid = vec![0.0f64; n];
    for r in 0..n {
        mid[r] = 0.5 * (at(&bk.bid_p, r, 0) + at(&bk.ask_p, r, 0));
    }
    for r in 0..n {
        let b0 = at(&bk.bid_p, r, 0);
        let a0 = at(&bk.ask_p, r, 0);
        let m = mid[r];
        let sm = if m > 0.0 { m } else { 1.0 };
        let base = r * N_TICK_FEAT;
        for k in 0..N_LEVELS {
            let d = at(&bk.bid_s, r, k) - at(&bk.ask_s, r, k);
            let sgn = if d > 0.0 {
                1.0
            } else if d < 0.0 {
                -1.0
            } else {
                0.0
            };
            f[base + k] = nz(sgn * (1.0 + d.abs()).ln());
            let lvl_mid = 0.5 * (at(&bk.bid_p, r, k) + at(&bk.ask_p, r, k));
            f[base + N_LEVELS + k] = nz((lvl_mid - m) / sm);
        }
        // [40] mid log-return since prev tick (0 at first)
        let lr = if r == 0 {
            0.0
        } else if mid[r] > 0.0 && mid[r - 1] > 0.0 {
            (mid[r] / mid[r - 1]).ln()
        } else {
            0.0
        };
        f[base + 40] = nz(lr);
        // [41] relative spread
        f[base + 41] = nz((a0 - b0) / sm);
        // [42] L5 depth imbalance, [43] L20
        let (mut bs5, mut as5, mut bs20, mut as20) = (0.0, 0.0, 0.0, 0.0);
        for k in 0..N_LEVELS {
            let bsk = at(&bk.bid_s, r, k);
            let ask = at(&bk.ask_s, r, k);
            if k < 5 {
                bs5 += bsk;
                as5 += ask;
            }
            bs20 += bsk;
            as20 += ask;
        }
        f[base + 42] = nz((bs5 - as5) / if bs5 + as5 > 0.0 { bs5 + as5 } else { 1.0 });
        f[base + 43] =
            nz((bs20 - as20) / if bs20 + as20 > 0.0 { bs20 + as20 } else { 1.0 });
        // [44] OFI increment (filled in second pass; needs r-1)
        // [45] microprice - mid (mid-relative)
        let bs0 = at(&bk.bid_s, r, 0);
        let as0 = at(&bk.ask_s, r, 0);
        let micro = if bs0 + as0 > 0.0 {
            (b0 * as0 + a0 * bs0) / (bs0 + as0)
        } else {
            m
        };
        f[base + 45] = nz((micro - m) / sm);
    }
    // [44] Cont top-of-book OFI: needs (r-1, r) level-0 price/size
    for r in 1..n {
        let b0 = at(&bk.bid_p, r, 0);
        let b0p = at(&bk.bid_p, r - 1, 0);
        let a0 = at(&bk.ask_p, r, 0);
        let a0p = at(&bk.ask_p, r - 1, 0);
        let bs0 = at(&bk.bid_s, r, 0);
        let bs0p = at(&bk.bid_s, r - 1, 0);
        let as0 = at(&bk.ask_s, r, 0);
        let as0p = at(&bk.ask_s, r - 1, 0);
        let db = b0 - b0p;
        let da = a0 - a0p;
        let e_b = if db > 0.0 {
            bs0
        } else if db < 0.0 {
            -bs0p
        } else {
            bs0 - bs0p
        };
        let e_a = if da < 0.0 {
            as0
        } else if da > 0.0 {
            -as0p
        } else {
            as0 - as0p
        };
        f[r * N_TICK_FEAT + 44] = nz(e_b - e_a);
    }
    f
}

/// searchsorted(ts, target, "left") == ts.partition_point(|x| x < target)
#[inline]
fn lower_bound(ts: &[i64], target: i64) -> usize {
    ts.partition_point(|&x| x < target)
}

fn main() -> Result<()> {
    let a = Args::parse();
    std::fs::create_dir_all(&a.out_dir)?;
    let bk = read_books(&a.book)?;
    let n_ticks = bk.ts.len();
    if n_ticks == 0 {
        return Err(anyhow!("empty book"));
    }
    let idx: Array1<i64> = read_npy(&a.indices).context("read indices.npy")?;

    // mid
    let mut mid = vec![0.0f64; n_ticks];
    for r in 0..n_ticks {
        mid[r] = 0.5 * (bk.bid_p[r * N_LEVELS] + bk.ask_p[r * N_LEVELS]);
    }

    // decision points: ok=(idx>0)&(idx<n-1); then [::STRIDE]; i=idx[sel]
    let nci = n_ticks as i64 - 1;
    let filtered: Vec<usize> = (0..idx.len())
        .filter(|&r| {
            let v = idx[r];
            v > 0 && v < nci
        })
        .collect();
    let i_book: Vec<i64> = filtered
        .iter()
        .step_by(STRIDE)
        .map(|&r| idx[r])
        .collect();
    let n_dp = i_book.len();
    let l = a.max_l;

    if n_dp == 0 {
        // emit empty-but-valid arrays so the orchestrator can skip cleanly
        write_npy(a.out_dir.join("i.npy"), &Array1::<i64>::zeros(0))?;
        write_npy(a.out_dir.join("t0.npy"), &Array1::<i64>::zeros(0))?;
        write_npy(a.out_dir.join("X.npy"), &Array3::<f32>::zeros((0, l, N_TICK_FEAT)))?;
        for h in HS {
            write_npy(a.out_dir.join(format!("y0_{h}.npy")), &Array1::<i8>::zeros(0))?;
            write_npy(a.out_dir.join(format!("rH_{h}.npy")), &Array1::<f32>::zeros(0))?;
        }
        println!("n_dp=0");
        return Ok(());
    }

    let tf = tick_features(&bk); // (n_ticks, 46) f32 row-major

    // causal window gather: rows i + (-(L-1)..=0); pad<0 -> zeros
    let mut x = Array3::<f32>::zeros((n_dp, l, N_TICK_FEAT));
    for (dp, &iv) in i_book.iter().enumerate() {
        for c in 0..l {
            let row = iv + c as i64 - (l as i64 - 1);
            if row < 0 {
                continue; // left pad zeros
            }
            let rc = row.min(n_ticks as i64 - 1) as usize;
            let src = rc * N_TICK_FEAT;
            for k in 0..N_TICK_FEAT {
                x[[dp, c, k]] = tf[src + k];
            }
        }
    }

    let t0: Vec<i64> = i_book.iter().map(|&iv| bk.ts[iv as usize]).collect();

    write_npy(a.out_dir.join("X.npy"), &x)?;
    write_npy(
        a.out_dir.join("i.npy"),
        &Array1::from(i_book.clone()),
    )?;
    write_npy(a.out_dir.join("t0.npy"), &Array1::from(t0.clone()))?;

    for h in HS {
        let mut y0 = Array1::<i8>::zeros(n_dp);
        let mut rh = Array1::<f32>::zeros(n_dp);
        for (dp, &iv) in i_book.iter().enumerate() {
            let i0 = iv as usize;
            let target = bk.ts[i0] + h * NS;
            let jh = lower_bound(&bk.ts, target).min(n_ticks - 1);
            let m0 = mid[i0];
            // first-passage: scan mid[i0+1 ..= jh]
            let up = m0 * (1.0 + F_T0);
            let dn = m0 * (1.0 - F_T0);
            let (mut u, mut d): (i64, i64) = (-1, -1);
            if i0 + 1 <= jh {
                'scan: for (off, r) in (i0 + 1..=jh).enumerate() {
                    let s = mid[r];
                    if u < 0 && s >= up {
                        u = off as i64;
                    }
                    if d < 0 && s <= dn {
                        d = off as i64;
                    }
                    if u >= 0 && d >= 0 {
                        break 'scan;
                    }
                }
            }
            y0[dp] = if u < 0 && d < 0 {
                0
            } else if d < 0 || (u >= 0 && u <= d) {
                1
            } else {
                -1
            };
            rh[dp] = if m0 > 0.0 {
                (mid[jh] / m0).ln() as f32
            } else {
                f32::NAN
            };
        }
        write_npy(a.out_dir.join(format!("y0_{h}.npy")), &y0)?;
        write_npy(a.out_dir.join(format!("rH_{h}.npy")), &rh)?;
    }

    println!("n_dp={n_dp} n_ticks={n_ticks} L={l}");
    Ok(())
}
