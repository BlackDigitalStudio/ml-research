//! build_samples — streaming builder for the training cache.
//!
//! Reads `depth.parquet` row-group by row-group and emits per-sample arrays
//! without ever holding the full depth data in memory. Peak RSS stays around
//! ~1.5 GB regardless of depth size, so 100 h (65M rows) runs comfortably
//! on a 62 GB box alongside Python orchestration.
//!
//! Inputs:
//!   --depth D.parquet       flat schema {ts, bp, bq, ap, aq} FixedSizeList<20>
//!   --trades T.parquet      optional {ts, price, quantity, is_buyer_maker}
//!   --out-dir OUT
//!   --window W              (default 50)
//!   --horizon H             (default 1300)
//!   --step S                (default 2)
//!   --max-samples N         (default 2_000_000)
//!
//! Outputs (all .npy in OUT):
//!   sample_starts.npy  (N,)            i64
//!   end_indices.npy    (N,)            i64
//!   sample_ts.npy      (N,)            i64
//!   entry_long.npy     (N,)            f64
//!   entry_short.npy    (N,)            f64
//!   mid.npy            (N,)            f64
//!   top5_bid.npy       (N,)            f64
//!   top5_ask.npy       (N,)            f64
//!   mid_paths.npy      (N, H)          f64
//!   book_paths.npy     (N, H, 2)       f64   [best_bid, best_ask] per forward tick
//!   entry_book.npy     (N, 2)          f64   [bid_at_entry, ask_at_entry]
//!   X_lob.npy          (N, 3, 20, W)   f32
//!
//! Memory envelope (streaming path):
//!   * depth_ts (i64)              : n × 8 B
//!   * tick_buy/sell_vol (f32)     : n × 8 B
//!   * sliding buffer of bp0/ap0/bq20/aq20 : ≤ 2 batches × ~1M × 336 B ≈ 670 MB
//!   * output X_lob / mid_paths    : mmap-backed .npy, disk, minimal RAM
//! Peak RSS ≈ 1.0-1.5 GB even at 65M depth rows.

use std::fs::File;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use arrow::array::{Array, BooleanArray, FixedSizeListArray, Float64Array, Int64Array};
use clap::Parser;
use ndarray::{Array1, Array2};
use ndarray_npy::{write_npy, WriteNpyExt};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

const DEPTH_LEVELS: usize = 20;

#[derive(Parser, Debug)]
#[command(about = "Streaming cache builder: depth + trades → per-sample .npy arrays.")]
struct Args {
    #[arg(long)]
    depth: PathBuf,
    #[arg(long)]
    trades: Option<PathBuf>,
    #[arg(long, value_name = "OUT_DIR")]
    out_dir: PathBuf,
    #[arg(long, default_value_t = 50)]
    window: i64,
    #[arg(long, default_value_t = 1300)]
    horizon: i64,
    #[arg(long, default_value_t = 2)]
    step: i64,
    #[arg(long, default_value_t = 2_000_000)]
    max_samples: i64,
}

/// Compact per-row slice of depth kept in RAM across batches: top-1 prices,
/// full top-20 qtys, timestamp.
struct SlimBatch {
    global_start: usize,  // global row index of row 0 of this batch
    ts: Vec<i64>,
    bp0: Vec<f64>,
    ap0: Vec<f64>,
    bq: Vec<f64>, // row-major (n × 20)
    aq: Vec<f64>,
}

impl SlimBatch {
    fn n(&self) -> usize {
        self.ts.len()
    }
}

/// First pass: stream depth, capture only the timestamp column into a
/// pre-sized Vec. Also counts total rows — reported by the reader's first
/// batch metadata.
fn read_depth_timestamps(path: &PathBuf) -> Result<Vec<i64>> {
    let file = File::open(path).with_context(|| format!("open {:?}", path))?;
    let metadata = parquet::file::reader::SerializedFileReader::new(
        file.try_clone()?,
    )?;
    use parquet::file::reader::FileReader;
    let total_rows: usize = metadata.metadata().file_metadata().num_rows() as usize;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    // Column projection: just `timestamp`.
    let schema = builder.schema();
    let ts_col = schema.index_of("timestamp")
        .context("depth parquet missing `timestamp`")?;
    let mask = parquet::arrow::ProjectionMask::roots(
        builder.parquet_schema(), vec![ts_col],
    );
    let reader = builder.with_projection(mask).build()?;

    let mut out = Vec::<i64>::with_capacity(total_rows);
    for b in reader {
        let b = b?;
        let ts = b.column(0).as_any().downcast_ref::<Int64Array>()
            .ok_or_else(|| anyhow!("timestamp column not Int64"))?;
        out.extend_from_slice(ts.values());
    }
    if out.len() != total_rows {
        return Err(anyhow!(
            "depth row count mismatch: metadata={}, scanned={}",
            total_rows, out.len(),
        ));
    }
    Ok(out)
}

/// Decode one RecordBatch into a SlimBatch (top-1 prices + top-20 qtys + ts).
fn decode_slim(
    b: &arrow::record_batch::RecordBatch,
    global_start: usize,
    ts_idx: usize, bp_idx: usize, bq_idx: usize, ap_idx: usize, aq_idx: usize,
) -> Result<SlimBatch> {
    let n = b.num_rows();
    let ts = b.column(ts_idx).as_any().downcast_ref::<Int64Array>()
        .ok_or_else(|| anyhow!("timestamp not Int64"))?;
    let fsl = |col_idx: usize, name: &str| -> Result<&FixedSizeListArray> {
        b.column(col_idx).as_any().downcast_ref::<FixedSizeListArray>()
            .ok_or_else(|| anyhow!("{} not FixedSizeList", name))
    };
    let bp_fsl = fsl(bp_idx, "bid_prices")?;
    let bq_fsl = fsl(bq_idx, "bid_qtys")?;
    let ap_fsl = fsl(ap_idx, "ask_prices")?;
    let aq_fsl = fsl(aq_idx, "ask_qtys")?;
    fn floats<'a>(fsl_arr: &'a FixedSizeListArray, n: usize, name: &str) -> Result<&'a [f64]> {
        let v = fsl_arr.values().as_any().downcast_ref::<Float64Array>()
            .ok_or_else(|| anyhow!("{} child not Float64", name))?;
        let off = fsl_arr.offset() * DEPTH_LEVELS;
        Ok(&v.values()[off..off + n * DEPTH_LEVELS])
    }
    let bp_all = floats(bp_fsl, n, "bid_prices")?;
    let ap_all = floats(ap_fsl, n, "ask_prices")?;
    let bq_all = floats(bq_fsl, n, "bid_qtys")?;
    let aq_all = floats(aq_fsl, n, "ask_qtys")?;

    let mut bp0 = Vec::with_capacity(n);
    let mut ap0 = Vec::with_capacity(n);
    for i in 0..n {
        bp0.push(bp_all[i * DEPTH_LEVELS]);
        ap0.push(ap_all[i * DEPTH_LEVELS]);
    }
    Ok(SlimBatch {
        global_start,
        ts: ts.values().to_vec(),
        bp0,
        ap0,
        bq: bq_all.to_vec(),
        aq: aq_all.to_vec(),
    })
}

fn main() -> Result<()> {
    let a = Args::parse();
    std::fs::create_dir_all(&a.out_dir)?;

    // === Pass 1: depth_ts + compute sample_starts ===
    let t_pass1 = Instant::now();
    let depth_ts = read_depth_timestamps(&a.depth)?;
    let n = depth_ts.len();
    eprintln!("build_samples: depth_ts loaded n={} in {:.2}s",
              n, t_pass1.elapsed().as_secs_f64());

    let w = a.window;
    let h = a.horizon;
    if (n as i64) <= w + h + 1 {
        anyhow::bail!("depth has {} rows, need > {}", n, w + h + 1);
    }
    let total = n as i64 - w - h;
    let mut step = a.step.max(1);
    if total > a.max_samples * 2 {
        step = 2 * ((total + a.max_samples - 1) / a.max_samples);
        step = step.max(2);
    }
    let sample_starts: Vec<i64> = (0..total).step_by(step as usize).collect();
    let ns = sample_starts.len();
    let end_indices: Vec<i64> = sample_starts.iter().map(|&s| s + w - 1).collect();
    eprintln!(
        "build_samples: step={} num_samples={} (max={}, total={})",
        step, ns, a.max_samples, total
    );

    // === Pass 2a: trades → tick_buy_vol / tick_sell_vol ===
    let mut tick_buy_vol = vec![0f32; n];
    let mut tick_sell_vol = vec![0f32; n];
    if let Some(trades_path) = a.trades.as_ref() {
        let t_trades = Instant::now();
        let file = File::open(trades_path)
            .with_context(|| format!("open {:?}", trades_path))?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let schema = builder.schema();
        let ts_col = schema.index_of("timestamp")?;
        let qty_col = schema.index_of("quantity")?;
        let side_col = schema.index_of("is_buyer_maker")?;
        let mask = parquet::arrow::ProjectionMask::roots(
            builder.parquet_schema(), vec![ts_col, qty_col, side_col],
        );
        let reader = builder.with_projection(mask).build()?;
        let mut nt = 0_usize;
        for b in reader {
            let b = b?;
            // Columns in projection order: ts, qty, side.
            let ts = b.column(0).as_any().downcast_ref::<Int64Array>()
                .ok_or_else(|| anyhow!("trades ts not Int64"))?;
            let qty = b.column(1).as_any().downcast_ref::<Float64Array>()
                .ok_or_else(|| anyhow!("trades quantity not Float64"))?;
            let sdl = b.column(2).as_any().downcast_ref::<BooleanArray>()
                .ok_or_else(|| anyhow!("trades is_buyer_maker not Bool"))?;
            for i in 0..b.num_rows() {
                let t = ts.value(i);
                let q = qty.value(i) as f32;
                // searchsorted_right(depth_ts, t) - 1, clipped.
                let r = depth_ts.partition_point(|probe| *probe <= t);
                let idx = if r == 0 { 0 } else { (r - 1).min(n - 1) };
                if sdl.value(i) {
                    tick_sell_vol[idx] += q;
                } else {
                    tick_buy_vol[idx] += q;
                }
            }
            nt += b.num_rows();
        }
        eprintln!(
            "build_samples: {} trades aggregated in {:.2}s",
            nt, t_trades.elapsed().as_secs_f64()
        );
    }

    // === Preallocate output arrays (small) + mmap .npy outputs for big ones ===
    let mut entry_long = Array1::<f64>::zeros(ns);
    let mut entry_short = Array1::<f64>::zeros(ns);
    let mut mid_at_sample = Array1::<f64>::zeros(ns);
    let mut sample_ts = Array1::<i64>::zeros(ns);
    let mut top5_bid = Array1::<f64>::zeros(ns);
    let mut top5_ask = Array1::<f64>::zeros(ns);

    // mid_paths and X_lob go to disk — open a sink file, we'll append rows
    // as samples complete.  To keep things simple we first collect samples
    // in order (they emit in increasing s order) and write via ndarray_npy's
    // streaming API (`WriteNpyExt`) only after all are built.  Since we do a
    // single monotonic pass, we can stream-write rows one-at-a-time with a
    // fixed .npy header (shape known upfront).
    let mut mid_paths_writer = NpyRowStream::<f64>::create(
        a.out_dir.join("mid_paths.npy"), &[ns, h as usize],
    )?;
    // book_paths[i, t] = [best_bid(t), best_ask(t)] for the forward path of
    // sample i. Consumed by sim_labels --book-paths (book-aware simulator).
    let mut book_paths_writer = NpyRowStream::<f64>::create(
        a.out_dir.join("book_paths.npy"), &[ns, h as usize, 2],
    )?;
    // flow_paths[i, t] = [tick_buy_vol, tick_sell_vol] at end+1+t — realized
    // taker flow per forward tick. Consumed by the maker-entry simulator
    // (live_sim::simulate_trade_maker) to model resting-limit fills + adverse
    // selection. Additive output; legacy consumers ignore it.
    let mut flow_paths_writer = NpyRowStream::<f32>::create(
        a.out_dir.join("flow_paths.npy"), &[ns, h as usize, 2],
    )?;
    let mut entry_book = Array2::<f64>::zeros((ns, 2));
    // entry_q[i] = [bid_qty0, ask_qty0] at entry — the top-1 resting size a
    // maker order must clear (queue-ahead) when it joins the near side.
    let mut entry_q = Array2::<f64>::zeros((ns, 2));
    let mut x_lob_writer = NpyRowStream::<f32>::create(
        a.out_dir.join("X_lob.npy"), &[ns, 3, DEPTH_LEVELS, w as usize],
    )?;

    // === Pass 2b: stream depth row groups, emit samples in order ===
    let t_stream = Instant::now();
    let file = File::open(&a.depth)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = builder.schema();
    let ts_idx = schema.index_of("timestamp")?;
    let bp_idx = schema.index_of("bid_prices")?;
    let bq_idx = schema.index_of("bid_qtys")?;
    let ap_idx = schema.index_of("ask_prices")?;
    let aq_idx = schema.index_of("ask_qtys")?;
    let reader = builder.build()?;

    // Rolling buffer of SlimBatches. We retain batches that still overlap
    // any "not-yet-emitted" sample's window [s, s+w+h).
    let mut buffer: std::collections::VecDeque<SlimBatch> = std::collections::VecDeque::new();
    let mut buffered_rows: usize = 0;  // sum of batch.n() across buffer
    let mut global_start: usize = 0;   // global row index of buffer.front()
    let mut next_sample_i: usize = 0;

    for b in reader {
        let b = b?;
        let n_b = b.num_rows();
        let slab_start = global_start + buffered_rows;
        let slab = decode_slim(&b, slab_start, ts_idx, bp_idx, bq_idx, ap_idx, aq_idx)?;
        buffered_rows += n_b;
        buffer.push_back(slab);

        // Process as many samples as possible given the current buffered range.
        while next_sample_i < ns {
            let s = sample_starts[next_sample_i] as usize;
            let needs_row = s + w as usize + h as usize - 1;  // inclusive last row used
            // Last buffered row index = global_start + buffered_rows - 1.
            if global_start + buffered_rows <= needs_row {
                break;
            }
            // Emit sample next_sample_i.
            emit_sample(
                next_sample_i,
                s,
                w as usize,
                h as usize,
                &buffer,
                global_start,
                &tick_buy_vol,
                &tick_sell_vol,
                &depth_ts,
                &mut entry_long,
                &mut entry_short,
                &mut mid_at_sample,
                &mut sample_ts,
                &mut top5_bid,
                &mut top5_ask,
                &mut entry_book,
                &mut entry_q,
                &mut mid_paths_writer,
                &mut book_paths_writer,
                &mut flow_paths_writer,
                &mut x_lob_writer,
            )?;
            next_sample_i += 1;
        }

        // Trim buffer: drop front batches whose last row global index is
        // less than the smallest-still-needed row (= sample_starts[next]).
        if next_sample_i < ns {
            let min_needed = sample_starts[next_sample_i] as usize;
            while let Some(front) = buffer.front() {
                let last_row_of_front = front.global_start + front.n() - 1;
                if last_row_of_front < min_needed {
                    let n_drop = front.n();
                    buffer.pop_front();
                    global_start += n_drop;
                    buffered_rows -= n_drop;
                } else {
                    break;
                }
            }
        } else {
            // All samples emitted; drain buffer.
            while let Some(front) = buffer.pop_front() {
                global_start += front.n();
                buffered_rows -= front.n();
            }
            break;
        }
    }
    if next_sample_i != ns {
        return Err(anyhow!(
            "emitted {} / {} samples; depth likely shorter than expected",
            next_sample_i, ns
        ));
    }
    eprintln!(
        "build_samples: streaming pass done in {:.2}s",
        t_stream.elapsed().as_secs_f64()
    );

    // === Write small arrays + finalise streaming writers ===
    let sample_starts_arr = Array1::<i64>::from_vec(sample_starts.clone());
    let end_indices_arr = Array1::<i64>::from_vec(end_indices.clone());
    write_npy(a.out_dir.join("sample_starts.npy"), &sample_starts_arr)?;
    write_npy(a.out_dir.join("end_indices.npy"), &end_indices_arr)?;
    write_npy(a.out_dir.join("sample_ts.npy"), &sample_ts)?;
    write_npy(a.out_dir.join("mid.npy"), &mid_at_sample)?;
    write_npy(a.out_dir.join("entry_long.npy"), &entry_long)?;
    write_npy(a.out_dir.join("entry_short.npy"), &entry_short)?;
    write_npy(a.out_dir.join("top5_bid.npy"), &top5_bid)?;
    write_npy(a.out_dir.join("top5_ask.npy"), &top5_ask)?;
    write_npy(a.out_dir.join("entry_book.npy"), &entry_book)?;
    write_npy(a.out_dir.join("entry_q.npy"), &entry_q)?;
    mid_paths_writer.finish()?;
    book_paths_writer.finish()?;
    flow_paths_writer.finish()?;
    x_lob_writer.finish()?;
    eprintln!("build_samples: all outputs written ({} samples)", ns);
    Ok(())
}

fn emit_sample(
    i: usize,
    s: usize,
    w: usize,
    h: usize,
    buffer: &std::collections::VecDeque<SlimBatch>,
    global_start: usize,
    tick_buy_vol: &[f32],
    tick_sell_vol: &[f32],
    depth_ts: &[i64],
    entry_long: &mut Array1<f64>,
    entry_short: &mut Array1<f64>,
    mid_at_sample: &mut Array1<f64>,
    sample_ts: &mut Array1<i64>,
    top5_bid: &mut Array1<f64>,
    top5_ask: &mut Array1<f64>,
    entry_book: &mut Array2<f64>,
    entry_q: &mut Array2<f64>,
    mid_paths_writer: &mut NpyRowStream<f64>,
    book_paths_writer: &mut NpyRowStream<f64>,
    flow_paths_writer: &mut NpyRowStream<f32>,
    x_lob_writer: &mut NpyRowStream<f32>,
) -> Result<()> {
    let end = s + w - 1;

    // Fetch per-row values from the buffer; helper maps global_row → (batch_idx, local_row).
    let get = |row: usize| -> (f64, f64, &[f64], &[f64]) {
        let mut off = global_start;
        for batch in buffer.iter() {
            let n = batch.n();
            if row < off + n {
                let l = row - off;
                let bq_row = &batch.bq[l * DEPTH_LEVELS..(l + 1) * DEPTH_LEVELS];
                let aq_row = &batch.aq[l * DEPTH_LEVELS..(l + 1) * DEPTH_LEVELS];
                return (batch.bp0[l], batch.ap0[l], bq_row, aq_row);
            }
            off += n;
        }
        panic!("row {} not in buffer (global_start={})", row, global_start);
    };

    // entry_long, entry_short, mid, sample_ts, top5_*, entry_book
    {
        let (bp0, ap0, bq_row, aq_row) = get(end);
        entry_long[i] = bp0;
        entry_short[i] = ap0;
        entry_book[[i, 0]] = bp0;
        entry_book[[i, 1]] = ap0;
        entry_q[[i, 0]] = bq_row[0];   // top-1 bid resting size (queue-ahead, long)
        entry_q[[i, 1]] = aq_row[0];   // top-1 ask resting size (queue-ahead, short)
        if bp0 > 0.0 && ap0 > 0.0 {
            mid_at_sample[i] = 0.5 * (bp0 + ap0);
        }
        sample_ts[i] = depth_ts[end];
        let mut sb = 0.0;
        let mut sa = 0.0;
        for k in 0..5 {
            sb += bq_row[k];
            sa += aq_row[k];
        }
        top5_bid[i] = sb;
        top5_ask[i] = sa;
    }

    // mid_paths[i, :h] = mid[end+1 .. end+1+h] — stream-write.
    // book_paths[i, :h, :2] = [bid, ask] at end+1+k — stream-write.
    let mut mid_path_row = vec![0f64; h];
    let mut book_path_row = vec![0f64; h * 2];
    let mut flow_path_row = vec![0f32; h * 2];
    for k in 0..h {
        let row = end + 1 + k;
        let (bp0, ap0, _bq, _aq) = get(row);
        if bp0 > 0.0 && ap0 > 0.0 {
            mid_path_row[k] = 0.5 * (bp0 + ap0);
        }
        book_path_row[k * 2] = bp0;
        book_path_row[k * 2 + 1] = ap0;
        flow_path_row[k * 2] = tick_buy_vol[row];
        flow_path_row[k * 2 + 1] = tick_sell_vol[row];
    }
    mid_paths_writer.write_row(&mid_path_row)?;
    book_paths_writer.write_row(&book_path_row)?;
    flow_paths_writer.write_row(&flow_path_row)?;

    // X_lob[i] = (3, 20, w) f32. Channel 0/1: bid/ask qtys per level per tick.
    // Channel 2 row 0/1: tick_buy/sell_vol at each tick. Rest zero.
    let mut x_lob_row = vec![0f32; 3 * DEPTH_LEVELS * w];
    // Row layout: channel × level × tick_offset (ndarray default C-order).
    for off in 0..w {
        let row = s + off;
        let (_bp0, _ap0, bq_row, aq_row) = get(row);
        for k in 0..DEPTH_LEVELS {
            // X[0, k, off]
            x_lob_row[0 * DEPTH_LEVELS * w + k * w + off] = bq_row[k] as f32;
            x_lob_row[1 * DEPTH_LEVELS * w + k * w + off] = aq_row[k] as f32;
        }
        x_lob_row[2 * DEPTH_LEVELS * w + 0 * w + off] = tick_buy_vol[row];
        x_lob_row[2 * DEPTH_LEVELS * w + 1 * w + off] = tick_sell_vol[row];
    }
    x_lob_writer.write_row(&x_lob_row)?;
    Ok(())
}

/// Minimal "write one row at a time" helper for .npy files with a known shape.
/// Writes the .npy header once at creation time using ndarray_npy's `WriteNpyExt`
/// on a zero-size array, then appends raw bytes for each row.
///
/// Rationale: ndarray_npy's official API expects a fully materialised array.
/// For samples × (3 × 20 × 50 × 4 B) ≈ 12 KB per sample at 1M samples that's
/// 12 GB — too much to hold before write. Row streaming keeps RAM at one row.
struct NpyRowStream<T: ndarray_npy::WritableElement + Copy + Default + 'static> {
    file: File,
    expected_rows: usize,
    rows_written: usize,
    row_len: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: ndarray_npy::WritableElement + Copy + Default + 'static> NpyRowStream<T> {
    fn create(path: std::path::PathBuf, shape: &[usize]) -> Result<Self> {
        assert!(!shape.is_empty(), "shape must have >=1 dim");
        let expected_rows = shape[0];
        let row_len: usize = shape[1..].iter().product::<usize>().max(1);
        // Write the full header + zero-init body via ndarray, then reopen for
        // append-overwrite. Cheap: (rows × row_len × sizeof<T>) zeros on disk
        // would be huge, so we instead write only the header by hand.
        //
        // Easier + safe approach: use `ndarray_npy::write_zeroed_npy` if
        // available, otherwise write an empty row-shaped file via
        // `WriteNpyExt` then seek past the body. We go with the explicit
        // header write to avoid the dependency on an unstable crate feature.
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .with_context(|| format!("open {:?}", path))?;
        write_npy_header::<T>(&file, shape)?;
        // File cursor is now just past the header; subsequent writes append
        // data in the correct spot.
        Ok(Self {
            file,
            expected_rows,
            rows_written: 0,
            row_len,
            _phantom: std::marker::PhantomData,
        })
    }

    fn write_row(&mut self, row: &[T]) -> Result<()> {
        if row.len() != self.row_len {
            return Err(anyhow!(
                "row len {} != expected {}",
                row.len(), self.row_len
            ));
        }
        use std::io::Write;
        // Cast &[T] to &[u8] in-place (T is Copy + plain-old-data for f32/f64/i64).
        let byte_len = row.len() * std::mem::size_of::<T>();
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(row.as_ptr() as *const u8, byte_len)
        };
        self.file.write_all(bytes)?;
        self.rows_written += 1;
        Ok(())
    }

    fn finish(self) -> Result<()> {
        if self.rows_written != self.expected_rows {
            return Err(anyhow!(
                "wrote {} rows, expected {}",
                self.rows_written, self.expected_rows
            ));
        }
        // File is closed on drop.
        Ok(())
    }
}

/// Write a .npy header directly to a freshly-created file. After this, the
/// file position is right after the header — data writes go straight into
/// the body. Matches numpy .npy v1.0 format.
fn write_npy_header<T: ndarray_npy::WritableElement + 'static>(
    file: &File,
    shape: &[usize],
) -> Result<()> {
    use std::io::{Seek, SeekFrom, Write};
    // Serialise a zero-size array of the right dtype/shape to capture the
    // header bytes, then rewrite with the real shape embedded.
    let dtype_str = ndarray_npy_dtype_string::<T>();
    let shape_str = {
        let mut s = String::from("(");
        for (i, d) in shape.iter().enumerate() {
            if i > 0 { s.push_str(", "); }
            s.push_str(&d.to_string());
        }
        if shape.len() == 1 { s.push_str(","); }
        s.push(')');
        s
    };
    let header_body = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}",
        dtype_str, shape_str
    );
    // Pad to 16-byte alignment (magic + version + header_len + header + '\n').
    let prefix_len = 10; // magic(6) + version(2) + header_len(2)
    let mut header_total = prefix_len + header_body.len() + 1; // +1 for '\n'
    let pad = (16 - header_total % 16) % 16;
    let pad_str: String = " ".repeat(pad);
    let header = format!("{}{}\n", header_body, pad_str);
    header_total = prefix_len + header.len();
    if header.len() > u16::MAX as usize {
        return Err(anyhow!("npy header too long: {}", header.len()));
    }

    let magic: [u8; 6] = [0x93, b'N', b'U', b'M', b'P', b'Y'];
    let version: [u8; 2] = [1, 0];
    let header_len_bytes = (header.len() as u16).to_le_bytes();

    let mut f = file;
    f.seek(SeekFrom::Start(0))?;
    f.write_all(&magic)?;
    f.write_all(&version)?;
    f.write_all(&header_len_bytes)?;
    f.write_all(header.as_bytes())?;
    // Sanity: cursor = header_total.
    let pos = f.stream_position()?;
    if pos as usize != header_total {
        return Err(anyhow!(
            "header position mismatch: {} != {}", pos, header_total
        ));
    }
    Ok(())
}

fn ndarray_npy_dtype_string<T: ndarray_npy::WritableElement + 'static>() -> &'static str {
    // Hard-map the types we actually use.
    let tid = std::any::TypeId::of::<T>();
    if tid == std::any::TypeId::of::<f64>() { "<f8" }
    else if tid == std::any::TypeId::of::<f32>() { "<f4" }
    else if tid == std::any::TypeId::of::<i64>() { "<i8" }
    else if tid == std::any::TypeId::of::<u8>() { "|u1" }
    else { panic!("unsupported npy dtype for streaming") }
}
