//! Tardis `incremental_book_L2` depth parser — single-binary, no Python.
//!
//! Reads a gzipped Tardis Binance-Futures depth CSV and writes fixed-size
//! binary records (one per 100ms snapshot) matching this layout:
//!
//!     i64     snapshot_timestamp_ms  (little-endian)
//!     [f64; 40] bids   (price0, qty0, price1, qty1, ..., price19, qty19) LE
//!     [f64; 40] asks   same layout, ascending price
//!
//! Total record size: 8 + 320 + 320 = 648 bytes. Zero-padded if the book
//! holds fewer than 20 levels on a side.
//!
//! The Python wrapper reads these records via `numpy.memmap`, reshapes into
//! list-of-tuples layout, and writes the final parquet matching our
//! recorder's schema.
//!
//! Usage:
//!     depth_parser <input.csv.gz> <output.bin>
//!
//! Performance target: ~500k-1M input rows/sec on a modern CPU — roughly
//! 1-2 min per day of BTCUSDT LOB data (~50M rows). Parallelism is at the
//! day-level (Python spawns one process per day).

use std::collections::BTreeMap;
use std::env;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use flate2::read::GzDecoder;
use ordered_float::OrderedFloat;
use serde::Deserialize;

const DEPTH_LEVELS: usize = 20;
const SNAPSHOT_INTERVAL_MS: i64 = 100;
const WARMUP_MINUTES: i64 = 30;

#[derive(Debug, Deserialize)]
struct DepthRow {
    // Tardis CSV columns we care about.
    timestamp: i64,       // microseconds
    #[serde(default)]
    is_snapshot: bool,    // not used for output logic (we always rebuild)
    side: String,         // "bid" or "ask"
    price: f64,
    amount: f64,          // 0 = delete level
}

fn parse_depth(input: &Path, output: &Path) -> Result<u64> {
    let t0 = Instant::now();

    let f = File::open(input).with_context(|| format!("open {:?}", input))?;
    let gz = GzDecoder::new(BufReader::with_capacity(16 << 20, f));
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(gz);

    let out_file = File::create(output).with_context(|| format!("create {:?}", output))?;
    let mut out = BufWriter::with_capacity(16 << 20, out_file);

    // Bids and asks keyed by OrderedFloat<f64> so BTreeMap orders them.
    // Iteration: bids in reverse (highest price first), asks forward.
    let mut bids: BTreeMap<OrderedFloat<f64>, f64> = BTreeMap::new();
    let mut asks: BTreeMap<OrderedFloat<f64>, f64> = BTreeMap::new();

    let mut next_snap_ts: Option<i64> = None;
    let mut snapshots: u64 = 0;
    let mut rows_read: u64 = 0;

    // Reuse one 648-byte buffer for each record.
    let mut rec_buf: [u8; 648] = [0u8; 648];

    // Use raw record iterator for speed — avoids string allocation on "side".
    // We still deserialize field-by-field with minimal allocations.
    let headers = rdr.headers()?.clone();
    let col = |name: &str| -> Option<usize> {
        headers.iter().position(|h| h == name)
    };
    let col_ts = col("timestamp").context("no 'timestamp' column")?;
    let col_side = col("side").context("no 'side' column")?;
    let col_price = col("price").context("no 'price' column")?;
    let col_amount = col("amount").context("no 'amount' column")?;

    let mut record = csv::ByteRecord::new();
    while rdr.read_byte_record(&mut record)? {
        rows_read += 1;
        // Parse the 4 fields we care about.
        let ts_us: i64 = atoi(record.get(col_ts).context("row too short")?)?;
        let ts_ms = ts_us / 1000;
        let side = record.get(col_side).context("row too short")?;
        let price: f64 = atof(record.get(col_price).context("row too short")?)?;
        let amount: f64 = atof(record.get(col_amount).context("row too short")?)?;

        if next_snap_ts.is_none() {
            let first = ts_ms + WARMUP_MINUTES * 60 * 1000;
            let aligned = (first / SNAPSHOT_INTERVAL_MS) * SNAPSHOT_INTERVAL_MS;
            next_snap_ts = Some(aligned);
        }

        // Emit snapshots for every 100ms boundary we've just passed.
        while let Some(snap_ts) = next_snap_ts {
            if ts_ms < snap_ts {
                break;
            }
            if !bids.is_empty() && !asks.is_empty() {
                write_snapshot(&mut out, &mut rec_buf, snap_ts, &bids, &asks)?;
                snapshots += 1;
            }
            next_snap_ts = Some(snap_ts + SNAPSHOT_INTERVAL_MS);
        }

        // Apply this update.
        let key = OrderedFloat(price);
        let book = match side {
            b"bid" => &mut bids,
            b"ask" => &mut asks,
            other => bail!("unknown side: {:?}", std::str::from_utf8(other)),
        };
        if amount > 0.0 {
            book.insert(key, amount);
        } else {
            book.remove(&key);
        }
    }

    out.flush()?;

    let dt = t0.elapsed();
    eprintln!(
        "depth_parser: {} rows in, {} snapshots out, {:.2}s ({:.0} rows/s)",
        rows_read,
        snapshots,
        dt.as_secs_f64(),
        rows_read as f64 / dt.as_secs_f64().max(1e-9),
    );
    Ok(snapshots)
}

#[inline]
fn write_snapshot(
    out: &mut BufWriter<File>,
    rec_buf: &mut [u8; 648],
    ts_ms: i64,
    bids: &BTreeMap<OrderedFloat<f64>, f64>,
    asks: &BTreeMap<OrderedFloat<f64>, f64>,
) -> Result<()> {
    // Zero out buffer (for level padding).
    rec_buf.fill(0);
    rec_buf[0..8].copy_from_slice(&ts_ms.to_le_bytes());

    // Bids: BTreeMap iterates ascending; reverse to get highest price first.
    let mut off = 8;
    let mut count = 0usize;
    for (p, q) in bids.iter().rev() {
        if count >= DEPTH_LEVELS {
            break;
        }
        rec_buf[off..off + 8].copy_from_slice(&p.into_inner().to_le_bytes());
        rec_buf[off + 8..off + 16].copy_from_slice(&q.to_le_bytes());
        off += 16;
        count += 1;
    }
    // Asks: BTreeMap iterates ascending (lowest price first).
    off = 8 + DEPTH_LEVELS * 16;
    count = 0;
    for (p, q) in asks.iter() {
        if count >= DEPTH_LEVELS {
            break;
        }
        rec_buf[off..off + 8].copy_from_slice(&p.into_inner().to_le_bytes());
        rec_buf[off + 8..off + 16].copy_from_slice(&q.to_le_bytes());
        off += 16;
        count += 1;
    }
    out.write_all(rec_buf)?;
    Ok(())
}

#[inline]
fn atoi(bytes: &[u8]) -> Result<i64> {
    std::str::from_utf8(bytes)?
        .parse::<i64>()
        .with_context(|| format!("atoi: {:?}", std::str::from_utf8(bytes)))
}

#[inline]
fn atof(bytes: &[u8]) -> Result<f64> {
    std::str::from_utf8(bytes)?
        .parse::<f64>()
        .with_context(|| format!("atof: {:?}", std::str::from_utf8(bytes)))
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        bail!(
            "usage: {} <input.csv.gz> <output.bin>\n\
             \n\
             Reads a Tardis `incremental_book_L2` CSV, emits 648-byte\n\
             fixed-size binary snapshots (i64 ts + 20 bid levels + 20 ask levels).",
            args.first().map(|s| s.as_str()).unwrap_or("depth_parser")
        );
    }
    let n = parse_depth(Path::new(&args[1]), Path::new(&args[2]))?;
    println!("{}", n);
    Ok(())
}
