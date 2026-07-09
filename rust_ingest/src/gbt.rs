//! Minimal bit-exact XGBoost (gbtree, binary:logistic) predictor for the h150
//! deploy bundle. Parses the standard `save_model(*.json)` format; traversal and
//! accumulation mirror xgboost's CPU predictor: f32 threshold compare (`x < cond`
//! -> left, NaN -> default_left), leaf value = `split_conditions` at leaf nodes,
//! margin accumulated in f32 in tree order, prob = sigmoid_f32(margin + logit(base)).
//! Verified bitwise against Python xgboost predictions by `bin/score_harness`.

use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Deserialize)]
struct ModelFile {
    learner: Learner,
}
#[derive(Deserialize)]
struct Learner {
    gradient_booster: Booster,
    learner_model_param: LearnerModelParam,
}
#[derive(Deserialize)]
struct Booster {
    model: BoosterModel,
}
#[derive(Deserialize)]
struct BoosterModel {
    trees: Vec<TreeJson>,
}
#[derive(Deserialize)]
struct LearnerModelParam {
    base_score: String,
}
#[derive(Deserialize)]
struct TreeJson {
    left_children: Vec<i32>,
    right_children: Vec<i32>,
    split_indices: Vec<u32>,
    split_conditions: Vec<f32>,
    default_left: Vec<i32>,
}

pub struct Tree {
    left: Vec<i32>,
    right: Vec<i32>,
    feat: Vec<u32>,
    cond: Vec<f32>,
    default_left: Vec<bool>,
}

pub struct Gbt {
    trees: Vec<Tree>,
    base_margin: f32,
}

impl Gbt {
    pub fn load_json(path: &std::path::Path) -> Result<Self> {
        let raw = std::fs::read(path).with_context(|| format!("read {:?}", path))?;
        let mf: ModelFile = serde_json::from_slice(&raw).context("parse xgboost json")?;
        // newer xgboost writes base_score as a bracketed vector, e.g. "[5E-1]"
        let bs_raw = mf.learner.learner_model_param.base_score;
        let base: f32 = bs_raw
            .trim_matches(|c| c == '[' || c == ']')
            .parse()
            .with_context(|| format!("base_score {:?}", bs_raw))?;
        // binary:logistic ProbToMargin float formula. xgboost's internal value can
        // differ by 1 ulp (observed on one of 8 deploy models) — for bit parity the
        // boot pipeline SOLVES the exact bits from a one-tree prediction and passes
        // them via `set_base_margin`; this formula is only the fallback.
        let base_margin = -(1.0f32 / base - 1.0f32).ln();
        let trees = mf
            .learner
            .gradient_booster
            .model
            .trees
            .into_iter()
            .map(|t| Tree {
                default_left: t.default_left.iter().map(|&v| v != 0).collect(),
                left: t.left_children,
                right: t.right_children,
                feat: t.split_indices,
                cond: t.split_conditions,
            })
            .collect();
        Ok(Self { trees, base_margin })
    }

    /// Override the base margin with exactly-solved bits (boot artifact).
    pub fn set_base_margin(&mut self, v: f32) {
        self.base_margin = v;
    }


    /// f32 sum of leaf values only (no base) in tree order.
    #[inline]
    pub fn leaf_sum(&self, x: &[f32]) -> f32 {
        self.leaf_sum_from(0.0f32, x)
    }

    /// f32 accumulation starting from `init` (base-first order test).
    #[inline]
    pub fn leaf_sum_from(&self, init: f32, x: &[f32]) -> f32 {
        let mut m: f32 = init;
        for t in &self.trees {
            let mut n: usize = 0;
            loop {
                let l = t.left[n];
                if l == -1 {
                    m += t.cond[n];
                    break;
                }
                let v = x[t.feat[n] as usize];
                n = if v.is_nan() {
                    if t.default_left[n] { l as usize } else { t.right[n] as usize }
                } else if v < t.cond[n] {
                    l as usize
                } else {
                    t.right[n] as usize
                };
            }
        }
        m
    }

    /// Margin (pre-sigmoid): f32 accumulation starting FROM base_margin, then leaf
    /// values in tree order — the empirically verified xgboost CPU order (base-first
    /// matched 28546/28546 on both deploy model families; base-last was 1-2 ulp off
    /// wherever base != -0.0).
    #[inline]
    pub fn margin(&self, x: &[f32]) -> f32 {
        self.leaf_sum_from(self.base_margin, x)
    }

    #[inline]
    pub fn predict_prob(&self, x: &[f32]) -> f32 {
        let m = self.margin(x);
        1.0f32 / (1.0f32 + (-m).exp())
    }
}
