#!/usr/bin/env python3
"""
LightGBM Walk-Forward Training Script

Replaces random-split training with proper time-series validation:
- Train on past data, test on future data, roll forward
- Handle class imbalance with scale_pos_weight
- Evaluate by actual PnL, not just accuracy
- Tune probability threshold for optimal trade filtering
- Only save models that improve walk-forward PnL

Usage:
    python scripts/lightgbm_walkforward.py --input data/learning-loop/training_data_*.csv
    python scripts/lightgbm_walkforward.py --input data/h2o-training/paper_scalp_*.csv
"""

import argparse
import json
import os
import sys
import glob
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, classification_report
    )
except ImportError:
    print("ERROR: Required packages not installed.")
    print("Run: pip install lightgbm scikit-learn pandas numpy")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

WALK_FORWARD_CONFIG = {
    'train_months': 3,           # Train on 3 months of data
    'test_months': 1,            # Test on next 1 month
    'min_train_samples': 200,    # Minimum trades needed to train
    'min_test_samples': 30,      # Minimum trades needed for valid test
    'embargo_days': 1,           # Gap between train/test to prevent leakage
    'threshold_search_min': 0.35,
    'threshold_search_max': 0.75,
    'threshold_search_step': 0.02,
    'min_filtered_trades': 10,   # Minimum trades after filtering for valid threshold
    'pnl_improvement_threshold': 0.05,  # 5% improvement needed to save
    'min_features_used': 3,      # Minimum features with importance > 0
}

# Small data mode for paper trading (days instead of months)
SMALL_DATA_CONFIG = {
    'train_pct': 0.70,           # Train on 70% of data
    'test_pct': 0.30,            # Test on 30% of data
    'min_train_samples': 50,     # Lower threshold for small data
    'min_test_samples': 20,      # Lower threshold for small data
    'min_total_samples': 100,    # Minimum total to attempt training
    'threshold_search_min': 0.35,
    'threshold_search_max': 0.75,
    'threshold_search_step': 0.02,
    'min_filtered_trades': 5,    # Lower threshold for small data
    'pnl_improvement_threshold': 0.0,  # Save any model for small data
    'min_features_used': 3,
}

LIGHTGBM_PARAMS = {
    'objective': 'binary',
    'metric': ['binary_logloss', 'auc'],
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'max_depth': 8,
    'min_data_in_leaf': 50,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 1,
    'learning_rate': 0.02,
    'lambda_l1': 0.5,
    'lambda_l2': 0.5,
    'max_bin': 255,
    'verbose': -1,
    'seed': 42,
    'force_col_wise': True,
}


# ═══════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════

@dataclass
class FoldMetrics:
    """Metrics for a single walk-forward fold"""
    fold_idx: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_samples: int
    test_samples: int
    train_win_rate: float
    test_win_rate: float
    scale_pos_weight: float

    # Standard metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    auc_roc: float = 0.0

    # PnL metrics (THE IMPORTANT ONES)
    baseline_trades: int = 0
    baseline_pnl: float = 0.0
    baseline_win_rate: float = 0.0
    baseline_avg_pnl: float = 0.0

    filtered_trades: int = 0
    filtered_pnl: float = 0.0
    filtered_win_rate: float = 0.0
    filtered_avg_pnl: float = 0.0

    pnl_improvement: float = 0.0
    win_rate_improvement: float = 0.0

    optimal_threshold: float = 0.5
    best_iteration: int = 0


@dataclass
class WalkForwardResults:
    """Aggregate results from all walk-forward folds"""
    folds: List[FoldMetrics] = field(default_factory=list)
    best_model: Optional[lgb.Booster] = None
    best_threshold: float = 0.5
    best_fold_idx: int = -1
    feature_names: List[str] = field(default_factory=list)

    @property
    def num_folds(self) -> int:
        return len(self.folds)

    @property
    def total_baseline_pnl(self) -> float:
        return sum(f.baseline_pnl for f in self.folds)

    @property
    def total_filtered_pnl(self) -> float:
        return sum(f.filtered_pnl for f in self.folds)

    @property
    def avg_filtered_win_rate(self) -> float:
        if not self.folds:
            return 0.0
        return np.mean([f.filtered_win_rate for f in self.folds])

    @property
    def avg_accuracy(self) -> float:
        if not self.folds:
            return 0.0
        return np.mean([f.accuracy for f in self.folds])

    @property
    def avg_auc(self) -> float:
        if not self.folds:
            return 0.0
        return np.mean([f.auc_roc for f in self.folds])

    @property
    def avg_fold_pnl(self) -> float:
        if not self.folds:
            return 0.0
        return np.mean([f.filtered_pnl for f in self.folds])

    def add_fold(self, metrics: FoldMetrics):
        self.folds.append(metrics)


# ═══════════════════════════════════════════════════════════════
# DATA LOADING AND PREPARATION
# ═══════════════════════════════════════════════════════════════

def load_training_data(input_path: str) -> pd.DataFrame:
    """Load training data from CSV, handling glob patterns"""

    # Handle glob patterns
    if '*' in input_path:
        files = sorted(glob.glob(input_path))
        if not files:
            raise FileNotFoundError(f"No files match pattern: {input_path}")
        # Use the most recent file
        input_path = files[-1]
        print(f"Using most recent file: {input_path}")

    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Validate required columns
    required = ['entry_time', 'outcome', 'pnl']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[int], List[str]]:
    """Prepare features and target for training"""

    # Target: outcome (WIN=1, LOSS=0)
    y = (df['outcome'] == 'WIN').astype(int)

    # Feature columns (same as original trainer)
    categorical_cols = [
        'trend_direction', 'ob_type', 'fvg_type', 'ema_trend',
        'rsi_state', 'direction', 'session', 'pullback_fib',
        'smart_money_direction', 'ob_state',
        # Momentum features
        'regime',  # 'TREND' or 'RANGE'
        'kill_zone',  # 'LONDON', 'NY_OPEN', 'NY_AFTERNOON', 'ASIA', 'OFF_HOURS'
        'ema_aligned',  # 'bullish', 'bearish', 'neutral' - NOT boolean!
    ]

    numeric_cols = [
        'trend_strength', 'ob_distance', 'ob_size', 'ob_age',
        'fvg_nearest_distance', 'fvg_size', 'fvg_count',
        'volatility', 'rsi_value', 'atr_value',
        'price_position', 'distance_to_high', 'distance_to_low',
        'volume_ratio', 'confluence_score', 'potential_rr',
        'pullback_depth', 'pullback_bars',
        'ob_test_count', 'ob_return_age', 'smc_signal_strength',
        'ob_impulse_size', 'bb_position', 'bb_width', 'volume_delta',
        # Momentum features
        'atr_percent', 'macd_line', 'macd_signal', 'macd_histogram',
        'vwap', 'vwap_deviation', 'vwap_deviation_std',
        'ema_fast', 'ema_slow', 'body_ratio', 'strength',
    ]

    bool_cols = [
        'ob_near', 'fvg_near', 'liquidity_near',
        'mtf_aligned', 'volume_spike', 'trend_bos_aligned', 'is_pullback',
        'ob_is_fresh', 'has_ob_return', 'has_fvg_fill',
        'ob_confirmed_mitigated', 'ob_in_mitigation',
        'ob_caused_bos', 'ob_has_displacement',
        'bb_squeeze', 'bb_breakout_upper', 'bb_breakout_lower',
        'large_volume_spike', 'accumulation_detected', 'distribution_detected',
        # Momentum features
        'macd_bullish_cross', 'macd_bearish_cross',
        'price_above_vwap', 'ema_bullish_cross', 'ema_bearish_cross',
        'is_kill_zone',  # True/False boolean
    ]

    # Filter to available columns
    available_categorical = [c for c in categorical_cols if c in df.columns]
    available_numeric = [c for c in numeric_cols if c in df.columns]
    available_bool = [c for c in bool_cols if c in df.columns]

    # Build feature matrix
    feature_dfs = []

    if available_numeric:
        feature_dfs.append(df[available_numeric].fillna(0))

    if available_bool:
        # Handle bool columns - check for string values that should be categorical
        bool_df = df[available_bool].copy()
        for col in available_bool:
            # Check if column has non-boolean string values
            sample_vals = bool_df[col].dropna().head(100)
            if sample_vals.dtype == object:
                unique_vals = set(str(v).lower() for v in sample_vals.unique())
                # If it has values like 'bullish', 'bearish', 'neutral' - it's categorical, not bool
                if unique_vals - {'true', 'false', '1', '0', '1.0', '0.0'}:
                    print(f"  Warning: '{col}' has string values {unique_vals}, treating as categorical")
                    available_categorical.append(col)
                    continue
            # Convert to int for proper boolean columns
            bool_df[col] = bool_df[col].fillna(False).astype(bool).astype(int)
        # Only keep actual boolean columns
        bool_cols_final = [c for c in available_bool if c not in available_categorical]
        if bool_cols_final:
            feature_dfs.append(bool_df[bool_cols_final])

    cat_feature_names = []
    if available_categorical:
        cat_df = df[available_categorical].copy()
        for col in available_categorical:
            cat_df[col] = cat_df[col].fillna('unknown').astype('category')
            cat_feature_names.append(col)
        feature_dfs.append(cat_df)

    X = pd.concat(feature_dfs, axis=1)
    cat_indices = [X.columns.get_loc(c) for c in cat_feature_names if c in X.columns]

    return X, y, cat_indices, list(X.columns)


def calculate_scale_pos_weight(y: pd.Series) -> float:
    """Calculate class weight for imbalanced data"""
    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    if pos_count == 0:
        return 1.0
    return neg_count / pos_count


# ═══════════════════════════════════════════════════════════════
# WALK-FORWARD SPLITTING
# ═══════════════════════════════════════════════════════════════

def generate_walk_forward_folds(
    min_date: datetime,
    max_date: datetime,
    train_months: int,
    test_months: int,
    embargo_days: int = 1
) -> List[Tuple[datetime, datetime, datetime, datetime]]:
    """
    Generate time-based train/test fold boundaries.

    Returns list of (train_start, train_end, test_start, test_end) tuples.
    """
    folds = []

    # Start first fold
    train_start = min_date

    while True:
        # Calculate boundaries
        train_end = train_start + timedelta(days=train_months * 30)
        test_start = train_end + timedelta(days=embargo_days)
        test_end = test_start + timedelta(days=test_months * 30)

        # Stop if test period exceeds data
        if test_end > max_date:
            break

        folds.append((train_start, train_end, test_start, test_end))

        # Roll forward by test_months
        train_start = train_start + timedelta(days=test_months * 30)

    return folds


# ═══════════════════════════════════════════════════════════════
# PNL METRICS
# ═══════════════════════════════════════════════════════════════

def compute_pnl_metrics(
    y_pred_proba: np.ndarray,
    pnl_values: np.ndarray,
    threshold: float
) -> Dict:
    """
    THE REAL TEST: What's actual P&L if we only take ML-approved trades?
    """
    # Trades the model would approve
    approved_mask = y_pred_proba >= threshold

    # Baseline: take ALL trades
    baseline_trades = len(pnl_values)
    baseline_pnl = float(pnl_values.sum())
    baseline_win_rate = float((pnl_values > 0).mean()) if baseline_trades > 0 else 0.0
    baseline_avg_pnl = baseline_pnl / baseline_trades if baseline_trades > 0 else 0.0

    # Filtered: only ML-approved trades
    filtered_pnl_values = pnl_values[approved_mask]
    filtered_trades = len(filtered_pnl_values)
    filtered_pnl = float(filtered_pnl_values.sum()) if filtered_trades > 0 else 0.0
    filtered_win_rate = float((filtered_pnl_values > 0).mean()) if filtered_trades > 0 else 0.0
    filtered_avg_pnl = filtered_pnl / filtered_trades if filtered_trades > 0 else 0.0

    return {
        'baseline_trades': baseline_trades,
        'baseline_pnl': baseline_pnl,
        'baseline_win_rate': baseline_win_rate,
        'baseline_avg_pnl': baseline_avg_pnl,
        'filtered_trades': filtered_trades,
        'filtered_pnl': filtered_pnl,
        'filtered_win_rate': filtered_win_rate,
        'filtered_avg_pnl': filtered_avg_pnl,
        'pnl_improvement': filtered_pnl - baseline_pnl,
        'win_rate_improvement': filtered_win_rate - baseline_win_rate,
        'trades_filtered_pct': (1 - filtered_trades / baseline_trades) * 100 if baseline_trades > 0 else 0,
    }


def tune_threshold(
    y_pred_proba: np.ndarray,
    pnl_values: np.ndarray,
    config: Dict
) -> Tuple[float, Dict]:
    """
    Find optimal probability threshold using training data.
    Optimizes for average PnL per trade while maintaining reasonable trade count.
    """
    thresholds = np.arange(
        config['threshold_search_min'],
        config['threshold_search_max'] + 0.001,
        config['threshold_search_step']
    )

    best_threshold = 0.5
    best_score = -float('inf')
    best_metrics = None

    for thresh in thresholds:
        metrics = compute_pnl_metrics(y_pred_proba, pnl_values, thresh)

        # Skip if too few trades after filtering
        if metrics['filtered_trades'] < config['min_filtered_trades']:
            continue

        # Scoring function: balance avg PnL with trade frequency
        # Primary: maximize average PnL per trade
        # Secondary: bonus for win rate improvement
        # Penalty: don't filter out too many trades
        trade_retention = 1 - (metrics['trades_filtered_pct'] / 100)

        score = (
            metrics['filtered_avg_pnl'] * 0.6 +
            metrics['win_rate_improvement'] * 100 * 0.25 +
            trade_retention * 5 * 0.15
        )

        if score > best_score:
            best_score = score
            best_threshold = thresh
            best_metrics = metrics

    if best_metrics is None:
        # Fallback to 0.5 if no valid threshold found
        best_threshold = 0.5
        best_metrics = compute_pnl_metrics(y_pred_proba, pnl_values, 0.5)

    return best_threshold, best_metrics


# ═══════════════════════════════════════════════════════════════
# MODEL TRAINING
# ═══════════════════════════════════════════════════════════════

def train_fold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cat_indices: List[int],
    scale_pos_weight: float
) -> lgb.Booster:
    """Train LightGBM model for a single fold"""

    params = LIGHTGBM_PARAMS.copy()
    params['scale_pos_weight'] = scale_pos_weight

    # Adjust for small data
    n_train = len(X_train)
    if n_train < 500:
        params['min_data_in_leaf'] = max(5, n_train // 20)
        params['num_leaves'] = 15
        params['max_depth'] = 4

    train_data = lgb.Dataset(
        X_train, label=y_train,
        categorical_feature=cat_indices if cat_indices else 'auto'
    )
    val_data = lgb.Dataset(
        X_val, label=y_val,
        reference=train_data,
        categorical_feature=cat_indices if cat_indices else 'auto'
    )

    callbacks = [
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=0)  # Silent
    ]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=callbacks
    )

    return model


# ═══════════════════════════════════════════════════════════════
# SMALL DATA MODE (for paper trading with limited history)
# ═══════════════════════════════════════════════════════════════

def small_data_train(df: pd.DataFrame, config: Dict) -> WalkForwardResults:
    """
    Training mode for small datasets (< 4 months of data).
    Uses a simple time-based 70/30 split instead of rolling windows.
    """
    print("\n" + "="*60)
    print("SMALL DATA MODE (time-based split)")
    print("="*60)

    # Sort by entry_time
    df = df.sort_values('entry_time').reset_index(drop=True)
    df['entry_date'] = pd.to_datetime(df['entry_time'], unit='ms')

    # Check minimum samples
    if len(df) < config['min_total_samples']:
        print(f"ERROR: Only {len(df)} samples (need {config['min_total_samples']}+)")
        return WalkForwardResults()

    # Prepare features
    X_full, y_full, cat_indices, feature_names = prepare_features(df)

    # Time-based split
    split_idx = int(len(df) * config['train_pct'])

    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]

    X_train = X_full.iloc[:split_idx].reset_index(drop=True)
    y_train = y_full.iloc[:split_idx].reset_index(drop=True)
    X_test = X_full.iloc[split_idx:].reset_index(drop=True)
    y_test = y_full.iloc[split_idx:].reset_index(drop=True)

    pnl_train = df_train['pnl'].values
    pnl_test = df_test['pnl'].values

    print(f"\nTrain: {len(df_train)} samples ({df_train['entry_date'].min().date()} to {df_train['entry_date'].max().date()})")
    print(f"Test:  {len(df_test)} samples ({df_test['entry_date'].min().date()} to {df_test['entry_date'].max().date()})")

    if len(df_train) < config['min_train_samples']:
        print(f"ERROR: Not enough train samples ({len(df_train)} < {config['min_train_samples']})")
        return WalkForwardResults(feature_names=feature_names)

    if len(df_test) < config['min_test_samples']:
        print(f"ERROR: Not enough test samples ({len(df_test)} < {config['min_test_samples']})")
        return WalkForwardResults(feature_names=feature_names)

    # Calculate class weight
    scale_pos_weight = calculate_scale_pos_weight(y_train)
    print(f"Train WIN rate: {y_train.mean():.1%}, scale_pos_weight: {scale_pos_weight:.3f}")

    # Train model
    val_split = int(len(X_train) * 0.85)
    X_tr, X_val = X_train.iloc[:val_split], X_train.iloc[val_split:]
    y_tr, y_val = y_train.iloc[:val_split], y_train.iloc[val_split:]

    print(f"\nTraining LightGBM...")
    model = train_fold(X_tr, y_tr, X_val, y_val, cat_indices, scale_pos_weight)
    print(f"Best iteration: {model.best_iteration}")

    # Tune threshold on training data
    y_train_proba = model.predict(X_train)
    print(f"\nTuning threshold...")
    optimal_threshold, train_metrics = tune_threshold(y_train_proba, pnl_train, config)
    print(f"Optimal threshold: {optimal_threshold:.2f}")

    # Predict on test
    y_test_proba = model.predict(X_test)
    y_test_pred = (y_test_proba >= optimal_threshold).astype(int)

    # Metrics
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, zero_division=0)
    recall = recall_score(y_test, y_test_pred, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, zero_division=0)
    try:
        auc_roc = roc_auc_score(y_test, y_test_proba)
    except ValueError:
        auc_roc = 0.5

    # PnL metrics
    test_metrics = compute_pnl_metrics(y_test_proba, pnl_test, optimal_threshold)

    # Create results
    train_start = df_train['entry_date'].min()
    train_end = df_train['entry_date'].max()
    test_start = df_test['entry_date'].min()
    test_end = df_test['entry_date'].max()

    fold_metrics = FoldMetrics(
        fold_idx=0,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        train_samples=len(df_train),
        test_samples=len(df_test),
        train_win_rate=float(y_train.mean()),
        test_win_rate=float(y_test.mean()),
        scale_pos_weight=scale_pos_weight,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        auc_roc=auc_roc,
        baseline_trades=test_metrics['baseline_trades'],
        baseline_pnl=test_metrics['baseline_pnl'],
        baseline_win_rate=test_metrics['baseline_win_rate'],
        baseline_avg_pnl=test_metrics['baseline_avg_pnl'],
        filtered_trades=test_metrics['filtered_trades'],
        filtered_pnl=test_metrics['filtered_pnl'],
        filtered_win_rate=test_metrics['filtered_win_rate'],
        filtered_avg_pnl=test_metrics['filtered_avg_pnl'],
        pnl_improvement=test_metrics['pnl_improvement'],
        win_rate_improvement=test_metrics['win_rate_improvement'],
        optimal_threshold=optimal_threshold,
        best_iteration=model.best_iteration
    )

    results = WalkForwardResults(feature_names=feature_names)
    results.add_fold(fold_metrics)
    results.best_model = model
    results.best_threshold = optimal_threshold
    results.best_fold_idx = 0

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.1%}, AUC: {auc_roc:.3f}")
    print(f"Baseline:  {test_metrics['baseline_trades']} trades, PnL: ${test_metrics['baseline_pnl']:,.0f}, Win: {test_metrics['baseline_win_rate']:.1%}")
    print(f"Filtered:  {test_metrics['filtered_trades']} trades, PnL: ${test_metrics['filtered_pnl']:,.0f}, Win: {test_metrics['filtered_win_rate']:.1%}")
    print(f"Improvement: ${test_metrics['pnl_improvement']:,.0f}, Win rate: {test_metrics['win_rate_improvement']:+.1%}")

    return results


# ═══════════════════════════════════════════════════════════════
# MAIN WALK-FORWARD LOOP
# ═══════════════════════════════════════════════════════════════

def walk_forward_train(df: pd.DataFrame, config: Dict) -> WalkForwardResults:
    """Main walk-forward training loop"""

    # Sort by entry_time
    df = df.sort_values('entry_time').reset_index(drop=True)

    # Convert entry_time to datetime
    df['entry_date'] = pd.to_datetime(df['entry_time'], unit='ms')

    # Date range
    min_date = df['entry_date'].min()
    max_date = df['entry_date'].max()

    print(f"\nData range: {min_date.date()} to {max_date.date()}")
    print(f"Total samples: {len(df)}")

    # Prepare features once (column structure)
    X_full, y_full, cat_indices, feature_names = prepare_features(df)

    # Generate folds
    folds = generate_walk_forward_folds(
        min_date, max_date,
        config['train_months'],
        config['test_months'],
        config['embargo_days']
    )

    print(f"Generated {len(folds)} walk-forward folds\n")

    if len(folds) == 0:
        print("ERROR: Not enough data for walk-forward validation")
        print(f"Need at least {config['train_months'] + config['test_months']} months of data")
        return WalkForwardResults(feature_names=feature_names)

    results = WalkForwardResults(feature_names=feature_names)
    best_pnl_improvement = -float('inf')

    for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(folds):
        print(f"{'='*60}")
        print(f"FOLD {fold_idx + 1}/{len(folds)}")
        print(f"{'='*60}")
        print(f"Train: {train_start.date()} to {train_end.date()}")
        print(f"Test:  {test_start.date()} to {test_end.date()}")

        # Split data temporally
        train_mask = (df['entry_date'] >= train_start) & (df['entry_date'] < train_end)
        test_mask = (df['entry_date'] >= test_start) & (df['entry_date'] < test_end)

        df_train = df[train_mask]
        df_test = df[test_mask]

        # Check minimum samples
        if len(df_train) < config['min_train_samples']:
            print(f"  SKIP: Only {len(df_train)} train samples (need {config['min_train_samples']})")
            continue
        if len(df_test) < config['min_test_samples']:
            print(f"  SKIP: Only {len(df_test)} test samples (need {config['min_test_samples']})")
            continue

        # Get features for this fold
        X_train = X_full.loc[train_mask].reset_index(drop=True)
        y_train = y_full.loc[train_mask].reset_index(drop=True)
        X_test = X_full.loc[test_mask].reset_index(drop=True)
        y_test = y_full.loc[test_mask].reset_index(drop=True)

        # Get PnL values
        pnl_train = df_train['pnl'].values
        pnl_test = df_test['pnl'].values

        # Calculate class weight
        scale_pos_weight = calculate_scale_pos_weight(y_train)

        print(f"\n  Train: {len(df_train)} samples, WIN rate: {y_train.mean():.1%}")
        print(f"  Test:  {len(df_test)} samples, WIN rate: {y_test.mean():.1%}")
        print(f"  scale_pos_weight: {scale_pos_weight:.3f}")

        # Train model (use small validation split from train for early stopping)
        val_split = int(len(X_train) * 0.85)
        X_tr, X_val = X_train.iloc[:val_split], X_train.iloc[val_split:]
        y_tr, y_val = y_train.iloc[:val_split], y_train.iloc[val_split:]

        print(f"\n  Training LightGBM...")
        model = train_fold(X_tr, y_tr, X_val, y_val, cat_indices, scale_pos_weight)
        print(f"  Best iteration: {model.best_iteration}")

        # Predict on train (for threshold tuning)
        y_train_proba = model.predict(X_train)

        # Tune threshold using TRAIN data only
        print(f"\n  Tuning threshold...")
        optimal_threshold, train_metrics = tune_threshold(y_train_proba, pnl_train, config)
        print(f"  Optimal threshold: {optimal_threshold:.2f}")

        # Predict on TEST
        y_test_proba = model.predict(X_test)
        y_test_pred = (y_test_proba >= optimal_threshold).astype(int)

        # Standard metrics
        accuracy = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred, zero_division=0)
        recall = recall_score(y_test, y_test_pred, zero_division=0)
        f1 = f1_score(y_test, y_test_pred, zero_division=0)
        try:
            auc_roc = roc_auc_score(y_test, y_test_proba)
        except ValueError:
            auc_roc = 0.5

        # PnL metrics on TEST
        test_metrics = compute_pnl_metrics(y_test_proba, pnl_test, optimal_threshold)

        # Create fold metrics
        fold_metrics = FoldMetrics(
            fold_idx=fold_idx,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            train_samples=len(df_train),
            test_samples=len(df_test),
            train_win_rate=float(y_train.mean()),
            test_win_rate=float(y_test.mean()),
            scale_pos_weight=scale_pos_weight,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            auc_roc=auc_roc,
            baseline_trades=test_metrics['baseline_trades'],
            baseline_pnl=test_metrics['baseline_pnl'],
            baseline_win_rate=test_metrics['baseline_win_rate'],
            baseline_avg_pnl=test_metrics['baseline_avg_pnl'],
            filtered_trades=test_metrics['filtered_trades'],
            filtered_pnl=test_metrics['filtered_pnl'],
            filtered_win_rate=test_metrics['filtered_win_rate'],
            filtered_avg_pnl=test_metrics['filtered_avg_pnl'],
            pnl_improvement=test_metrics['pnl_improvement'],
            win_rate_improvement=test_metrics['win_rate_improvement'],
            optimal_threshold=optimal_threshold,
            best_iteration=model.best_iteration
        )

        results.add_fold(fold_metrics)

        # Print fold results
        print(f"\n  RESULTS:")
        print(f"  - Accuracy: {accuracy:.1%}, AUC: {auc_roc:.3f}")
        print(f"  - Baseline:  {test_metrics['baseline_trades']} trades, "
              f"PnL: ${test_metrics['baseline_pnl']:,.0f}, "
              f"Win: {test_metrics['baseline_win_rate']:.1%}")
        print(f"  - Filtered:  {test_metrics['filtered_trades']} trades, "
              f"PnL: ${test_metrics['filtered_pnl']:,.0f}, "
              f"Win: {test_metrics['filtered_win_rate']:.1%}")
        print(f"  - Improvement: ${test_metrics['pnl_improvement']:,.0f}, "
              f"Win rate: {test_metrics['win_rate_improvement']:+.1%}")

        # Track best model by PnL improvement
        if test_metrics['pnl_improvement'] > best_pnl_improvement:
            best_pnl_improvement = test_metrics['pnl_improvement']
            results.best_model = model
            results.best_threshold = optimal_threshold
            results.best_fold_idx = fold_idx
            print(f"\n  *** NEW BEST MODEL (Fold {fold_idx + 1}) ***")

    return results


# ═══════════════════════════════════════════════════════════════
# MODEL PERSISTENCE
# ═══════════════════════════════════════════════════════════════

def get_feature_importance(model: lgb.Booster, feature_names: List[str]) -> Dict[str, float]:
    """Get feature importance as dict"""
    importance = model.feature_importance(importance_type='gain')
    return dict(sorted(
        zip(feature_names, importance),
        key=lambda x: x[1],
        reverse=True
    ))


def count_features_used(importance_dict: Dict[str, float]) -> int:
    """Count features with non-zero importance"""
    return sum(1 for v in importance_dict.values() if v > 0)


def should_save_model(results: WalkForwardResults, current_best_path: str, config: Dict) -> bool:
    """Decide if new model should replace current best"""

    if results.best_model is None:
        print("\n  Not saving: No model trained")
        return False

    # Check feature diversity
    importance = get_feature_importance(results.best_model, results.feature_names)
    num_features = count_features_used(importance)

    if num_features < config['min_features_used']:
        print(f"\n  Not saving: Only {num_features} features used (need {config['min_features_used']}+)")
        return False

    # Check if current best exists
    metadata_path = os.path.join(current_best_path, 'lightgbm_metadata.json')
    if not os.path.exists(metadata_path):
        print("\n  Saving: No existing model")
        return True

    # Compare to current best
    try:
        with open(metadata_path, 'r') as f:
            current_meta = json.load(f)

        current_pnl = current_meta.get('walk_forward', {}).get('total_filtered_pnl', 0)
        new_pnl = results.total_filtered_pnl

        if current_pnl == 0:
            improvement = 1.0 if new_pnl > 0 else 0.0
        else:
            improvement = (new_pnl - current_pnl) / abs(current_pnl)

        if improvement < config['pnl_improvement_threshold']:
            print(f"\n  Not saving: Only {improvement:.1%} improvement "
                  f"(need {config['pnl_improvement_threshold']:.0%}+)")
            return False

        print(f"\n  Saving: {improvement:.1%} improvement over current best")
        return True

    except (json.JSONDecodeError, KeyError):
        print("\n  Saving: Could not read current best metadata")
        return True


def save_model(results: WalkForwardResults, output_dir: str):
    """Save model with walk-forward metadata"""

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')

    model = results.best_model

    # Save LightGBM model
    model_path = os.path.join(output_dir, 'lightgbm_model.txt')
    model.save_model(model_path)

    # Save as JSON
    model_json_path = os.path.join(output_dir, 'lightgbm_model.json')
    with open(model_json_path, 'w') as f:
        json.dump(model.dump_model(), f)

    # Get feature importance
    importance = get_feature_importance(model, results.feature_names)

    # Build metadata
    metadata = {
        'modelId': f'walkforward_{timestamp}',
        'trainedAt': datetime.now().isoformat(),
        'algorithm': 'lightgbm-walkforward',

        # Walk-forward metrics (KEY DATA)
        'walk_forward': {
            'num_folds': results.num_folds,
            'total_filtered_pnl': results.total_filtered_pnl,
            'total_baseline_pnl': results.total_baseline_pnl,
            'pnl_improvement': results.total_filtered_pnl - results.total_baseline_pnl,
            'avg_fold_pnl': results.avg_fold_pnl,
            'avg_filtered_win_rate': results.avg_filtered_win_rate,
            'optimal_threshold': results.best_threshold,
            'best_fold_idx': results.best_fold_idx,
        },

        # Standard metrics
        'accuracy': results.avg_accuracy,
        'auc': results.avg_auc,

        # Model info
        'bestIteration': model.best_iteration,
        'numFeatures': len(results.feature_names),
        'numFeaturesUsed': count_features_used(importance),
        'features': results.feature_names,
        'featureImportance': importance,

        # Config
        'config': {
            'train_months': WALK_FORWARD_CONFIG['train_months'],
            'test_months': WALK_FORWARD_CONFIG['test_months'],
            'threshold_tuned': True,
            'scale_pos_weight_used': True,
        },

        # Per-fold details
        'folds': [
            {
                'fold': f.fold_idx + 1,
                'train_period': f"{f.train_start.date()} to {f.train_end.date()}",
                'test_period': f"{f.test_start.date()} to {f.test_end.date()}",
                'test_samples': f.test_samples,
                'filtered_pnl': f.filtered_pnl,
                'filtered_win_rate': f.filtered_win_rate,
                'threshold': f.optimal_threshold,
            }
            for f in results.folds
        ]
    }

    metadata_path = os.path.join(output_dir, 'lightgbm_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Saved: {model_path}")
    print(f"  Saved: {model_json_path}")
    print(f"  Saved: {metadata_path}")


# ═══════════════════════════════════════════════════════════════
# SUMMARY REPORT
# ═══════════════════════════════════════════════════════════════

def print_summary(results: WalkForwardResults):
    """Print comprehensive walk-forward summary"""

    print("\n" + "="*65)
    print("WALK-FORWARD VALIDATION RESULTS")
    print("="*65)

    if not results.folds:
        print("\nNo folds completed.")
        return

    # Fold summary table
    print(f"\n{'FOLD SUMMARY':^65}")
    print("-"*65)
    print(f"{'Fold':<5} {'Test Period':<22} {'Trades':>7} {'PnL':>10} {'Win%':>7}")
    print("-"*65)

    for f in results.folds:
        period = f"{f.test_start.strftime('%b %Y')}"
        print(f"{f.fold_idx+1:<5} {period:<22} {f.filtered_trades:>7} "
              f"${f.filtered_pnl:>9,.0f} {f.filtered_win_rate:>6.1%}")

    print("-"*65)
    total_trades = sum(f.filtered_trades for f in results.folds)
    print(f"{'TOTAL':<5} {'':<22} {total_trades:>7} "
          f"${results.total_filtered_pnl:>9,.0f} {results.avg_filtered_win_rate:>6.1%}")

    # Baseline comparison
    print(f"\n{'BASELINE COMPARISON':^65}")
    print("-"*65)
    print(f"{'':20} {'Baseline':>15} {'Filtered':>15} {'Change':>12}")
    print("-"*65)

    baseline_trades = sum(f.baseline_trades for f in results.folds)
    baseline_pnl = results.total_baseline_pnl
    baseline_win = np.mean([f.baseline_win_rate for f in results.folds])

    filtered_trades = total_trades
    filtered_pnl = results.total_filtered_pnl
    filtered_win = results.avg_filtered_win_rate

    trade_change = ((filtered_trades - baseline_trades) / baseline_trades * 100) if baseline_trades else 0
    pnl_change = ((filtered_pnl - baseline_pnl) / abs(baseline_pnl) * 100) if baseline_pnl else 0
    win_change = (filtered_win - baseline_win) * 100

    print(f"{'Trades':<20} {baseline_trades:>15,} {filtered_trades:>15,} {trade_change:>+11.1f}%")
    print(f"{'Total PnL':<20} ${baseline_pnl:>14,.0f} ${filtered_pnl:>14,.0f} {pnl_change:>+11.1f}%")
    print(f"{'Win Rate':<20} {baseline_win:>14.1%} {filtered_win:>14.1%} {win_change:>+11.1f}%")

    if baseline_trades > 0 and filtered_trades > 0:
        baseline_avg = baseline_pnl / baseline_trades
        filtered_avg = filtered_pnl / filtered_trades
        avg_change = ((filtered_avg - baseline_avg) / abs(baseline_avg) * 100) if baseline_avg else 0
        print(f"{'Avg PnL/Trade':<20} ${baseline_avg:>14,.2f} ${filtered_avg:>14,.2f} {avg_change:>+11.1f}%")

    # Model details
    if results.best_model:
        print(f"\n{'MODEL DETAILS':^65}")
        print("-"*65)
        print(f"Optimal Threshold: {results.best_threshold:.2f}")
        print(f"Best Fold: {results.best_fold_idx + 1}")
        print(f"Best Iteration: {results.best_model.best_iteration}")

        importance = get_feature_importance(results.best_model, results.feature_names)
        num_used = count_features_used(importance)
        print(f"Features Used: {num_used} / {len(results.feature_names)}")

        print(f"\nTop 10 Features:")
        for i, (feat, imp) in enumerate(list(importance.items())[:10]):
            print(f"  {i+1:2}. {feat:<30} {imp:>10,.0f}")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Train LightGBM with walk-forward validation'
    )
    parser.add_argument('--input', '-i', required=True,
                        help='Path to training CSV (supports glob patterns)')
    parser.add_argument('--output', '-o', default='data/models',
                        help='Output directory for model')
    parser.add_argument('--train-months', type=int, default=3,
                        help='Months of training data per fold')
    parser.add_argument('--test-months', type=int, default=1,
                        help='Months of test data per fold')
    parser.add_argument('--force-save', action='store_true',
                        help='Save model even if not improving')
    args = parser.parse_args()

    # Update config
    config = WALK_FORWARD_CONFIG.copy()
    config['train_months'] = args.train_months
    config['test_months'] = args.test_months

    print("="*65)
    print("LIGHTGBM WALK-FORWARD TRAINING")
    print("="*65)

    # Load data
    df = load_training_data(args.input)

    # Check data density to decide mode
    df_temp = df.copy()
    df_temp['entry_date'] = pd.to_datetime(df_temp['entry_time'], unit='ms')

    # Count unique months with significant trades (>20 trades)
    monthly_counts = df_temp.groupby(df_temp['entry_date'].dt.to_period('M')).size()
    months_with_data = (monthly_counts >= 20).sum()

    print(f"\nTotal samples: {len(df)}")
    print(f"Months with 20+ trades: {months_with_data}")

    # Use small data mode if < 4 months with sufficient data
    use_small_data = months_with_data < 4

    if use_small_data:
        print("\n*** Using SMALL DATA MODE (time-based split) ***")
        print("(Not enough months with 20+ trades for walk-forward validation)")
        small_config = SMALL_DATA_CONFIG.copy()
        results = small_data_train(df, small_config)
    else:
        print(f"\nConfig:")
        print(f"  Train months: {config['train_months']}")
        print(f"  Test months: {config['test_months']}")
        print(f"  Min train samples: {config['min_train_samples']}")
        print(f"  Min test samples: {config['min_test_samples']}")
        # Run walk-forward training
        results = walk_forward_train(df, config)

    # Print summary
    print_summary(results)

    # Save model
    if args.force_save or should_save_model(results, args.output, config):
        if results.best_model:
            save_model(results, args.output)
            print("\n" + "="*65)
            print("MODEL SAVED")
            print("="*65)
        else:
            print("\n  No model to save (all folds failed)")
    else:
        print("\n" + "="*65)
        print("MODEL NOT SAVED (did not meet criteria)")
        print("="*65)

    print("\nDone.")


if __name__ == '__main__':
    main()
