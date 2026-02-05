#!/usr/bin/env node
/**
 * Export SCALP paper trades to a TradeFeatures training CSV.
 *
 * Reads per-symbol states from data/paper-trades-scalp/*.json,
 * aggregates partial exits into a single label per entry,
 * reconstructs entry-time features by fetching candles from Binance,
 * and writes a CSV compatible with scripts/lightgbm_trainer.py.
 *
 * Usage:
 *   npm run export-paper-trades-scalp
 *   node dist/export-paper-trades-scalp-to-training.js --lookback 200 --limit 500
 */

import { createRequire } from 'module';
import fs from 'fs';
import path from 'path';

const require = createRequire(import.meta.url);
const Binance = require('binance-api-node').default;

import { Candle, SMCIndicators } from './smc-indicators.js';
import { ICTIndicators } from './ict-indicators.js';
import { FeatureExtractor, TradeFeatures, BacktestTrade } from './trade-features.js';
import { UnifiedScoring } from './unified-scoring.js';

// ═══════════════════════════════════════════════════════════════
// MOMENTUM CALCULATIONS (same as scalper)
// ═══════════════════════════════════════════════════════════════

function calculateEMA(candles: Candle[], period: number): number[] {
  const ema: number[] = [];
  const multiplier = 2 / (period + 1);
  let sum = 0;
  for (let i = 0; i < period && i < candles.length; i++) {
    sum += candles[i].close;
  }
  ema.push(sum / Math.min(period, candles.length));
  for (let i = period; i < candles.length; i++) {
    const value = (candles[i].close - ema[ema.length - 1]) * multiplier + ema[ema.length - 1];
    ema.push(value);
  }
  return ema;
}

function calculateMACD(candles: Candle[], fastPeriod = 12, slowPeriod = 26, signalPeriod = 9) {
  const emaFast = calculateEMA(candles, fastPeriod);
  const emaSlow = calculateEMA(candles, slowPeriod);
  const macdLine: number[] = [];
  const startIdx = slowPeriod - fastPeriod;
  for (let i = 0; i < emaSlow.length; i++) {
    const fastIdx = i + startIdx;
    if (fastIdx >= 0 && fastIdx < emaFast.length) {
      macdLine.push(emaFast[fastIdx] - emaSlow[i]);
    }
  }
  const signalLine: number[] = [];
  if (macdLine.length >= signalPeriod) {
    const mult = 2 / (signalPeriod + 1);
    let s = 0;
    for (let i = 0; i < signalPeriod; i++) s += macdLine[i];
    signalLine.push(s / signalPeriod);
    for (let i = signalPeriod; i < macdLine.length; i++) {
      const val = (macdLine[i] - signalLine[signalLine.length - 1]) * mult + signalLine[signalLine.length - 1];
      signalLine.push(val);
    }
  }
  const histogram: number[] = [];
  const offset = macdLine.length - signalLine.length;
  for (let i = 0; i < signalLine.length; i++) {
    histogram.push(macdLine[i + offset] - signalLine[i]);
  }
  return { macdLine, signalLine, histogram };
}

function calculateVWAP(candles: Candle[]): { vwap: number; stdDev: number } {
  let cumulativePV = 0;
  let cumulativeVolume = 0;
  const typicalPrices: number[] = [];
  for (const c of candles) {
    const tp = (c.high + c.low + c.close) / 3;
    typicalPrices.push(tp);
    cumulativePV += tp * c.volume;
    cumulativeVolume += c.volume;
  }
  const vwap = cumulativeVolume > 0 ? cumulativePV / cumulativeVolume : candles[candles.length - 1]?.close || 0;
  const deviations = typicalPrices.map(p => Math.pow(p - vwap, 2));
  const variance = deviations.reduce((a, b) => a + b, 0) / deviations.length;
  return { vwap, stdDev: Math.sqrt(variance) };
}

function calculateATR(candles: Candle[], period = 14): number {
  if (candles.length < period + 1) return 0;
  const trueRanges: number[] = [];
  for (let i = 1; i < candles.length; i++) {
    const tr = Math.max(
      candles[i].high - candles[i].low,
      Math.abs(candles[i].high - candles[i - 1].close),
      Math.abs(candles[i].low - candles[i - 1].close)
    );
    trueRanges.push(tr);
  }
  const recentTR = trueRanges.slice(-period);
  return recentTR.reduce((a, b) => a + b, 0) / recentTR.length;
}

function calculateBollingerBands(candles: Candle[], period = 20, stdDev = 2) {
  if (candles.length < period) return { upper: 0, middle: 0, lower: 0 };
  const slice = candles.slice(-period);
  const mean = slice.reduce((s, c) => s + c.close, 0) / period;
  const variance = slice.reduce((s, c) => s + Math.pow(c.close - mean, 2), 0) / period;
  const std = Math.sqrt(variance);
  return { upper: mean + stdDev * std, middle: mean, lower: mean - stdDev * std };
}

interface MomentumFeatures {
  regime: 'TREND' | 'RANGE';
  atr_percent: number;
  macd_line: number;
  macd_signal: number;
  macd_histogram: number;
  macd_bullish_cross: boolean;
  macd_bearish_cross: boolean;
  vwap: number;
  vwap_deviation: number;
  vwap_deviation_std: number;
  price_above_vwap: boolean;
  ema_fast: number;
  ema_slow: number;
  ema_bullish_cross: boolean;
  ema_bearish_cross: boolean;
  body_ratio: number;
  bb_position: number;
  bb_width: number;
}

function extractMomentumFeatures(candles: Candle[]): MomentumFeatures {
  const len = candles.length;
  const current = candles[len - 1];
  const prev = candles[len - 2];

  // ATR & Regime
  const atr = calculateATR(candles, 14);
  const atr_percent = current.close > 0 ? atr / current.close : 0;
  const regime: 'TREND' | 'RANGE' = atr_percent >= 0.015 ? 'TREND' : 'RANGE';

  // MACD
  const macd = calculateMACD(candles);
  const macd_line = macd.macdLine[macd.macdLine.length - 1] || 0;
  const macd_signal = macd.signalLine[macd.signalLine.length - 1] || 0;
  const macd_histogram = macd.histogram[macd.histogram.length - 1] || 0;
  const prevMacdLine = macd.macdLine[macd.macdLine.length - 2] || macd_line;
  const prevMacdSignal = macd.signalLine[macd.signalLine.length - 2] || macd_signal;
  const macd_bullish_cross = prevMacdLine <= prevMacdSignal && macd_line > macd_signal;
  const macd_bearish_cross = prevMacdLine >= prevMacdSignal && macd_line < macd_signal;

  // VWAP
  const vwapData = calculateVWAP(candles);
  const vwap = vwapData.vwap;
  const vwap_deviation = vwap > 0 ? ((current.close - vwap) / vwap) * 100 : 0;
  const vwap_deviation_std = vwapData.stdDev > 0 ? (current.close - vwap) / vwapData.stdDev : 0;
  const price_above_vwap = current.close > vwap;

  // EMA
  const emaFastValues = calculateEMA(candles, 9);
  const emaSlowValues = calculateEMA(candles, 21);
  const ema_fast = emaFastValues[emaFastValues.length - 1] || current.close;
  const ema_slow = emaSlowValues[emaSlowValues.length - 1] || current.close;
  const prevEmaFast = emaFastValues[emaFastValues.length - 2] || ema_fast;
  const prevEmaSlow = emaSlowValues[emaSlowValues.length - 2] || ema_slow;
  const ema_bullish_cross = prevEmaFast <= prevEmaSlow && ema_fast > ema_slow;
  const ema_bearish_cross = prevEmaFast >= prevEmaSlow && ema_fast < ema_slow;

  // Body ratio
  const candleRange = current.high - current.low;
  const candleBody = Math.abs(current.close - current.open);
  const body_ratio = candleRange > 0 ? candleBody / candleRange : 0;

  // BB Position
  const bb = calculateBollingerBands(candles);
  const bbRange = bb.upper - bb.lower;
  const bb_position = bbRange > 0 ? Math.max(0, Math.min(1, (current.close - bb.lower) / bbRange)) : 0.5;
  const bb_width = current.close > 0 ? (bbRange / current.close) * 100 : 0;

  return {
    regime, atr_percent,
    macd_line, macd_signal, macd_histogram,
    macd_bullish_cross, macd_bearish_cross,
    vwap, vwap_deviation, vwap_deviation_std, price_above_vwap,
    ema_fast, ema_slow, ema_bullish_cross, ema_bearish_cross,
    body_ratio, bb_position, bb_width
  };
}

type PaperTradeDirection = 'LONG' | 'SHORT';

type PaperTrade = {
  id: string;
  symbol: string;
  direction: PaperTradeDirection;
  entryPrice: number;
  entryTime: number;
  originalPositionSize: number;
  status: 'OPEN' | 'CLOSED';
  exitPrice?: number;
  exitTime?: number;
  exitReason?: string;
  pnl?: number;
  pnlPercent?: number;
};

type ExportRow = TradeFeatures;

type GroupedTrade = {
  symbol: string;
  direction: PaperTradeDirection;
  entryPrice: number;
  entryTime: number;
  originalPositionSize: number;
  exitTime: number;
  exitReason: string;
  pnl: number;
  pnlPercent: number;
  holdingPeriods: number;
  outcome: 'WIN' | 'LOSS';
};

const DEFAULTS = {
  interval: '15m',
  lookback: 200,
  limit: 500,
  outputDir: path.join(process.cwd(), 'data', 'h2o-training'),
  tradesDir: path.join(process.cwd(), 'data', 'paper-trades-scalp'),
};

function intervalToMs(interval: string): number {
  const match = interval.match(/^(\d+)([mhdw])$/);
  if (!match) throw new Error(`Unsupported interval: ${interval}`);
  const n = parseInt(match[1], 10);
  const unit = match[2];
  if (unit === 'm') return n * 60_000;
  if (unit === 'h') return n * 60 * 60_000;
  if (unit === 'd') return n * 24 * 60 * 60_000;
  if (unit === 'w') return n * 7 * 24 * 60 * 60_000;
  throw new Error(`Unsupported interval unit: ${unit}`);
}

function parseArgs(argv: string[]): Record<string, string> {
  const out: Record<string, string> = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith('--')) continue;
    const key = a.slice(2);
    const val = argv[i + 1];
    if (!val || val.startsWith('--')) {
      out[key] = 'true';
    } else {
      out[key] = val;
      i++;
    }
  }
  return out;
}

function exportToCSV(trades: ExportRow[], outputPath: string): void {
  if (trades.length === 0) {
    throw new Error('No rows to export');
  }

  const headers = Object.keys(trades[0]).join(',');
  const rows = trades.map(trade => {
    const values = Object.values(trade).map(val => {
      if (typeof val === 'string') return `"${String(val).replaceAll('"', '""')}"`;
      if (typeof val === 'boolean') return val ? 1 : 0;
      if (typeof val === 'number') return Number.isFinite(val) ? val : 0;
      return `"${String(val)}"`;
    });
    return values.join(',');
  });

  fs.writeFileSync(outputPath, [headers, ...rows].join('\n'));
}

function findEntryIndex(candles: Candle[], entryTime: number): number {
  // candle.timestamp is openTime (ms). Choose the last candle whose openTime <= entryTime.
  let lo = 0;
  let hi = candles.length - 1;
  let best = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (candles[mid].timestamp <= entryTime) {
      best = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return best;
}

function groupClosedTrades(allTrades: PaperTrade[], intervalMs: number): GroupedTrade[] {
  const closed = allTrades.filter(t => t.status === 'CLOSED' && !!t.exitTime && !!t.entryTime);
  const groups = new Map<string, PaperTrade[]>();

  for (const t of closed) {
    const key = `${t.symbol}::${t.entryTime}::${t.direction}`;
    const arr = groups.get(key) ?? [];
    arr.push(t);
    groups.set(key, arr);
  }

  const result: GroupedTrade[] = [];

  for (const [key, trades] of groups.entries()) {
    trades.sort((a, b) => (a.exitTime ?? 0) - (b.exitTime ?? 0));

    const first = trades[0];
    const last = trades[trades.length - 1];

    const entryPrice = first.entryPrice;
    const entryTime = first.entryTime;
    const symbol = first.symbol;
    const direction = first.direction;

    const originalPositionSize = trades.reduce(
      (m, t) => Math.max(m, Number(t.originalPositionSize) || 0),
      0
    );

    const pnl = trades.reduce((s, t) => s + (Number(t.pnl) || 0), 0);

    const entryNotional = entryPrice > 0 ? entryPrice * originalPositionSize : 0;
    const pnlPercent = entryNotional > 0 ? (pnl / entryNotional) * 100 : (last.pnlPercent ?? 0);

    const exitTime = last.exitTime ?? entryTime;
    const exitReason = (last.exitReason ?? 'UNKNOWN').toString();

    const holdingPeriods = Math.max(1, Math.ceil((exitTime - entryTime) / intervalMs));
    const outcome: 'WIN' | 'LOSS' = pnl > 0 ? 'WIN' : 'LOSS';

    result.push({
      symbol,
      direction,
      entryPrice,
      entryTime,
      originalPositionSize,
      exitTime,
      exitReason,
      pnl,
      pnlPercent,
      holdingPeriods,
      outcome,
    });
  }

  // Stable sort by time
  result.sort((a, b) => a.entryTime - b.entryTime);
  return result;
}

async function fetchCandles(client: any, symbol: string, interval: string, limit: number): Promise<Candle[]> {
  const klines = await client.candles({ symbol, interval, limit });
  return klines.map((k: any) => ({
    timestamp: k.openTime,
    open: parseFloat(k.open),
    high: parseFloat(k.high),
    low: parseFloat(k.low),
    close: parseFloat(k.close),
    volume: parseFloat(k.volume),
  }));
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const interval = args.interval ?? DEFAULTS.interval;
  const lookback = args.lookback ? Math.max(50, parseInt(args.lookback, 10)) : DEFAULTS.lookback;
  const limit = args.limit ? Math.max(200, parseInt(args.limit, 10)) : DEFAULTS.limit;
  const tradesDir = args.tradesDir ? path.resolve(args.tradesDir) : DEFAULTS.tradesDir;
  const outputDir = args.outputDir ? path.resolve(args.outputDir) : DEFAULTS.outputDir;

  const intervalMs = intervalToMs(interval);

  if (!fs.existsSync(tradesDir)) {
    throw new Error(`Trades directory not found: ${tradesDir}`);
  }
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  const files = fs
    .readdirSync(tradesDir)
    .filter(f => f.endsWith('.json') && !f.includes('summary'))
    .map(f => path.join(tradesDir, f));

  if (files.length === 0) {
    throw new Error(`No state files found in: ${tradesDir}`);
  }

  let rawTrades: PaperTrade[] = [];
  for (const file of files) {
    try {
      const state = JSON.parse(fs.readFileSync(file, 'utf-8'));
      const trades = (state?.trades ?? []) as PaperTrade[];
      rawTrades.push(...trades);
    } catch (e: any) {
      console.warn(`[WARN] Failed to read ${file}: ${e?.message ?? e}`);
    }
  }

  const grouped = groupClosedTrades(rawTrades, intervalMs);

  console.log(`\n=== Export Paper Trades (SCALP) ===`);
  console.log(`State files: ${files.length}`);
  console.log(`Raw trades: ${rawTrades.length}`);
  console.log(`Grouped closed entries: ${grouped.length}`);
  if (grouped.length < 20) {
    console.log(`⚠️  Very small dataset (${grouped.length}). Treat any retraining as experimental.`);
  }

  const client = Binance();

  // Cache candles per symbol to reduce API calls.
  const candleCache = new Map<string, Candle[]>();

  const weights = {
    trend_structure: 40,
    order_blocks: 30,
    fvgs: 20,
    ema_alignment: 15,
    liquidity: 10,
    mtf_bonus: 35,
    rsi_penalty: 15,
  };

  const rows: ExportRow[] = [];
  const skipped: Array<{ symbol: string; entryTime: number; reason: string }> = [];

  for (const g of grouped) {
    try {
      let candles = candleCache.get(g.symbol);
      if (!candles) {
        candles = await fetchCandles(client, g.symbol, interval, limit);
        candleCache.set(g.symbol, candles);
      }

      const entryIndex = findEntryIndex(candles, g.entryTime);
      if (entryIndex < 0) {
        skipped.push({ symbol: g.symbol, entryTime: g.entryTime, reason: 'entryTime before candle window' });
        continue;
      }
      if (entryIndex < lookback) {
        skipped.push({ symbol: g.symbol, entryTime: g.entryTime, reason: `not enough lookback (${entryIndex}/${lookback})` });
        continue;
      }

      const window = candles.slice(entryIndex - lookback, entryIndex + 1);
      const analysis = SMCIndicators.analyze(window);
      const scoring = UnifiedScoring.calculateConfluence(analysis, g.entryPrice, weights);
      const dir = g.direction === 'LONG' ? 'long' : 'short';
      const ictAnalysis = ICTIndicators.analyzeFast(window, analysis);

      const features = FeatureExtractor.extractFeatures(
        window,
        window.length - 1,
        analysis,
        scoring.score,
        dir,
        ictAnalysis
      );

      // Extract momentum features for scalp trades
      const momentumFeatures = extractMomentumFeatures(window);

      const outcomeTrade: BacktestTrade = {
        outcome: g.outcome,
        pnl: g.pnl,
        pnl_percent: g.pnlPercent,
        exit_reason: g.exitReason,
        holding_periods: g.holdingPeriods,
      };

      const row = FeatureExtractor.addOutcome(features, outcomeTrade);

      // Add momentum features to the row
      const rowWithMomentum = {
        ...row,
        regime: momentumFeatures.regime,
        atr_percent: momentumFeatures.atr_percent,
        macd_line: momentumFeatures.macd_line,
        macd_signal: momentumFeatures.macd_signal,
        macd_histogram: momentumFeatures.macd_histogram,
        macd_bullish_cross: momentumFeatures.macd_bullish_cross,
        macd_bearish_cross: momentumFeatures.macd_bearish_cross,
        vwap: momentumFeatures.vwap,
        vwap_deviation: momentumFeatures.vwap_deviation,
        vwap_deviation_std: momentumFeatures.vwap_deviation_std,
        price_above_vwap: momentumFeatures.price_above_vwap,
        ema_fast: momentumFeatures.ema_fast,
        ema_slow: momentumFeatures.ema_slow,
        ema_bullish_cross: momentumFeatures.ema_bullish_cross,
        ema_bearish_cross: momentumFeatures.ema_bearish_cross,
        body_ratio: momentumFeatures.body_ratio,
        // BB features may already exist but update with momentum calc
        bb_position: momentumFeatures.bb_position,
        bb_width: momentumFeatures.bb_width,
      };

      rows.push(rowWithMomentum as ExportRow);
    } catch (e: any) {
      skipped.push({ symbol: g.symbol, entryTime: g.entryTime, reason: e?.message ?? String(e) });
    }
  }

  const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
  const outCsv = path.join(outputDir, `paper_scalp_${timestamp}.csv`);
  const outJson = path.join(outputDir, `paper_scalp_${timestamp}.json`);

  if (rows.length === 0) {
    console.log('\n❌ No rows were exportable.');
    if (skipped.length) {
      console.log(`Skipped (${skipped.length}):`);
      for (const s of skipped.slice(0, 10)) {
        console.log(`  - ${s.symbol} @ ${new Date(s.entryTime).toISOString()} : ${s.reason}`);
      }
      if (skipped.length > 10) console.log(`  ... +${skipped.length - 10} more`);
    }
    process.exit(1);
  }

  exportToCSV(rows, outCsv);
  fs.writeFileSync(outJson, JSON.stringify({
    meta: {
      interval,
      lookback,
      limit,
      groupedTrades: grouped.length,
      exportedRows: rows.length,
      skipped: skipped.length,
      generatedAt: new Date().toISOString(),
    },
    rows,
    skipped,
  }, null, 2));

  console.log(`\n✅ Exported: ${rows.length} rows`);
  console.log(`CSV:  ${outCsv}`);
  console.log(`JSON: ${outJson}`);
  if (skipped.length) {
    console.log(`Skipped: ${skipped.length} (see JSON for details)`);
  }

  console.log(`\nNext (optional):`);
  console.log(`  npm run train-lgbm -- --input "${outCsv}"`);
  console.log('');
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
