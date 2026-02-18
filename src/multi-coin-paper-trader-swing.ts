#!/usr/bin/env node
/**
 * Multi-Coin Paper Trading - SWING TRADER (4h Primary)
 *
 * Systematic regime-based entries:
 * - TREND mode: EMA/breakout entry + VWAP confirmation (MACD filter only)
 * - RANGE mode: BB extremes + volume spike (mean reversion)
 * - CHOP mode: Skip (no edge)
 *
 * SMC/ICT concepts are parameter ADJUSTERS, not entry gates.
 * ML model optional (threshold 0 = bypass for data collection).
 *
 * Usage: npm run paper-trade-multi-swing
 */

import { createRequire } from 'module';
import fs from 'fs';
import path from 'path';

const require = createRequire(import.meta.url);
const Binance = require('binance-api-node').default;
import { Candle, SMCIndicators, OrderBlock } from './smc-indicators.js';
import { ICTIndicators } from './ict-indicators.js';
import { FeatureExtractor, TradeFeatures } from './trade-features.js';
import { UnifiedScoring } from './unified-scoring.js';
import { TradingMLModel } from './ml-model.js';
import { LightGBMPredictor } from './lightgbm-predictor.js';

// Top 30 coins
const SYMBOLS = [
  'BTCUSDT',  'ETHUSDT',
  'BNBUSDT',  'ADAUSDT',
  'SOLUSDT',  'XRPUSDT',
  'DOGEUSDT', 'DOTUSDT',
  'AVAXUSDT', 'LINKUSDT',
  'ATOMUSDT', 'NEARUSDT',
  'MATICUSDT', 'UNIUSDT',
  'LDOUSDT',  'ARBUSDT',
  'OPUSDT',   'SUIUSDT',
  'INJUSDT',  'TONUSDT',
  'APTUSDT',  'PEPEUSDT',
  'FETUSDT',  'RNDRUSDT',
  'WIFUSDT',  'TIAUSDT',
  'SEIUSDT',  'MINAUSDT',
  'IMXUSDT',  'GMTUSDT',
] as const;

// ═══════════════════════════════════════════════════════════════
// CONFIGURATION - Regime-Based Swing Trading
// ═══════════════════════════════════════════════════════════════
const CONFIG = {
  mode: 'REGIME_SWING',
  intervals: ['1w', '1d', '4h'] as const,  // Weekly added for MTF alignment
  primaryInterval: '4h' as const,
  checkIntervalMs: 30000,
  minCandlesRequired: 50,

  // ═══════════════════════════════════════════════════════════════
  // REPORTING MODE
  // ═══════════════════════════════════════════════════════════════
  reporting: {
    mode: 'verbose',               // 'verbose' = all coins (default), 'summary' = clean output
    showOpenTrades: true,          // Always show open trade details
    showStreaks: true,             // Show current win/loss streak
    showRecent: true,              // Show last 10 trades performance
    summaryFile: path.join(process.cwd(), 'data', 'paper-trades-summary-swing-live.json'),  // Live summary for OpenClaw
  },

  // Regime detection
  regime: {
    volatilityThreshold: 0.025,  // 2.5% ATR/price = TREND mode (4H scale)
    minVolatility: 0.005,        // 0.5% = floor. Below = CHOP, skip (lowered from 0.8%).
    atrPeriod: 14,
  },

  // ═══════════════════════════════════════════════════════════════
  // QUANT-LEVEL MOMENTUM THRESHOLDS
  // ═══════════════════════════════════════════════════════════════
  momentum: {
    // Volume spike: 1.1x for 4H timeframe (QUANT-LEVEL: optimized for swing entries)
    volumeSpikeMultiple: 1.1,
    volumeAvgPeriod: 20,

    // RSI: Standard settings for swing timeframe
    rsiPeriod: 14,
    rsiBullishCross: 50,
    rsiBearishCross: 50,
    rsiOverbought: 70,
    rsiOversold: 30,

    // EMA: Fast/slow for trend detection
    emaFast: 9,
    emaSlow: 21,

    // Bollinger Bands: Mean reversion extremes (25%/75% for 4H swing)
    bbPeriod: 20,
    bbStdDev: 2,
    bbExtremeLong: 0.25,   // 25% = LONG entry in RANGE mode (widened from 20%)
    bbExtremeShort: 0.75,  // 75% = SHORT entry in RANGE mode (widened from 80%)

    // MACD: Momentum confirmation
    macdFast: 12,
    macdSlow: 26,
    macdSignal: 9,

    // Price breakout: Lookback for swing high/low breaks
    breakoutLookback: 10,

    // Candle momentum: Body ratio for strong candles
    minBodyRatio: 0.6,

    // Minimum signals for direction confirmation
    minSignals: 1,
  },

  // ML - set to 0 for data collection (bypass)
  minWinProbability: 0,

  // Trade lifecycle
  maxHoldHours: 72,

  // Risk management
  virtualBalancePerCoin: 10000,
  leverage: 1,
  takerFeeRate: 0.00025,
  stopLossATRMultiple: 2.0,

  // Trailing stop (activated after TP1)
  trailingStopPct: 1.0,  // 1% trail distance for 4H swing

  // Tracking
  tradesDir: path.join(process.cwd(), 'data', 'paper-trades-swing'),
  summaryFile: path.join(process.cwd(), 'data', 'paper-trades-summary-swing.json'),

  // Persistence
  enableStateBackups: false,
  backupIntervalMs: 12 * 60 * 60_000,
  maxBackupsPerSymbol: 24,

  // AUTO-LEARNING
  autoLearn: {
    enabled: false,                // DISABLED - Collecting data with new quant features (Williams %R, OFI, 1.1x vol, 25% Kelly)
    triggerEveryNTrades: 50,       // Retrain after every 50 closed trades (swing trades less frequent)
    minTradesForTraining: 30,     // Need at least 30 trades to train
  },
};

// ═══════════════════════════════════════════════════════════════
// MOMENTUM INDICATORS (ported from scalp trader)
// ═══════════════════════════════════════════════════════════════

interface MomentumSignals {
  volumeSpike: boolean;
  volumeRatio: number;
  rsiValue: number;
  rsiBullishCross: boolean;
  rsiBearishCross: boolean;
  rsiOverbought: boolean;
  rsiOversold: boolean;
  williamsR: number;              // Williams %R: -100 to 0, <-80 oversold, >-20 overbought
  williamsROversold: boolean;
  williamsROverbought: boolean;
  emaFast: number;
  emaSlow: number;
  emaBullishCross: boolean;
  emaBearishCross: boolean;
  emaAligned: 'bullish' | 'bearish' | 'neutral';
  bbUpper: number;
  bbLower: number;
  bbMiddle: number;
  bbPosition: number;
  bbBreakoutUp: boolean;
  bbBreakoutDown: boolean;
  priceBreakoutUp: boolean;
  priceBreakoutDown: boolean;
  candleMomentum: 'bullish' | 'bearish' | 'neutral';
  macdLine: number;
  macdSignal: number;
  macdHistogram: number;
  macdBullishCross: boolean;
  macdBearishCross: boolean;
  atr: number;
  atrPercent: number;
  regime: 'TREND' | 'RANGE' | 'CHOP';
  adx: number;
  plusDI: number;
  minusDI: number;
  vwap: number;
  vwapDeviation: number;
  vwapDeviationStd: number;
  priceAboveVwap: boolean;
  // Multi-session VWAP bands
  sessionVwapAsia: number;
  sessionVwapLondon: number;
  sessionVwapNy: number;
  sessionVwapUpper: number;
  sessionVwapLower: number;
  currentSession: 'asia' | 'london' | 'ny';
  killZone: 'LONDON' | 'NY_OPEN' | 'NY_AFTERNOON' | 'ASIA' | 'OFF_HOURS';
  isKillZone: boolean;
  swingHigh: number | null;
  swingLow: number | null;
  bullishSignals: number;
  bearishSignals: number;
  direction: 'LONG' | 'SHORT' | 'NEUTRAL';
  strength: number;
}

function calculateRSI(candles: Candle[], period: number): number[] {
  const rsi: number[] = [];
  let gainsSum = 0;   // Cumulative sum for initial average only
  let lossesSum = 0;  // Cumulative sum for initial average only
  let avgGain = 0;    // Running Wilder's average
  let avgLoss = 0;    // Running Wilder's average

  for (let i = 1; i < candles.length; i++) {
    const change = candles[i].close - candles[i - 1].close;
    const gain = change > 0 ? change : 0;
    const loss = change < 0 ? -change : 0;

    if (i <= period) {
      // Build cumulative sums for initial average
      gainsSum += gain;
      lossesSum += loss;

      if (i === period) {
        // Initialize running averages
        avgGain = gainsSum / period;
        avgLoss = lossesSum / period;
        const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
        rsi.push(100 - (100 / (1 + rs)));
      }
    } else {
      // Wilder's smoothing: use previous average, not cumulative sum
      avgGain = ((avgGain * (period - 1)) + gain) / period;
      avgLoss = ((avgLoss * (period - 1)) + loss) / period;
      const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
      rsi.push(100 - (100 / (1 + rs)));
    }
  }

  return rsi;
}

function calculateWilliamsR(candles: Candle[], period: number): number[] {
  const williamsR: number[] = [];

  if (candles.length < period) {
    return williamsR;
  }

  for (let i = period - 1; i < candles.length; i++) {
    const slice = candles.slice(i - period + 1, i + 1);
    const highestHigh = Math.max(...slice.map(c => c.high));
    const lowestLow = Math.min(...slice.map(c => c.low));
    const currentClose = candles[i].close;

    // Williams %R: -100 * (highestHigh - currentClose) / (highestHigh - lowestLow)
    // Range: -100 (most oversold) to 0 (most overbought)
    const range = highestHigh - lowestLow;
    const wr = range === 0 ? -50 : -100 * (highestHigh - currentClose) / range;
    williamsR.push(wr);
  }

  return williamsR;
}

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

function calculateBollingerBands(candles: Candle[], period: number, stdDev: number): {
  upper: number[];
  middle: number[];
  lower: number[];
} {
  const upper: number[] = [];
  const middle: number[] = [];
  const lower: number[] = [];

  for (let i = period - 1; i < candles.length; i++) {
    const slice = candles.slice(i - period + 1, i + 1);
    const mean = slice.reduce((s, c) => s + c.close, 0) / period;
    const variance = slice.reduce((s, c) => s + Math.pow(c.close - mean, 2), 0) / period;
    const std = Math.sqrt(variance);

    middle.push(mean);
    upper.push(mean + stdDev * std);
    lower.push(mean - stdDev * std);
  }

  return { upper, middle, lower };
}

function calculateMACD(candles: Candle[], fastPeriod: number, slowPeriod: number, signalPeriod: number): {
  macdLine: number[];
  signalLine: number[];
  histogram: number[];
} {
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
    const multiplier = 2 / (signalPeriod + 1);
    let sum = 0;
    for (let i = 0; i < signalPeriod; i++) {
      sum += macdLine[i];
    }
    signalLine.push(sum / signalPeriod);

    for (let i = signalPeriod; i < macdLine.length; i++) {
      const value = (macdLine[i] - signalLine[signalLine.length - 1]) * multiplier + signalLine[signalLine.length - 1];
      signalLine.push(value);
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

  for (const candle of candles) {
    const typicalPrice = (candle.high + candle.low + candle.close) / 3;
    typicalPrices.push(typicalPrice);
    cumulativePV += typicalPrice * candle.volume;
    cumulativeVolume += candle.volume;
  }

  const vwap = cumulativeVolume > 0 ? cumulativePV / cumulativeVolume : candles[candles.length - 1]?.close || 0;

  const deviations = typicalPrices.map(p => Math.pow(p - vwap, 2));
  const variance = deviations.reduce((a, b) => a + b, 0) / deviations.length;
  const stdDev = Math.sqrt(variance);

  return { vwap, stdDev };
}

interface MultiSessionVWAP {
  asia: { vwap: number; upper: number; lower: number };
  london: { vwap: number; upper: number; lower: number };
  ny: { vwap: number; upper: number; lower: number };
  currentSession: 'asia' | 'london' | 'ny';
}

function calculateMultiSessionVWAP(candles: Candle[], numStdDev: number = 1.5): MultiSessionVWAP {
  const asia: { vwap: number; upper: number; lower: number } = { vwap: 0, upper: 0, lower: 0 };
  const london: { vwap: number; upper: number; lower: number } = { vwap: 0, upper: 0, lower: 0 };
  const ny: { vwap: number; upper: number; lower: number } = { vwap: 0, upper: 0, lower: 0 };

  const asiaCandles: Candle[] = [];
  const londonCandles: Candle[] = [];
  const nyCandles: Candle[] = [];

  for (const candle of candles) {
    const hour = new Date(candle.timestamp).getUTCHours();

    if (hour >= 0 && hour < 8) {
      asiaCandles.push(candle);
    }
    if (hour >= 7 && hour < 16) {
      londonCandles.push(candle);
    }
    if (hour >= 13 && hour < 22) {
      nyCandles.push(candle);
    }
  }

  const asiaData = asiaCandles.length > 0 ? calculateVWAPWithBands(asiaCandles, numStdDev) : { vwap: 0, upper: 0, lower: 0 };
  const londonData = londonCandles.length > 0 ? calculateVWAPWithBands(londonCandles, numStdDev) : { vwap: 0, upper: 0, lower: 0 };
  const nyData = nyCandles.length > 0 ? calculateVWAPWithBands(nyCandles, numStdDev) : { vwap: 0, upper: 0, lower: 0 };

  const currentHour = new Date(candles[candles.length - 1].timestamp).getUTCHours();
  let currentSession: 'asia' | 'london' | 'ny' = 'asia';
  if (currentHour >= 13 && currentHour < 22) {
    currentSession = 'ny';
  } else if (currentHour >= 7 && currentHour < 16) {
    currentSession = 'london';
  }

  return {
    asia: asiaData,
    london: londonData,
    ny: nyData,
    currentSession,
  };
}

function calculateVWAPWithBands(candles: Candle[], numStdDev: number): { vwap: number; upper: number; lower: number } {
  let cumulativePV = 0;
  let cumulativeVolume = 0;
  const typicalPrices: number[] = [];

  for (const candle of candles) {
    const typicalPrice = (candle.high + candle.low + candle.close) / 3;
    typicalPrices.push(typicalPrice);
    cumulativePV += typicalPrice * candle.volume;
    cumulativeVolume += candle.volume;
  }

  const vwap = cumulativeVolume > 0 ? cumulativePV / cumulativeVolume : candles[candles.length - 1]?.close || 0;

  const deviations = typicalPrices.map(p => Math.pow(p - vwap, 2));
  const variance = deviations.reduce((a, b) => a + b, 0) / deviations.length;
  const stdDev = Math.sqrt(variance);

  return {
    vwap,
    upper: vwap + numStdDev * stdDev,
    lower: vwap - numStdDev * stdDev,
  };
}

function calculateATR(candles: Candle[], period: number): number[] {
  const atr: number[] = [];
  const trueRanges: number[] = [];

  for (let i = 1; i < candles.length; i++) {
    const high = candles[i].high;
    const low = candles[i].low;
    const prevClose = candles[i - 1].close;

    const tr = Math.max(
      high - low,
      Math.abs(high - prevClose),
      Math.abs(low - prevClose)
    );
    trueRanges.push(tr);

    if (trueRanges.length >= period) {
      if (atr.length === 0) {
        const sum = trueRanges.slice(-period).reduce((a, b) => a + b, 0);
        atr.push(sum / period);
      } else {
        const prevATR = atr[atr.length - 1];
        atr.push((prevATR * (period - 1) + tr) / period);
      }
    }
  }

  return atr;
}

function calculateADX(candles: Candle[], period: number = 14): { adx: number[]; plusDI: number[]; minusDI: number[] } {
  const adx: number[] = [];
  const plusDI: number[] = [];
  const minusDI: number[] = [];

  if (candles.length < period * 2) {
    return { adx: [0], plusDI: [0], minusDI: [0] };
  }

  const tr: number[] = [];
  const plusDM: number[] = [];
  const minusDM: number[] = [];

  for (let i = 1; i < candles.length; i++) {
    const high = candles[i].high;
    const low = candles[i].low;
    const prevClose = candles[i - 1].close;
    const prevHigh = candles[i - 1].high;
    const prevLow = candles[i - 1].low;

    const trueRange = Math.max(high - low, Math.abs(high - prevClose), Math.abs(low - prevClose));
    const upMove = high - prevHigh;
    const downMove = prevLow - low;

    const positiveDM = (upMove > downMove && upMove > 0) ? upMove : 0;
    const negativeDM = (downMove > upMove && downMove > 0) ? downMove : 0;

    tr.push(trueRange);
    plusDM.push(positiveDM);
    minusDM.push(negativeDM);
  }

  let smoothedTR = 0;
  let smoothedPlusDM = 0;
  let smoothedMinusDM = 0;

  for (let i = 0; i < period; i++) {
    smoothedTR += tr[i];
    smoothedPlusDM += plusDM[i];
    smoothedMinusDM += minusDM[i];
  }

  const di: number[] = [];
  for (let i = period; i < tr.length; i++) {
    smoothedPlusDM = smoothedPlusDM - (smoothedPlusDM / period) + plusDM[i];
    smoothedMinusDM = smoothedMinusDM - (smoothedMinusDM / period) + minusDM[i];
    smoothedTR = smoothedTR - (smoothedTR / period) + tr[i];

    const plusDI_val = 100 * (smoothedPlusDM / smoothedTR);
    const minusDI_val = 100 * (smoothedMinusDM / smoothedTR);

    plusDI.push(plusDI_val);
    minusDI.push(minusDI_val);

    const dx = 100 * Math.abs(plusDI_val - minusDI_val) / (plusDI_val + minusDI_val || 1);
    di.push(dx);
  }

  if (di.length >= period) {
    let adxSum = 0;
    for (let i = 0; i < period; i++) {
      adxSum += di[i];
    }
    adx.push(adxSum / period);

    for (let i = period; i < di.length; i++) {
      const newAdx = (adx[adx.length - 1] * (period - 1) + di[i]) / period;
      adx.push(newAdx);
    }
  }

  return { adx, plusDI, minusDI };
}

function getKillZone(timestamp: number): { zone: 'LONDON' | 'NY_OPEN' | 'NY_AFTERNOON' | 'ASIA' | 'OFF_HOURS'; isActive: boolean } {
  const date = new Date(timestamp);
  const hour = date.getUTCHours();

  if (hour >= 7 && hour < 10) {
    return { zone: 'LONDON', isActive: true };
  } else if (hour >= 13 && hour < 16) {
    return { zone: 'NY_OPEN', isActive: true };
  } else if (hour >= 18 && hour < 20) {
    return { zone: 'NY_AFTERNOON', isActive: true };
  } else if (hour >= 0 && hour < 3) {
    return { zone: 'ASIA', isActive: true };
  } else {
    return { zone: 'OFF_HOURS', isActive: false };
  }
}

interface SwingPoints {
  swingHigh: number | null;
  swingLow: number | null;
}

function findSwingPoints(candles: Candle[], lookback: number = 5): SwingPoints {
  let swingHigh: number | null = null;
  let swingLow: number | null = null;

  const len = candles.length;
  if (len < lookback * 2 + 1) {
    return { swingHigh: null, swingLow: null };
  }

  for (let i = len - lookback - 1; i >= lookback; i--) {
    const high = candles[i].high;
    const low = candles[i].low;

    let isSwingHigh = true;
    let isSwingLow = true;

    for (let j = i - lookback; j <= i + lookback; j++) {
      if (j === i) continue;
      if (candles[j].high >= high) isSwingHigh = false;
      if (candles[j].low <= low) isSwingLow = false;
    }

    if (isSwingHigh && swingHigh === null) {
      swingHigh = high;
    }
    if (isSwingLow && swingLow === null) {
      swingLow = low;
    }

    if (swingHigh !== null && swingLow !== null) {
      break;
    }
  }

  return { swingHigh, swingLow };
}

function analyzeMomentum(candles: Candle[]): MomentumSignals {
  const cfg = CONFIG.momentum;
  const len = candles.length;

  if (len < 30) {
    return {
      volumeSpike: false, volumeRatio: 1,
      rsiValue: 50, rsiBullishCross: false, rsiBearishCross: false,
      rsiOverbought: false, rsiOversold: false,
      williamsR: -50, williamsROversold: false, williamsROverbought: false,
      emaFast: 0, emaSlow: 0, emaBullishCross: false, emaBearishCross: false,
      emaAligned: 'neutral',
      bbUpper: 0, bbLower: 0, bbMiddle: 0, bbPosition: 0.5,
      bbBreakoutUp: false, bbBreakoutDown: false,
      priceBreakoutUp: false, priceBreakoutDown: false,
      candleMomentum: 'neutral',
      macdLine: 0, macdSignal: 0, macdHistogram: 0,
      macdBullishCross: false, macdBearishCross: false,
      atr: 0, atrPercent: 0, regime: 'CHOP' as const,
      adx: 0, plusDI: 0, minusDI: 0,
      vwap: 0, vwapDeviation: 0, vwapDeviationStd: 0, priceAboveVwap: false,
      sessionVwapAsia: 0, sessionVwapLondon: 0, sessionVwapNy: 0,
      sessionVwapUpper: 0, sessionVwapLower: 0,
      currentSession: 'asia' as 'asia' | 'london' | 'ny',
      killZone: 'OFF_HOURS' as const, isKillZone: false,
      swingHigh: null, swingLow: null,
      bullishSignals: 0, bearishSignals: 0,
      direction: 'NEUTRAL', strength: 0,
    };
  }

  const current = candles[len - 1];
  const prev = candles[len - 2];

  // Volume spike
  const volumeSlice = candles.slice(-cfg.volumeAvgPeriod - 1, -1);
  const avgVolume = volumeSlice.reduce((sum, c) => sum + c.volume, 0) / volumeSlice.length;
  const volumeRatio = current.volume / avgVolume;
  const volumeSpike = volumeRatio >= cfg.volumeSpikeMultiple;

  // RSI
  const rsiValues = calculateRSI(candles, cfg.rsiPeriod);
  const rsiValue = rsiValues[rsiValues.length - 1] || 50;
  const prevRsi = rsiValues[rsiValues.length - 2] || 50;
  const rsiBullishCross = prevRsi < cfg.rsiBullishCross && rsiValue >= cfg.rsiBullishCross;
  const rsiBearishCross = prevRsi > cfg.rsiBearishCross && rsiValue <= cfg.rsiBearishCross;
  const rsiOverbought = rsiValue >= cfg.rsiOverbought;
  const rsiOversold = rsiValue <= cfg.rsiOversold;

  // Williams %R (momentum oscillator: -100 to 0, <-80 oversold, >-20 overbought)
  const williamsRValues = calculateWilliamsR(candles, cfg.rsiPeriod);
  const williamsR = williamsRValues[williamsRValues.length - 1] || -50;
  const prevWilliamsR = williamsRValues[williamsRValues.length - 2] || -50;
  const williamsROverbought = williamsR > -20;
  const williamsROversold = williamsR < -80;

  // EMA crossover
  const emaFastValues = calculateEMA(candles, cfg.emaFast);
  const emaSlowValues = calculateEMA(candles, cfg.emaSlow);
  const emaFast = emaFastValues[emaFastValues.length - 1] || current.close;
  const emaSlow = emaSlowValues[emaSlowValues.length - 1] || current.close;
  const prevEmaFast = emaFastValues[emaFastValues.length - 2] || emaFast;
  const prevEmaSlow = emaSlowValues[emaSlowValues.length - 2] || emaSlow;
  const emaBullishCross = prevEmaFast <= prevEmaSlow && emaFast > emaSlow;
  const emaBearishCross = prevEmaFast >= prevEmaSlow && emaFast < emaSlow;
  const emaAligned: 'bullish' | 'bearish' | 'neutral' =
    emaFast > emaSlow ? 'bullish' :
    emaFast < emaSlow ? 'bearish' : 'neutral';

  // Bollinger Bands
  const bb = calculateBollingerBands(candles, cfg.bbPeriod, cfg.bbStdDev);
  const bbUpper = bb.upper[bb.upper.length - 1] || current.close * 1.02;
  const bbLower = bb.lower[bb.lower.length - 1] || current.close * 0.98;
  const bbMiddle = bb.middle[bb.middle.length - 1] || current.close;
  const bbBreakoutUp = current.close > bbUpper && prev.close <= (bb.upper[bb.upper.length - 2] || bbUpper);
  const bbBreakoutDown = current.close < bbLower && prev.close >= (bb.lower[bb.lower.length - 2] || bbLower);
  const bbRange = bbUpper - bbLower;
  const bbPosition = bbRange > 0 ? Math.max(0, Math.min(1, (current.close - bbLower) / bbRange)) : 0.5;

  // Price breakout
  const lookbackCandles = candles.slice(-cfg.breakoutLookback - 1, -1);
  const recentHigh = Math.max(...lookbackCandles.map(c => c.high));
  const recentLow = Math.min(...lookbackCandles.map(c => c.low));
  const priceBreakoutUp = current.close > recentHigh;
  const priceBreakoutDown = current.close < recentLow;

  // Candle momentum
  const candleRange = current.high - current.low;
  const candleBody = Math.abs(current.close - current.open);
  const bodyRatio = candleRange > 0 ? candleBody / candleRange : 0;
  const isBullishCandle = current.close > current.open;
  const isBearishCandle = current.close < current.open;
  const candleMomentum: 'bullish' | 'bearish' | 'neutral' =
    bodyRatio >= cfg.minBodyRatio
      ? (isBullishCandle ? 'bullish' : isBearishCandle ? 'bearish' : 'neutral')
      : 'neutral';

  // MACD
  const macd = calculateMACD(candles, cfg.macdFast, cfg.macdSlow, cfg.macdSignal);
  const macdLine = macd.macdLine[macd.macdLine.length - 1] || 0;
  const macdSignalLine = macd.signalLine[macd.signalLine.length - 1] || 0;
  const macdHistogram = macd.histogram[macd.histogram.length - 1] || 0;
  const prevMacdLine = macd.macdLine[macd.macdLine.length - 2] || macdLine;
  const prevMacdSignal = macd.signalLine[macd.signalLine.length - 2] || macdSignalLine;
  const macdBullishCross = prevMacdLine <= prevMacdSignal && macdLine > macdSignalLine;
  const macdBearishCross = prevMacdLine >= prevMacdSignal && macdLine < macdSignalLine;

  // ATR and ADX-based Regime Detection
  const atrValues = calculateATR(candles, CONFIG.regime.atrPeriod);
  const atr = atrValues[atrValues.length - 1] || 0;
  const atrPercent = current.close > 0 ? (atr / current.close) : 0;

  const adxResult = calculateADX(candles, 14);
  const adx = adxResult.adx[adxResult.adx.length - 1] || 0;
  const plusDI = adxResult.plusDI[adxResult.plusDI.length - 1] || 0;
  const minusDI = adxResult.minusDI[adxResult.minusDI.length - 1] || 0;

  // Enhanced regime detection using both ATR volatility and ADX trend strength
  // CHOP: Low volatility AND low ADX (both conditions - dead market)
  // TREND: High volatility AND high ADX (strong trend)
  // RANGE: Between the two
  let regime: 'TREND' | 'RANGE' | 'CHOP';
  if (atrPercent < CONFIG.regime.minVolatility && adx < 15) {
    regime = 'CHOP';  // Low volatility AND very weak trend = choppy (was OR, now AND)
  } else if (atrPercent >= CONFIG.regime.volatilityThreshold && adx >= 20) {
    regime = 'TREND';  // High volatility AND strong trend = trending (lowered ADX from 25 to 20)
  } else {
    regime = 'RANGE';  // Between = range-bound
  }

  // VWAP - Multi-session
  const multiSessionVwap = calculateMultiSessionVWAP(candles, 1.5);
  const vwap = multiSessionVwap[multiSessionVwap.currentSession]?.vwap || current.close;
  const vwapDeviation = vwap > 0 ? ((current.close - vwap) / vwap) * 100 : 0;
  const vwapDeviationStd = 1.0;
  const priceAboveVwap = current.close > vwap;
  const currentSessionVwap = multiSessionVwap[multiSessionVwap.currentSession];
  const sessionVwapUpper = currentSessionVwap?.upper || current.close * 1.01;
  const sessionVwapLower = currentSessionVwap?.lower || current.close * 0.99;

  // Kill Zone
  const killZoneData = getKillZone(current.timestamp);
  const killZone = killZoneData.zone;
  const isKillZone = killZoneData.isActive;

  // Count signals
  let bullishSignals = 0;
  let bearishSignals = 0;

  if (volumeSpike && candleMomentum === 'bullish') bullishSignals++;
  if (volumeSpike && candleMomentum === 'bearish') bearishSignals++;
  if (rsiBullishCross || rsiOversold) bullishSignals++;
  if (rsiBearishCross || rsiOverbought) bearishSignals++;
  if (williamsROversold) bullishSignals++;        // Williams %R < -80 = oversold
  if (williamsROverbought) bearishSignals++;      // Williams %R > -20 = overbought
  if (emaBullishCross || emaAligned === 'bullish') bullishSignals++;
  if (emaBearishCross || emaAligned === 'bearish') bearishSignals++;
  if (bbBreakoutUp) bullishSignals++;
  if (bbBreakoutDown) bearishSignals++;
  if (priceBreakoutUp) bullishSignals++;
  if (priceBreakoutDown) bearishSignals++;
  if (candleMomentum === 'bullish') bullishSignals++;
  if (candleMomentum === 'bearish') bearishSignals++;
  if (macdBullishCross || macdHistogram > 0) bullishSignals++;
  if (macdBearishCross || macdHistogram < 0) bearishSignals++;

  let direction: 'LONG' | 'SHORT' | 'NEUTRAL' = 'NEUTRAL';
  const maxSignals = Math.max(bullishSignals, bearishSignals);

  if (bullishSignals >= cfg.minSignals && bullishSignals > bearishSignals) {
    direction = 'LONG';
  } else if (bearishSignals >= cfg.minSignals && bearishSignals > bullishSignals) {
    direction = 'SHORT';
  }

  const strength = maxSignals / 7;

  // Find swing points
  const swingPoints = findSwingPoints(candles, 5);

  return {
    volumeSpike, volumeRatio,
    rsiValue, rsiBullishCross, rsiBearishCross, rsiOverbought, rsiOversold,
    williamsR, williamsROversold, williamsROverbought,
    emaFast, emaSlow, emaBullishCross, emaBearishCross, emaAligned,
    bbUpper, bbLower, bbMiddle, bbPosition, bbBreakoutUp, bbBreakoutDown,
    priceBreakoutUp, priceBreakoutDown,
    candleMomentum,
    macdLine, macdSignal: macdSignalLine, macdHistogram,
    macdBullishCross, macdBearishCross,
    atr, atrPercent, regime,
    adx, plusDI, minusDI,
    vwap, vwapDeviation, vwapDeviationStd, priceAboveVwap,
    sessionVwapAsia: multiSessionVwap.asia.vwap,
    sessionVwapLondon: multiSessionVwap.london.vwap,
    sessionVwapNy: multiSessionVwap.ny.vwap,
    sessionVwapUpper,
    sessionVwapLower,
    currentSession: multiSessionVwap.currentSession,
    killZone, isKillZone,
    swingHigh: swingPoints.swingHigh,
    swingLow: swingPoints.swingLow,
    bullishSignals, bearishSignals,
    direction, strength,
  };
}

// ═══════════════════════════════════════════════════════════════
// SMC CONTEXT (adjusts trades, never blocks them)
// ═══════════════════════════════════════════════════════════════

interface SMCContext {
  stopTighten: number;      // Multiplier for stop distance (< 1 = tighter)
  sizeMultiplier: number;   // Multiplier for position size
  confidenceBoost: number;  // Points to add to quality score
  adjustments: string[];    // Log of what was applied
}

function applySMCContext(
  direction: 'LONG' | 'SHORT',
  smcAnalysis: any,
  ictAnalysis: any,
  isKillZone: boolean,
  weeklyAlignmentScore: number = 0.5,  // 0 = opposing, 0.5 = neutral, 1 = aligned
): SMCContext {
  const ctx: SMCContext = {
    stopTighten: 1.0,
    sizeMultiplier: 1.0,
    confidenceBoost: 0,
    adjustments: [],
  };

  // ═══════════════════════════════════════════════════════════════
  // QUANT-LEVEL: Weekly trend is ALWAYS an adjustment, never a gate
  // ═══════════════════════════════════════════════════════════════
  if (weeklyAlignmentScore < 0.3) {
    // Opposing weekly trend -> smaller position, tighter stop (risk management)
    ctx.sizeMultiplier *= 0.6;
    ctx.stopTighten *= 0.85;
    ctx.adjustments.push(`WeeklyOpposing: -40%size,SL-15%`);
  } else if (weeklyAlignmentScore >= 0.8) {
    // Aligned weekly trend -> confidence boost
    ctx.confidenceBoost += 5;
    ctx.adjustments.push(`WeeklyAligned: +5conf`);
  }

  if (!smcAnalysis || !ictAnalysis) return ctx;

  const entryCriteria = ictAnalysis?.entryCriteria;

  // Fresh OB nearby -> tighten stop 15%, +5 confidence
  const freshOBs = smcAnalysis?.orderBlocks?.filter((ob: OrderBlock) => {
    const isFresh = (ob.testCount ?? 0) <= 1;
    const rightDirection = (direction === 'LONG' && ob.type === 'bull') ||
                          (direction === 'SHORT' && ob.type === 'bear');
    const tradeable = ob.state !== 'INVALIDATED' && ob.state !== 'NEW_OB';
    return isFresh && rightDirection && tradeable;
  }) || [];

  if (freshOBs.length > 0) {
    ctx.stopTighten *= 0.85;
    ctx.confidenceBoost += 5;
    ctx.adjustments.push('FreshOB: SL-15%,+5conf');
  }

  // Displacement detected -> +20% position size, +5 confidence
  const hasDisplacement = entryCriteria?.displacementPresent || false;
  if (hasDisplacement) {
    ctx.sizeMultiplier *= 1.2;
    ctx.confidenceBoost += 5;
    ctx.adjustments.push('Displacement: +20%size,+5conf');
  }

  // In OTE zone -> +5 confidence
  const inOTE = entryCriteria?.inOTEZone || false;
  if (inOTE) {
    ctx.confidenceBoost += 5;
    ctx.adjustments.push('OTE: +5conf');
  }

  // OB/FVG confluence -> tighten stop 20%, +10% size, +5 confidence
  const hasOBFVGConfluence = entryCriteria?.obFvgConfluence || false;
  if (hasOBFVGConfluence) {
    ctx.stopTighten *= 0.80;
    ctx.sizeMultiplier *= 1.1;
    ctx.confidenceBoost += 5;
    ctx.adjustments.push('OB/FVG: SL-20%,+10%size,+5conf');
  }

  // Kill zone active -> +3 confidence
  if (isKillZone) {
    ctx.confidenceBoost += 3;
    ctx.adjustments.push('KillZone: +3conf');
  }

  return ctx;
}

// ═══════════════════════════════════════════════════════════════
// MARKET SNAPSHOT (for ML training - saved at entry & exit)
// ═══════════════════════════════════════════════════════════════

interface MarketSnapshot {
  price: number;
  timestamp: number;
  regime: 'TREND' | 'RANGE' | 'CHOP';
  atrPercent: number;
  bbPosition: number;
  bbWidth: number;
  rsiValue: number;
  emaFast: number;
  emaSlow: number;
  emaAligned: string;
  macdLine: number;
  macdSignal: number;
  macdHistogram: number;
  vwap: number;
  vwapDeviation: number;
  vwapDeviationStd: number;
  priceAboveVwap: boolean;
  volumeRatio: number;
  volumeSpike: boolean;
  killZone: string;
  isKillZone: boolean;
  direction: string;
  strength: number;
}

function captureSnapshot(m: MomentumSignals, price: number): MarketSnapshot {
  const bbRange = m.bbUpper - m.bbLower;
  return {
    price,
    timestamp: Date.now(),
    regime: m.regime,
    atrPercent: m.atrPercent,
    bbPosition: m.bbPosition,
    bbWidth: bbRange > 0 && price > 0 ? (bbRange / price) * 100 : 0,
    rsiValue: m.rsiValue,
    emaFast: m.emaFast,
    emaSlow: m.emaSlow,
    emaAligned: m.emaAligned,
    macdLine: m.macdLine,
    macdSignal: m.macdSignal,
    macdHistogram: m.macdHistogram,
    vwap: m.vwap,
    vwapDeviation: m.vwapDeviation,
    vwapDeviationStd: m.vwapDeviationStd,
    priceAboveVwap: m.priceAboveVwap,
    volumeRatio: m.volumeRatio,
    volumeSpike: m.volumeSpike,
    killZone: m.killZone,
    isKillZone: m.isKillZone,
    direction: m.direction,
    strength: m.strength,
  };
}

// ═══════════════════════════════════════════════════════════════
// LIVE SUMMARY (for OpenClaw integration)
// ═══════════════════════════════════════════════════════════════

interface LiveSummary {
  timestamp: string;
  cycle: number;
  trader: 'swing';
  openTrades: number;
  totalTrades: number;
  totalWins: number;
  totalLosses: number;
  winRate: number;
  totalPnl: number;
  currentStreak: { type: 'WIN' | 'LOSS' | 'NONE'; count: number; };
  recentPerformance: { trades: number; wins: number; winRate: number; pnl: number; };
  openPositions: Array<{
    symbol: string;
    direction: 'LONG' | 'SHORT';
    entryPrice: number;
    currentPrice: number;
    unrealizedPnl: number;
    pnlPercent: number;
  }>;
}

function writeLiveSummary(
  traders: CoinTrader[],
  iteration: number,
  results: Array<{ symbol: string; price: number; trader: CoinTrader }>
): void {
  // Calculate summary metrics
  const openCount = traders.filter(t => t.state.openTrade).length;
  const totalPnl = traders.reduce((sum, t) => sum + t.state.stats.totalPnl, 0);
  const totalTrades = traders.reduce((sum, t) => sum + t.state.stats.totalTrades, 0);
  const totalWins = traders.reduce((sum, t) => sum + t.state.stats.wins, 0);
  const totalLosses = traders.reduce((sum, t) => sum + t.state.stats.losses, 0);
  const winRate = totalTrades > 0 ? (totalWins / totalTrades) * 100 : 0;

  // Calculate current streak
  const allTrades = traders.flatMap(t => t.state.trades).filter(t => t.status === 'CLOSED');
  let currentStreak = 0;
  let currentStreakType: 'WIN' | 'LOSS' | 'NONE' = 'NONE';
  for (let i = allTrades.length - 1; i >= 0; i--) {
    const pnl = allTrades[i].pnl || 0;
    if (currentStreak === 0) {
      currentStreakType = pnl >= 0 ? 'WIN' : 'LOSS';
      currentStreak++;
    } else if ((currentStreakType === 'WIN' && pnl >= 0) || (currentStreakType === 'LOSS' && pnl < 0)) {
      currentStreak++;
    } else {
      break;
    }
  }

  // Calculate recent performance (last 10 trades)
  const recentTrades = allTrades.slice(-10);
  const recentWins = recentTrades.filter(t => (t.pnl || 0) > 0).length;
  const recentWinRate = recentTrades.length > 0 ? (recentWins / recentTrades.length) * 100 : 0;
  const recentPnl = recentTrades.reduce((sum, t) => sum + (t.pnl || 0), 0);

  // Open positions
  const openPositions: LiveSummary['openPositions'] = [];
  for (const { symbol, price, trader } of results) {
    if (trader.state.openTrade) {
      const trade = trader.state.openTrade;
      const isLong = trade.direction === 'LONG';
      const currentPrice = price > 0 ? price : trade.entryPrice;
      const priceDiff = isLong ? currentPrice - trade.entryPrice : trade.entryPrice - currentPrice;
      const unrealizedPnl = priceDiff * (trade.currentPositionSize || 1);
      const pnlPercent = trade.entryPrice > 0 ? (priceDiff / trade.entryPrice) * 100 : 0;
      openPositions.push({
        symbol,
        direction: trade.direction,
        entryPrice: trade.entryPrice,
        currentPrice,
        unrealizedPnl: Math.round(unrealizedPnl * 100) / 100,
        pnlPercent: Math.round(pnlPercent * 100) / 100,
      });
    }
  }

  const summary: LiveSummary = {
    timestamp: new Date().toISOString(),
    cycle: iteration,
    trader: 'swing',
    openTrades: openCount,
    totalTrades,
    totalWins,
    totalLosses,
    winRate: Math.round(winRate * 10) / 10,
    totalPnl: Math.round(totalPnl * 100) / 100,
    currentStreak: {
      type: currentStreakType,
      count: currentStreak,
    },
    recentPerformance: {
      trades: recentTrades.length,
      wins: recentWins,
      winRate: Math.round(recentWinRate * 10) / 10,
      pnl: Math.round(recentPnl * 100) / 100,
    },
    openPositions,
  };

  // Write to file
  try {
    const summaryDir = path.dirname(CONFIG.reporting.summaryFile);
    if (!fs.existsSync(summaryDir)) {
      fs.mkdirSync(summaryDir, { recursive: true });
    }
    fs.writeFileSync(CONFIG.reporting.summaryFile, JSON.stringify(summary, null, 2));
  } catch (e) {
    // Silently fail - this is optional reporting
  }
}

// ═══════════════════════════════════════════════════════════════
// PAPER TRADE TYPES
// ═══════════════════════════════════════════════════════════════

interface PaperTrade {
  id: string;
  symbol: string;
  direction: 'LONG' | 'SHORT';
  entryPrice: number;
  entryTime: number;
  stopLoss: number;
  originalStopLoss: number;
  takeProfit1: number;
  takeProfit2: number;
  takeProfit3: number;
  trailingStop: number | null;
  originalPositionSize: number;
  currentPositionSize: number;
  tp1Hit: boolean;
  tp2Hit: boolean;
  tp3Hit: boolean;
  stopLossMovedToBreakeven: boolean;
  status: 'OPEN' | 'CLOSED';
  exitPrice?: number;
  exitTime?: number;
  exitReason?: 'TP1' | 'TP2' | 'TP3' | 'SL' | 'TRAILING' | 'MANUAL' | 'TIMEOUT';
  pnl?: number;
  pnlPercent?: number;
  mlPrediction: number;
  regime: 'TREND' | 'RANGE' | 'CHOP';
  qualityScore: number;
  smcContext: string[];
  entryFeatures?: MarketSnapshot;
  exitFeatures?: MarketSnapshot;
  entryFee?: number;  // Track entry fee for proper PnL calculation
}

interface TimeframeData {
  interval: string;
  candles: Candle[];
  momentum: MomentumSignals;
  smcAnalysis: any;
  ictAnalysis: any;
  bias: 'bullish' | 'bearish' | 'neutral';
}

interface CoinTradingState {
  symbol: string;
  balance: number;
  trades: PaperTrade[];
  openTrade: PaperTrade | null;
  stats: {
    totalTrades: number;
    wins: number;
    losses: number;
    totalPnl: number;
    winRate: number;
  };
  timeframes: Map<string, TimeframeData>;
  lastCheckTime: number;
  // Performance analytics
  performance: {
    byRegime: Record<string, { trades: number; wins: number; totalPnl: number }>;
    bySession: Record<string, { trades: number; wins: number; totalPnl: number }>;
    byDayOfWeek: Record<number, { trades: number; wins: number; totalPnl: number }>;
    recentEdge: number;  // Win rate last 20 trades
    historicalEdge: number;  // Win rate all trades
    edgeDecay: boolean;
  };
  lastCandleCloseTime: number;  // Track last 4H candle close time for trailing stop
}

// ═══════════════════════════════════════════════════════════════
// COIN TRADER CLASS
// ═══════════════════════════════════════════════════════════════

class CoinTrader {
  private static readonly POSITION_EPS = 1e-9;
  private static readonly lastBackupAtBySymbol: Map<string, number> = new Map();
  public state: CoinTradingState;
  private mlModel: TradingMLModel;
  public lgbmPredictor: LightGBMPredictor;
  public useLightGBM: boolean = false;

  constructor(symbol: typeof SYMBOLS[number]) {
    this.state = {
      symbol,
      balance: CONFIG.virtualBalancePerCoin,
      trades: [],
      openTrade: null,
      stats: {
        totalTrades: 0,
        wins: 0,
        losses: 0,
        totalPnl: 0,
        winRate: 0,
      },
      timeframes: new Map(),
      lastCheckTime: 0,
      performance: {
        byRegime: { TREND: { trades: 0, wins: 0, totalPnl: 0 }, RANGE: { trades: 0, wins: 0, totalPnl: 0 }, CHOP: { trades: 0, wins: 0, totalPnl: 0 } },
        bySession: { LONDON: { trades: 0, wins: 0, totalPnl: 0 }, NY_OPEN: { trades: 0, wins: 0, totalPnl: 0 }, NY_AFTERNOON: { trades: 0, wins: 0, totalPnl: 0 }, ASIA: { trades: 0, wins: 0, totalPnl: 0 }, OFF_HOURS: { trades: 0, wins: 0, totalPnl: 0 } },
        byDayOfWeek: { 0: { trades: 0, wins: 0, totalPnl: 0 }, 1: { trades: 0, wins: 0, totalPnl: 0 }, 2: { trades: 0, wins: 0, totalPnl: 0 }, 3: { trades: 0, wins: 0, totalPnl: 0 }, 4: { trades: 0, wins: 0, totalPnl: 0 }, 5: { trades: 0, wins: 0, totalPnl: 0 }, 6: { trades: 0, wins: 0, totalPnl: 0 } },
        recentEdge: 0,
        historicalEdge: 0,
        edgeDecay: false,
      },
      lastCandleCloseTime: 0,
    };
    this.mlModel = new TradingMLModel();
    this.lgbmPredictor = new LightGBMPredictor();
  }

  loadState(loadedStates: Map<string, any>): void {
    const tradesFile = path.join(CONFIG.tradesDir, `${this.state.symbol}.json`);
    if (fs.existsSync(tradesFile)) {
      try {
        const savedState = JSON.parse(fs.readFileSync(tradesFile, 'utf-8'));

        // Add performance tracking if loading from old state
        if (!savedState.performance) {
          savedState.performance = {
            byRegime: { TREND: { trades: 0, wins: 0, totalPnl: 0 }, RANGE: { trades: 0, wins: 0, totalPnl: 0 }, CHOP: { trades: 0, wins: 0, totalPnl: 0 } },
            bySession: { LONDON: { trades: 0, wins: 0, totalPnl: 0 }, NY_OPEN: { trades: 0, wins: 0, totalPnl: 0 }, NY_AFTERNOON: { trades: 0, wins: 0, totalPnl: 0 }, ASIA: { trades: 0, wins: 0, totalPnl: 0 }, OFF_HOURS: { trades: 0, wins: 0, totalPnl: 0 } },
            byDayOfWeek: { 0: { trades: 0, wins: 0, totalPnl: 0 }, 1: { trades: 0, wins: 0, totalPnl: 0 }, 2: { trades: 0, wins: 0, totalPnl: 0 }, 3: { trades: 0, wins: 0, totalPnl: 0 }, 4: { trades: 0, wins: 0, totalPnl: 0 }, 5: { trades: 0, wins: 0, totalPnl: 0 }, 6: { trades: 0, wins: 0, totalPnl: 0 } },
            recentEdge: 0,
            historicalEdge: 0,
            edgeDecay: false,
          };
          savedState.lastCandleCloseTime = savedState.lastCandleCloseTime || 0;
        }

        if (savedState.openTrade) {
          if (savedState.openTrade.originalStopLoss === undefined) {
            savedState.openTrade.originalStopLoss = savedState.openTrade.stopLoss;
          }
          if (savedState.openTrade.stopLossMovedToBreakeven === undefined) {
            savedState.openTrade.stopLossMovedToBreakeven = false;
          }
          if (savedState.openTrade.currentPositionSize === undefined) {
            savedState.openTrade.currentPositionSize = savedState.openTrade.originalPositionSize || savedState.openTrade.positionSize;
          }
          if (savedState.openTrade.originalPositionSize === undefined) {
            savedState.openTrade.originalPositionSize = savedState.openTrade.currentPositionSize || savedState.openTrade.positionSize;
          }
          if (savedState.openTrade.trailingStop === undefined) {
            savedState.openTrade.trailingStop = null;
          }
          if (savedState.openTrade.regime === undefined) {
            savedState.openTrade.regime = 'RANGE';
          }
          if (savedState.openTrade.qualityScore === undefined) {
            savedState.openTrade.qualityScore = 50;
          }
          if (savedState.openTrade.smcContext === undefined) {
            savedState.openTrade.smcContext = [];
          }

          const trade = savedState.openTrade;
          if (trade.takeProfit1 === undefined || trade.takeProfit2 === undefined || trade.takeProfit3 === undefined) {
            const stopDistance = Math.abs(trade.entryPrice - trade.stopLoss);
            trade.takeProfit1 = trade.direction === 'LONG'
              ? trade.entryPrice + stopDistance * 1.5
              : trade.entryPrice - stopDistance * 1.5;
            trade.takeProfit2 = trade.direction === 'LONG'
              ? trade.entryPrice + stopDistance * 3.0
              : trade.entryPrice - stopDistance * 3.0;
            trade.takeProfit3 = trade.direction === 'LONG'
              ? trade.entryPrice + stopDistance * 4.5
              : trade.entryPrice - stopDistance * 4.5;
          }
        }

        this.state = {
          ...this.state,
          ...savedState,
          timeframes: new Map(),
        };
        console.log(`  ${this.state.symbol}: Loaded ${this.state.stats.totalTrades} trades, Balance: $${this.state.balance.toFixed(2)}`);
      } catch (e) {
        console.log(`  ${this.state.symbol}: Could not load state, starting fresh`);
      }
    }
  }

  // Update performance analytics after a trade closes
  private updatePerformanceAnalytics(trade: PaperTrade): void {
    if (trade.status !== 'CLOSED' || trade.pnl === undefined) return;

    const pnl = trade.pnl;
    const isWin = pnl > 0;
    const regime = trade.regime || 'RANGE';
    const session = trade.entryFeatures?.killZone || 'OFF_HOURS';
    const dayOfWeek = new Date(trade.entryTime).getDay();

    // Update by regime
    if (!this.state.performance.byRegime[regime]) {
      this.state.performance.byRegime[regime] = { trades: 0, wins: 0, totalPnl: 0 };
    }
    this.state.performance.byRegime[regime].trades++;
    if (isWin) this.state.performance.byRegime[regime].wins++;
    this.state.performance.byRegime[regime].totalPnl += pnl;

    // Update by session
    if (!this.state.performance.bySession[session]) {
      this.state.performance.bySession[session] = { trades: 0, wins: 0, totalPnl: 0 };
    }
    this.state.performance.bySession[session].trades++;
    if (isWin) this.state.performance.bySession[session].wins++;
    this.state.performance.bySession[session].totalPnl += pnl;

    // Update by day of week
    if (!this.state.performance.byDayOfWeek[dayOfWeek]) {
      this.state.performance.byDayOfWeek[dayOfWeek] = { trades: 0, wins: 0, totalPnl: 0 };
    }
    this.state.performance.byDayOfWeek[dayOfWeek].trades++;
    if (isWin) this.state.performance.byDayOfWeek[dayOfWeek].wins++;
    this.state.performance.byDayOfWeek[dayOfWeek].totalPnl += pnl;

    // Calculate edge decay
    const closedTrades = this.state.trades.filter(t => t.status === 'CLOSED');
    this.state.performance.historicalEdge = this.state.stats.winRate;

    const recentTrades = closedTrades.slice(-20);
    if (recentTrades.length >= 10) {
      const recentWins = recentTrades.filter(t => (t.pnl || 0) > 0).length;
      this.state.performance.recentEdge = recentWins / recentTrades.length;
      this.state.performance.edgeDecay = this.state.performance.recentEdge < this.state.performance.historicalEdge * 0.7;
    }
  }

  saveState(): void {
    if (!fs.existsSync(CONFIG.tradesDir)) {
      fs.mkdirSync(CONFIG.tradesDir, { recursive: true });
    }

    const tradesFile = path.join(CONFIG.tradesDir, `${this.state.symbol}.json`);

    if (CONFIG.enableStateBackups && fs.existsSync(tradesFile)) {
      try {
        const now = Date.now();
        const lastBackupAt = CoinTrader.lastBackupAtBySymbol.get(this.state.symbol) ?? 0;
        if ((now - lastBackupAt) >= CONFIG.backupIntervalMs) {
          const backupDir = path.join(CONFIG.tradesDir, 'backups');
          if (!fs.existsSync(backupDir)) {
            fs.mkdirSync(backupDir, { recursive: true });
          }

          const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
          const backupFile = path.join(backupDir, `${this.state.symbol}-${timestamp}.json`);
          fs.copyFileSync(tradesFile, backupFile);
          CoinTrader.lastBackupAtBySymbol.set(this.state.symbol, now);

          const prefix = `${this.state.symbol}-`;
          const backups = fs
            .readdirSync(backupDir)
            .filter(f => f.startsWith(prefix) && f.endsWith('.json'))
            .sort()
            .reverse();
          for (const old of backups.slice(CONFIG.maxBackupsPerSymbol)) {
            try {
              fs.unlinkSync(path.join(backupDir, old));
            } catch {
              // ignore
            }
          }
        }
      } catch (e) {
        console.error(`Backup failed for ${this.state.symbol}:`, e);
      }
    }

    fs.writeFileSync(tradesFile, JSON.stringify(this.state, null, 2));
  }

  resetState(): void {
    // Backup current state before reset
    const tradesFile = path.join(CONFIG.tradesDir, `${this.state.symbol}.json`);
    if (fs.existsSync(tradesFile)) {
      const backupFile = tradesFile.replace('.json', '-backup.json');
      fs.copyFileSync(tradesFile, backupFile);
      console.log(`  ${this.state.symbol}: State backed up to ${path.basename(backupFile)}`);
    }

    // Initialize new timeframes Map with all intervals
    const newTimeframes = new Map();
    for (const interval of CONFIG.intervals) {
      newTimeframes.set(interval, {
        candles: [],
        momentum: analyzeMomentum([]),
        lastUpdate: 0,
      });
    }

    // Reset to initial state (preserve virtualBalancePerCoin from CONFIG)
    const initialBalance = CONFIG.virtualBalancePerCoin;
    this.state = {
      symbol: this.state.symbol,
      balance: initialBalance,
      openTrade: null,
      trades: [],
      stats: {
        totalTrades: 0,
        wins: 0,
        losses: 0,
        totalPnl: 0,
        winRate: 0,
      },
      timeframes: newTimeframes,
      lastCheckTime: 0,
      performance: {
        byRegime: {},
        bySession: {},
        byDayOfWeek: {},
        recentEdge: 0,
        historicalEdge: 0,
        edgeDecay: false,
      },
      lastCandleCloseTime: 0,
    };
    this.saveState();
    console.log(`  ${this.state.symbol}: State reset to $${initialBalance} balance`);
  }

  async initialize(client: any, modelWeights: any): Promise<void> {
    this.useLightGBM = this.lgbmPredictor.load();

    if (!this.useLightGBM) {
      if (modelWeights) {
        this.mlModel.importWeights(modelWeights);
      }
    }

    await this.fetchAllTimeframes(client);
  }

  async fetchAllTimeframes(client: any): Promise<void> {
    for (const interval of CONFIG.intervals) {
      await this.fetchCandles(client, interval);
    }
  }

  async fetchCandles(client: any, interval: string): Promise<void> {
    try {
      const klines = await client.candles({
        symbol: this.state.symbol,
        interval,
        limit: 500,
      });

      const candles = klines.map((k: any) => ({
        timestamp: k.openTime,
        open: parseFloat(k.open),
        high: parseFloat(k.high),
        low: parseFloat(k.low),
        close: parseFloat(k.close),
        volume: parseFloat(k.volume),
      }));

      // Compute momentum signals on candles
      const momentum = analyzeMomentum(candles);

      let tfData = this.state.timeframes.get(interval);
      if (!tfData) {
        tfData = {
          interval,
          candles: [],
          momentum,
          smcAnalysis: null,
          ictAnalysis: null,
          bias: 'neutral',
        };
      }

      tfData.candles = candles;
      tfData.momentum = momentum;

      // Track last candle close time for trailing stop updates
      if (candles.length > 0) {
        const latestCandle = candles[candles.length - 1];
        this.state.lastCandleCloseTime = latestCandle.timestamp;
      }

      // Derive bias from momentum direction
      if (momentum.direction === 'LONG') {
        tfData.bias = 'bullish';
      } else if (momentum.direction === 'SHORT') {
        tfData.bias = 'bearish';
      } else {
        tfData.bias = 'neutral';
      }

      // SMC/ICT analysis is lazy - only computed when entry signal detected (in analyzeForEntry)
      this.state.timeframes.set(interval, tfData);
    } catch (error: any) {
      console.error(`Error fetching candles for ${this.state.symbol} ${interval}:`, error.message);
    }
  }

  async tick(client: any, timestamp: string): Promise<{ status: string; details: string; regime: string }> {
    if (this.useLightGBM) {
      const updated = this.lgbmPredictor.checkForUpdates();
      if (updated) {
        console.log(`  ${this.state.symbol}: LightGBM model reloaded`);
      }
    }

    await this.fetchAllTimeframes(client);

    const primaryTf = this.state.timeframes.get(CONFIG.primaryInterval);
    if (!primaryTf || primaryTf.candles.length === 0) {
      return { status: 'ERROR', details: 'No candles', regime: '?' };
    }

    const currentPrice = primaryTf.candles[primaryTf.candles.length - 1].close;
    const regime = primaryTf.momentum.regime;

    if (this.state.openTrade) {
      const result = await this.checkOpenTrade(currentPrice);
      return {
        status: result.closed ? 'CLOSED' : 'OPEN',
        details: `${result.message} | P&L: ${result.pnlSign}$${result.pnl.toFixed(2)} (${result.pnlSign}${result.pnlPercent.toFixed(2)}%)`,
        regime,
      };
    }

    const analysis = this.analyzeForEntry();

    if (analysis.shouldEnter) {
      await this.enterTrade(analysis, currentPrice);
      return { status: 'ENTERED', details: analysis.reason, regime };
    }

    return { status: 'MONITOR', details: analysis.reason, regime };
  }

  // ═══════════════════════════════════════════════════════════════
  // REGIME-BASED ENTRY ANALYSIS (replaces ICT confluence scoring)
  // ═══════════════════════════════════════════════════════════════

  private analyzeForEntry(): {
    shouldEnter: boolean;
    direction: 'LONG' | 'SHORT' | 'NEUTRAL';
    regime: 'TREND' | 'RANGE' | 'CHOP';
    mlPrediction: number;
    qualityScore: number;
    smcContext: SMCContext;
    signals: string[];
    reason: string;
  } {
    const noEntry = (reason: string, direction: 'LONG' | 'SHORT' | 'NEUTRAL' = 'NEUTRAL', regime: 'TREND' | 'RANGE' | 'CHOP' = 'CHOP', mlPrediction: number = 0.5, qualityScore: number = 0, smcContext: any = { stopTighten: 1, sizeMultiplier: 1, confidenceBoost: 0, adjustments: [] }) => ({
      shouldEnter: false,
      direction,
      regime,
      mlPrediction,
      qualityScore,
      smcContext,
      signals: [] as string[],
      reason,
    });

    const tf4h = this.state.timeframes.get('4h');
    if (!tf4h || tf4h.candles.length < CONFIG.minCandlesRequired) {
      return noEntry('Insufficient 4h data');
    }

    const m4h = tf4h.momentum;
    const regime = m4h.regime;
    let direction = m4h.direction;
    const signals: string[] = [];

    // ═══════════════════════════════════════════════════════════════
    // WEEKLY TREND ALIGNMENT CHECK
    // ═══════════════════════════════════════════════════════════════
    const tfWeekly = this.state.timeframes.get('1w');
    let weeklyTrend: 'UP' | 'DOWN' | 'SIDE' = 'SIDE';
    let weeklyAlignmentScore = 0;  // 0 = aligned, 1 = opposite

    if (tfWeekly && tfWeekly.candles.length >= 20) {
      // Simple weekly trend: compare current close to SMA(20) of weekly candles
      const weeklyCloses = tfWeekly.candles.map(c => c.close);
      const weeklySMA20 = weeklyCloses.slice(-20).reduce((a, b) => a + b, 0) / 20;
      const weeklyClose = weeklyCloses[weeklyCloses.length - 1];

      if (weeklyClose > weeklySMA20 * 1.02) {  // 2% above SMA = clear uptrend
        weeklyTrend = 'UP';
      } else if (weeklyClose < weeklySMA20 * 0.98) {  // 2% below SMA = clear downtrend
        weeklyTrend = 'DOWN';
      } else {
        weeklyTrend = 'SIDE';
      }

      // Calculate alignment: 1 = aligned, 0 = neutral, -1 = opposite
      if (weeklyTrend === 'UP' && direction === 'LONG') {
        weeklyAlignmentScore = 1;
      } else if (weeklyTrend === 'DOWN' && direction === 'SHORT') {
        weeklyAlignmentScore = 1;
      } else if (weeklyTrend === 'SIDE') {
        weeklyAlignmentScore = 0.5;  // Neutral when weekly is ranging
      } else {
        weeklyAlignmentScore = 0;  // Opposing weekly trend
      }

      signals.push(`W:${weeklyTrend}`);
    } else {
      // No weekly data available
      weeklyAlignmentScore = 0.5;  // Neutral default
    }

    // ═══════════════════════════════════════════════════════════════
    // QUANT-LEVEL: Weekly trend is ADVISORY, not a hard gate
    // - Opposing weekly trend → smaller position, tighter stop (risk management)
    // - Aligned weekly trend → confidence boost
    // - Never block trades purely on weekly alignment (miss too many opportunities)
    // ═══════════════════════════════════════════════════════════════

    // CHOP: Don't trade (no edge in dead markets)
    if (regime === 'CHOP') {
      return noEntry(`CHOP: ATR ${(m4h.atrPercent * 100).toFixed(2)}% < ${(CONFIG.regime.minVolatility * 100).toFixed(1)}% - skip`, direction, 'CHOP');
    }

    // TREND mode: Allow NEUTRAL direction to pass through to entry logic
    // Swing trades can work even without perfect 4H alignment
    // RANGE mode derives direction from BB position (handled below)
    signals.push(`ATR:${(m4h.atrPercent * 100).toFixed(2)}%`);
    signals.push(`BB:${(m4h.bbPosition * 100).toFixed(0)}%`);

    signals.push(`ATR:${(m4h.atrPercent * 100).toFixed(2)}%`);
    signals.push(`BB:${(m4h.bbPosition * 100).toFixed(0)}%`);

    if (regime === 'TREND') {
      // ═══════════════════════════════════════════════════════════════
      // TREND MODE: EMA/breakout entry + VWAP (MACD as filter, not gate)
      // ═══════════════════════════════════════════════════════════════

      const hasEmaAlign = direction === 'LONG'
        ? m4h.emaAligned === 'bullish'
        : m4h.emaAligned === 'bearish';

      const hasBreakout = direction === 'LONG'
        ? (m4h.priceBreakoutUp || m4h.bbBreakoutUp)
        : (m4h.priceBreakoutDown || m4h.bbBreakoutDown);

      if (!hasEmaAlign && !hasBreakout) {
        return { ...noEntry(`TREND: Need EMA align or breakout`, direction, 'TREND'), signals };
      }
      signals.push(hasEmaAlign ? 'EMA' : 'BRK');

      // VWAP confirmation (soft - not a hard gate for swing trading)
      const hasVwapConfirm = direction === 'LONG' ? m4h.priceAboveVwap : !m4h.priceAboveVwap;
      if (hasVwapConfirm) signals.push('VWAP✓');
      // No rejection for wrong VWAP side - swing trades can work against VWAP

      // MACD filter (not gate): only block if strongly against direction (>0.5%)
      const macdAgainst = direction === 'LONG'
        ? -m4h.macdHistogram // Negative histogram = bearish pressure
        : m4h.macdHistogram;  // Positive histogram = bullish pressure
      const macdThreshold = 0.005; // 0.5% threshold

      if (macdAgainst > macdThreshold) {
        return { ...noEntry(`TREND: MACD strongly against (${(macdAgainst * 100).toFixed(2)}%)`, direction, 'TREND'), signals };
      }
      if (Math.abs(m4h.macdHistogram) > 0.0001) {
        signals.push(`MACD:${(m4h.macdHistogram * 100).toFixed(3)}%`);
      }

    } else {
      // ═══════════════════════════════════════════════════════════════
      // RANGE MODE: Mean reversion at BB extremes
      // Direction OVERRIDDEN by BB position (mean reversion logic):
      //   BB < 25% → LONG (buy oversold), BB > 75% → SHORT (sell overbought)
      // Volume spike is BONUS (confidence boost), NOT a requirement
      // ═══════════════════════════════════════════════════════════════

      let rangeDirection: 'LONG' | 'SHORT' | 'SKIP' = 'SKIP';
      if (m4h.bbPosition < CONFIG.momentum.bbExtremeLong) {
        rangeDirection = 'LONG';
      } else if (m4h.bbPosition > CONFIG.momentum.bbExtremeShort) {
        rangeDirection = 'SHORT';
      }

      if (rangeDirection === 'SKIP') {
        return { ...noEntry(`RANGE: BB in middle (${(m4h.bbPosition * 100).toFixed(0)}%) - need <${(CONFIG.momentum.bbExtremeLong * 100).toFixed(0)}% or >${(CONFIG.momentum.bbExtremeShort * 100).toFixed(0)}%`, direction, 'RANGE'), signals };
      }

      // Override direction for mean reversion
      direction = rangeDirection;

      signals.push(`BB-extreme:${(m4h.bbPosition * 100).toFixed(0)}%`);

      // Volume spike is OPTIONAL - bonus if present, not a gate
      if (m4h.volumeSpike) {
        signals.push(`VOL:${m4h.volumeRatio.toFixed(1)}x`);
      }
    }

    // Entry signal passed - now compute SMC context (lazy)
    if (!tf4h.smcAnalysis && tf4h.candles.length >= CONFIG.minCandlesRequired) {
      tf4h.smcAnalysis = SMCIndicators.analyze(tf4h.candles);
      tf4h.ictAnalysis = ICTIndicators.analyzeFast(tf4h.candles, tf4h.smcAnalysis);
    }

    const smcContext = applySMCContext(direction as 'LONG' | 'SHORT', tf4h.smcAnalysis, tf4h.ictAnalysis, m4h.isKillZone, weeklyAlignmentScore);

    // ═══════════════════════════════════════════════════════════════
    // QUANT-LEVEL: Multi-Timeframe (MTF) alignment position sizing
    // Daily alignment affects position size, not entry decision
    // ═══════════════════════════════════════════════════════════════
    const dailyTf = this.state.timeframes.get('1d');
    const dailyAligns = dailyTf && dailyTf.momentum.direction === direction;

    // Apply MTF position sizing adjustment
    if (!dailyAligns && dailyTf) {
      // Daily doesn't align → smaller position (50%)
      smcContext.sizeMultiplier *= 0.5;
      smcContext.adjustments.push(`MTF-4h/1d-mismatch: -50%size`);
    } else if (dailyAligns) {
      // Daily aligns → confidence boost
      smcContext.confidenceBoost += 3;
      signals.push('1d');
    }

    // Calculate quality score (will be updated with weekly alignment later)
    const qualityScore = this.calculateTradeQuality(m4h, weeklyAlignmentScore, smcContext);

    // ML prediction (optional)
    let mlPrediction = 0.5;
    if (tf4h.smcAnalysis) {
      const currentPrice = tf4h.candles[tf4h.candles.length - 1].close;
      const scoring = UnifiedScoring.calculateConfluence(tf4h.smcAnalysis, currentPrice, {
        trend_structure: 40, order_blocks: 30, fvgs: 20,
        ema_alignment: 15, liquidity: 10, mtf_bonus: 35, rsi_penalty: 15,
      });
      const smcScore = scoring.score;

      try {
        const features = FeatureExtractor.extractFeatures(
          tf4h.candles,
          tf4h.candles.length - 1,
          tf4h.smcAnalysis,
          smcScore,
          direction === 'LONG' ? 'long' : 'short',
          tf4h.ictAnalysis
        );

        const prediction = this.useLightGBM
          ? this.lgbmPredictor.predict(features as TradeFeatures)
          : this.mlModel.predict(features as TradeFeatures);
        mlPrediction = prediction.winProbability;
      } catch {
        // ML prediction failed, proceed with default 0.5
      }
    }

    // ML filter (bypass if threshold = 0)
    if (CONFIG.minWinProbability > 0 && mlPrediction < CONFIG.minWinProbability) {
      return {
        shouldEnter: false,
        direction,
        regime,
        mlPrediction,
        qualityScore,
        smcContext,
        signals,
        reason: `ML reject: ${(mlPrediction * 100).toFixed(0)}% < ${(CONFIG.minWinProbability * 100).toFixed(0)}%`,
      };
    }

    if (m4h.isKillZone) signals.push(`KZ:${m4h.killZone}`);
    if (m4h.volumeSpike && regime === 'TREND') signals.push(`VOL:${m4h.volumeRatio.toFixed(1)}x`);

    // Pre-validate stop distance (avoid ghost signals for low-price coins)
    const currentPrice = tf4h.candles[tf4h.candles.length - 1].close;
    const mDaily = dailyTf?.momentum;
    const isLong = direction === 'LONG';

    // Check if we can find a reasonable stop (ATR or swing point)
    const swingHighForStop = (mDaily?.swingHigh && mDaily.swingHigh > currentPrice)
      ? mDaily.swingHigh
      : (m4h.swingHigh && m4h.swingHigh > currentPrice) ? m4h.swingHigh : null;
    const swingLowForStop = (mDaily?.swingLow && mDaily.swingLow < currentPrice)
      ? mDaily.swingLow
      : (m4h.swingLow && m4h.swingLow < currentPrice) ? m4h.swingLow : null;

    let estimatedStopDistance: number;
    if (isLong && swingLowForStop) {
      estimatedStopDistance = currentPrice - (swingLowForStop * 0.998);
    } else if (!isLong && swingHighForStop) {
      estimatedStopDistance = (swingHighForStop * 1.002) - currentPrice;
    } else {
      // ATR-based stop, capped to 5% max for low-price coins
      const atrStopDistance = m4h.atr * CONFIG.stopLossATRMultiple;
      const atrRiskPct = (atrStopDistance / currentPrice) * 100;
      const maxAtrStopPct = 0.05; // 5% max for ATR-based stops
      estimatedStopDistance = atrRiskPct <= maxAtrStopPct * 100 ? atrStopDistance : currentPrice * maxAtrStopPct;
    }

    const estimatedRiskPct = (estimatedStopDistance / currentPrice) * 100;
    // For swing trading, allow wider stops (up to 15% risk) - swing needs room to breathe
    if (estimatedRiskPct > 15) {
      return { ...noEntry(`Stop too far (${estimatedRiskPct.toFixed(1)}% risk)`, direction, regime), signals, mlPrediction, qualityScore, smcContext };
    }
    if (estimatedRiskPct < 0.3) {
      return { ...noEntry(`Stop too tight (${estimatedRiskPct.toFixed(2)}% risk)`, direction, regime), signals, mlPrediction, qualityScore, smcContext };
    }

    const kzBonus = m4h.isKillZone ? '+KZ' : '';
    const reason = `${regime} ${direction} (${signals.join('+')})${kzBonus}`;

    return {
      shouldEnter: true,
      direction,
      regime,
      mlPrediction,
      qualityScore,
      smcContext,
      signals,
      reason,
    };
  }

  // ═══════════════════════════════════════════════════════════════
  // TRADE QUALITY SCORING (regime-based + MTF alignment)
  // ═══════════════════════════════════════════════════════════════

  private calculateTradeQuality(m: MomentumSignals, weeklyAlignmentScore: number, smcCtx: SMCContext): number {
    let score = 50;  // Base

    // +15 for volume spike
    if (m.volumeSpike) score += 15;

    // +10 for strong BB position or high strength
    if (m.bbPosition < 0.15 || m.bbPosition > 0.85 || m.strength > 0.7) score += 10;

    // +5 for VWAP alignment
    const vwapAligned = m.direction === 'LONG' ? m.priceAboveVwap : !m.priceAboveVwap;
    if (vwapAligned) score += 5;

    // +5 for kill zone
    if (m.isKillZone) score += 5;

    // +10 for daily alignment
    const dailyTf = this.state.timeframes.get('1d');
    if (dailyTf && dailyTf.momentum.direction === m.direction) score += 10;

    // +10 for weekly alignment (new!)
    if (weeklyAlignmentScore >= 0.8) score += 10;
    else if (weeklyAlignmentScore >= 0.5) score += 5;
    // Penalty for opposing weekly trend (shouldn't happen since we filter above)
    else if (weeklyAlignmentScore === 0) score -= 10;

    // +0-5 for ML prediction (scaled: 0.5 = 0, 1.0 = 5)
    // ML gets minimal weight since threshold is 0 for data collection

    // +0-20 from SMC context confidence boost
    score += Math.min(20, smcCtx.confidenceBoost);

    return Math.min(100, Math.max(0, score));
  }

  // ═══════════════════════════════════════════════════════════════
  // ENTER TRADE
  // ═══════════════════════════════════════════════════════════════

  private async enterTrade(analysis: ReturnType<CoinTrader['analyzeForEntry']>, currentPrice: number): Promise<void> {
    const primaryTf = this.state.timeframes.get(CONFIG.primaryInterval)!;
    const dailyTf = this.state.timeframes.get('1d');
    const m = primaryTf.momentum;
    const mDaily = dailyTf?.momentum;
    const isLong = analysis.direction === 'LONG';

    // Structure-based stops: prefer 1D swing points, fallback to 4H
    // Clamp swing stops to max 8% — if further, fall back to ATR
    const maxSwingStopPct = 0.08; // 8% max for swing point stops

    const swingHighForStop = (mDaily?.swingHigh && mDaily.swingHigh > currentPrice)
      ? mDaily.swingHigh
      : (m.swingHigh && m.swingHigh > currentPrice) ? m.swingHigh : null;

    const swingLowForStop = (mDaily?.swingLow && mDaily.swingLow < currentPrice)
      ? mDaily.swingLow
      : (m.swingLow && m.swingLow < currentPrice) ? m.swingLow : null;

    // Check if swing point stop is within acceptable range
    let useSwingStop = false;
    let stopLoss: number;
    let stopDistance: number;
    let stopType: string = 'ATR';

    // Default to ATR-based stop (will be overridden if swing point is used)
    // Cap ATR stop to max 5% for low-price coins
    const atrStopDistance = m.atr * CONFIG.stopLossATRMultiple;
    const atrRiskPct = (atrStopDistance / currentPrice) * 100;
    const maxAtrStopPct = 0.05; // 5% max for ATR-based stops

    if (atrRiskPct <= maxAtrStopPct * 100) {
      stopDistance = atrStopDistance;
    } else {
      stopDistance = currentPrice * maxAtrStopPct;
    }
    stopLoss = isLong ? currentPrice - stopDistance : currentPrice + stopDistance;

    if (isLong && swingLowForStop) {
      const potentialStop = swingLowForStop * 0.998;
      const potentialDistance = currentPrice - potentialStop;
      const potentialRiskPct = (potentialDistance / currentPrice) * 100;
      if (potentialRiskPct <= maxSwingStopPct * 100) {
        stopLoss = potentialStop;
        stopDistance = potentialDistance;
        stopType = mDaily?.swingLow ? 'SWING_1D' : 'SWING_4H';
        useSwingStop = true;
      }
    } else if (!isLong && swingHighForStop) {
      const potentialStop = swingHighForStop * 1.002;
      const potentialDistance = potentialStop - currentPrice;
      const potentialRiskPct = (potentialDistance / currentPrice) * 100;
      if (potentialRiskPct <= maxSwingStopPct * 100) {
        stopLoss = potentialStop;
        stopDistance = potentialDistance;
        stopType = mDaily?.swingHigh ? 'SWING_1D' : 'SWING_4H';
        useSwingStop = true;
      }
    }

    // Fall back to ATR if swing stop is too far or unavailable
    if (!useSwingStop) {
      stopDistance = m.atr * CONFIG.stopLossATRMultiple;
      stopLoss = isLong ? currentPrice - stopDistance : currentPrice + stopDistance;
    }

    // Apply SMC context stop tightening
    stopDistance *= analysis.smcContext.stopTighten;
    stopLoss = isLong ? currentPrice - stopDistance : currentPrice + stopDistance;

    // Cap risk - 10% max for 4H swing trades (position sizing controls actual $ risk)
    const riskPct = (stopDistance / currentPrice) * 100;
    if (riskPct > 10) {
      console.log(`  ${this.state.symbol}: Skip - stop too far (${riskPct.toFixed(1)}% risk)`);
      return;
    }
    if (riskPct < 0.3) {
      console.log(`  ${this.state.symbol}: Skip - stop too tight (${riskPct.toFixed(2)}% risk)`);
      return;
    }

    // ═══════════════════════════════════════════════════════════════
    // KELLY CRITERION POSITION SIZING (with real trade statistics)
    // f* = (bp - q) / b where b = R:R, p = win prob, q = 1-p
    // ═══════════════════════════════════════════════════════════════

    // Calculate real statistics from recent trades
    const recentTrades = this.state.trades.slice(-50);  // Last 50 trades
    const closedTrades = recentTrades.filter(t => t.status === 'CLOSED' && t.pnl !== undefined);
    const wins = closedTrades.filter(t => (t.pnl || 0) > 0);
    const losses = closedTrades.filter(t => (t.pnl || 0) <= 0);

    // Calculate win rate (p) and payoff ratio (b)
    let winRate: number;
    let payoffRatio: number;

    if (closedTrades.length >= 20) {
      // Use real statistics
      winRate = wins.length / closedTrades.length;

      const avgWin = wins.length > 0
        ? wins.reduce((sum, t) => sum + (t.pnl || 0), 0) / wins.length
        : 0;
      const avgLoss = losses.length > 0
        ? Math.abs(losses.reduce((sum, t) => sum + (t.pnl || 0), 0)) / losses.length
        : 1; // Avoid division by zero

      payoffRatio = avgLoss > 0 ? avgWin / avgLoss : 1;
    } else {
      // Conservative fallback with not enough data
      winRate = 0.40;  // 40% assumed win rate
      payoffRatio = 1.5;  // 1.5:1 reward-risk
    }

    // Kelly fraction: f = (bp - q) / b
    const p = winRate;
    const q = 1 - p;
    const b = payoffRatio;

    // Kelly calculation (can be negative if edge is negative)
    let kellyFraction = (b * p - q) / b;

    // Safety: don't over-bet, use quarter-Kelly for better risk-adjusted returns
    kellyFraction = Math.max(0, Math.min(0.15, kellyFraction * 0.25));  // Cap at 15% max (quarter-Kelly)

    // Convert Kelly fraction to risk % (5% base when Kelly = 10%)
    let kellyRiskPct = 5.0 * (kellyFraction / 0.10);
    kellyRiskPct = Math.max(0.5, Math.min(5.0, kellyRiskPct));  // Cap at 0.5%-5% risk

    // Quality-based risk multiplier
    const qualityMultiplier = 0.5 + (analysis.qualityScore / 100) * 0.5;
    const qualityRiskPct = 5.0 * qualityMultiplier;

    // Use the more conservative of Kelly and quality
    let adjustedRiskPct = Math.min(kellyRiskPct, qualityRiskPct);

    // Apply SMC context size boost
    adjustedRiskPct *= analysis.smcContext.sizeMultiplier;

    const riskAmount = this.state.balance * (adjustedRiskPct / 100);
    const positionSize = riskAmount / stopDistance;

    // Dynamic leverage based on quality score: 1x (low) to 3x (high)
    const dynamicLeverage = 1 + Math.floor((analysis.qualityScore / 100) * 3);  // 0-33=1x, 34-66=2x, 67-100=3x
    const maxNotional = this.state.balance * dynamicLeverage;
    const notional = currentPrice * positionSize;
    const cappedPositionSize = notional > maxNotional ? maxNotional / currentPrice : positionSize;

    // Regime-specific TP levels
    let tp1R: number, tp2R: number, tp3R: number;
    if (analysis.regime === 'TREND') {
      tp1R = 1.5; tp2R = 3.0; tp3R = 4.5;  // Let trends run
    } else {
      tp1R = 1.5; tp2R = 2.0; tp3R = 3.5;  // Mean reversion has ceilings
    }

    let rawTp1 = isLong ? currentPrice + stopDistance * tp1R : currentPrice - stopDistance * tp1R;
    const rawTp2 = isLong ? currentPrice + stopDistance * tp2R : currentPrice - stopDistance * tp2R;
    const rawTp3 = isLong ? currentPrice + stopDistance * tp3R : currentPrice - stopDistance * tp3R;

    // Structure-aware TP1: snap to nearby round numbers or structural levels
    const findNearestRound = (price: number, dir: 'above' | 'below'): number => {
      let interval: number;
      if (price > 10000) interval = 500;
      else if (price > 1000) interval = 100;
      else if (price > 100) interval = 10;
      else if (price > 10) interval = 1;
      else if (price > 1) interval = 0.1;
      else interval = 0.01;
      return dir === 'below' ? Math.floor(price / interval) * interval : Math.ceil(price / interval) * interval;
    };

    const snapRange = stopDistance * 0.5;
    const minTp1Dist = stopDistance * 1.0;  // At least 1R for swing
    const roundTarget = findNearestRound(rawTp1, isLong ? 'above' : 'below');

    const structLow = mDaily?.swingLow || m.swingLow;
    const structHigh = mDaily?.swingHigh || m.swingHigh;
    const structTarget = isLong
      ? (structHigh && structHigh > currentPrice + minTp1Dist ? structHigh : null)
      : (structLow && structLow < currentPrice - minTp1Dist ? structLow : null);

    if (Math.abs(roundTarget - rawTp1) <= snapRange && Math.abs(roundTarget - currentPrice) >= minTp1Dist) {
      rawTp1 = roundTarget;
    } else if (structTarget && Math.abs(structTarget - rawTp1) <= snapRange) {
      rawTp1 = structTarget;
    }

    const takeProfit1 = rawTp1;
    const takeProfit2 = rawTp2;
    const takeProfit3 = rawTp3;

    // Calculate entry fee (will be deducted from PnL at exit, NOT from balance now)
    const entryNotional = currentPrice * cappedPositionSize;
    const entryFee = entryNotional * CONFIG.takerFeeRate;

    const trade: PaperTrade = {
      id: `${this.state.symbol}-${Date.now()}`,
      symbol: this.state.symbol,
      direction: analysis.direction as 'LONG' | 'SHORT',
      entryPrice: currentPrice,
      entryTime: Date.now(),
      stopLoss,
      originalStopLoss: stopLoss,
      takeProfit1,
      takeProfit2,
      takeProfit3,
      trailingStop: null,
      originalPositionSize: cappedPositionSize,
      currentPositionSize: cappedPositionSize,
      tp1Hit: false,
      tp2Hit: false,
      tp3Hit: false,
      stopLossMovedToBreakeven: false,
      status: 'OPEN',
      mlPrediction: analysis.mlPrediction,
      regime: analysis.regime,
      qualityScore: analysis.qualityScore,
      smcContext: analysis.smcContext.adjustments,
      entryFee,  // Track entry fee for proper PnL accounting
    };

    // Save market snapshot at entry for ML training
    trade.entryFeatures = captureSnapshot(m, currentPrice);

    this.state.openTrade = trade;
    this.state.trades.push(trade);
    this.saveState();

    console.log(`\n${analysis.regime === 'TREND' ? '🔥' : '📊'} ${this.state.symbol}: ENTERED ${trade.direction} [${analysis.regime} | ${stopType} STOP]`);
    console.log(`   Entry: $${trade.entryPrice.toFixed(2)} | SL: $${trade.stopLoss.toFixed(2)} (${riskPct.toFixed(1)}% risk)`);
    console.log(`   TP1: $${trade.takeProfit1.toFixed(2)} (${tp1R}R) | TP2: $${trade.takeProfit2.toFixed(2)} (${tp2R}R) | TP3: $${trade.takeProfit3.toFixed(2)} (${tp3R}R)`);
    console.log(`   Quality: ${analysis.qualityScore}/100 | ML: ${(analysis.mlPrediction * 100).toFixed(0)}% | Signals: ${analysis.signals.join('+')}`);
    if (analysis.smcContext.adjustments.length > 0) {
      console.log(`   SMC Context: ${analysis.smcContext.adjustments.join(' | ')}`);
    }
    console.log(`   Risk: ${adjustedRiskPct.toFixed(1)}% ($${riskAmount.toFixed(2)}) | Size: ${trade.currentPositionSize.toFixed(6)}\n`);
  }

  // ═══════════════════════════════════════════════════════════════
  // CHECK OPEN TRADE (with trailing stop)
  // ═══════════════════════════════════════════════════════════════

  private async checkOpenTrade(currentPrice: number): Promise<{
    closed: boolean;
    message: string;
    pnl: number;
    pnlPercent: number;
    pnlSign: string;
  }> {
    const trade = this.state.openTrade!;
    const isLong = trade.direction === 'LONG';

    const priceDiff = isLong
      ? currentPrice - trade.entryPrice
      : trade.entryPrice - currentPrice;
    const unrealizedPnl = priceDiff * trade.currentPositionSize;
    const pnlPercent = (priceDiff / trade.entryPrice) * 100;

    if (trade.currentPositionSize <= CoinTrader.POSITION_EPS) {
      trade.currentPositionSize = 0;
      trade.exitPrice = currentPrice;
      trade.exitTime = Date.now();
      trade.exitReason = trade.tp3Hit ? 'TP3' : trade.tp2Hit ? 'TP2' : trade.tp1Hit ? 'TP1' : 'MANUAL';
      trade.pnl = 0;
      trade.pnlPercent = 0;
      trade.status = 'CLOSED';
      this.state.openTrade = null;
      this.saveState();
      return {
        closed: true,
        message: `CLOSED (flat) ${trade.exitReason}`,
        pnl: 0, pnlPercent: 0, pnlSign: '',
      };
    }

    // Check stop loss
    if ((isLong && currentPrice <= trade.stopLoss) ||
        (!isLong && currentPrice >= trade.stopLoss)) {
      await this.closeTrade(trade.stopLoss, 'SL');
      return {
        closed: true,
        message: 'STOP LOSS',
        pnl: unrealizedPnl, pnlPercent, pnlSign: '',
      };
    }

    // Check trailing stop
    if (trade.trailingStop !== null) {
      if ((isLong && currentPrice <= trade.trailingStop) ||
          (!isLong && currentPrice >= trade.trailingStop)) {
        await this.closeTrade(trade.trailingStop, 'TRAILING');
        return {
          closed: true,
          message: 'TRAILING STOP',
          pnl: unrealizedPnl, pnlPercent,
          pnlSign: unrealizedPnl >= 0 ? '+' : '',
        };
      }
    }

    // Check TP1 (33% close)
    if (!trade.tp1Hit) {
      if ((isLong && currentPrice >= trade.takeProfit1) ||
          (!isLong && currentPrice <= trade.takeProfit1)) {
        await this.closePartialTrade(currentPrice, 0.33, 'TP1');
        trade.tp1Hit = true;
        return {
          closed: false,
          message: `TP1 HIT (+33% closed) | SL->BE`,
          pnl: unrealizedPnl, pnlPercent, pnlSign: '+',
        };
      }
    }

    // Check TP2 (33% close)
    if (!trade.tp2Hit && trade.tp1Hit) {
      if ((isLong && currentPrice >= trade.takeProfit2) ||
          (!isLong && currentPrice <= trade.takeProfit2)) {
        await this.closePartialTrade(currentPrice, 0.33, 'TP2');
        trade.tp2Hit = true;
        if (trade.currentPositionSize <= CoinTrader.POSITION_EPS) {
          trade.currentPositionSize = 0;
          trade.exitPrice = currentPrice;
          trade.exitTime = Date.now();
          trade.exitReason = 'TP2';
          trade.pnl = 0;
          trade.pnlPercent = 0;
          trade.status = 'CLOSED';
          this.state.openTrade = null;
          this.saveState();
          return {
            closed: true,
            message: `TP2 HIT (closed)`,
            pnl: unrealizedPnl, pnlPercent, pnlSign: '+',
          };
        }
        return {
          closed: false,
          message: `TP2 HIT (+33% closed)`,
          pnl: unrealizedPnl, pnlPercent, pnlSign: '+',
        };
      }
    }

    // Check TP3 (remaining 34% close)
    if (!trade.tp3Hit && trade.tp2Hit) {
      if ((isLong && currentPrice >= trade.takeProfit3) ||
          (!isLong && currentPrice <= trade.takeProfit3)) {
        await this.closePartialTrade(currentPrice, 1.0, 'TP3');  // Close remaining
        trade.tp3Hit = true;
        trade.exitPrice = currentPrice;
        trade.exitTime = Date.now();
        trade.exitReason = 'TP3';
        trade.status = 'CLOSED';
        this.state.openTrade = null;
        this.saveState();
        return {
          closed: true,
          message: `TP3 HIT (closed)`,
          pnl: unrealizedPnl, pnlPercent, pnlSign: '+',
        };
      }
    }

    // Trailing stop: after TP1, trail by CONFIG.trailingStopPct below/above price
    // ONLY update on new 4H candle close (not every tick) to avoid noise
    if (trade.tp1Hit) {
      const primaryTf = this.state.timeframes.get(CONFIG.primaryInterval);
      const isNewCandle = primaryTf && primaryTf.candles.length > 1 &&
        primaryTf.candles[primaryTf.candles.length - 1].timestamp !== this.state.lastCandleCloseTime;

      if (isNewCandle) {
        const trailDistance = currentPrice * (CONFIG.trailingStopPct / 100);
        const newTrailing = isLong ? currentPrice - trailDistance : currentPrice + trailDistance;

        // Only tighten, never loosen
        if (trade.trailingStop === null ||
            (isLong && newTrailing > trade.trailingStop) ||
            (!isLong && newTrailing < trade.trailingStop)) {
          trade.trailingStop = newTrailing;
        }
      }
    }

    // Timeout
    if (CONFIG.maxHoldHours > 0) {
      const heldMs = Date.now() - trade.entryTime;
      const maxHoldMs = CONFIG.maxHoldHours * 60 * 60 * 1000;
      if (heldMs >= maxHoldMs) {
        await this.closeTrade(currentPrice, 'TIMEOUT');
        return {
          closed: true,
          message: `TIMEOUT (${CONFIG.maxHoldHours}h)`,
          pnl: unrealizedPnl, pnlPercent,
          pnlSign: unrealizedPnl >= 0 ? '+' : '',
        };
      }
    }

    const pnlSign = unrealizedPnl >= 0 ? '+' : '';
    const tpInfo = [trade.tp1Hit ? 'TP1' : '', trade.tp2Hit ? 'TP2' : ''].filter(Boolean).join('+') || '';
    const trailInfo = trade.trailingStop ? ` TRAIL:$${trade.trailingStop.toFixed(2)}` : '';
    return {
      closed: false,
      message: `${trade.direction} @ $${trade.entryPrice.toFixed(0)} | Now: $${currentPrice.toFixed(0)}${tpInfo ? ` | ${tpInfo}` : ''}${trailInfo}`,
      pnl: unrealizedPnl, pnlPercent, pnlSign,
    };
  }

  private async closePartialTrade(exitPrice: number, closeFraction: number, reason: 'TP1' | 'TP2' | 'TP3'): Promise<void> {
    const trade = this.state.openTrade!;
    const isLong = trade.direction === 'LONG';

    // Calculate close amount with safety check
    let closeAmount: number;
    if (reason === 'TP3') {
      closeAmount = trade.currentPositionSize;  // Close all remaining
    } else {
      // For TP1/TP2, close fraction of ORIGINAL position
      closeAmount = trade.originalPositionSize * closeFraction;
    }

    // Safety: never close more than we actually have
    closeAmount = Math.min(closeAmount, trade.currentPositionSize);

    trade.currentPositionSize -= closeAmount;
    if (trade.currentPositionSize < CoinTrader.POSITION_EPS) {
      trade.currentPositionSize = 0;
    }

    // Calculate gross PnL (before fees)
    const priceDiff = isLong
      ? exitPrice - trade.entryPrice
      : trade.entryPrice - exitPrice;
    const grossPnl = priceDiff * closeAmount;
    const pnlPercent = (priceDiff / trade.entryPrice) * 100;

    // Calculate fees
    const exitNotional = exitPrice * closeAmount;
    const exitFee = exitNotional * CONFIG.takerFeeRate;

    // Allocate entry fee proportionally based on % of position closing
    const totalEntryFee = trade.entryFee || (trade.entryPrice * trade.originalPositionSize * CONFIG.takerFeeRate);
    const entryFeeAllocation = (closeAmount / trade.originalPositionSize) * totalEntryFee;

    // Net PnL = gross - entry fee portion - exit fee
    const netPnl = grossPnl - entryFeeAllocation - exitFee;

    this.state.balance += netPnl;

    // Capture exit features for ML training
    const primaryTf = this.state.timeframes.get(CONFIG.primaryInterval);
    const exitFeatures = primaryTf?.momentum ? captureSnapshot(primaryTf.momentum, exitPrice) : undefined;

    const closedTrade: PaperTrade = {
      id: `${this.state.symbol}-${Date.now()}-${reason}`,
      symbol: trade.symbol,
      direction: trade.direction,
      entryPrice: trade.entryPrice,
      entryTime: trade.entryTime,
      stopLoss: trade.stopLoss,
      originalStopLoss: trade.originalStopLoss,
      takeProfit1: trade.takeProfit1,
      takeProfit2: trade.takeProfit2,
      takeProfit3: trade.takeProfit3,
      trailingStop: trade.trailingStop,
      originalPositionSize: closeAmount,
      currentPositionSize: 0,
      tp1Hit: reason === 'TP1' ? true : trade.tp1Hit,
      tp2Hit: reason === 'TP2' ? true : trade.tp2Hit,
      tp3Hit: reason === 'TP3' ? true : trade.tp3Hit,
      stopLossMovedToBreakeven: trade.stopLossMovedToBreakeven,
      status: 'CLOSED',
      exitPrice,
      exitTime: Date.now(),
      exitReason: reason,
      pnl: netPnl,
      pnlPercent,
      mlPrediction: trade.mlPrediction,
      regime: trade.regime,
      qualityScore: trade.qualityScore,
      smcContext: trade.smcContext,
      entryFeatures: trade.entryFeatures,
      exitFeatures,
      entryFee: entryFeeAllocation,  // Track entry fee portion
    };

    this.state.trades.push(closedTrade);

    // Update performance analytics
    this.updatePerformanceAnalytics(closedTrade);

    const emoji = netPnl > 0 ? '✅' : '❌';
    const pnlSign = netPnl >= 0 ? '+' : '';

    console.log(`${emoji} ${trade.symbol}: ${reason} HIT!`);
    console.log(`   Entry: $${trade.entryPrice.toFixed(2)} | Exit: $${exitPrice.toFixed(2)}`);
    console.log(`   Closed ${(closeFraction * 100).toFixed(0)}% (${closeAmount.toFixed(6)}) | P&L: ${pnlSign}$${netPnl.toFixed(2)} (${pnlSign}${pnlPercent.toFixed(2)}%)`);
    console.log(`   Remaining: ${trade.currentPositionSize.toFixed(6)}`);

    if (reason === 'TP1' && !trade.stopLossMovedToBreakeven) {
      trade.stopLoss = trade.entryPrice;
      trade.stopLossMovedToBreakeven = true;
      console.log(`   SL moved to breakeven ($${trade.entryPrice.toFixed(2)})\n`);
    } else {
      console.log();
    }

    this.state.stats.totalTrades++;
    this.state.stats.totalPnl += netPnl;
    if (netPnl > 0) {
      this.state.stats.wins++;
    } else {
      this.state.stats.losses++;
    }
    this.state.stats.winRate = this.state.stats.wins / this.state.stats.totalTrades;

    this.saveState();
  }

  private async closeTrade(exitPrice: number, reason: 'TP1' | 'TP2' | 'TP3' | 'SL' | 'TRAILING' | 'MANUAL' | 'TIMEOUT'): Promise<void> {
    const trade = this.state.openTrade!;
    const isLong = trade.direction === 'LONG';

    // Calculate gross PnL (before fees)
    const priceDiff = isLong
      ? exitPrice - trade.entryPrice
      : trade.entryPrice - exitPrice;
    const grossPnl = priceDiff * trade.currentPositionSize;
    const pnlPercent = (priceDiff / trade.entryPrice) * 100;

    // Calculate fees
    const exitNotional = exitPrice * trade.currentPositionSize;
    const exitFee = exitNotional * CONFIG.takerFeeRate;

    // Calculate remaining entry fee portion
    const totalEntryFee = trade.entryFee || (trade.entryPrice * trade.originalPositionSize * CONFIG.takerFeeRate);
    const remainingEntryFee = (trade.currentPositionSize / trade.originalPositionSize) * totalEntryFee;

    // Net PnL = gross - remaining entry fee - exit fee
    const netPnl = grossPnl - remainingEntryFee - exitFee;

    trade.exitPrice = exitPrice;
    trade.exitTime = Date.now();
    trade.exitReason = reason;
    trade.pnl = netPnl;
    trade.pnlPercent = pnlPercent;
    trade.status = 'CLOSED';

    // Save market snapshot at exit for ML training
    const primaryTf = this.state.timeframes.get(CONFIG.primaryInterval);
    if (primaryTf?.momentum) {
      trade.exitFeatures = captureSnapshot(primaryTf.momentum, exitPrice);
    }

    this.state.balance += netPnl;

    // Update performance analytics
    this.updatePerformanceAnalytics(trade);

    this.state.stats.totalTrades++;
    this.state.stats.totalPnl += netPnl;
    if (netPnl > 0) {
      this.state.stats.wins++;
    } else {
      this.state.stats.losses++;
    }
    this.state.stats.winRate = this.state.stats.wins / this.state.stats.totalTrades;

    const emoji = netPnl > 0 ? '✅' : '❌';
    const pnlSign = netPnl >= 0 ? '+' : '';

    console.log(`${emoji} ${trade.symbol}: CLOSED ${trade.direction} (${reason})`);
    console.log(`   Entry: $${trade.entryPrice.toFixed(2)} | Exit: $${exitPrice.toFixed(2)}`);
    console.log(`   P&L: ${pnlSign}$${netPnl.toFixed(2)} (${pnlSign}${pnlPercent.toFixed(2)}%)\n`);
  }

  // Print performance analytics summary
  printPerformanceAnalytics(): void {
    if (this.state.stats.totalTrades < 5) return;  // Need minimum trades

    const p = this.state.performance;

    console.log(`\n   📊 ${this.state.symbol} Performance Analytics:`);

    // By regime
    console.log(`   By Regime:`);
    for (const [regime, data] of Object.entries(p.byRegime)) {
      if (data.trades > 0) {
        const wr = data.wins / data.trades;
        console.log(`     ${regime}: ${data.trades} trades | WR: ${(wr * 100).toFixed(0)}% | P&L: $${data.totalPnl.toFixed(2)}`);
      }
    }

    // By session
    console.log(`   By Session:`);
    for (const [session, data] of Object.entries(p.bySession)) {
      if (data.trades > 0) {
        const wr = data.wins / data.trades;
        console.log(`     ${session}: ${data.trades} trades | WR: ${(wr * 100).toFixed(0)}% | P&L: $${data.totalPnl.toFixed(2)}`);
      }
    }

    // Edge decay detection
    if (p.edgeDecay) {
      console.log(`   ⚠️  EDGE DECAY: Recent ${(p.recentEdge * 100).toFixed(0)}% < Historical ${(p.historicalEdge * 100).toFixed(0)}%`);
    }

    console.log('');
  }
}

// ═══════════════════════════════════════════════════════════════
// ORCHESTRATOR
// ═══════════════════════════════════════════════════════════════

class MultiCoinOrchestrator {
  private client: any;
  private traders: Map<string, CoinTrader> = new Map();
  private modelWeights: any = null;
  public running: boolean = false;
  private cycleCount: number = 0;

  constructor() {
    this.client = Binance();
  }

  async initialize(): Promise<void> {
    console.log('\n╔═══════════════════════════════════════════════════════════════╗');
    console.log(`║   REGIME SWING TRADER - ${CONFIG.primaryInterval} Primary (Systematic Entry)      ║`);
    console.log('╚═══════════════════════════════════════════════════════════════╝\n');

    console.log('Entry System:');
    console.log(`  Mode:            Regime-based (TREND/RANGE/CHOP)`);
    console.log(`  TREND entry:     MACD + (EMA align OR breakout) + VWAP`);
    console.log(`  RANGE entry:     BB extreme (<20% / >80%) + Volume spike`);
    console.log(`  CHOP:            Skip (ATR < ${(CONFIG.regime.minVolatility * 100).toFixed(1)}%)`);
    console.log(`  SMC/ICT:         Context adjusters (stop/size/confidence)`);
    console.log(`  Trailing Stop:   ${CONFIG.trailingStopPct}% after TP1`);
    console.log(`  ML Threshold:    ${CONFIG.minWinProbability === 0 ? 'BYPASSED (data collection)' : `${(CONFIG.minWinProbability * 100).toFixed(0)}%`}`);
    console.log(`  Max Hold:        ${CONFIG.maxHoldHours}h`);
    console.log('');

    // Check for --reset flag
    const shouldReset = process.argv.includes('--reset');
    if (shouldReset) {
      console.log('🔄 RESET MODE: Backing up and resetting all trader states...\n');
    }

    const weightsFile = path.join(process.cwd(), 'data', 'models', 'model-weights.json');
    if (fs.existsSync(weightsFile)) {
      this.modelWeights = JSON.parse(fs.readFileSync(weightsFile, 'utf-8'));
      console.log('ML model loaded');
    } else {
      console.log('No model weights found. Running without ML filter.\n');
    }

    console.log('Initializing traders for all coins...\n');
    for (const symbol of SYMBOLS) {
      const trader = new CoinTrader(symbol);
      trader.loadState(new Map());
      await trader.initialize(this.client, this.modelWeights);
      if (shouldReset) {
        trader.resetState();
      }
      this.traders.set(symbol, trader);
    }

    this.printSummary();
  }

  private printSummary(): void {
    let totalBalance = 0;
    let totalTrades = 0;
    let totalWins = 0;
    let totalLosses = 0;
    let totalPnl = 0;
    let unrealizedPnl = 0;
    let openTrades = 0;
    let modelType = 'None';
    let modelAccuracy = 'N/A';

    for (const trader of this.traders.values()) {
      totalBalance += trader.state.balance;
      totalTrades += trader.state.stats.totalTrades;
      totalWins += trader.state.stats.wins;
      totalLosses += trader.state.stats.losses;
      totalPnl += trader.state.stats.totalPnl;

      if (modelType === 'None') {
        if (trader.useLightGBM && trader.lgbmPredictor.isLoaded()) {
          modelType = 'LightGBM';
          const metadata = trader.lgbmPredictor.getMetadata();
          modelAccuracy = metadata?.validation_accuracy
            ? `${(metadata.validation_accuracy * 100).toFixed(1)}%`
            : '73%';
        } else if (this.modelWeights) {
          modelType = 'Gradient Descent';
          modelAccuracy = '59%';
        }
      }

      if (trader.state.openTrade) {
        openTrades++;
        const trade = trader.state.openTrade;
        const primaryTf = trader.state.timeframes.get(CONFIG.primaryInterval);
        if (primaryTf && primaryTf.candles.length > 0) {
          const currentPrice = primaryTf.candles[primaryTf.candles.length - 1].close;
          const isLong = trade.direction === 'LONG';
          const priceDiff = isLong
            ? currentPrice - trade.entryPrice
            : trade.entryPrice - currentPrice;
          unrealizedPnl += priceDiff * trade.currentPositionSize;
        }
      }
    }

    const winRate = totalTrades > 0 ? (totalWins / totalTrades) : 0;
    const startingBalance = SYMBOLS.length * CONFIG.virtualBalancePerCoin;
    const realizedPnl = totalBalance - startingBalance;
    const equity = totalBalance + unrealizedPnl;
    const totalPnlAll = equity - startingBalance;
    const realizedReturn = (realizedPnl / startingBalance) * 100;
    const totalReturn = (totalPnlAll / startingBalance) * 100;
    const realizedSign = realizedPnl >= 0 ? '+' : '';
    const unrealizedSign = unrealizedPnl >= 0 ? '+' : '';
    const totalSign = totalPnlAll >= 0 ? '+' : '';

    console.log('═══════════════════════════════════════════════════════════');
    console.log(`PORTFOLIO SUMMARY (REGIME SWING)`);
    console.log(`   Balance: $${totalBalance.toFixed(2)} (Started: $${startingBalance})`);
    console.log(`   Realized P&L: ${realizedSign}$${realizedPnl.toFixed(2)} (${realizedReturn >= 0 ? '+' : ''}${realizedReturn.toFixed(2)}%)`);
    if (openTrades > 0) {
      console.log(`   Unrealized: ${unrealizedSign}$${unrealizedPnl.toFixed(2)} (${openTrades} open)`);
    }
    console.log(`   Equity: $${equity.toFixed(2)} | Total P&L: ${totalSign}$${totalPnlAll.toFixed(2)} (${totalReturn >= 0 ? '+' : ''}${totalReturn.toFixed(2)}%)`);
    console.log(`   Trades: ${totalTrades} (${totalWins}W/${totalLosses}L) | Win Rate: ${(winRate * 100).toFixed(1)}%`);
    console.log(`   Open Positions: ${openTrades}`);
    console.log(`   ML Model: ${modelType} (${modelAccuracy})`);
    console.log('═════════════════════════════════════════════════════════════\n');
  }

  async run(): Promise<void> {
    const getDecimalPlaces = (p: number): number => {
      if (p >= 1000) return 2;
      if (p >= 100) return 2;
      if (p >= 10) return 3;
      if (p >= 1) return 4;
      return 5;
    };

    this.running = true;
    this.running = true;
    console.log('Starting regime swing trading loop...');
    console.log(`   Checking every ${CONFIG.checkIntervalMs / 1000}s`);
    console.log('   Press Ctrl+C to stop\n\n');

    while (this.running) {
      this.cycleCount++;
      const timestamp = new Date().toLocaleTimeString();
      const cycleStart = Date.now();

      process.stdout.write('\x1B[2J\x1B[0f');
      console.log('\n╔═══════════════════════════════════════════════════════════╗');
      console.log(`║  REGIME SWING TRADER - Cycle: ${this.cycleCount} | ${timestamp}      ║`);
      console.log('╚═══════════════════════════════════════════════════════════╝\n');

      const results: Array<{ symbol: string; price: number; result: any }> = [];
      for (const symbol of SYMBOLS) {
        const trader = this.traders.get(symbol)!;
        const primaryTf = trader.state.timeframes.get(CONFIG.primaryInterval);
        const currentPrice = (primaryTf && primaryTf.candles && primaryTf.candles.length > 0)
          ? primaryTf.candles[primaryTf.candles.length - 1].close
          : 0;

        const result = await trader.tick(this.client, timestamp);
        results.push({ symbol, price: currentPrice, result });

        trader.saveState();
      }

      // ═══════════════════════════════════════════════════════════════
      // DISPLAY: Summary mode (clean) or Verbose mode (all coins)
      // ═══════════════════════════════════════════════════════════════
      const isVerbose = CONFIG.reporting.mode === 'verbose';

      // Show open trades (always shown)
      const openTrades = results.filter(({ symbol }) => {
        const trader = this.traders.get(symbol)!;
        return trader.state.openTrade;
      });
      if (openTrades.length > 0) {
        console.log('⚡ OPEN TRADES:');
        for (const { symbol, price } of openTrades) {
          const trader = this.traders.get(symbol)!;
          const trade = trader.state.openTrade!;
          const isLong = trade.direction === 'LONG';
          const priceDiff = isLong ? price - trade.entryPrice : trade.entryPrice - price;
          const unrealizedPnl = priceDiff * trade.currentPositionSize;
          const pnlPercent = (priceDiff / trade.entryPrice) * 100;
          const pnlSign = unrealizedPnl >= 0 ? '+' : '';
          const pricePrecision = getDecimalPlaces(price);

          // TP hit indicators
          const tp1 = trade.tp1Hit ? '✓' : ' ';
          const tp2 = trade.tp2Hit ? '✓' : ' ';
          const tp3 = trade.tp3Hit ? '✓' : ' ';

          const currentPriceDisplay = price > 0 ? `${price.toFixed(pricePrecision)}` : '---';
          console.log(`   ${symbol}: ${trade.direction} | ${currentPriceDisplay} | ${pnlSign}$${unrealizedPnl.toFixed(2)} (${pnlSign}${pnlPercent.toFixed(1)}%) | TP: [${tp1}]$${trade.takeProfit1.toFixed(pricePrecision)} [${tp2}]$${trade.takeProfit2.toFixed(pricePrecision)} [${tp3}]$${trade.takeProfit3.toFixed(pricePrecision)} | SL:$${trade.stopLoss.toFixed(pricePrecision)}`);
        }
      }

      // Show all coins only in verbose mode
      if (isVerbose) {
        console.log('\n📡 SCANNING:');
        for (const { symbol, price, result } of results) {
          const trader = this.traders.get(symbol)!;
          if (trader.state.openTrade) continue; // Skip open trades (already shown)
          const priceDisplay = price > 0 ? `$${price.toFixed(getDecimalPlaces(price))}` : 'N/A';
          const regimeDisplay = (result.regime || '?').padEnd(5);
          console.log(`   ${symbol.padEnd(10)}: ${priceDisplay.padEnd(12)} | ${regimeDisplay} | ${result.details}`);
        }
      } else {
        const scanCount = results.filter(({ symbol }) => {
          const trader = this.traders.get(symbol)!;
          return !trader.state.openTrade;
        }).length;
        console.log(`\n📡 SCANNING: ${scanCount} coins (use verbose mode for details)`);
      }

      // ═══════════════════════════════════════════════════════════════
      // SUMMARY: Enhanced metrics (every cycle, not every 5)
      // ═══════════════════════════════════════════════════════════════
      if (true) {  // Show summary every cycle
        const openCount = openTrades.length;
        let totalPnl = 0, totalTrades = 0, totalWins = 0, totalLosses = 0;
        for (const [symbol, trader] of this.traders) {
          totalPnl += trader.state.stats.totalPnl;
          totalTrades += trader.state.stats.totalTrades;
          totalWins += trader.state.stats.wins;
          totalLosses += trader.state.stats.losses;
        }
        const winRate = totalTrades > 0 ? (totalWins / totalTrades * 100).toFixed(1) : '0.0';
        const pnlSign = totalPnl >= 0 ? '+' : '';
        const pnlColor = totalPnl >= 0 ? '🟢' : '🔴';

        // Calculate current streak
        const allTrades: PaperTrade[] = [];
        for (const [, trader] of this.traders) {
          allTrades.push(...trader.state.trades.filter(t => t.status === 'CLOSED'));
        }
        allTrades.sort((a, b) => (a.exitTime || 0) - (b.exitTime || 0));

        let currentStreak = 0;
        let currentStreakType = 'NONE';
        for (let i = allTrades.length - 1; i >= 0; i--) {
          const pnl = allTrades[i].pnl || 0;
          if (currentStreak === 0) {
            currentStreakType = pnl >= 0 ? 'WIN' : 'LOSS';
            currentStreak++;
          } else if ((currentStreakType === 'WIN' && pnl >= 0) || (currentStreakType === 'LOSS' && pnl < 0)) {
            currentStreak++;
          } else {
            break;
          }
        }

        // Calculate unrealized P&L from open trades
        let unrealizedPnl = 0;
        for (const { symbol, price } of results) {
          const trader = this.traders.get(symbol)!;
          if (trader.state.openTrade) {
            const trade = trader.state.openTrade;
            const isLong = trade.direction === 'LONG';
            const priceDiff = isLong ? price - trade.entryPrice : trade.entryPrice - price;
            unrealizedPnl += priceDiff * trade.currentPositionSize;
          }
        }
        const unrealizedSign = unrealizedPnl >= 0 ? '+' : '';
        const unrealizedColor = unrealizedPnl >= 0 ? '🟢' : '🔴';

        console.log('\n═══════════════════════════════════════════════════════════════');
        console.log(`📊 SUMMARY: Open: ${openCount} | Trades: ${totalTrades} (${totalWins}W/${totalLosses}L) | Win: ${winRate}%`);
        console.log(`           PnL: ${pnlColor} ${pnlSign}$${totalPnl.toFixed(2)} | Unrealized: ${unrealizedColor} ${unrealizedSign}$${unrealizedPnl.toFixed(2)}`);
        console.log(`           Streak: ${currentStreakType} (${currentStreak})`);
        console.log('═══════════════════════════════════════════════════════════════\n');
      }

      // ═══════════════════════════════════════════════════════════════
      // LIVE SUMMARY: Write to JSON file for OpenClaw integration
      // ═══════════════════════════════════════════════════════════════
      const tradersArray = Array.from(this.traders.values());
      const resultsWithTraders = results.map(({ symbol, price, result }) => ({
        symbol,
        price,
        trader: this.traders.get(symbol)!,
      }));
      writeLiveSummary(tradersArray, this.cycleCount, resultsWithTraders);

      // Regime distribution + filter diagnostic every 10 cycles
      if (this.cycleCount % 10 === 0) {
        const regimeCounts: Record<string, number> = { TREND: 0, RANGE: 0, CHOP: 0 };
        const blockerCounts: Record<string, number> = {};

        for (const { result } of results) {
          const regime = result.regime || '?';
          if (regime in regimeCounts) regimeCounts[regime]++;

          if (result.status === 'MONITOR') {
            const details = result.details || '';
            let blocker = details;
            // Truncate long reasons
            if (blocker.length > 50) blocker = blocker.slice(0, 50) + '...';
            blockerCounts[blocker] = (blockerCounts[blocker] || 0) + 1;
          }
        }

        console.log(`\nREGIME DISTRIBUTION: TREND:${regimeCounts.TREND} | RANGE:${regimeCounts.RANGE} | CHOP:${regimeCounts.CHOP}`);

        const monitorCount = results.filter(r => r.result.status === 'MONITOR').length;
        if (monitorCount > 0) {
          console.log(`\nFILTER DIAGNOSTIC: ${monitorCount}/${results.length} coins not trading`);
          const sorted = Object.entries(blockerCounts).sort((a, b) => b[1] - a[1]);
          for (const [reason, count] of sorted.slice(0, 8)) {
            console.log(`   ${count} coins: ${reason}`);
          }
        }
        console.log();
      }

      // ═══════════════════════════════════════════════════════════════
      // AUTO-LEARNING: Trigger retraining every N trades
      // ═══════════════════════════════════════════════════════════════
      if (CONFIG.autoLearn.enabled) {
        let totalTrades = 0;
        for (const trader of this.traders.values()) {
          totalTrades += trader.state.stats.totalTrades;
        }

        if (totalTrades > 0) {
          const tradesAtLastLearn = (global as any).__lastLearnedAtSwing || 0;
          const tradesSinceLearn = totalTrades - tradesAtLastLearn;

          if (tradesSinceLearn >= CONFIG.autoLearn.triggerEveryNTrades && totalTrades >= CONFIG.autoLearn.minTradesForTraining) {
            console.log(`\n🧠 AUTO-LEARN: ${tradesSinceLearn} new trades - triggering learning loop...`);
            (global as any).__lastLearnedAtSwing = totalTrades;

            try {
              const { execSync } = await import('child_process');
              const cwd = process.cwd();

              // Export trades
              console.log('   Exporting trades...');
              execSync('npm run export-paper-trades-swing', { cwd, stdio: 'pipe' });

              // Find the latest export file
              const exportDir = path.join(cwd, 'data', 'h2o-training');
              const files = fs.readdirSync(exportDir)
                .filter((f: string) => f.startsWith('paper_swing_') && f.endsWith('.csv'))
                .sort()
                .reverse();

              if (files.length > 0) {
                const latestFile = path.join(exportDir, files[0]);
                console.log(`   Training on: ${files[0]}`);

                const paperModelDir = path.join(cwd, 'data', 'models-paper-swing');
                if (!fs.existsSync(paperModelDir)) fs.mkdirSync(paperModelDir, { recursive: true });
                const trainOutput = execSync(`python scripts/lightgbm_walkforward.py --input "${latestFile}" --output "${paperModelDir}"`, {
                  cwd,
                  encoding: 'utf-8',
                  timeout: 300000
                });

                const lines = trainOutput.split('\n');
                for (const line of lines) {
                  const t = line.trim();
                  if (!t || t.startsWith('Loading') || t.startsWith('Loaded') || t.startsWith('Config:')) continue;
                  if (t.includes('WALK-FORWARD') || t.includes('RESULTS') || t.includes('FOLD') ||
                      t.includes('MODEL') || t.includes('SMALL DATA') || t.includes('Optimal threshold') ||
                      t.includes('Accuracy') || t.includes('AUC') || t.includes('Baseline') ||
                      t.includes('Filtered') || t.includes('Improvement') || t.includes('Win rate') ||
                      t.includes('Win Rate') || t.includes('PnL') || t.includes('Top 10') ||
                      t.includes('Features Used') || t.includes('Best Fold') || t.includes('Best iteration') ||
                      t.includes('scale_pos_weight') || t.includes('Train:') || t.includes('Test:') ||
                      t.includes('Trades') || t.includes('Model saved') || t.includes('Saved') ||
                      t.includes('Not saving') || t.includes('improvement') ||
                      t.match(/^\d+\./) || t.startsWith('---') || t.startsWith('===')) {
                    console.log(`   ${t}`);
                  }
                }
                console.log('');
              }
            } catch (e: any) {
              console.log(`   ⚠️ Learning failed: ${e.message?.slice(0, 80)}\n`);
            }
          }
        }
      }

      const elapsed = Date.now() - cycleStart;
      const waitTime = Math.max(0, CONFIG.checkIntervalMs - elapsed);
      // Sleep in 1s chunks so SIGINT is responsive
      const deadline = Date.now() + waitTime;
      while (this.running && Date.now() < deadline) {
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }

    console.log('\nTrading loop stopped. Performing final save...');
    this.stop();
  }

  stop(): void {
    this.running = false;
    console.log('\n\nStopping regime swing trader...');

    for (const trader of this.traders.values()) {
      trader.saveState();
    }

    this.printSummary();
  }
}

async function main() {
  const orchestrator = new MultiCoinOrchestrator();

  process.on('SIGINT', () => {
    console.log('\n\nSIGINT received. Stopping gracefully...');
    orchestrator.running = false;
  });

  await orchestrator.initialize();
  await orchestrator.run();
  console.log('\nRegime swing trader stopped successfully.\n');
  process.exit(0);
}

main().catch(console.error);
