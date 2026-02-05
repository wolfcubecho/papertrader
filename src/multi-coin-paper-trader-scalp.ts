#!/usr/bin/env node
/**
 * Multi-Coin Paper Trading - MOMENTUM SCALPER (5m Primary)
 *
 * TRUE SCALPING: Momentum-based entries, NOT structure-based.
 * - Volume spikes + price breakouts
 * - RSI momentum crossovers
 * - Bollinger Band breakouts
 * - EMA crossovers (9/21)
 * - Quick fixed % targets (0.3-0.5%)
 *
 * This is DIFFERENT from the day trader which uses SMC/ICT.
 *
 * Usage: npm run paper-trade-multi-scalp
 */

import { createRequire } from 'module';
import fs from 'fs';
import path from 'path';

const require = createRequire(import.meta.url);
const Binance = require('binance-api-node').default;
import { Candle, SMCIndicators } from './smc-indicators.js';
import { FeatureExtractor, TradeFeatures } from './trade-features.js';
import { LightGBMPredictor } from './lightgbm-predictor.js';

// Top 20 coins for scalping
const SYMBOLS = [
  'BTCUSDT',  'ETHUSDT',
  'BNBUSDT',  'SOLUSDT',
  'XRPUSDT',  'DOGEUSDT',
  'ADAUSDT',  'AVAXUSDT',
  'LINKUSDT', 'DOTUSDT',
  'MATICUSDT', 'ATOMUSDT',
  'NEARUSDT', 'ARBUSDT',
  'OPUSDT',   'INJUSDT',
  'LDOUSDT',  'SUIUSDT',
  'UNIUSDT',  'TONUSDT',
] as const;

// ═══════════════════════════════════════════════════════════════
// SCALPER CONFIGURATION - Momentum-based
// ═══════════════════════════════════════════════════════════════
const CONFIG = {
  mode: 'MOMENTUM_SCALP',
  intervals: ['1m', '5m', '15m'] as const,
  primaryInterval: '5m' as const,  // 5m for scalping momentum
  checkIntervalMs: 5000,           // Check every 5s for quick entries
  minCandlesRequired: 50,          // Need less history for momentum

  // ═══════════════════════════════════════════════════════════════
  // MOMENTUM THRESHOLDS
  // ═══════════════════════════════════════════════════════════════
  momentum: {
    // Volume spike detection
    volumeSpikeMultiple: 1.8,      // Volume > 1.8x average = spike
    volumeAvgPeriod: 20,           // 20-candle average for comparison

    // RSI settings
    rsiPeriod: 14,
    rsiBullishCross: 50,           // RSI crossing above 50 = bullish momentum
    rsiBearishCross: 50,           // RSI crossing below 50 = bearish momentum
    rsiOverbought: 70,
    rsiOversold: 30,

    // EMA crossover
    emaFast: 9,
    emaSlow: 21,

    // Bollinger Bands
    bbPeriod: 20,
    bbStdDev: 2,

    // Price breakout
    breakoutLookback: 10,          // Look for break of 10-candle high/low

    // Candle momentum
    minBodyRatio: 0.6,             // Body must be > 60% of candle range

    // Confluence required
    minSignals: 2,                 // Need at least 2 momentum signals

    // MACD settings
    macdFast: 12,
    macdSlow: 26,
    macdSignal: 9,
  },

  // ═══════════════════════════════════════════════════════════════
  // DUAL MODE: TREND vs RANGE
  // ═══════════════════════════════════════════════════════════════
  regime: {
    // Volatility threshold to switch modes (ATR % of price)
    volatilityThreshold: 0.015,    // 1.5% = high volatility = TREND mode
    atrPeriod: 14,

    // TREND mode: ride momentum breakouts
    // RANGE mode: mean reversion at BB extremes
  },

  // ═══════════════════════════════════════════════════════════════
  // AUTO-LEARNING
  // ═══════════════════════════════════════════════════════════════
  autoLearn: {
    enabled: true,
    triggerEveryNTrades: 100,      // Retrain after every 100 closed trades
    minTradesForTraining: 50,     // Need at least 50 trades to train
  },

  // ═══════════════════════════════════════════════════════════════
  // ENTRY/EXIT - 3 TP levels for scalping
  // ═══════════════════════════════════════════════════════════════
  targets: {
    stopLossPct: 0.25,             // 0.25% stop (tight)
    tp1Pct: 0.25,                  // TP1: 0.25% - quick partial (50%)
    tp2Pct: 0.40,                  // TP2: 0.40% - second partial (25%)
    tp3Pct: 0.60,                  // TP3: 0.60% - runner (25%)
    trailingActivatePct: 0.25,    // Activate trailing at 0.25%
    trailingDistancePct: 0.15,    // Trail by 0.15%
  },

  // ═══════════════════════════════════════════════════════════════
  // TIMING
  // ═══════════════════════════════════════════════════════════════
  maxHoldMinutes: 30,              // Max hold 30 mins (it's a scalp!)
  cooldownMs: 60_000,              // 1 minute cooldown between trades
  onlyEnterOnCandleClose: false,   // Scalper can enter mid-candle

  // ═══════════════════════════════════════════════════════════════
  // RISK & FEES
  // ═══════════════════════════════════════════════════════════════
  virtualBalancePerCoin: 10000,
  riskPerTradePct: 1.0,            // Risk 1% per scalp
  leverage: 1,
  takerFeeRate: 0.0004,            // 4 bps
  slippageBps: 2,

  // Data refresh
  refreshMsByInterval: {
    '1m': 5_000,
    '5m': 5_000,
    '15m': 15_000,
  } as Record<string, number>,
};

// ═══════════════════════════════════════════════════════════════
// MOMENTUM INDICATORS
// ═══════════════════════════════════════════════════════════════

interface MomentumSignals {
  volumeSpike: boolean;
  volumeRatio: number;
  rsiValue: number;
  rsiBullishCross: boolean;
  rsiBearishCross: boolean;
  rsiOverbought: boolean;
  rsiOversold: boolean;
  emaFast: number;
  emaSlow: number;
  emaBullishCross: boolean;
  emaBearishCross: boolean;
  emaAligned: 'bullish' | 'bearish' | 'neutral';
  bbUpper: number;
  bbLower: number;
  bbMiddle: number;
  bbPosition: number;  // 0 = at lower band, 0.5 = middle, 1 = at upper band
  bbBreakoutUp: boolean;
  bbBreakoutDown: boolean;
  priceBreakoutUp: boolean;
  priceBreakoutDown: boolean;
  candleMomentum: 'bullish' | 'bearish' | 'neutral';

  // MACD
  macdLine: number;
  macdSignal: number;
  macdHistogram: number;
  macdBullishCross: boolean;
  macdBearishCross: boolean;

  // Volatility / Regime
  atr: number;
  atrPercent: number;  // ATR as % of price
  regime: 'TREND' | 'RANGE';

  // VWAP (Institutional anchor)
  vwap: number;
  vwapDeviation: number;      // (price - vwap) / vwap as %
  vwapDeviationStd: number;   // How many std devs from VWAP
  priceAboveVwap: boolean;

  // Kill Zone (Session timing)
  killZone: 'LONDON' | 'NY_OPEN' | 'NY_AFTERNOON' | 'ASIA' | 'OFF_HOURS';
  isKillZone: boolean;        // True if in high-probability session

  // Structure-based stops (swing points)
  swingHigh: number | null;   // Most recent swing high for SHORT stops
  swingLow: number | null;    // Most recent swing low for LONG stops

  // Aggregated
  bullishSignals: number;
  bearishSignals: number;
  direction: 'LONG' | 'SHORT' | 'NEUTRAL';
  strength: number;  // 0-1
}

function calculateRSI(candles: Candle[], period: number): number[] {
  const rsi: number[] = [];
  let gains = 0;
  let losses = 0;

  for (let i = 1; i < candles.length; i++) {
    const change = candles[i].close - candles[i - 1].close;

    if (i <= period) {
      if (change > 0) gains += change;
      else losses -= change;

      if (i === period) {
        const avgGain = gains / period;
        const avgLoss = losses / period;
        const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
        rsi.push(100 - (100 / (1 + rs)));
      }
    } else {
      const avgGain = ((gains * (period - 1)) + (change > 0 ? change : 0)) / period;
      const avgLoss = ((losses * (period - 1)) + (change < 0 ? -change : 0)) / period;
      gains = avgGain;
      losses = avgLoss;
      const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
      rsi.push(100 - (100 / (1 + rs)));
    }
  }

  return rsi;
}

function calculateEMA(candles: Candle[], period: number): number[] {
  const ema: number[] = [];
  const multiplier = 2 / (period + 1);

  // Start with SMA
  let sum = 0;
  for (let i = 0; i < period && i < candles.length; i++) {
    sum += candles[i].close;
  }
  ema.push(sum / Math.min(period, candles.length));

  // Calculate EMA
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
    const mean = slice.reduce((sum, c) => sum + c.close, 0) / period;
    const variance = slice.reduce((sum, c) => sum + Math.pow(c.close - mean, 2), 0) / period;
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

  // MACD Line = Fast EMA - Slow EMA
  const macdLine: number[] = [];
  const startIdx = slowPeriod - fastPeriod;

  for (let i = 0; i < emaSlow.length; i++) {
    const fastIdx = i + startIdx;
    if (fastIdx >= 0 && fastIdx < emaFast.length) {
      macdLine.push(emaFast[fastIdx] - emaSlow[i]);
    }
  }

  // Signal Line = EMA of MACD Line
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

  // Histogram = MACD Line - Signal Line
  const histogram: number[] = [];
  const offset = macdLine.length - signalLine.length;
  for (let i = 0; i < signalLine.length; i++) {
    histogram.push(macdLine[i + offset] - signalLine[i]);
  }

  return { macdLine, signalLine, histogram };
}

function calculateVWAP(candles: Candle[]): { vwap: number; stdDev: number } {
  // VWAP = Cumulative(Price * Volume) / Cumulative(Volume)
  // Use typical price: (High + Low + Close) / 3
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

  // Calculate standard deviation of price from VWAP
  const deviations = typicalPrices.map(p => Math.pow(p - vwap, 2));
  const variance = deviations.reduce((a, b) => a + b, 0) / deviations.length;
  const stdDev = Math.sqrt(variance);

  return { vwap, stdDev };
}

function getKillZone(timestamp: number): { zone: 'LONDON' | 'NY_OPEN' | 'NY_AFTERNOON' | 'ASIA' | 'OFF_HOURS'; isActive: boolean } {
  // Kill zones based on UTC hours (high-probability trading windows)
  // London: 07:00-10:00 UTC (8-11 AM London)
  // NY Open: 13:00-16:00 UTC (8-11 AM EST)
  // NY Afternoon: 18:00-20:00 UTC (2-4 PM EST)
  // Asia: 00:00-03:00 UTC (Tokyo/Sydney overlap)

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
        // First ATR is simple average
        const sum = trueRanges.slice(-period).reduce((a, b) => a + b, 0);
        atr.push(sum / period);
      } else {
        // Subsequent ATR uses smoothing
        const prevATR = atr[atr.length - 1];
        atr.push((prevATR * (period - 1) + tr) / period);
      }
    }
  }

  return atr;
}

// ═══════════════════════════════════════════════════════════════
// STRUCTURE-BASED STOPS - Find swing highs/lows
// ═══════════════════════════════════════════════════════════════

interface SwingPoints {
  recentSwingHigh: number | null;
  recentSwingLow: number | null;
  swingHighIdx: number;
  swingLowIdx: number;
}

function findSwingPoints(candles: Candle[], lookback: number = 5): SwingPoints {
  /**
   * Find the most recent swing high and swing low.
   * A swing high is a candle whose high is higher than the surrounding candles.
   * A swing low is a candle whose low is lower than the surrounding candles.
   */
  let recentSwingHigh: number | null = null;
  let recentSwingLow: number | null = null;
  let swingHighIdx = -1;
  let swingLowIdx = -1;

  const len = candles.length;
  if (len < lookback * 2 + 1) {
    return { recentSwingHigh: null, recentSwingLow: null, swingHighIdx: -1, swingLowIdx: -1 };
  }

  // Search backwards from most recent candle (skip last few as they can't be confirmed yet)
  for (let i = len - lookback - 1; i >= lookback; i--) {
    const high = candles[i].high;
    const low = candles[i].low;

    // Check if this is a swing high (higher than all candles in lookback range)
    let isSwingHigh = true;
    let isSwingLow = true;

    for (let j = i - lookback; j <= i + lookback; j++) {
      if (j === i) continue;
      if (candles[j].high >= high) isSwingHigh = false;
      if (candles[j].low <= low) isSwingLow = false;
    }

    if (isSwingHigh && recentSwingHigh === null) {
      recentSwingHigh = high;
      swingHighIdx = i;
    }
    if (isSwingLow && recentSwingLow === null) {
      recentSwingLow = low;
      swingLowIdx = i;
    }

    // Once we have both, we can stop
    if (recentSwingHigh !== null && recentSwingLow !== null) {
      break;
    }
  }

  return { recentSwingHigh, recentSwingLow, swingHighIdx, swingLowIdx };
}

function analyzeMomentum(candles: Candle[]): MomentumSignals {
  const cfg = CONFIG.momentum;
  const len = candles.length;

  if (len < 30) {
    return {
      volumeSpike: false, volumeRatio: 1,
      rsiValue: 50, rsiBullishCross: false, rsiBearishCross: false,
      rsiOverbought: false, rsiOversold: false,
      emaFast: 0, emaSlow: 0, emaBullishCross: false, emaBearishCross: false,
      emaAligned: 'neutral',
      bbUpper: 0, bbLower: 0, bbMiddle: 0, bbPosition: 0.5,
      bbBreakoutUp: false, bbBreakoutDown: false,
      priceBreakoutUp: false, priceBreakoutDown: false,
      candleMomentum: 'neutral',
      macdLine: 0, macdSignal: 0, macdHistogram: 0,
      macdBullishCross: false, macdBearishCross: false,
      atr: 0, atrPercent: 0, regime: 'RANGE' as const,
      vwap: 0, vwapDeviation: 0, vwapDeviationStd: 0, priceAboveVwap: false,
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
    emaFast > emaSlow * 1.001 ? 'bullish' :
    emaFast < emaSlow * 0.999 ? 'bearish' : 'neutral';

  // Bollinger Bands
  const bb = calculateBollingerBands(candles, cfg.bbPeriod, cfg.bbStdDev);
  const bbUpper = bb.upper[bb.upper.length - 1] || current.close * 1.02;
  const bbLower = bb.lower[bb.lower.length - 1] || current.close * 0.98;
  const bbMiddle = bb.middle[bb.middle.length - 1] || current.close;
  const bbBreakoutUp = current.close > bbUpper && prev.close <= bb.upper[bb.upper.length - 2];
  const bbBreakoutDown = current.close < bbLower && prev.close >= bb.lower[bb.lower.length - 2];
  // BB Position: 0 = at lower band, 0.5 = middle, 1 = at upper band
  const bbRange = bbUpper - bbLower;
  const bbPosition = bbRange > 0 ? Math.max(0, Math.min(1, (current.close - bbLower) / bbRange)) : 0.5;

  // Price breakout (new high/low)
  const lookbackCandles = candles.slice(-cfg.breakoutLookback - 1, -1);
  const recentHigh = Math.max(...lookbackCandles.map(c => c.high));
  const recentLow = Math.min(...lookbackCandles.map(c => c.low));
  const priceBreakoutUp = current.close > recentHigh;
  const priceBreakoutDown = current.close < recentLow;

  // Candle momentum (big body in direction)
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

  // ATR and Regime Detection
  const atrValues = calculateATR(candles, CONFIG.regime.atrPeriod);
  const atr = atrValues[atrValues.length - 1] || 0;
  const atrPercent = current.close > 0 ? (atr / current.close) : 0;
  const regime: 'TREND' | 'RANGE' = atrPercent >= CONFIG.regime.volatilityThreshold ? 'TREND' : 'RANGE';

  // VWAP (Institutional anchor)
  const vwapData = calculateVWAP(candles);
  const vwap = vwapData.vwap;
  const vwapDeviation = vwap > 0 ? ((current.close - vwap) / vwap) * 100 : 0;  // as percentage
  const vwapDeviationStd = vwapData.stdDev > 0 ? (current.close - vwap) / vwapData.stdDev : 0;  // in std devs
  const priceAboveVwap = current.close > vwap;

  // Kill Zone (Session timing)
  const killZoneData = getKillZone(current.timestamp);
  const killZone = killZoneData.zone;
  const isKillZone = killZoneData.isActive;

  // Count signals
  let bullishSignals = 0;
  let bearishSignals = 0;

  // Volume spike is directional based on candle
  if (volumeSpike && candleMomentum === 'bullish') bullishSignals++;
  if (volumeSpike && candleMomentum === 'bearish') bearishSignals++;

  // RSI
  if (rsiBullishCross || rsiOversold) bullishSignals++;
  if (rsiBearishCross || rsiOverbought) bearishSignals++;

  // EMA
  if (emaBullishCross || emaAligned === 'bullish') bullishSignals++;
  if (emaBearishCross || emaAligned === 'bearish') bearishSignals++;

  // Bollinger
  if (bbBreakoutUp) bullishSignals++;
  if (bbBreakoutDown) bearishSignals++;

  // Price breakout
  if (priceBreakoutUp) bullishSignals++;
  if (priceBreakoutDown) bearishSignals++;

  // Candle momentum
  if (candleMomentum === 'bullish') bullishSignals++;
  if (candleMomentum === 'bearish') bearishSignals++;

  // MACD
  if (macdBullishCross || macdHistogram > 0) bullishSignals++;
  if (macdBearishCross || macdHistogram < 0) bearishSignals++;

  // Direction
  let direction: 'LONG' | 'SHORT' | 'NEUTRAL' = 'NEUTRAL';
  const maxSignals = Math.max(bullishSignals, bearishSignals);

  if (bullishSignals >= cfg.minSignals && bullishSignals > bearishSignals) {
    direction = 'LONG';
  } else if (bearishSignals >= cfg.minSignals && bearishSignals > bullishSignals) {
    direction = 'SHORT';
  }

  const strength = maxSignals / 7;  // 7 possible signals now (added MACD)

  // Find swing points for structure-based stops (lookback=5 for 5m candles)
  const swingPoints = findSwingPoints(candles, 5);

  return {
    volumeSpike, volumeRatio,
    rsiValue, rsiBullishCross, rsiBearishCross, rsiOverbought, rsiOversold,
    emaFast, emaSlow, emaBullishCross, emaBearishCross, emaAligned,
    bbUpper, bbLower, bbMiddle, bbPosition, bbBreakoutUp, bbBreakoutDown,
    priceBreakoutUp, priceBreakoutDown,
    candleMomentum,
    macdLine, macdSignal: macdSignalLine, macdHistogram,
    macdBullishCross, macdBearishCross,
    atr, atrPercent, regime,
    vwap, vwapDeviation, vwapDeviationStd, priceAboveVwap,
    killZone, isKillZone,
    swingHigh: swingPoints.recentSwingHigh,
    swingLow: swingPoints.recentSwingLow,
    bullishSignals, bearishSignals,
    direction, strength,
  };
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
  pnl?: number;
  pnlPercent?: number;
  exitPrice?: number;
  exitTime?: number;
  exitReason?: 'TP1' | 'TP2' | 'TP3' | 'SL' | 'TRAILING' | 'TIMEOUT' | 'MANUAL';
  feesPaid: number;
  momentumStrength: number;
  signals: string[];
}

interface TimeframeData {
  candles: Candle[];
  momentum: MomentumSignals;
  lastUpdate: number;
}

interface CoinTradingState {
  symbol: string;
  balance: number;
  timeframes: Map<string, TimeframeData>;
  openTrade: PaperTrade | null;
  trades: PaperTrade[];
  cooldownUntil: number;
  stats: {
    totalTrades: number;
    wins: number;
    losses: number;
    totalPnl: number;
    winRate: number;
  };
}

// ═══════════════════════════════════════════════════════════════
// COIN TRADER CLASS
// ═══════════════════════════════════════════════════════════════

class CoinTrader {
  public state: CoinTradingState;
  public lgbmPredictor: LightGBMPredictor;
  private stateFile: string;

  constructor(symbol: typeof SYMBOLS[number]) {
    this.stateFile = path.join(process.cwd(), 'data', 'paper-trades-scalp', `${symbol}.json`);
    this.lgbmPredictor = new LightGBMPredictor();

    this.state = {
      symbol,
      balance: CONFIG.virtualBalancePerCoin,
      timeframes: new Map(),
      openTrade: null,
      trades: [],
      cooldownUntil: 0,
      stats: {
        totalTrades: 0,
        wins: 0,
        losses: 0,
        totalPnl: 0,
        winRate: 0,
      },
    };

    // Initialize timeframes
    for (const interval of CONFIG.intervals) {
      this.state.timeframes.set(interval, {
        candles: [],
        momentum: analyzeMomentum([]),
        lastUpdate: 0,
      });
    }
  }

  loadState(): void {
    try {
      if (fs.existsSync(this.stateFile)) {
        const saved = JSON.parse(fs.readFileSync(this.stateFile, 'utf-8'));
        this.state.balance = saved.balance || CONFIG.virtualBalancePerCoin;
        this.state.openTrade = saved.openTrade || null;
        this.state.trades = saved.trades || [];
        this.state.cooldownUntil = saved.cooldownUntil || 0;
        this.state.stats = saved.stats || this.state.stats;
      }
    } catch (e) {
      console.log(`  ${this.state.symbol}: Fresh start (no saved state)`);
    }
  }

  saveState(): void {
    try {
      const dir = path.dirname(this.stateFile);
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }

      const toSave = {
        symbol: this.state.symbol,
        balance: this.state.balance,
        openTrade: this.state.openTrade,
        trades: this.state.trades.slice(-100),  // Keep last 100
        cooldownUntil: this.state.cooldownUntil,
        stats: this.state.stats,
        savedAt: new Date().toISOString(),
      };

      fs.writeFileSync(this.stateFile, JSON.stringify(toSave, null, 2));
    } catch (e) {
      console.error(`  ${this.state.symbol}: Failed to save state:`, e);
    }
  }

  async initialize(client: any): Promise<void> {
    this.lgbmPredictor.load();
    this.loadState();

    // Fetch initial candles
    for (const interval of CONFIG.intervals) {
      await this.fetchCandles(client, interval);
    }
  }

  async fetchCandles(client: any, interval: string): Promise<void> {
    const tf = this.state.timeframes.get(interval);
    if (!tf) return;

    const now = Date.now();
    const refreshMs = CONFIG.refreshMsByInterval[interval] || 15000;

    if (now - tf.lastUpdate < refreshMs) return;

    try {
      const rawCandles = await client.candles({
        symbol: this.state.symbol,
        interval,
        limit: 100,
      });

      tf.candles = rawCandles.map((c: any) => ({
        timestamp: c.openTime,
        open: parseFloat(c.open),
        high: parseFloat(c.high),
        low: parseFloat(c.low),
        close: parseFloat(c.close),
        volume: parseFloat(c.volume),
      }));

      tf.momentum = analyzeMomentum(tf.candles);
      tf.lastUpdate = now;
    } catch (e: any) {
      if (!e.message?.includes('ENOTFOUND')) {
        console.error(`  ${this.state.symbol}/${interval}: Fetch error:`, e.message);
      }
    }
  }

  async tick(client: any): Promise<{ status: string; details: string }> {
    // Fetch candles for all timeframes
    for (const interval of CONFIG.intervals) {
      await this.fetchCandles(client, interval);
    }

    const primary = this.state.timeframes.get(CONFIG.primaryInterval);
    if (!primary || primary.candles.length < CONFIG.minCandlesRequired) {
      return { status: 'WAIT', details: 'Insufficient data' };
    }

    const currentPrice = primary.candles[primary.candles.length - 1].close;

    // Check open trade
    if (this.state.openTrade) {
      const result = this.checkOpenTrade(currentPrice);
      if (result.closed) {
        return { status: 'CLOSED', details: result.message };
      }
      return { status: 'HOLDING', details: result.message };
    }

    // Check cooldown
    if (Date.now() < this.state.cooldownUntil) {
      const remaining = Math.round((this.state.cooldownUntil - Date.now()) / 1000);
      return { status: 'COOLDOWN', details: `${remaining}s remaining` };
    }

    // Analyze for entry
    const analysis = this.analyzeForEntry();

    if (analysis.shouldEnter && analysis.direction !== 'NEUTRAL') {
      this.enterTrade({ ...analysis, direction: analysis.direction as 'LONG' | 'SHORT' }, currentPrice);
      return { status: 'ENTERED', details: analysis.reason };
    }

    return { status: 'SCAN', details: `${analysis.direction} (${analysis.signals.join(', ')})` };
  }

  private analyzeForEntry(): {
    shouldEnter: boolean;
    direction: 'LONG' | 'SHORT' | 'NEUTRAL';
    strength: number;
    signals: string[];
    reason: string;
  } {
    const tf5m = this.state.timeframes.get('5m');
    const tf1m = this.state.timeframes.get('1m');
    const tf15m = this.state.timeframes.get('15m');

    if (!tf5m || tf5m.candles.length < CONFIG.minCandlesRequired) {
      return { shouldEnter: false, direction: 'NEUTRAL', strength: 0, signals: [], reason: 'No data' };
    }

    const m5 = tf5m.momentum;
    const m1 = tf1m?.momentum;
    const m15 = tf15m?.momentum;

    const signals: string[] = [];

    // Collect active signals
    if (m5.volumeSpike) signals.push(`VOL:${m5.volumeRatio.toFixed(1)}x`);
    if (m5.rsiBullishCross) signals.push('RSI↑50');
    if (m5.rsiBearishCross) signals.push('RSI↓50');
    if (m5.rsiOversold) signals.push('RSI<30');
    if (m5.rsiOverbought) signals.push('RSI>70');
    if (m5.emaBullishCross) signals.push('EMA↑');
    if (m5.emaBearishCross) signals.push('EMA↓');
    if (m5.bbBreakoutUp) signals.push('BB↑');
    if (m5.bbBreakoutDown) signals.push('BB↓');
    if (m5.priceBreakoutUp) signals.push('BRK↑');
    if (m5.priceBreakoutDown) signals.push('BRK↓');
    if (m5.candleMomentum !== 'neutral') signals.push(`CANDLE:${m5.candleMomentum[0].toUpperCase()}`);

    // Check direction
    const direction = m5.direction;

    if (direction === 'NEUTRAL') {
      return { shouldEnter: false, direction: 'NEUTRAL', strength: 0, signals, reason: 'No clear direction' };
    }

    // Confirm with 1m (optional boost)
    let strengthBoost = 0;
    if (m1 && m1.direction === direction) {
      strengthBoost += 0.1;
      signals.push('1m✓');
    }

    // Confirm with 15m trend (optional boost)
    if (m15 && m15.emaAligned === (direction === 'LONG' ? 'bullish' : 'bearish')) {
      strengthBoost += 0.1;
      signals.push('15m✓');
    }

    const strength = Math.min(1, m5.strength + strengthBoost);

    // Need minimum signals
    const signalCount = direction === 'LONG' ? m5.bullishSignals : m5.bearishSignals;
    if (signalCount < CONFIG.momentum.minSignals) {
      return {
        shouldEnter: false,
        direction,
        strength,
        signals,
        reason: `Only ${signalCount}/${CONFIG.momentum.minSignals} signals`
      };
    }

    // ═══════════════════════════════════════════════════════════════
    // VWAP & KILL ZONE CONTEXT
    // ═══════════════════════════════════════════════════════════════
    const vwapConf = direction === 'LONG' ? m5.priceAboveVwap : !m5.priceAboveVwap;
    if (vwapConf) signals.push('VWAP✓');
    if (Math.abs(m5.vwapDeviationStd) > 1.5) signals.push(`VWAP:${m5.vwapDeviationStd.toFixed(1)}σ`);
    if (m5.isKillZone) signals.push(`KZ:${m5.killZone}`);

    // ═══════════════════════════════════════════════════════════════
    // DUAL MODE: TREND vs RANGE
    // ═══════════════════════════════════════════════════════════════
    const regime = m5.regime;
    const bbPos = m5.bbPosition;
    signals.push(`${regime}`);
    signals.push(`BB:${(bbPos * 100).toFixed(0)}%`);

    if (regime === 'TREND') {
      // ═══════════════════════════════════════════════════════════════
      // TREND MODE: High volatility - ride breakouts
      // Entry: MACD crossover + EMA alignment + VWAP confirmation
      // Prefer Kill Zone timing for institutional activity
      // ═══════════════════════════════════════════════════════════════
      signals.push(`ATR:${(m5.atrPercent * 100).toFixed(2)}%`);

      const hasMacdConfirm = direction === 'LONG'
        ? (m5.macdBullishCross || m5.macdHistogram > 0)
        : (m5.macdBearishCross || m5.macdHistogram < 0);

      const hasEmaAlign = direction === 'LONG'
        ? m5.emaAligned === 'bullish'
        : m5.emaAligned === 'bearish';

      const hasBreakout = direction === 'LONG'
        ? (m5.priceBreakoutUp || m5.bbBreakoutUp)
        : (m5.priceBreakoutDown || m5.bbBreakoutDown);

      // VWAP confirmation: price should be on correct side of VWAP
      const hasVwapConfirm = direction === 'LONG' ? m5.priceAboveVwap : !m5.priceAboveVwap;

      // TREND mode: need MACD + (EMA or Breakout) + preferably volume
      if (!hasMacdConfirm) {
        return {
          shouldEnter: false,
          direction,
          strength,
          signals,
          reason: `TREND: Need MACD confirm`
        };
      }

      if (!hasEmaAlign && !hasBreakout) {
        return {
          shouldEnter: false,
          direction,
          strength,
          signals,
          reason: `TREND: Need EMA align or breakout`
        };
      }

      // Require VWAP alignment for cleaner entries
      if (!hasVwapConfirm) {
        return {
          shouldEnter: false,
          direction,
          strength,
          signals,
          reason: `TREND: Wrong side of VWAP`
        };
      }

      // Extra confidence with volume
      if (m5.volumeSpike) signals.push('VOL✓');
      if (m5.macdBullishCross) signals.push('MACD↑');
      if (m5.macdBearishCross) signals.push('MACD↓');

      // Kill zone bonus (prefer but don't require)
      const kzBonus = m5.isKillZone ? '+KZ' : '';

      return {
        shouldEnter: true,
        direction,
        strength: m5.isKillZone ? Math.min(1, strength + 0.1) : strength,
        signals,
        reason: `TREND ${direction} (MACD+${hasEmaAlign ? 'EMA' : 'BRK'}+VWAP${kzBonus})`
      };

    } else {
      // ═══════════════════════════════════════════════════════════════
      // RANGE MODE: Low volatility - mean reversion at BB extremes
      // Entry: BB extreme + VWAP deviation + direction confirm
      // Based on 443 trade analysis:
      // - LONG only at lower BB (< 0.4) = profitable
      // - SHORT only at upper BB (> 0.6) = profitable
      // - Skip middle zone (0.4-0.6) = losing trades
      // VWAP deviation >1σ = extra confluence for mean reversion
      // ═══════════════════════════════════════════════════════════════

      // Skip middle zone - no edge
      if (bbPos >= 0.4 && bbPos <= 0.6) {
        return {
          shouldEnter: false,
          direction,
          strength,
          signals,
          reason: `RANGE: BB middle (${(bbPos * 100).toFixed(0)}%) - skip`
        };
      }

      // LONG only at lower BB (mean reversion: expect bounce up)
      if (direction === 'LONG' && bbPos > 0.4) {
        return {
          shouldEnter: false,
          direction,
          strength,
          signals,
          reason: `RANGE: LONG needs BB<40%`
        };
      }

      // SHORT only at upper BB (mean reversion: expect drop)
      if (direction === 'SHORT' && bbPos < 0.6) {
        return {
          shouldEnter: false,
          direction,
          strength,
          signals,
          reason: `RANGE: SHORT needs BB>60%`
        };
      }

      // VWAP deviation adds confluence (extended = better reversion opportunity)
      // For LONG at lower BB, price should be below VWAP (oversold vs VWAP)
      // For SHORT at upper BB, price should be above VWAP (overbought vs VWAP)
      const vwapExtended = Math.abs(m5.vwapDeviationStd) > 1.0;
      const vwapCorrectSide = direction === 'LONG' ? m5.vwapDeviationStd < 0 : m5.vwapDeviationStd > 0;
      const vwapBonus = vwapExtended && vwapCorrectSide;

      // Kill zone timing (optional but preferred)
      const kzBonus = m5.isKillZone ? '+KZ' : '';

      // Passed RANGE filters - take the trade!
      const bbZone = bbPos < 0.4 ? 'lower' : 'upper';
      return {
        shouldEnter: true,
        direction,
        strength: (vwapBonus || m5.isKillZone) ? Math.min(1, strength + 0.1) : strength,
        signals,
        reason: `RANGE ${direction} @ BB ${bbZone}${vwapBonus ? '+VWAP' : ''}${kzBonus}`
      };
    }
  }

  private enterTrade(analysis: { direction: 'LONG' | 'SHORT'; strength: number; signals: string[] }, currentPrice: number): void {
    const isLong = analysis.direction === 'LONG';
    const tf5m = this.state.timeframes.get('5m');
    const momentum = tf5m?.momentum;

    // ═══════════════════════════════════════════════════════════════
    // STRUCTURE-BASED STOPS: Use swing high/low instead of fixed %
    // ═══════════════════════════════════════════════════════════════
    let stopLoss: number;
    let stopDistance: number;

    if (isLong && momentum?.swingLow) {
      // LONG: Stop below recent swing low (with small buffer)
      stopLoss = momentum.swingLow * 0.999;
      stopDistance = currentPrice - stopLoss;
    } else if (!isLong && momentum?.swingHigh) {
      // SHORT: Stop above recent swing high (with small buffer)
      stopLoss = momentum.swingHigh * 1.001;
      stopDistance = stopLoss - currentPrice;
    } else {
      // Fallback to fixed % if no swing points found
      stopDistance = currentPrice * (CONFIG.targets.stopLossPct / 100);
      stopLoss = isLong ? currentPrice - stopDistance : currentPrice + stopDistance;
    }

    // Skip if structure stop is too far (> 2% risk) or too tight (< 0.1%)
    const riskPct = (stopDistance / currentPrice) * 100;
    if (riskPct > 2.0) {
      console.log(`  ${this.state.symbol}: Skip - structure stop too far (${riskPct.toFixed(2)}% risk)`);
      return;
    }
    if (riskPct < 0.1) {
      console.log(`  ${this.state.symbol}: Skip - structure stop too tight (${riskPct.toFixed(2)}% risk)`);
      return;
    }

    // Target: middle Bollinger Band (mean reversion target)
    const bbMiddle = momentum?.bbMiddle || currentPrice;
    const tpDistance = Math.abs(bbMiddle - currentPrice);

    // If target is too small, use minimum R:R of 1.5:1
    const minTpDistance = stopDistance * 1.5;
    const actualTpDistance = Math.max(tpDistance, minTpDistance);

    const takeProfit1 = isLong ? currentPrice + actualTpDistance * 0.5 : currentPrice - actualTpDistance * 0.5;
    const takeProfit2 = isLong ? currentPrice + actualTpDistance * 0.75 : currentPrice - actualTpDistance * 0.75;
    const takeProfit3 = isLong ? currentPrice + actualTpDistance : currentPrice - actualTpDistance;

    // Position sizing based on risk (with leverage cap)
    const riskAmount = this.state.balance * (CONFIG.riskPerTradePct / 100);
    let positionSize = riskAmount / stopDistance;

    // Cap notional to max leverage × balance to prevent huge positions with tight stops
    const maxNotional = this.state.balance * CONFIG.leverage;
    const uncappedNotional = currentPrice * positionSize;
    if (uncappedNotional > maxNotional) {
      positionSize = maxNotional / currentPrice;
      console.log(`  ${this.state.symbol}: Position capped to ${CONFIG.leverage}x leverage`);
    }

    // Apply slippage
    const slippage = currentPrice * (CONFIG.slippageBps / 10000);
    const fillPrice = isLong ? currentPrice + slippage : currentPrice - slippage;

    // Entry fee
    const notional = fillPrice * positionSize;
    const entryFee = notional * CONFIG.takerFeeRate;
    this.state.balance -= entryFee;

    const trade: PaperTrade = {
      id: `${this.state.symbol}-${Date.now()}`,
      symbol: this.state.symbol,
      direction: analysis.direction,
      entryPrice: fillPrice,
      entryTime: Date.now(),
      stopLoss,
      originalStopLoss: stopLoss,
      takeProfit1,
      takeProfit2,
      takeProfit3,
      trailingStop: null,
      originalPositionSize: positionSize,
      currentPositionSize: positionSize,
      tp1Hit: false,
      tp2Hit: false,
      tp3Hit: false,
      stopLossMovedToBreakeven: false,
      status: 'OPEN',
      feesPaid: entryFee,
      momentumStrength: analysis.strength,
      signals: analysis.signals,
    };

    this.state.openTrade = trade;
    this.state.trades.push(trade);
    this.saveState();

    const swingInfo = momentum?.swingLow || momentum?.swingHigh
      ? `STRUCTURE (swing ${isLong ? 'low' : 'high'})`
      : 'FIXED %';

    console.log(`\n⚡ ${this.state.symbol}: SCALP ${trade.direction} [${swingInfo}]`);
    console.log(`   Entry: $${fillPrice.toFixed(4)} | SL: $${stopLoss.toFixed(4)} (${riskPct.toFixed(2)}% risk)`);
    console.log(`   TP1: $${takeProfit1.toFixed(4)} | TP2: $${takeProfit2.toFixed(4)} | TP3: $${takeProfit3.toFixed(4)}`);
    console.log(`   Signals: ${analysis.signals.join(', ')}`);
    console.log(`   Strength: ${(analysis.strength * 100).toFixed(0)}% | Size: ${positionSize.toFixed(4)}`);
  }

  private checkOpenTrade(currentPrice: number): { closed: boolean; message: string } {
    const trade = this.state.openTrade!;
    const isLong = trade.direction === 'LONG';

    const priceDiff = isLong ? currentPrice - trade.entryPrice : trade.entryPrice - currentPrice;
    const pnlPercent = (priceDiff / trade.entryPrice) * 100;
    const pnl = priceDiff * trade.currentPositionSize;

    // Check if position is flat (all TPs hit)
    if (trade.currentPositionSize <= 0.0000001) {
      trade.status = 'CLOSED';
      this.state.openTrade = null;
      this.saveState();
      return { closed: true, message: 'Position closed (flat)' };
    }

    // Check timeout
    const holdTime = Date.now() - trade.entryTime;
    const maxHoldMs = CONFIG.maxHoldMinutes * 60 * 1000;

    if (holdTime > maxHoldMs) {
      return this.closeTrade(currentPrice, 'TIMEOUT', pnl, pnlPercent);
    }

    // Check stop loss
    if ((isLong && currentPrice <= trade.stopLoss) || (!isLong && currentPrice >= trade.stopLoss)) {
      return this.closeTrade(trade.stopLoss, 'SL', pnl, pnlPercent);
    }

    // Check trailing stop
    if (trade.trailingStop !== null) {
      if ((isLong && currentPrice <= trade.trailingStop) || (!isLong && currentPrice >= trade.trailingStop)) {
        return this.closeTrade(trade.trailingStop, 'TRAILING', pnl, pnlPercent);
      }
    }

    // Check TP1 (50% close)
    if (!trade.tp1Hit) {
      if ((isLong && currentPrice >= trade.takeProfit1) || (!isLong && currentPrice <= trade.takeProfit1)) {
        trade.tp1Hit = true;
        const closeAmount = trade.originalPositionSize * 0.50;
        this.closePartialTrade(currentPrice, closeAmount, 'TP1');
        // Move stop to breakeven
        trade.stopLoss = trade.entryPrice;
        trade.stopLossMovedToBreakeven = true;
        return { closed: false, message: `TP1 HIT (+50%) | SL→BE` };
      }
    }

    // Check TP2 (25% close)
    if (!trade.tp2Hit && trade.tp1Hit) {
      if ((isLong && currentPrice >= trade.takeProfit2) || (!isLong && currentPrice <= trade.takeProfit2)) {
        trade.tp2Hit = true;
        const closeAmount = trade.originalPositionSize * 0.25;
        this.closePartialTrade(currentPrice, closeAmount, 'TP2');
        return { closed: false, message: `TP2 HIT (+25%)` };
      }
    }

    // Check TP3 (final 25% close)
    if (!trade.tp3Hit && trade.tp2Hit) {
      if ((isLong && currentPrice >= trade.takeProfit3) || (!isLong && currentPrice <= trade.takeProfit3)) {
        trade.tp3Hit = true;
        const closeAmount = trade.currentPositionSize; // Close remaining
        this.closePartialTrade(currentPrice, closeAmount, 'TP3');
        trade.status = 'CLOSED';
        this.state.openTrade = null;
        this.saveState();
        return { closed: true, message: `TP3 HIT (closed)` };
      }
    }

    // Update trailing stop if past TP1
    if (trade.tp1Hit) {
      const trailDistance = currentPrice * (CONFIG.targets.trailingDistancePct / 100);
      const newTrailing = isLong ? currentPrice - trailDistance : currentPrice + trailDistance;

      if (trade.trailingStop === null ||
          (isLong && newTrailing > trade.trailingStop) ||
          (!isLong && newTrailing < trade.trailingStop)) {
        trade.trailingStop = newTrailing;
      }
    }

    const tpStatus = [trade.tp1Hit ? 'TP1' : '', trade.tp2Hit ? 'TP2' : ''].filter(Boolean).join('+') || 'HOLD';
    const trailInfo = trade.trailingStop ? ` TRAIL:$${trade.trailingStop.toFixed(4)}` : '';
    return {
      closed: false,
      message: `${tpStatus}${trailInfo} | PnL: ${pnlPercent >= 0 ? '+' : ''}${pnlPercent.toFixed(2)}%`
    };
  }

  private closePartialTrade(exitPrice: number, closeAmount: number, reason: 'TP1' | 'TP2' | 'TP3'): void {
    const trade = this.state.openTrade!;
    const isLong = trade.direction === 'LONG';

    // Apply slippage
    const slippage = exitPrice * (CONFIG.slippageBps / 10000);
    const fillPrice = isLong ? exitPrice - slippage : exitPrice + slippage;

    // Exit fee for partial
    const notional = fillPrice * closeAmount;
    const exitFee = notional * CONFIG.takerFeeRate;

    // Calculate PnL for this partial
    const priceDiff = isLong ? fillPrice - trade.entryPrice : trade.entryPrice - fillPrice;
    const partialPnl = (priceDiff * closeAmount) - exitFee;
    const partialPnlPercent = (priceDiff / trade.entryPrice) * 100;

    // Update position
    trade.currentPositionSize -= closeAmount;
    trade.feesPaid += exitFee;

    // Update stats
    this.state.balance += partialPnl;
    this.state.stats.totalPnl += partialPnl;
    this.state.stats.totalTrades++;
    if (partialPnl > 0) {
      this.state.stats.wins++;
    } else {
      this.state.stats.losses++;
    }
    this.state.stats.winRate = this.state.stats.wins / Math.max(1, this.state.stats.totalTrades);

    const pnlSign = partialPnl >= 0 ? '+' : '';
    console.log(`✅ ${trade.symbol}: ${reason} HIT | ${pnlSign}$${partialPnl.toFixed(2)} (${pnlSign}${partialPnlPercent.toFixed(2)}%) | Remaining: ${(trade.currentPositionSize / trade.originalPositionSize * 100).toFixed(0)}%`);

    this.saveState();
  }

  private closeTrade(exitPrice: number, reason: PaperTrade['exitReason'], pnl: number, pnlPercent: number): { closed: boolean; message: string } {
    const trade = this.state.openTrade!;

    // Apply slippage on exit
    const slippage = exitPrice * (CONFIG.slippageBps / 10000);
    const fillPrice = trade.direction === 'LONG' ? exitPrice - slippage : exitPrice + slippage;

    // Exit fee
    const notional = fillPrice * trade.currentPositionSize;
    const exitFee = notional * CONFIG.takerFeeRate;

    // Final PnL
    const isLong = trade.direction === 'LONG';
    const actualPriceDiff = isLong ? fillPrice - trade.entryPrice : trade.entryPrice - fillPrice;
    const finalPnl = (actualPriceDiff * trade.currentPositionSize) - exitFee;
    const finalPnlPercent = (actualPriceDiff / trade.entryPrice) * 100;

    trade.exitPrice = fillPrice;
    trade.exitTime = Date.now();
    trade.exitReason = reason;
    trade.pnl = finalPnl;
    trade.pnlPercent = finalPnlPercent;
    trade.feesPaid += exitFee;
    trade.status = 'CLOSED';

    // Update balance and stats
    this.state.balance += finalPnl;
    this.state.stats.totalTrades++;
    this.state.stats.totalPnl += finalPnl;

    if (finalPnl > 0) {
      this.state.stats.wins++;
    } else {
      this.state.stats.losses++;
    }

    this.state.stats.winRate = this.state.stats.totalTrades > 0
      ? (this.state.stats.wins / this.state.stats.totalTrades) * 100
      : 0;

    // Set cooldown
    this.state.cooldownUntil = Date.now() + CONFIG.cooldownMs;

    this.state.openTrade = null;
    this.saveState();

    const emoji = finalPnl > 0 ? '✅' : '❌';
    const pnlSign = finalPnl >= 0 ? '+' : '';
    const holdMins = Math.round((trade.exitTime - trade.entryTime) / 60000);

    console.log(`\n${emoji} ${this.state.symbol}: CLOSED ${reason}`);
    console.log(`   PnL: ${pnlSign}$${finalPnl.toFixed(2)} (${pnlSign}${finalPnlPercent.toFixed(2)}%)`);
    console.log(`   Held: ${holdMins}m | Fees: $${trade.feesPaid.toFixed(2)}`);
    console.log(`   Stats: ${this.state.stats.wins}W/${this.state.stats.losses}L (${this.state.stats.winRate.toFixed(1)}%)`);
    console.log(`   Balance: $${this.state.balance.toFixed(2)}\n`);

    return { closed: true, message: `${reason} ${pnlSign}${finalPnlPercent.toFixed(2)}%` };
  }
}

// ═══════════════════════════════════════════════════════════════
// MAIN TRADING LOOP
// ═══════════════════════════════════════════════════════════════

async function main() {
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('     STRUCTURE-BASED SCALPER - Paper Trading');
  console.log('═══════════════════════════════════════════════════════════════');
  console.log(`Mode: ${CONFIG.mode}`);
  console.log(`Dual Mode: TREND (breakouts) / RANGE (BB reversion)`);
  console.log(`Primary TF: ${CONFIG.primaryInterval}`);
  console.log(`STOPS: STRUCTURE-BASED (swing high/low, max 2% risk)`);
  console.log(`TARGET: Middle Bollinger Band (min 1.5:1 R:R)`);
  console.log(`Cooldown: ${CONFIG.cooldownMs / 1000}s`);
  console.log(`Auto-Learn: Every ${CONFIG.autoLearn.triggerEveryNTrades} trades`);
  console.log(`Symbols: ${SYMBOLS.length}`);
  console.log('═══════════════════════════════════════════════════════════════\n');

  const client = Binance();

  // Initialize traders
  const traders: CoinTrader[] = [];
  for (const symbol of SYMBOLS) {
    const trader = new CoinTrader(symbol);
    await trader.initialize(client);
    traders.push(trader);
  }

  console.log(`Initialized ${traders.length} traders\n`);
  console.log('Starting momentum scalper loop...\n');

  // Helper for price display
  const getDecimalPlaces = (p: number): number => {
    if (p >= 1000) return 2;
    if (p >= 100) return 2;
    if (p >= 10) return 3;
    if (p >= 1) return 4;
    return 5;
  };

  // Main loop
  let iteration = 0;

  while (true) {
    iteration++;
    const now = new Date().toLocaleTimeString();

    // Clear screen and show header
    process.stdout.write('\x1B[2J\x1B[0f');
    console.log('\n╔═══════════════════════════════════════════════════════════════╗');
    console.log(`║  DUAL-MODE SCALPER - Cycle: ${iteration} | ${now}            ║`);
    console.log('╚═══════════════════════════════════════════════════════════════╝\n');

    // Process each trader and collect results
    const results: Array<{ symbol: string; price: number; result: any; trader: typeof traders[0] }> = [];

    for (const trader of traders) {
      try {
        const tf5m = trader.state.timeframes.get('5m');
        const price = tf5m?.candles?.length ? tf5m.candles[tf5m.candles.length - 1].close : 0;
        const result = await trader.tick(client);
        results.push({ symbol: trader.state.symbol, price, result, trader });
      } catch (e: any) {
        results.push({
          symbol: trader.state.symbol,
          price: 0,
          result: { status: 'ERROR', details: e.message?.slice(0, 30) || 'Unknown error' },
          trader
        });
      }
    }

    // Display each coin status
    for (const { symbol, price, result, trader } of results) {
      const priceDisplay = price > 0 ? `$${price.toFixed(getDecimalPlaces(price))}` : 'N/A';
      let statusLine = `${symbol.padEnd(10)}: ${priceDisplay.padEnd(14)} | `;

      if (trader.state.openTrade) {
        const trade = trader.state.openTrade;
        const isLong = trade.direction === 'LONG';
        const currentPrice = price > 0 ? price : trade.entryPrice;  // Fallback to entry if no price
        const priceDiff = isLong ? currentPrice - trade.entryPrice : trade.entryPrice - currentPrice;
        const unrealizedPnl = priceDiff * (trade.currentPositionSize || 1);
        const pnlPercent = trade.entryPrice > 0 ? (priceDiff / trade.entryPrice) * 100 : 0;
        const pnlSign = unrealizedPnl >= 0 ? '+' : '';
        const pricePrecision = getDecimalPlaces(trade.entryPrice);
        statusLine += `OPEN ${trade.direction.padEnd(5)} | ${pnlSign}$${unrealizedPnl.toFixed(2)} (${pnlSign}${pnlPercent.toFixed(2)}%) | SL:$${trade.stopLoss.toFixed(pricePrecision)} TP:$${trade.takeProfit3.toFixed(pricePrecision)}`;
      } else {
        statusLine += `${result.status.padEnd(10)} | ${result.details}`;
      }

      console.log(statusLine);
    }

    // Summary
    const openCount = traders.filter(t => t.state.openTrade).length;
    const totalPnl = traders.reduce((sum, t) => sum + t.state.stats.totalPnl, 0);
    const totalTrades = traders.reduce((sum, t) => sum + t.state.stats.totalTrades, 0);
    const totalWins = traders.reduce((sum, t) => sum + t.state.stats.wins, 0);
    const winRate = totalTrades > 0 ? (totalWins / totalTrades * 100).toFixed(1) : '0.0';
    const pnlSign = totalPnl >= 0 ? '+' : '';

    console.log('\n═══════════════════════════════════════════════════════════════');
    console.log(`📊 SUMMARY: Open: ${openCount} | Trades: ${totalTrades} (${totalWins}W) | Win: ${winRate}% | PnL: ${pnlSign}$${totalPnl.toFixed(2)}`);
    console.log('═══════════════════════════════════════════════════════════════\n');

    // ═══════════════════════════════════════════════════════════════
    // AUTO-LEARNING: Trigger retraining every N trades
    // ═══════════════════════════════════════════════════════════════
    if (CONFIG.autoLearn.enabled && totalTrades > 0) {
      const tradesAtLastLearn = (global as any).__lastLearnedAt || 0;
      const tradesSinceLearn = totalTrades - tradesAtLastLearn;

      if (tradesSinceLearn >= CONFIG.autoLearn.triggerEveryNTrades && totalTrades >= CONFIG.autoLearn.minTradesForTraining) {
        console.log(`\n🧠 AUTO-LEARN: ${tradesSinceLearn} new trades - triggering learning loop...`);
        (global as any).__lastLearnedAt = totalTrades;

        try {
          const { execSync } = await import('child_process');
          const cwd = process.cwd();

          // Export trades
          console.log('   Exporting trades...');
          execSync('npm run export-paper-trades-scalp', { cwd, stdio: 'pipe' });

          // Find the latest export file
          const exportDir = path.join(cwd, 'data', 'h2o-training');
          const files = fs.readdirSync(exportDir)
            .filter(f => f.startsWith('paper_scalp_') && f.endsWith('.csv'))
            .sort()
            .reverse();

          if (files.length > 0) {
            const latestFile = path.join(exportDir, files[0]);
            console.log(`   Training on: ${files[0]}`);

            // Run walk-forward training and capture output
            const trainOutput = execSync(`python scripts/lightgbm_walkforward.py --input "${latestFile}"`, {
              cwd,
              encoding: 'utf-8',
              timeout: 300000  // 5 min timeout
            });

            // Parse and display key metrics from training output
            const lines = trainOutput.split('\n');
            let showOutput = false;
            for (const line of lines) {
              // Show summary lines
              if (line.includes('WALK-FORWARD') ||
                  line.includes('Optimal threshold') ||
                  line.includes('Baseline PnL') ||
                  line.includes('Filtered PnL') ||
                  line.includes('Win rate') ||
                  line.includes('Model saved') ||
                  line.includes('improvement')) {
                console.log(`   ${line.trim()}`);
                showOutput = true;
              }
            }

            if (!showOutput) {
              console.log('   ✅ Training completed (no improvement over current model)');
            }
            console.log('');
          }
        } catch (e: any) {
          console.log(`   ⚠️ Learning failed: ${e.message?.slice(0, 50)}\n`);
        }
      }
    }

    // Wait before next check
    await new Promise(resolve => setTimeout(resolve, CONFIG.checkIntervalMs));
  }
}

main().catch(console.error);
