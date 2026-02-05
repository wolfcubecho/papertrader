/**
 * Trade Feature Extractor
 * Extracts detailed features from each trade setup for ML training
 *
 * Includes ICT (Inner Circle Trader) institutional features:
 * - OTE Zone (Optimal Trade Entry)
 * - Displacement Detection
 * - Liquidity Grabs
 * - Judas Swings
 * - Killzone Timing
 */

import { Candle } from './smc-indicators.js';
import { SMCAnalysis, SMCIndicators } from './smc-indicators.js';
import { ICTIndicators, ICTAnalysis } from './ict-indicators.js';
import fs from 'fs';

// Trade outcome data (for adding to features after backtest)
export interface BacktestTrade {
  outcome: 'WIN' | 'LOSS';
  pnl: number;
  pnl_percent: number;
  exit_reason: string;
  holding_periods: number;
}
import path from 'path';

// Load configuration
let featureConfig: any;
try {
  const configPath = path.join(process.cwd(), 'config', 'features.json');
  featureConfig = JSON.parse(fs.readFileSync(configPath, 'utf-8'));
} catch (error) {
  console.warn('Warning: Could not load features.json config, using defaults');
  featureConfig = {
    feature_extraction: {
      lookback_periods: { default: 200 },
      thresholds: {
        ob_near_percent: 5,
        ob_distance_max: 1,
        fvg_near_percent: 2,
        fvg_distance_max: 1,
        fvg_size_max: 0.5,
        ob_age_max: 100,
        rsi_overbought: 70,
        rsi_oversold: 30,
        mtf_trend_threshold: 0.5
      }
    }
  };
}

export interface TradeFeatures {
  // Entry conditions
  entry_price: number;
  entry_time: number;
  direction: 'long' | 'short';
  
  // Trend features
  trend_direction: 'up' | 'down' | 'neutral';
  trend_strength: number;  // 0-1, based on SMA slope
  trend_bos_aligned: boolean;  // Does BOS match trend?
  
  // Order Block features
  ob_near: boolean;
  ob_distance: number;  // % away from price
  ob_type: 'bull' | 'bear' | 'none';
  ob_size: number;  // % of candle body
  ob_age: number;  // How many candles ago (0-100+)
  
  // FVG features
  fvg_near: boolean;
  fvg_count: number;  // Total relevant FVGs
  fvg_nearest_distance: number;  // % away
  fvg_size: number;  // Average gap size %
  fvg_type: 'bull' | 'bear' | 'mixed';
  
  // EMA alignment
  ema_aligned: boolean;
  ema_trend: 'up' | 'down' | 'neutral';
  
  // RSI state
  rsi_value: number;
  rsi_state: 'overbought' | 'oversold' | 'neutral';
  
  // Liquidity
  liquidity_near: boolean;
  liquidity_count: number;
  
  // Multi-timeframe
  mtf_aligned: boolean;
  
  // Market state
  volatility: number;  // ATR % of price
  atr_value: number;
  
  // Price position
  price_position: number;  // 0-1, where in recent range
  distance_to_high: number;  // % from recent high
  distance_to_low: number;  // % from recent low
  
  // Volume features
  volume_spike: boolean;  // Is volume 2x+ average?
  volume_ratio: number;  // Current / avg volume
  
  // Confluence
  confluence_score: number;  // 0-1, how many signals align
  confluence_count: number;  // Number of confluence factors
  
  // Market session
  session: 'asian' | 'london' | 'newyork' | 'overlap' | 'off-hours';
  
  // Psychological
  days_since_loss: number;  // Days since last loss (0-100+)
  streak_type: 'win' | 'loss' | 'none';
  
  // Risk/Reward
  potential_rr: number;  // Potential risk/reward ratio

  // Pullback features (key for entry timing)
  is_pullback: boolean;          // Is price in a pullback?
  pullback_depth: number;        // 0-1, how deep the pullback
  pullback_fib: string;          // Which fib level ('0.382', '0.5', '0.618', '0.786', 'none')
  pullback_bars: number;         // How many bars into pullback

  // NEW: SMC Return signals (correct OB/FVG usage)
  has_ob_return: boolean;        // Is there an OB return signal?
  has_fvg_fill: boolean;         // Is there an FVG fill signal?
  smc_signal_strength: number;   // 0-1, strength of best SMC signal
  ob_return_age: number;         // Bars since OB formed (if OB return)
  ob_test_count: number;         // How many times OB tested (0 = fresh = BEST)
  ob_is_fresh: boolean;          // Is this a fresh, untested OB?
  fvg_fill_percent: number;      // How much of FVG is already filled

  // OB State Machine (proper SMC logic)
  ob_state: string;              // NEW_OB, WAITING, IN_MITIGATION, CONFIRMED, INVALIDATED
  ob_confirmed_mitigated: boolean;  // BEST: Rejection + BOS confirmed
  ob_in_mitigation: boolean;     // Price currently testing zone
  ob_caused_bos: boolean;        // Did OB cause break of structure?
  ob_has_displacement: boolean;  // Was there displacement after OB?
  ob_impulse_size: number;       // Size of move after OB (ATR multiples)

  // NEW: Swing timing
  bars_since_swing_high: number; // Bars since recent swing high
  bars_since_swing_low: number;  // Bars since recent swing low

  // ═══════════════════════════════════════════════════════════════
  // ICT (Inner Circle Trader) Institutional Features
  // ═══════════════════════════════════════════════════════════════

  // OTE (Optimal Trade Entry) - 61.8%-79% Fib zone
  ote_in_zone: boolean;           // Is price currently in OTE zone?
  ote_zone_direction: 'bullish' | 'bearish' | 'none';

  // Displacement (Institutional momentum)
  displacement_detected: boolean;
  displacement_direction: 'up' | 'down' | 'none';
  displacement_strength: number;  // ATR multiple of move

  // Liquidity Grab (Stop hunt)
  liquidity_grab_detected: boolean;
  liquidity_grab_type: 'buy_side' | 'sell_side' | 'none';
  liquidity_grab_volume_spike: number;

  // Judas Swing (Asian session fakeout)
  judas_swing_detected: boolean;
  judas_swing_direction: 'bullish' | 'bearish' | 'none';
  judas_swing_complete: boolean;

  // Killzone (Optimal trading hours)
  killzone_active: boolean;
  killzone_name: 'london' | 'new_york' | 'asian' | 'none';

  // Market Structure (BOS/CHoCH)
  market_structure_trend: 'bullish' | 'bearish' | 'ranging';
  bos_detected: boolean;          // Break of Structure
  choch_detected: boolean;        // Change of Character
  in_discount: boolean;           // Price below equilibrium
  in_premium: boolean;            // Price above equilibrium

  // Zone Confluence
  ob_in_ote: boolean;             // Order Block overlaps OTE
  fvg_in_ote: boolean;            // FVG overlaps OTE

  // Market Regime
  efficiency_ratio: number;       // Net move / total move (0-1)
  is_trending_market: boolean;    // Efficiency ratio >= 0.30

  // ═══════════════════════════════════════════════════════════════
  // Bollinger Bands Features
  // ═══════════════════════════════════════════════════════════════
  bb_position: number;            // 0-1, where price is in bands (0=lower, 1=upper)
  bb_width: number;               // Band width as % of price (volatility)
  bb_squeeze: boolean;            // Bands tight (low volatility, breakout coming)
  bb_breakout_upper: boolean;     // Price above upper band
  bb_breakout_lower: boolean;     // Price below lower band

  // ═══════════════════════════════════════════════════════════════
  // Momentum Scalping Features (dual-mode: TREND vs RANGE)
  // ═══════════════════════════════════════════════════════════════
  regime: 'TREND' | 'RANGE';      // Market regime based on ATR volatility
  atr_percent: number;            // ATR as % of price (regime detection)

  // MACD
  macd_line: number;              // Fast EMA - Slow EMA
  macd_signal: number;            // Signal line (EMA of MACD)
  macd_histogram: number;         // MACD - Signal
  macd_bullish_cross: boolean;    // MACD crossed above signal
  macd_bearish_cross: boolean;    // MACD crossed below signal

  // VWAP (Institutional anchor)
  vwap: number;                   // Volume-weighted average price
  vwap_deviation: number;         // % deviation from VWAP
  vwap_deviation_std: number;     // Std deviations from VWAP
  price_above_vwap: boolean;      // Price above VWAP

  // EMA Momentum
  ema_fast: number;               // 9 EMA
  ema_slow: number;               // 21 EMA
  ema_bullish_cross: boolean;     // Fast EMA crossed above slow
  ema_bearish_cross: boolean;     // Fast EMA crossed below slow

  // Body ratio (candle momentum)
  body_ratio: number;             // Body / Range (0-1), high = strong momentum

  // ═══════════════════════════════════════════════════════════════
  // Institutional Flow Features (volume-based detection)
  // ═══════════════════════════════════════════════════════════════
  large_volume_spike: boolean;    // Volume 3x+ average (institutional activity)
  volume_delta: number;           // Buy vs sell pressure estimate
  accumulation_detected: boolean; // High volume + price holding (accumulation)
  distribution_detected: boolean; // High volume + price falling (distribution)
  smart_money_direction: 'buy' | 'sell' | 'neutral';  // Inferred institutional direction

  // ICT Entry Quality
  ict_entry_score: number;        // 0-100 ICT-specific score
  ict_entry_valid: boolean;       // Meets ICT entry criteria

  // HTF Cascade
  htf_cascade_aligned: boolean;   // Multi-timeframe alignment
  htf_cascade_score: number;      // 0-1 alignment score

  // Entry Checklist (12 conditions)
  checklist_pass_rate: number;    // 0-1, how many conditions passed
  checklist_grade: 'A' | 'B' | 'C' | 'D' | 'F';

  // Outcome (for training)
  outcome: 'WIN' | 'LOSS';
  pnl: number;
  pnl_percent: number;
  exit_reason: string;
  holding_periods: number;
}

export class FeatureExtractor {
  /**
   * Extract features from a trade entry point
   * Includes both SMC and ICT institutional features
   */
  static extractFeatures(
    candles: Candle[],
    index: number,
    analysis: SMCAnalysis,
    score: number,
    direction: 'long' | 'short',
    ictAnalysis?: ICTAnalysis
  ): Omit<TradeFeatures, 'outcome' | 'pnl' | 'pnl_percent' | 'exit_reason' | 'holding_periods'> {
    const currentCandle = candles[index];
    const config = featureConfig.feature_extraction;
    const lookback = config.lookback_periods.default;
    const historicalCandles = candles.slice(Math.max(0, index - lookback), index + 1);
    
    // Trend features
    const trend_direction = analysis.trend || 'neutral';
    const trend_strength = this.calculateTrendStrength(historicalCandles);
    const trend_bos_aligned = analysis.bos === analysis.trend;
    
    // Order Block features
    const relevantOBs = analysis.orderBlocks.filter(ob => {
      const obHigh = Math.max(ob.open, ob.close);
      const obLow = Math.min(ob.open, ob.close);
      const nearPercent = 1 + (config.thresholds.ob_near_percent / 100);
      return currentCandle.close >= obLow * (2 - nearPercent) && currentCandle.close <= obHigh * nearPercent;
    });
    
    const bullOBs = relevantOBs.filter(ob => ob.type === 'bull');
    const bearOBs = relevantOBs.filter(ob => ob.type === 'bear');
    const matchingOBs = direction === 'long' ? bullOBs : bearOBs;
    
    const ob_near = matchingOBs.length > 0;
    let ob_distance = 999;
    let ob_size = 0;
    let ob_age = 0;
    let ob_type: 'bull' | 'bear' | 'none' = 'none';
    
    if (matchingOBs.length > 0) {
      const nearestOB = matchingOBs.reduce((nearest, ob) => {
        const dist = direction === 'long'
          ? Math.abs(currentCandle.close - Math.min(ob.open, ob.close))
          : Math.abs(currentCandle.close - Math.max(ob.open, ob.close));
        const nearestDist = direction === 'long'
          ? Math.abs(currentCandle.close - Math.min(nearest.open, nearest.close))
          : Math.abs(currentCandle.close - Math.max(nearest.open, nearest.close));
        return dist < nearestDist ? ob : nearest;
      });
      
      const obHigh = Math.max(nearestOB.open, nearestOB.close);
      const obLow = Math.min(nearestOB.open, nearestOB.close);
      ob_distance = direction === 'long'
        ? Math.abs(currentCandle.close - obLow) / currentCandle.close
        : Math.abs(currentCandle.close - obHigh) / currentCandle.close;
      
      ob_size = Math.abs(nearestOB.close - nearestOB.open) / nearestOB.open;
      ob_age = index - nearestOB.index;
      ob_type = nearestOB.type;
    }
    
    // FVG features
    const relevantFVGs = analysis.fvg.filter((fvg: any) => {
      return currentCandle.close >= fvg.from * 0.98 && currentCandle.close <= fvg.to * 1.02;
    });
    
    const bullFVGs = relevantFVGs.filter((fvg: any) => fvg.type === 'bull');
    const bearFVGs = relevantFVGs.filter((fvg: any) => fvg.type === 'bear');
    const matchingFVGs = direction === 'long' ? bullFVGs : bearFVGs;
    
    const fvg_near = relevantFVGs.length > 0;
    const fvg_count = relevantFVGs.length;
    
    let fvg_nearest_distance = 999;
    let fvg_size = 0;
    let fvg_type: 'bull' | 'bear' | 'mixed' = 'mixed';
    
    if (matchingFVGs.length > 0) {
      const nearestFVG = matchingFVGs[0];
      fvg_nearest_distance = Math.abs(currentCandle.close - nearestFVG.from) / currentCandle.close;
      fvg_size = (nearestFVG.to - nearestFVG.from) / nearestFVG.from;
      fvg_type = direction === 'long' ? 'bull' : 'bear';
    }
    
    // EMA alignment
    const ema_aligned: boolean = !!(analysis.ema50 && analysis.ema200 && 
      ((direction === 'long' && analysis.ema50 > analysis.ema200) ||
       (direction === 'short' && analysis.ema50 < analysis.ema200)));
    const ema_trend = analysis.ema50 && analysis.ema200 
      ? (analysis.ema50 > analysis.ema200 ? 'up' : 'down')
      : 'neutral';
    
    // RSI state
    const rsi_value = analysis.rsi || 50;
    let rsi_state: 'overbought' | 'oversold' | 'neutral' = 'neutral';
    if (rsi_value > 70) rsi_state = 'overbought';
    else if (rsi_value < 30) rsi_state = 'oversold';
    
    // Liquidity
    const liquidity_near = analysis.liquidityZones.highs.length > 0 || 
                         analysis.liquidityZones.lows.length > 0;
    const liquidity_count = analysis.liquidityZones.highs.length + 
                         analysis.liquidityZones.lows.length;
    
    // Multi-timeframe alignment check
    const mtf_aligned: boolean = this.checkMTFAlignment(
      trend_direction,
      direction,
      currentCandle,
      historicalCandles
    );
    
    // Market state
    const atr = analysis.atr || (currentCandle.high - currentCandle.low);
    const volatility = atr / currentCandle.close;
    
    // Price position (where in recent range)
    const recentHigh = Math.max(...historicalCandles.slice(-50).map(c => c.high));
    const recentLow = Math.min(...historicalCandles.slice(-50).map(c => c.low));
    const range = recentHigh - recentLow;
    const price_position = range > 0 ? (currentCandle.close - recentLow) / range : 0.5;
    const distance_to_high = range > 0 ? (recentHigh - currentCandle.close) / currentCandle.close : 0;
    const distance_to_low = range > 0 ? (currentCandle.close - recentLow) / currentCandle.close : 0;
    
    // Volume features
    const recentVolumes = historicalCandles.slice(-20).map(c => c.volume);
    const avgVolume = recentVolumes.reduce((sum, v) => sum + v, 0) / recentVolumes.length;
    const volume_ratio = avgVolume > 0 ? currentCandle.volume / avgVolume : 1;
    const volume_spike = volume_ratio >= 2;
    
    // Confluence score
    const confluenceFactors = [
      ob_near,
      fvg_near,
      ema_aligned,
      liquidity_near,
      mtf_aligned,
      trend_bos_aligned
    ].filter(Boolean);
    const confluence_count = confluenceFactors.length;
    const confluence_score = Math.min(confluence_count / 6, 1);
    
    // Market session
    const hour = new Date(currentCandle.timestamp).getUTCHours();
    let session: 'asian' | 'london' | 'newyork' | 'overlap' | 'off-hours' = 'off-hours';
    if (hour >= 0 && hour < 6) session = 'asian';
    else if (hour >= 6 && hour < 8) session = 'london';
    else if (hour >= 8 && hour < 12) session = 'overlap';
    else if (hour >= 12 && hour < 16) session = 'newyork';
    else if (hour >= 16 && hour < 20) session = 'newyork';
    
    // Psychological (simplified - would need trade history)
    const days_since_loss = 0;  // Would track from trade journal
    const streak_type: 'win' | 'loss' | 'none' = 'none';
    
    // Risk/Reward potential (based on ATR and nearest resistance)
    const potential_rr = 2;  // Default 1:2 RR, would calculate from actual levels

    // Pullback features (crucial for entry timing)
    const is_pullback = analysis.pullback?.isPullback || false;
    const pullback_depth = analysis.pullback?.pullbackDepth || 0;
    const pullback_fib = analysis.pullback?.fibLevel || 'none';
    const pullback_bars = analysis.pullback?.pullbackBars || 0;

    // SMC Return signals (correct OB/FVG usage)
    const bestSignal = analysis.bestSignal;
    const has_ob_return = bestSignal?.type === 'ob_return' || false;
    const has_fvg_fill = bestSignal?.type === 'fvg_fill' || false;
    const smc_signal_strength = bestSignal?.strength || 0;

    // Get OB details from the signal (state machine fields)
    let ob_return_age = 0;
    let ob_test_count = 0;
    let ob_is_fresh = false;
    let fvg_fill_percent = 0;
    let ob_state = 'none';
    let ob_confirmed_mitigated = false;
    let ob_in_mitigation = false;
    let ob_caused_bos = false;
    let ob_has_displacement = false;
    let ob_impulse_size = 0;

    if (bestSignal && bestSignal.zone) {
      if (bestSignal.type === 'ob_return') {
        const ob = bestSignal.zone as any;
        ob_return_age = ob.barsAgo || 0;
        ob_test_count = ob.testCount || 0;
        ob_is_fresh = ob_test_count === 0;

        // State machine fields
        ob_state = ob.state || 'unknown';
        ob_confirmed_mitigated = ob.state === 'CONFIRMED_MITIGATED';
        ob_in_mitigation = ob.state === 'IN_MITIGATION';
        ob_caused_bos = ob.causedBOS || false;
        ob_has_displacement = ob.hasDisplacement || false;
        ob_impulse_size = ob.impulseSize || 0;
      }
      if (bestSignal.type === 'fvg_fill' && 'fillPercent' in bestSignal.zone) {
        fvg_fill_percent = (bestSignal.zone as any).fillPercent || 0;
      }
    }

    // ═══════════════════════════════════════════════════════════════
    // Bollinger Bands Calculation (20-period, 2 std dev)
    // ═══════════════════════════════════════════════════════════════
    const bbPeriod = 20;
    const bbStdDev = 2;
    let bb_position = 0.5;
    let bb_width = 0;
    let bb_squeeze = false;
    let bb_breakout_upper = false;
    let bb_breakout_lower = false;

    if (historicalCandles.length >= bbPeriod) {
      const bbCandles = historicalCandles.slice(-bbPeriod);
      const closes = bbCandles.map(c => c.close);
      const bbSma = closes.reduce((a, b) => a + b, 0) / bbPeriod;
      const variance = closes.reduce((sum, c) => sum + Math.pow(c - bbSma, 2), 0) / bbPeriod;
      const stdDev = Math.sqrt(variance);

      const upperBand = bbSma + (stdDev * bbStdDev);
      const lowerBand = bbSma - (stdDev * bbStdDev);
      const bandWidth = upperBand - lowerBand;

      // Position within bands (0 = at lower, 1 = at upper)
      bb_position = bandWidth > 0 ? (currentCandle.close - lowerBand) / bandWidth : 0.5;
      bb_position = Math.max(0, Math.min(1, bb_position));

      // Width as percentage of price
      bb_width = (bandWidth / currentCandle.close) * 100;

      // Squeeze detection (tight bands = low volatility = potential breakout)
      const avgWidth = bb_width;  // Could compare to historical average
      bb_squeeze = bb_width < 3;  // Less than 3% width = squeeze

      // Breakout detection
      bb_breakout_upper = currentCandle.close > upperBand;
      bb_breakout_lower = currentCandle.close < lowerBand;
    }

    // ═══════════════════════════════════════════════════════════════
    // Institutional Flow Detection (volume-based)
    // ═══════════════════════════════════════════════════════════════
    const instVolLookback = Math.min(20, historicalCandles.length);
    const instRecentVolumes = historicalCandles.slice(-instVolLookback).map(c => c.volume);
    const instAvgVolume = instRecentVolumes.reduce((a, b) => a + b, 0) / instVolLookback;
    const currentVolume = currentCandle.volume;

    // Large volume spike = potential institutional activity
    const large_volume_spike = currentVolume >= instAvgVolume * 3;

    // Volume delta estimate (bullish vs bearish pressure)
    // If close > open with high volume = buying pressure
    // If close < open with high volume = selling pressure
    const priceChange = currentCandle.close - currentCandle.open;
    const volume_delta = currentVolume > instAvgVolume
      ? (priceChange > 0 ? 1 : priceChange < 0 ? -1 : 0) * (currentVolume / instAvgVolume)
      : 0;

    // Accumulation: High volume + price holding/rising (smart money buying)
    const recentPriceChange = historicalCandles.length >= 5
      ? (currentCandle.close - historicalCandles[historicalCandles.length - 5].close) / historicalCandles[historicalCandles.length - 5].close
      : 0;
    const accumulation_detected = large_volume_spike && recentPriceChange >= -0.01;  // High vol, price not falling

    // Distribution: High volume + price falling (smart money selling)
    const distribution_detected = large_volume_spike && recentPriceChange < -0.02;  // High vol, price dropping

    // Smart money direction inference
    let smart_money_direction: 'buy' | 'sell' | 'neutral' = 'neutral';
    if (accumulation_detected) smart_money_direction = 'buy';
    else if (distribution_detected) smart_money_direction = 'sell';

    // Swing timing - find bars since swing high/low
    let bars_since_swing_high = 999;
    let bars_since_swing_low = 999;
    const swingLookback = Math.min(50, historicalCandles.length);
    for (let j = historicalCandles.length - 1; j >= historicalCandles.length - swingLookback; j--) {
      const c = historicalCandles[j];
      // Simple swing detection
      if (j > 2 && j < historicalCandles.length - 2) {
        const isSwingHigh = c.high >= historicalCandles[j-1].high &&
                           c.high >= historicalCandles[j-2].high &&
                           c.high >= historicalCandles[j+1]?.high &&
                           c.high >= historicalCandles[j+2]?.high;
        const isSwingLow = c.low <= historicalCandles[j-1].low &&
                          c.low <= historicalCandles[j-2].low &&
                          c.low <= historicalCandles[j+1]?.low &&
                          c.low <= historicalCandles[j+2]?.low;

        if (isSwingHigh && bars_since_swing_high === 999) {
          bars_since_swing_high = historicalCandles.length - 1 - j;
        }
        if (isSwingLow && bars_since_swing_low === 999) {
          bars_since_swing_low = historicalCandles.length - 1 - j;
        }
      }
    }

    // ═══════════════════════════════════════════════════════════════
    // ICT (Inner Circle Trader) Feature Extraction
    // ═══════════════════════════════════════════════════════════════

    // If ICT analysis provided, use it; otherwise compute fast version
    const ict = ictAnalysis || ICTIndicators.analyzeFast(historicalCandles, analysis);

    // OTE Zone features
    const ote_in_zone = ict.ote.valid &&
      currentCandle.close >= ict.ote.bottom &&
      currentCandle.close <= ict.ote.top;
    const ote_zone_direction = ict.ote.valid ? ict.ote.direction : 'none';

    // Displacement features
    const displacement_detected = ict.displacement.detected;
    const displacement_direction = ict.displacement.detected ? ict.displacement.direction : 'none';
    const displacement_strength = ict.displacement.atrMultiple;

    // Liquidity Grab features
    const liquidity_grab_detected = ict.liquidityGrab.detected;
    const liquidity_grab_type = ict.liquidityGrab.detected ? ict.liquidityGrab.type : 'none';
    const liquidity_grab_volume_spike = ict.liquidityGrab.volumeSpike;

    // Judas Swing features
    const judas_swing_detected = ict.judasSwing.detected;
    const judas_swing_direction = ict.judasSwing.detected ? ict.judasSwing.direction : 'none';
    const judas_swing_complete = ict.judasSwing.manipulationComplete;

    // Killzone features
    const killzone_active = ict.killzone.active;
    const killzone_name = ict.killzone.name;

    // Market Structure features
    const market_structure_trend = ict.structure.trend;
    const bos_detected = ict.structure.lastBOS !== null;
    const choch_detected = ict.structure.lastCHoCH !== null;
    const in_discount = ict.structure.inDiscount;
    const in_premium = ict.structure.inPremium;

    // Zone Confluence features
    const ob_in_ote = ict.obInOTE.length > 0;
    const fvg_in_ote = ict.fvgInOTE.length > 0;

    // Market Regime features
    const efficiency_ratio = ICTIndicators.calculateEfficiencyRatio(historicalCandles, 30);
    const is_trending_market = efficiency_ratio >= 0.30;

    // ICT Entry Quality
    const ict_entry_score = ict.entryScore;
    const ict_entry_valid = ict.entryValid;

    // HTF Cascade
    const htf_cascade_aligned = ict.htfCascade.aligned;
    const htf_cascade_score = ict.htfCascade.alignmentScore;

    // Entry Checklist
    const checklist_pass_rate = ict.entryChecklist.passRate;
    const checklist_grade = ict.entryChecklist.entryGrade;

    return {
      entry_price: currentCandle.close,
      entry_time: currentCandle.timestamp,
      direction,
      trend_direction,
      trend_strength,
      trend_bos_aligned,
      ob_near,
      ob_distance: Math.min(ob_distance, 1),
      ob_type,
      ob_size,
      ob_age: Math.min(ob_age, 100),
      fvg_near,
      fvg_count,
      fvg_nearest_distance: Math.min(fvg_nearest_distance, 1),
      fvg_size: Math.min(fvg_size, 0.5),
      fvg_type,
      ema_aligned,
      ema_trend,
      rsi_value,
      rsi_state,
      liquidity_near,
      liquidity_count,
      mtf_aligned,
      volatility,
      atr_value: atr,
      price_position,
      distance_to_high,
      distance_to_low,
      volume_spike,
      volume_ratio,
      confluence_score,
      confluence_count,
      session,
      days_since_loss,
      streak_type,
      potential_rr,
      is_pullback,
      pullback_depth,
      pullback_fib,
      pullback_bars,
      has_ob_return,
      has_fvg_fill,
      smc_signal_strength,
      ob_return_age,
      ob_test_count,
      ob_is_fresh,
      fvg_fill_percent,
      ob_state,
      ob_confirmed_mitigated,
      ob_in_mitigation,
      ob_caused_bos,
      ob_has_displacement,
      ob_impulse_size,
      bars_since_swing_high: Math.min(bars_since_swing_high, 100),
      bars_since_swing_low: Math.min(bars_since_swing_low, 100),

      // Bollinger Bands
      bb_position,
      bb_width,
      bb_squeeze,
      bb_breakout_upper,
      bb_breakout_lower,

      // Institutional Flow
      large_volume_spike,
      volume_delta,
      accumulation_detected,
      distribution_detected,
      smart_money_direction,

      // ICT Institutional Features
      ote_in_zone,
      ote_zone_direction,
      displacement_detected,
      displacement_direction,
      displacement_strength,
      liquidity_grab_detected,
      liquidity_grab_type,
      liquidity_grab_volume_spike,
      judas_swing_detected,
      judas_swing_direction,
      judas_swing_complete,
      killzone_active,
      killzone_name,
      market_structure_trend,
      bos_detected,
      choch_detected,
      in_discount,
      in_premium,
      ob_in_ote,
      fvg_in_ote,
      efficiency_ratio,
      is_trending_market,
      ict_entry_score,
      ict_entry_valid,
      htf_cascade_aligned,
      htf_cascade_score,
      checklist_pass_rate,
      checklist_grade,

      // Momentum Scalping Features (defaults - can be overridden by scalp export)
      regime: is_trending_market ? 'TREND' : 'RANGE',
      atr_percent: volatility,  // Already have volatility as ATR %
      macd_line: 0,
      macd_signal: 0,
      macd_histogram: 0,
      macd_bullish_cross: false,
      macd_bearish_cross: false,
      vwap: currentCandle.close,
      vwap_deviation: 0,
      vwap_deviation_std: 0,
      price_above_vwap: true,
      ema_fast: currentCandle.close,
      ema_slow: currentCandle.close,
      ema_bullish_cross: false,
      ema_bearish_cross: false,
      body_ratio: Math.abs(currentCandle.close - currentCandle.open) / Math.max(0.0001, currentCandle.high - currentCandle.low),
    };
  }
  
  /**
   * Calculate trend strength from SMA slope
   */
  private static calculateTrendStrength(candles: Candle[]): number {
    if (candles.length < 20) return 0;
    
    // Calculate 20-period SMA
    const period = 20;
    const sma: number[] = [];
    for (let i = period - 1; i < candles.length; i++) {
      let sum = 0;
      for (let j = 0; j < period; j++) {
        sum += candles[i - j].close;
      }
      sma.push(sum / period);
    }
    
    if (sma.length < 5) return 0;
    
    // Calculate slope of last 5 SMAs
    const slope = (sma[sma.length - 1] - sma[sma.length - 5]) / 5;
    const avgPrice = candles[candles.length - 1].close;
    
    // Normalize to 0-1
    const normalizedSlope = Math.abs(slope / avgPrice) * 100;
    return Math.min(Math.max(normalizedSlope, 0), 1);
  }
  
  /**
   * Add outcome to features (after trade completes)
   */
  static addOutcome(
    features: Omit<TradeFeatures, 'outcome' | 'pnl' | 'pnl_percent' | 'exit_reason' | 'holding_periods'>,
    trade: BacktestTrade
  ): TradeFeatures {
    return {
      ...features,
      outcome: trade.outcome,
      pnl: trade.pnl,
      pnl_percent: trade.pnl_percent,
      exit_reason: trade.exit_reason,
      holding_periods: trade.holding_periods
    };
  }
  
  /**
   * Check multi-timeframe alignment
   * Simulates checking if higher timeframes agree with current trend
   */
  private static checkMTFAlignment(
    trend_direction: 'up' | 'down' | 'neutral',
    direction: 'long' | 'short',
    currentCandle: Candle,
    historicalCandles: Candle[]
  ): boolean {
    if (historicalCandles.length < 50) return true; // Not enough data
    
    // Get trend on different lookback periods (simulating MTF)
    const shortTermTrend = this.getTrendForPeriod(historicalCandles, 20);
    const mediumTermTrend = this.getTrendForPeriod(historicalCandles, 50);
    const longTermTrend = this.getTrendForPeriod(historicalCandles, 100);
    
    // Check alignment
    const isLong = direction === 'long';
    
    // For long trades: want higher timeframes to show upward trend
    if (isLong) {
      const shortAligned = shortTermTrend === 'up' || shortTermTrend === 'neutral';
      const mediumAligned = mediumTermTrend === 'up' || mediumTermTrend === 'neutral';
      const longAligned = longTermTrend === 'up' || longTermTrend === 'neutral';
      
      // At least 2 out of 3 should be aligned
      return [shortAligned, mediumAligned, longAligned].filter(Boolean).length >= 2;
    } 
    // For short trades: want higher timeframes to show downward trend
    else {
      const shortAligned = shortTermTrend === 'down' || shortTermTrend === 'neutral';
      const mediumAligned = mediumTermTrend === 'down' || mediumTermTrend === 'neutral';
      const longAligned = longTermTrend === 'down' || longTermTrend === 'neutral';
      
      // At least 2 out of 3 should be aligned
      return [shortAligned, mediumAligned, longAligned].filter(Boolean).length >= 2;
    }
  }
  
  /**
   * Get trend direction for a specific period
   */
  private static getTrendForPeriod(candles: Candle[], period: number): 'up' | 'down' | 'neutral' {
    if (candles.length < period) return 'neutral';
    
    const recentCandles = candles.slice(-period);
    const firstPrice = recentCandles[0].close;
    const lastPrice = recentCandles[recentCandles.length - 1].close;
    
    const percentChange = (lastPrice - firstPrice) / firstPrice * 100;
    
    if (percentChange > 0.5) return 'up';
    if (percentChange < -0.5) return 'down';
    return 'neutral';
  }
}
