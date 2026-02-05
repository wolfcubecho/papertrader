/**
 * Unified Scoring System v3
 *
 * REWRITTEN based on ML model weights analysis:
 *
 * WHAT WORKS (positive weights):
 * - distance_to_low: +0.058 (buying dips in uptrends)
 * - volatility: +0.056 (higher volatility = opportunity)
 * - session_off-hours: +0.036 (quieter markets work better)
 * - session_asian: +0.025 (Asian session favorable)
 * - rsi_value: +0.019 (momentum following works)
 * - fvg_size: +0.018 (larger gaps = stronger signal)
 *
 * WHAT HURTS (negative weights):
 * - Being "near" OBs/FVGs: -0.07 (chasing, not waiting for pullback)
 * - session_overlap: -0.047 (too noisy)
 * - trend_strength: -0.027 (entering during impulse, not pullback)
 *
 * KEY INSIGHT: The problem isn't long vs short, it's WHEN we enter.
 * Best entries happen during PULLBACKS, not during impulse moves.
 */

import { SMCWeights } from './types.js';
import { SMCAnalysis, Candle, PullbackInfo, SMCReturnSignal } from './smc-indicators.js';

export interface ConfluenceFactors {
  description: string;
  weight: number;
  value: any;
}

export interface UnifiedScore {
  score: number;
  breakdown: Record<string, number>;
  confluence: string[];
  bias: 'bullish' | 'bearish' | 'neutral';
  macroScore: number;
  entryQuality: 'A' | 'B' | 'C' | 'D';  // NEW: Grade the entry
}

export class UnifiedScoring {
  /**
   * Calculate score based on what ML model ACTUALLY learned works
   * @param timestamp - Optional timestamp for session calculation (for backtesting)
   */
  static calculateConfluence(
    analysis: SMCAnalysis,
    currentPrice: number,
    weights: SMCWeights,
    timestamp?: number
  ): UnifiedScore {
    const breakdown: Record<string, number> = {};
    const confluence: string[] = [];
    let totalScore = 0;

    // Determine bias from trend
    let bias: 'bullish' | 'bearish' | 'neutral' = 'neutral';
    if (analysis.trend === 'up') {
      bias = 'bullish';
    } else if (analysis.trend === 'down') {
      bias = 'bearish';
    }

    // ═══════════════════════════════════════════════════════════════
    // 1. PULLBACK DETECTION (0-35 pts) - THE MOST IMPORTANT FACTOR
    // ═══════════════════════════════════════════════════════════════
    // Don't enter during impulse moves. Wait for pullbacks!
    if (analysis.pullback && analysis.pullback.isPullback) {
      const pb = analysis.pullback;

      // Best entries at key fib levels
      if (pb.fibLevel === '0.618') {
        breakdown.pullback = 35;
        confluence.push(`Golden ratio pullback (61.8%)`);
      } else if (pb.fibLevel === '0.5') {
        breakdown.pullback = 30;
        confluence.push(`50% pullback retracement`);
      } else if (pb.fibLevel === '0.382') {
        breakdown.pullback = 25;
        confluence.push(`Shallow pullback (38.2%)`);
      } else if (pb.fibLevel === '0.786') {
        breakdown.pullback = 20;
        confluence.push(`Deep pullback (78.6%) - higher risk`);
      } else {
        // Valid pullback but not at fib level
        breakdown.pullback = 15;
        confluence.push(`Pullback in progress (${(pb.pullbackDepth * 100).toFixed(0)}%)`);
      }

      // Bonus for fresh pullbacks (not too old)
      if (pb.pullbackBars <= 5) {
        breakdown.pullback += 5;
        confluence.push(`Fresh pullback (${pb.pullbackBars} bars)`);
      }
    } else {
      // NO PULLBACK = BAD ENTRY (chasing impulse)
      breakdown.pullback = -10;
      confluence.push(`WARNING: No pullback - entering during impulse`);
    }
    totalScore += breakdown.pullback;

    // ═══════════════════════════════════════════════════════════════
    // 2. PRICE POSITION / DISCOUNT ENTRY (0-25 pts)
    // ═══════════════════════════════════════════════════════════════
    // ML shows distance_to_low is POSITIVE - buying dips works
    if (analysis.liquidityZones.highs.length > 0 && analysis.liquidityZones.lows.length > 0) {
      const recentHighs = analysis.liquidityZones.highs.slice(-5);
      const recentLows = analysis.liquidityZones.lows.slice(-5);
      const avgHigh = recentHighs.reduce((s, h) => s + h.price, 0) / recentHighs.length;
      const avgLow = recentLows.reduce((s, l) => s + l.price, 0) / recentLows.length;
      const range = avgHigh - avgLow;

      if (range > 0) {
        const pricePosition = (currentPrice - avgLow) / range;

        if (bias === 'bullish' && pricePosition < 0.35) {
          breakdown.price_position = 25;
          confluence.push(`Discount entry - ${(pricePosition * 100).toFixed(0)}% of range`);
        } else if (bias === 'bearish' && pricePosition > 0.65) {
          breakdown.price_position = 25;
          confluence.push(`Premium entry - ${(pricePosition * 100).toFixed(0)}% of range`);
        } else if (bias === 'bullish' && pricePosition < 0.5) {
          breakdown.price_position = 15;
          confluence.push(`Below midpoint (${(pricePosition * 100).toFixed(0)}%)`);
        } else if (bias === 'bearish' && pricePosition > 0.5) {
          breakdown.price_position = 15;
          confluence.push(`Above midpoint (${(pricePosition * 100).toFixed(0)}%)`);
        } else {
          // Chasing - price already moved
          breakdown.price_position = -5;
          confluence.push(`WARNING: Chasing (${(pricePosition * 100).toFixed(0)}% of range)`);
        }
      } else {
        breakdown.price_position = 0;
      }
    } else {
      breakdown.price_position = 0;
    }
    totalScore += breakdown.price_position;

    // ═══════════════════════════════════════════════════════════════
    // 3. VOLATILITY (0-20 pts) - ML says HIGHER is BETTER
    // ═══════════════════════════════════════════════════════════════
    // This is counterintuitive but the data shows it
    if (analysis.atr && analysis.ema50) {
      const atrPercent = (analysis.atr / analysis.ema50) * 100;

      if (atrPercent >= 3 && atrPercent <= 6) {
        // Sweet spot - enough volatility for movement, not crazy
        breakdown.volatility = 20;
        confluence.push(`Good volatility (${atrPercent.toFixed(1)}% ATR)`);
      } else if (atrPercent >= 2 && atrPercent < 3) {
        breakdown.volatility = 15;
        confluence.push(`Moderate volatility (${atrPercent.toFixed(1)}%)`);
      } else if (atrPercent > 6 && atrPercent <= 10) {
        breakdown.volatility = 10;
        confluence.push(`High volatility (${atrPercent.toFixed(1)}%) - wider stops needed`);
      } else if (atrPercent < 2) {
        breakdown.volatility = 0;
        confluence.push(`Low volatility (${atrPercent.toFixed(1)}%) - may not move`);
      } else {
        breakdown.volatility = -5;
        confluence.push(`Extreme volatility (${atrPercent.toFixed(1)}%) - risky`);
      }
    } else {
      breakdown.volatility = 0;
    }
    totalScore += breakdown.volatility;

    // ═══════════════════════════════════════════════════════════════
    // 4. TREND PRESENCE (0-15 pts) - Simple, not overweighted
    // ═══════════════════════════════════════════════════════════════
    if (analysis.trend) {
      breakdown.trend = 15;
      confluence.push(`${analysis.trend}trend established`);
    } else {
      breakdown.trend = -5;
      confluence.push(`No clear trend - ranging`);
    }
    totalScore += breakdown.trend;

    // ═══════════════════════════════════════════════════════════════
    // 5. RSI MOMENTUM (0-15 pts or penalty)
    // ═══════════════════════════════════════════════════════════════
    // ML shows rsi_value is slightly positive - momentum works
    // But NOT extremes - look for RSI moving WITH trend
    if (analysis.rsi) {
      if (bias === 'bullish' && analysis.rsi >= 45 && analysis.rsi <= 65) {
        // Healthy momentum in uptrend
        breakdown.rsi = 15;
        confluence.push(`Healthy RSI momentum (${analysis.rsi.toFixed(0)})`);
      } else if (bias === 'bearish' && analysis.rsi >= 35 && analysis.rsi <= 55) {
        // Healthy momentum in downtrend
        breakdown.rsi = 15;
        confluence.push(`Healthy RSI momentum (${analysis.rsi.toFixed(0)})`);
      } else if (bias === 'bullish' && analysis.rsi < 35) {
        // Oversold in uptrend - could be pullback or reversal
        breakdown.rsi = 5;
        confluence.push(`RSI oversold (${analysis.rsi.toFixed(0)}) - confirm support`);
      } else if (bias === 'bearish' && analysis.rsi > 65) {
        // Overbought in downtrend - could be pullback or reversal
        breakdown.rsi = 5;
        confluence.push(`RSI overbought (${analysis.rsi.toFixed(0)}) - confirm resistance`);
      } else if (analysis.rsi > 75 || analysis.rsi < 25) {
        // Extreme RSI - exhaustion likely
        breakdown.rsi = -10;
        confluence.push(`WARNING: RSI extreme (${analysis.rsi.toFixed(0)}) - exhaustion`);
      } else {
        breakdown.rsi = 0;
      }
    } else {
      breakdown.rsi = 0;
    }
    totalScore += breakdown.rsi;

    // ═══════════════════════════════════════════════════════════════
    // 6. SESSION TIMING (0-10 pts or penalty)
    // ═══════════════════════════════════════════════════════════════
    // ML shows Asian and off-hours are BETTER, overlap is WORSE
    const hour = timestamp ? new Date(timestamp).getUTCHours() : new Date().getUTCHours();
    if (hour >= 0 && hour < 6) {
      // Asian session
      breakdown.session = 10;
      confluence.push(`Asian session (favorable)`);
    } else if (hour >= 20 || hour >= 16 && hour < 20) {
      // Off hours
      breakdown.session = 8;
      confluence.push(`Off-hours (less noise)`);
    } else if (hour >= 6 && hour < 8) {
      // London open
      breakdown.session = 5;
      confluence.push(`London session`);
    } else if (hour >= 12 && hour < 16) {
      // NY session
      breakdown.session = 3;
      confluence.push(`NY session`);
    } else if (hour >= 8 && hour < 12) {
      // Overlap - WORST per ML
      breakdown.session = -5;
      confluence.push(`Session overlap (noisy)`);
    } else {
      breakdown.session = 0;
    }
    totalScore += breakdown.session;

    // ═══════════════════════════════════════════════════════════════
    // 7. EMA STRUCTURE (0-10 pts)
    // ═══════════════════════════════════════════════════════════════
    if (analysis.ema50 && analysis.ema200) {
      const emaAligned = (bias === 'bullish' && analysis.ema50 > analysis.ema200) ||
                         (bias === 'bearish' && analysis.ema50 < analysis.ema200);

      // Price should be near EMA (not too far extended)
      const distFromEma50 = Math.abs(currentPrice - analysis.ema50) / analysis.ema50;

      if (emaAligned && distFromEma50 < 0.02) {
        breakdown.ema = 10;
        confluence.push(`Price at EMA50 (good entry zone)`);
      } else if (emaAligned && distFromEma50 < 0.05) {
        breakdown.ema = 5;
        confluence.push(`EMAs aligned, price near EMA50`);
      } else if (emaAligned && distFromEma50 > 0.08) {
        breakdown.ema = -5;
        confluence.push(`WARNING: Extended from EMA50 (${(distFromEma50 * 100).toFixed(1)}%)`);
      } else {
        breakdown.ema = 0;
      }
    } else {
      breakdown.ema = 0;
    }
    totalScore += breakdown.ema;

    // ═══════════════════════════════════════════════════════════════
    // 8. SMC RETURN SIGNALS (0-30 pts) - NEW CORRECT LOGIC
    // ═══════════════════════════════════════════════════════════════
    // OBs and FVGs ARE valuable - but only when price RETURNS to them
    // Not when price is just "near" them (that was our mistake)
    if (analysis.bestSignal) {
      const signal = analysis.bestSignal;

      // Only score if signal direction matches our bias
      const signalMatchesBias =
        (bias === 'bullish' && signal.direction === 'long') ||
        (bias === 'bearish' && signal.direction === 'short');

      if (signalMatchesBias) {
        // Base score from signal strength
        const signalScore = Math.round(signal.strength * 30);
        breakdown.smc_return = signalScore;

        if (signal.type === 'ob_return') {
          confluence.push(`OB return signal (${(signal.strength * 100).toFixed(0)}% strength)`);
        } else {
          confluence.push(`FVG fill signal (${(signal.strength * 100).toFixed(0)}% strength)`);
        }

        // Extra bonus for high-tier signals
        if (signal.tier === 1) {
          breakdown.smc_return += 10;
          confluence.push(`TIER 1 signal (HTF confirmed)`);
        } else if (signal.tier === 2) {
          breakdown.smc_return += 5;
          confluence.push(`TIER 2 signal (strong confluence)`);
        }
      } else {
        // Signal exists but wrong direction - slight penalty
        breakdown.smc_return = -5;
        confluence.push(`SMC signal against bias`);
      }
    } else {
      breakdown.smc_return = 0;
    }
    totalScore += breakdown.smc_return;

    // ═══════════════════════════════════════════════════════════════
    // NOTE: We now properly use OBs and FVGs - but only RETURN signals
    // Simple "proximity" to zones is NOT scored (that was hurting us)
    // ═══════════════════════════════════════════════════════════════

    // Calculate entry quality grade
    const finalScore = Math.max(0, totalScore);
    let entryQuality: 'A' | 'B' | 'C' | 'D';
    if (finalScore >= 70) entryQuality = 'A';
    else if (finalScore >= 50) entryQuality = 'B';
    else if (finalScore >= 30) entryQuality = 'C';
    else entryQuality = 'D';

    return {
      score: finalScore,
      breakdown,
      confluence,
      bias,
      macroScore: 0,
      entryQuality
    };
  }

  /**
   * Calculate MTF alignment - simplified
   */
  static calculateMTFBonus(
    daily: SMCAnalysis,
    hourly: SMCAnalysis | null,
    fiveMin: SMCAnalysis | null
  ): { bonus: number; factors: string[] } {
    let bonus = 0;
    const factors: string[] = [];

    // Simple check: does daily have a trend?
    if (daily.trend) {
      bonus += 10;
      factors.push(`Daily trend: ${daily.trend}`);

      // Mild bonus if hourly agrees
      if (hourly && hourly.trend === daily.trend) {
        bonus += 5;
        factors.push(`Hourly confirms`);
      }

      // Check if hourly is in pullback (ideal entry)
      if (hourly && hourly.pullback?.isPullback) {
        bonus += 10;
        factors.push(`Hourly pullback active`);
      }
    }

    return { bonus, factors };
  }

  /**
   * Quick check: Is this a valid entry?
   * Use this as a gate before detailed scoring
   */
  static isValidEntry(analysis: SMCAnalysis, bias: 'bullish' | 'bearish'): { valid: boolean; reason: string } {
    // Must have a trend
    if (!analysis.trend) {
      return { valid: false, reason: 'No trend' };
    }

    // Trend must match bias
    if ((bias === 'bullish' && analysis.trend !== 'up') ||
        (bias === 'bearish' && analysis.trend !== 'down')) {
      return { valid: false, reason: 'Bias against trend' };
    }

    // Should be in a pullback (not chasing)
    if (!analysis.pullback?.isPullback) {
      return { valid: false, reason: 'No pullback - would be chasing' };
    }

    // RSI shouldn't be extreme
    if (analysis.rsi && (analysis.rsi > 80 || analysis.rsi < 20)) {
      return { valid: false, reason: 'RSI extreme' };
    }

    return { valid: true, reason: 'Valid entry setup' };
  }
}
