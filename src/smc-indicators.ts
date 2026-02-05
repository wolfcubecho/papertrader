/**
 * SMC (Smart Money Concepts) Technical Indicators
 * Calculates Order Blocks, Fair Value Gaps, EMAs, Liquidity, etc.
 */

export interface Candle {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

// OB State Machine (proper SMC logic)
// "An order block is only tradable after mitigation, not when it first forms"
export type OBState =
  | 'NEW_OB'                  // Just formed - NEVER trade here
  | 'WAITING_FOR_MITIGATION'  // Price moved away, waiting for return
  | 'IN_MITIGATION'           // Price has returned to zone
  | 'CONFIRMED_MITIGATED'     // LTF rejection + BOS confirmed - ENTRY ALLOWED
  | 'INVALIDATED';            // Price closed through - ignore this OB

export interface OrderBlock {
  index: number;
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  mid: number;                // (high + low) / 2 - key level for mitigation
  type: 'bull' | 'bear';

  // State Machine (THE KEY TO PROPER SMC)
  state: OBState;

  // Formation quality (only trade strong OBs)
  strength: number;           // 0-5 strength score
  causedBOS: boolean;         // Did this OB cause a break of structure?
  hasDisplacement: boolean;   // Was there displacement after?
  hasFVG: boolean;            // Did it create an FVG?
  impulseSize: number;        // Size of move after OB (in ATR multiples)

  // Mitigation tracking
  testCount: number;          // How many times tested (0 = fresh, 1 = first mitig = BEST)
  firstTouchIndex: number;    // When price first returned
  rejectionConfirmed: boolean; // Did price reject from zone?
  ltfBOSConfirmed: boolean;   // Lower timeframe BOS confirmation?

  // Legacy fields (for compatibility)
  tested: boolean;
  broken: boolean;
  priceLeftZone: boolean;
  barsAgo: number;
}

// FVG State Machine
// "Trade STRONG_FVG + rejection at 50% fill level"
export type FVGState =
  | 'NEW_FVG'           // Just formed - waiting
  | 'PENDING_FILL'      // Monitoring for price return
  | 'FILLING'           // Price has entered gap
  | 'PARTIAL_FILL'      // 50%+ filled but rejected
  | 'STRONG_FVG'        // Rejected at 50% - TRADEABLE
  | 'FULLY_FILLED'      // Completely filled - dead
  | 'DEAD';             // Invalid/expired

export interface FVG {
  index: number;
  from: number;           // Gap start price
  to: number;             // Gap end price
  mid: number;            // 50% level (key for rejection)
  type: 'bull' | 'bear';
  size: number;           // Gap size in price
  sizeATR: number;        // Gap size in ATR multiples

  // State Machine
  state: FVGState;

  // Fill tracking
  filled: boolean;
  partiallyFilled: boolean;
  fillPercent: number;    // 0-1 how much filled
  rejectedAt50: boolean;  // Key signal - rejected at midpoint

  // Quality metrics
  formedAfterBOS: boolean;    // Higher quality if after structure break
  hasDisplacement: boolean;   // Formed during displacement
  overlapsOB: boolean;        // Overlaps an order block (high confluence)
  inOTE: boolean;             // Inside OTE zone

  barsAgo: number;
}

// Liquidity Zone State Machine
// "Trade AFTER liquidity grab + reversal, not before"
export type LiquidityState =
  | 'TARGET_ZONE'       // Unraided - waiting for sweep
  | 'GRABBED'           // Wick swept zone
  | 'VALID_LIQUIDITY'   // Grab + reversal + BOS = TRADEABLE
  | 'DEAD';             // No reaction after grab

export interface LiquidityZone {
  price: number;
  type: 'high' | 'low';
  touches: number;        // How many times tested (more = more liquidity)
  state: LiquidityState;
  grabbedIndex: number;   // When was it swept
  reversalConfirmed: boolean;
  sessionType: 'asian' | 'london' | 'newyork' | 'other';
}

// Breaker Block - Failed OB that flips to S/R
export type BreakerState =
  | 'NEW_BREAKER'       // Just formed
  | 'TESTING'           // Price returning to test
  | 'FLIPPED_BLOCK'     // Acting as new S/R - TRADEABLE
  | 'FAKEOUT';          // Failed to hold

export interface BreakerBlock {
  index: number;
  high: number;
  low: number;
  type: 'bull' | 'bear';  // Direction it now supports
  state: BreakerState;
  originalOBType: 'bull' | 'bear';  // What it was before breaking
  breakIndex: number;     // When the OB failed
  testCount: number;
}

// NEW: Return-to-zone signals
export interface SMCReturnSignal {
  type: 'ob_return' | 'fvg_fill';
  zone: OrderBlock | FVG;
  direction: 'long' | 'short';
  strength: number;         // 0-1 based on zone age, size, trend alignment
  priceInZone: boolean;     // Is current price inside the zone?
  distanceToZone: number;   // % distance to zone edge
}

export interface LiquidityZones {
  highs: Array<{ price: number; touches: number; lastTouch: number }>;
  lows: Array<{ price: number; touches: number; lastTouch: number }>;
  // Enhanced liquidity tracking with state
  zones: LiquidityZone[];
}

export interface PullbackInfo {
  isPullback: boolean;
  pullbackDepth: number;  // 0-1, how deep into the pullback (0.5 = 50% retracement)
  pullbackBars: number;   // How many bars into pullback
  swingHigh: number;      // Recent swing high
  swingLow: number;       // Recent swing low
  fibLevel: '0.382' | '0.5' | '0.618' | '0.786' | 'none';  // Which fib level price is near
}

// Signal Tier for hierarchy-based entries
export type SignalTier = 1 | 2 | 3 | 4;

export interface SMCSignal {
  tier: SignalTier;
  type: 'ob_return' | 'fvg_fill' | 'liquidity_grab' | 'breaker_test' | 'ote_entry';
  direction: 'long' | 'short';
  strength: number;       // 0-1
  zone: OrderBlock | FVG | LiquidityZone | BreakerBlock | null;
  confluence: string[];   // What's aligned
}

export interface SMCAnalysis {
  trend: 'up' | 'down' | null;
  bos: 'up' | 'down' | null;
  choch: boolean;         // Change of Character detected
  ema50: number | null;
  ema200: number | null;
  rsi: number | null;
  orderBlocks: OrderBlock[];
  fvg: FVG[];
  liquidityZones: LiquidityZones;
  breakerBlocks: BreakerBlock[];
  atr: number | null;
  vwap: number | null;
  pullback: PullbackInfo;

  // Signal Generation (Tiered Hierarchy)
  // TIER 1 (90%): HTF OB/FVG confirmed + structure alignment
  // TIER 2 (70%): OTE overlap + liquidity grabbed
  // TIER 3 (50%): LTF BOS + displacement confirmation
  // TIER 4 (30%): ML probability >65%
  signals: SMCSignal[];
  bestSignal: SMCSignal | null;

  // Legacy
  returnSignals: SMCReturnSignal[];
}

export class SMCIndicators {
  /**
   * Calculate Simple Moving Average
   */
  static sma(data: number[], period: number): number[] {
    const result: number[] = [];
    for (let i = period - 1; i < data.length; i++) {
      const sum = data.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
      result.push(sum / period);
    }
    return result;
  }

  /**
   * Calculate Exponential Moving Average
   */
  static ema(data: number[], period: number): number[] {
    const result: number[] = [];
    const multiplier = 2 / (period + 1);
    
    // Start with SMA for first value
    let ema = data.slice(0, period).reduce((a, b) => a + b, 0) / period;
    result.push(ema);
    
    for (let i = period; i < data.length; i++) {
      ema = (data[i] - ema) * multiplier + ema;
      result.push(ema);
    }
    
    return result;
  }

  /**
   * Calculate RSI
   */
  static rsi(candles: Candle[], period: number = 14): number[] {
    const result: number[] = [];
    const closes = candles.map(c => c.close);
    
    for (let i = period; i < closes.length; i++) {
      const slice = closes.slice(i - period, i);
      const gains: number[] = [];
      const losses: number[] = [];
      
      for (let j = 1; j < slice.length; j++) {
        const diff = slice[j] - slice[j - 1];
        if (diff >= 0) {
          gains.push(diff);
          losses.push(0);
        } else {
          gains.push(0);
          losses.push(Math.abs(diff));
        }
      }
      
      const avgGain = gains.reduce((a, b) => a + b, 0) / period;
      const avgLoss = losses.reduce((a, b) => a + b, 0) / period;
      
      if (avgLoss === 0) {
        result.push(100);
      } else {
        const rs = avgGain / avgLoss;
        result.push(100 - (100 / (1 + rs)));
      }
    }
    
    return result;
  }

  /**
   * Calculate ATR (Average True Range)
   */
  static atr(candles: Candle[], period: number = 14): number[] {
    const result: number[] = [];
    
    for (let i = period - 1; i < candles.length; i++) {
      const trueRanges: number[] = [];
      
      for (let j = i - period + 1; j <= i; j++) {
        const c = candles[j];
        const tr = Math.max(
          c.high - c.low,
          Math.abs(c.high - c.close),
          Math.abs(c.low - c.close)
        );
        trueRanges.push(tr);
      }
      
      result.push(trueRanges.reduce((a, b) => a + b, 0) / period);
    }
    
    return result;
  }

  /**
   * Calculate VWAP
   */
  static vwap(candles: Candle[]): number[] {
    const result: number[] = [];
    let cumVolPrice = 0;
    let cumVol = 0;
    
    for (let i = 0; i < candles.length; i++) {
      const c = candles[i];
      const typicalPrice = (c.high + c.low + c.close) / 3;
      cumVolPrice += typicalPrice * c.volume;
      cumVol += c.volume;
      result.push(cumVolPrice / cumVol);
    }
    
    return result;
  }

  /**
   * Detect Order Blocks with return tracking
   * An OB is valid for entry when:
   * 1. Price LEFT the zone after it formed
   * 2. Price is now RETURNING to the zone
   * 3. The zone hasn't been broken (invalidated)
   */
  static detectOrderBlocks(candles: Candle[], lookback: number = 10): OrderBlock[] {
    const orderBlocks: OrderBlock[] = [];
    const currentPrice = candles[candles.length - 1].close;
    const currentIdx = candles.length - 1;

    for (let i = lookback; i < candles.length - 2; i++) {
      const candle = candles[i];
      const nextCandle = candles[i + 1];
      const nextNextCandle = candles[i + 2];

      const bodySize = Math.abs(candle.close - candle.open);
      const avgBody = candles.slice(i - lookback, i)
        .reduce((sum, c) => sum + Math.abs(c.close - c.open), 0) / lookback;

      let ob: OrderBlock | null = null;

      // Calculate ATR for impulse measurement
      const atrPeriod = Math.min(14, i);
      let atr = 0;
      if (atrPeriod > 0) {
        for (let k = i - atrPeriod; k < i; k++) {
          if (k >= 0) atr += candles[k].high - candles[k].low;
        }
        atr /= atrPeriod;
      }

      // Bullish OB: last DOWN candle before bullish impulse that breaks structure
      // "The last opposing candle before a strong impulsive move that breaks structure"
      if (candle.close < candle.open && bodySize > avgBody * 1.5) {
        if (nextCandle.close > nextCandle.open &&
            nextNextCandle.close > nextNextCandle.open) {

          // Check for displacement (fast, large candles)
          const impulseMove = nextNextCandle.close - candle.low;
          const impulseATR = atr > 0 ? impulseMove / atr : 1;
          const hasDisplacement = impulseATR > 1.5;

          // Check if this caused a BOS (broke prior high)
          let causedBOS = false;
          for (let k = Math.max(0, i - 20); k < i; k++) {
            if (nextNextCandle.close > candles[k].high) {
              causedBOS = true;
              break;
            }
          }

          // Calculate strength score (0-5)
          let strengthScore = 0;
          if (impulseATR > 1.5) strengthScore++;  // displacement
          if (causedBOS) strengthScore++;          // structure break
          if (bodySize > avgBody * 2) strengthScore++;  // strong OB candle
          // FVG check done later

          ob = {
            index: i,
            timestamp: candle.timestamp,
            open: candle.open,
            high: candle.high,
            low: candle.low,
            close: candle.close,
            mid: (candle.high + candle.low) / 2,
            type: 'bull',
            // State Machine - starts as NEW_OB
            state: 'NEW_OB' as OBState,
            // Strength metrics
            strength: strengthScore,
            causedBOS,
            hasDisplacement,
            hasFVG: false,  // Set later when checking FVGs
            impulseSize: impulseATR,
            // Mitigation tracking
            testCount: 0,
            firstTouchIndex: -1,
            rejectionConfirmed: false,
            ltfBOSConfirmed: false,
            // Legacy
            tested: false,
            broken: false,
            priceLeftZone: false,
            barsAgo: currentIdx - i
          };
        }
      }

      // Bearish OB: last UP candle before bearish impulse that breaks structure
      if (candle.close > candle.open && bodySize > avgBody * 1.5) {
        if (nextCandle.close < nextCandle.open &&
            nextNextCandle.close < nextNextCandle.open) {

          // Check for displacement
          const impulseMove = candle.high - nextNextCandle.close;
          const impulseATR = atr > 0 ? impulseMove / atr : 1;
          const hasDisplacement = impulseATR > 1.5;

          // Check if this caused a BOS (broke prior low)
          let causedBOS = false;
          for (let k = Math.max(0, i - 20); k < i; k++) {
            if (nextNextCandle.close < candles[k].low) {
              causedBOS = true;
              break;
            }
          }

          // Calculate strength score
          let strengthScore = 0;
          if (impulseATR > 1.5) strengthScore++;
          if (causedBOS) strengthScore++;
          if (bodySize > avgBody * 2) strengthScore++;

          ob = {
            index: i,
            timestamp: candle.timestamp,
            open: candle.open,
            high: candle.high,
            low: candle.low,
            close: candle.close,
            mid: (candle.high + candle.low) / 2,
            type: 'bear',
            state: 'NEW_OB' as OBState,
            strength: strengthScore,
            causedBOS,
            hasDisplacement,
            hasFVG: false,
            impulseSize: impulseATR,
            testCount: 0,
            firstTouchIndex: -1,
            rejectionConfirmed: false,
            ltfBOSConfirmed: false,
            tested: false,
            broken: false,
            priceLeftZone: false,
            barsAgo: currentIdx - i
          };
        }
      }

      if (ob) {
        // ═══════════════════════════════════════════════════════════════
        // OB STATE MACHINE - Proper SMC logic
        // ═══════════════════════════════════════════════════════════════
        const zoneHigh = Math.max(ob.open, ob.close);
        const zoneLow = Math.min(ob.open, ob.close);
        const zoneMid = ob.mid;

        for (let j = i + 3; j < candles.length; j++) {
          const c = candles[j];
          const prevCandle = candles[j - 1];

          if (ob.type === 'bull') {
            // ─────────────────────────────────────────────────────────
            // BULLISH OB STATE TRANSITIONS
            // ─────────────────────────────────────────────────────────

            // NEW_OB → WAITING_FOR_MITIGATION (price leaves zone going UP)
            if (ob.state === 'NEW_OB' && c.low > zoneHigh) {
              ob.state = 'WAITING_FOR_MITIGATION';
              ob.priceLeftZone = true;
            }

            // WAITING_FOR_MITIGATION → IN_MITIGATION (price returns to zone)
            if (ob.state === 'WAITING_FOR_MITIGATION') {
              if (c.low <= zoneHigh && c.low >= zoneLow) {
                ob.state = 'IN_MITIGATION';
                ob.firstTouchIndex = j;
                ob.testCount++;
                ob.tested = true;
              }
            }

            // IN_MITIGATION → CONFIRMED_MITIGATED (rejection + bullish continuation)
            // "A rejection candle (close back above mid or above previous candle high)"
            if (ob.state === 'IN_MITIGATION') {
              // Check for rejection: price touched zone then closed above it
              const touchedZone = c.low <= zoneHigh && c.low >= zoneLow;
              const rejectedUp = c.close > zoneMid && c.close > c.open;  // Bullish close above mid
              const madeHigherHigh = c.high > prevCandle.high;

              if (touchedZone && rejectedUp && madeHigherHigh) {
                ob.rejectionConfirmed = true;
                ob.ltfBOSConfirmed = true;  // Simplified - in real impl check LTF
                ob.state = 'CONFIRMED_MITIGATED';
              }

              // Still in zone but no rejection yet - stay IN_MITIGATION
            }

            // ANY STATE → INVALIDATED (price closes below OB low)
            // "If price deeply closes through the OB with no bullish rejection"
            if (c.close < zoneLow - (zoneLow * 0.005)) {  // 0.5% tolerance
              ob.state = 'INVALIDATED';
              ob.broken = true;
            }

          } else {
            // ─────────────────────────────────────────────────────────
            // BEARISH OB STATE TRANSITIONS
            // ─────────────────────────────────────────────────────────

            // NEW_OB → WAITING_FOR_MITIGATION (price leaves zone going DOWN)
            if (ob.state === 'NEW_OB' && c.high < zoneLow) {
              ob.state = 'WAITING_FOR_MITIGATION';
              ob.priceLeftZone = true;
            }

            // WAITING_FOR_MITIGATION → IN_MITIGATION (price returns to zone)
            if (ob.state === 'WAITING_FOR_MITIGATION') {
              if (c.high >= zoneLow && c.high <= zoneHigh) {
                ob.state = 'IN_MITIGATION';
                ob.firstTouchIndex = j;
                ob.testCount++;
                ob.tested = true;
              }
            }

            // IN_MITIGATION → CONFIRMED_MITIGATED (rejection + bearish continuation)
            if (ob.state === 'IN_MITIGATION') {
              const touchedZone = c.high >= zoneLow && c.high <= zoneHigh;
              const rejectedDown = c.close < zoneMid && c.close < c.open;  // Bearish close below mid
              const madeLowerLow = c.low < prevCandle.low;

              if (touchedZone && rejectedDown && madeLowerLow) {
                ob.rejectionConfirmed = true;
                ob.ltfBOSConfirmed = true;
                ob.state = 'CONFIRMED_MITIGATED';
              }
            }

            // ANY STATE → INVALIDATED (price closes above OB high)
            if (c.close > zoneHigh + (zoneHigh * 0.005)) {
              ob.state = 'INVALIDATED';
              ob.broken = true;
            }
          }
        }

        // Include all OBs except INVALIDATED ones
        if (ob.state !== 'INVALIDATED') {
          orderBlocks.push(ob);
        }
      }
    }

    return orderBlocks;
  }

  /**
   * Detect Fair Value Gaps with STATE MACHINE
   * FVG States: NEW_FVG → PENDING_FILL → FILLING → PARTIAL_FILL/STRONG_FVG → FULLY_FILLED
   * Trade: STRONG_FVG (rejected at 50% fill level)
   */
  static detectFVG(candles: Candle[]): FVG[] {
    const fvgList: FVG[] = [];
    const currentIdx = candles.length - 1;
    const currentPrice = candles[currentIdx].close;

    // Calculate ATR for size filtering
    let atr = 0;
    const atrPeriod = Math.min(14, candles.length - 1);
    for (let i = candles.length - atrPeriod; i < candles.length; i++) {
      if (i >= 0) atr += candles[i].high - candles[i].low;
    }
    atr = atr / atrPeriod || 1;

    for (let i = 1; i < candles.length - 1; i++) {
      const prev = candles[i - 1];
      const curr = candles[i];
      const next = candles[i + 1];

      let fvg: FVG | null = null;

      // Bullish FVG: gap between prev candle's high and next candle's low
      if (prev.high < next.low) {
        const size = next.low - prev.high;
        const sizeATR = size / atr;

        // Only include significant gaps (> 0.3 ATR)
        if (sizeATR > 0.3) {
          fvg = {
            index: i,
            from: prev.high,    // Bottom of gap
            to: next.low,       // Top of gap
            mid: (prev.high + next.low) / 2,  // 50% level
            type: 'bull',
            size,
            sizeATR,
            state: 'NEW_FVG' as FVGState,
            filled: false,
            partiallyFilled: false,
            fillPercent: 0,
            rejectedAt50: false,
            formedAfterBOS: false,  // Set later
            hasDisplacement: curr.close > curr.open && (curr.high - curr.low) > atr,
            overlapsOB: false,      // Set later
            inOTE: false,           // Set later
            barsAgo: currentIdx - i
          };
        }
      }

      // Bearish FVG: gap between prev candle's low and next candle's high
      if (!fvg && prev.low > next.high) {
        const size = prev.low - next.high;
        const sizeATR = size / atr;

        if (sizeATR > 0.3) {
          fvg = {
            index: i,
            from: next.high,
            to: prev.low,
            mid: (next.high + prev.low) / 2,
            type: 'bear',
            size,
            sizeATR,
            state: 'NEW_FVG' as FVGState,
            filled: false,
            partiallyFilled: false,
            fillPercent: 0,
            rejectedAt50: false,
            formedAfterBOS: false,
            hasDisplacement: curr.close < curr.open && (curr.high - curr.low) > atr,
            overlapsOB: false,
            inOTE: false,
            barsAgo: currentIdx - i
          };
        }
      }

      if (fvg) {
        // ═══════════════════════════════════════════════════════════════
        // FVG STATE MACHINE
        // ═══════════════════════════════════════════════════════════════
        let prevCandleWasInGap = false;

        for (let j = i + 2; j < candles.length; j++) {
          const c = candles[j];
          const prevC = candles[j - 1];

          if (fvg.type === 'bull') {
            // Bullish FVG fills when price drops INTO the gap
            const priceInGap = c.low <= fvg.to && c.low >= fvg.from;
            const priceBelowMid = c.low <= fvg.mid;

            // State transitions
            if (fvg.state === 'NEW_FVG' && !priceInGap) {
              fvg.state = 'PENDING_FILL';
            }

            if (fvg.state === 'PENDING_FILL' && priceInGap) {
              fvg.state = 'FILLING';
              fvg.partiallyFilled = true;
            }

            if (fvg.state === 'FILLING') {
              const filledAmount = fvg.to - Math.max(c.low, fvg.from);
              fvg.fillPercent = Math.min(1, filledAmount / fvg.size);

              // Check for rejection at 50% level (STRONG_FVG)
              if (priceBelowMid && c.close > fvg.mid && c.close > c.open) {
                fvg.rejectedAt50 = true;
                fvg.state = 'STRONG_FVG';  // TRADEABLE
              }

              // Fully filled
              if (c.low <= fvg.from) {
                fvg.filled = true;
                fvg.fillPercent = 1;
                fvg.state = 'FULLY_FILLED';
              }
            }

          } else {
            // Bearish FVG fills when price rises INTO the gap
            const priceInGap = c.high >= fvg.from && c.high <= fvg.to;
            const priceAboveMid = c.high >= fvg.mid;

            if (fvg.state === 'NEW_FVG' && !priceInGap) {
              fvg.state = 'PENDING_FILL';
            }

            if (fvg.state === 'PENDING_FILL' && priceInGap) {
              fvg.state = 'FILLING';
              fvg.partiallyFilled = true;
            }

            if (fvg.state === 'FILLING') {
              const filledAmount = Math.min(c.high, fvg.to) - fvg.from;
              fvg.fillPercent = Math.min(1, filledAmount / fvg.size);

              // Check for rejection at 50% level (STRONG_FVG)
              if (priceAboveMid && c.close < fvg.mid && c.close < c.open) {
                fvg.rejectedAt50 = true;
                fvg.state = 'STRONG_FVG';  // TRADEABLE
              }

              // Fully filled
              if (c.high >= fvg.to) {
                fvg.filled = true;
                fvg.fillPercent = 1;
                fvg.state = 'FULLY_FILLED';
              }
            }
          }
        }

        // Include FVGs that aren't FULLY_FILLED or DEAD
        if (fvg.state !== 'FULLY_FILLED' && fvg.state !== 'DEAD') {
          fvgList.push(fvg);
        }
      }
    }

    return fvgList;
  }

  /**
   * Detect Liquidity Zones with STATE MACHINE
   * Trade AFTER liquidity grab + reversal, not before
   */
  static detectLiquidity(candles: Candle[], swingPeriod: number = 5): LiquidityZones {
    const highs: Array<{ price: number; touches: number; lastTouch: number }> = [];
    const lows: Array<{ price: number; touches: number; lastTouch: number }> = [];
    const zones: LiquidityZone[] = [];

    const currentIdx = candles.length - 1;

    // Find swing highs and lows
    for (let i = swingPeriod; i < candles.length - swingPeriod; i++) {
      const isSwingHigh = candles.slice(i - swingPeriod, i + swingPeriod + 1)
        .every(c => c.high <= candles[i].high);

      if (isSwingHigh) {
        highs.push({
          price: candles[i].high,
          touches: 0,
          lastTouch: i
        });

        // Create liquidity zone with state
        const zone: LiquidityZone = {
          price: candles[i].high,
          type: 'high',
          touches: 1,
          state: 'TARGET_ZONE',
          grabbedIndex: -1,
          reversalConfirmed: false,
          sessionType: 'other'
        };

        // Check if this liquidity was grabbed (swept) and what happened after
        for (let j = i + 1; j < candles.length; j++) {
          const c = candles[j];

          // Wick swept the level (liquidity grab)
          if (c.high > zone.price && zone.state === 'TARGET_ZONE') {
            zone.state = 'GRABBED';
            zone.grabbedIndex = j;
          }

          // After grab, check for reversal (close back below + bearish candle)
          if (zone.state === 'GRABBED' && j > zone.grabbedIndex) {
            if (c.close < zone.price && c.close < c.open) {
              zone.reversalConfirmed = true;
              zone.state = 'VALID_LIQUIDITY';  // TRADEABLE
              break;
            }
            // If no reversal within 5 candles, dead
            if (j - zone.grabbedIndex > 5) {
              zone.state = 'DEAD';
              break;
            }
          }
        }

        zones.push(zone);
      }

      const isSwingLow = candles.slice(i - swingPeriod, i + swingPeriod + 1)
        .every(c => c.low >= candles[i].low);

      if (isSwingLow) {
        lows.push({
          price: candles[i].low,
          touches: 0,
          lastTouch: i
        });

        const zone: LiquidityZone = {
          price: candles[i].low,
          type: 'low',
          touches: 1,
          state: 'TARGET_ZONE',
          grabbedIndex: -1,
          reversalConfirmed: false,
          sessionType: 'other'
        };

        for (let j = i + 1; j < candles.length; j++) {
          const c = candles[j];

          // Wick swept the level
          if (c.low < zone.price && zone.state === 'TARGET_ZONE') {
            zone.state = 'GRABBED';
            zone.grabbedIndex = j;
          }

          // After grab, check for reversal
          if (zone.state === 'GRABBED' && j > zone.grabbedIndex) {
            if (c.close > zone.price && c.close > c.open) {
              zone.reversalConfirmed = true;
              zone.state = 'VALID_LIQUIDITY';
              break;
            }
            if (j - zone.grabbedIndex > 5) {
              zone.state = 'DEAD';
              break;
            }
          }
        }

        zones.push(zone);
      }
    }

    return { highs, lows, zones };
  }

  /**
   * Determine trend using SMA crossovers
   */
  static getTrend(candles: Candle[]): 'up' | 'down' | null {
    if (candles.length < 200) return null;
    
    const closes = candles.map(c => c.close);
    const sma50 = this.sma(closes, 50);
    const sma200 = this.sma(closes, 200);
    
    const lastSMA50 = sma50[sma50.length - 1];
    const lastSMA200 = sma200[sma200.length - 1];
    
    if (lastSMA50 > lastSMA200) return 'up';
    if (lastSMA50 < lastSMA200) return 'down';
    return null;
  }

  /**
   * Detect Break of Structure (BOS)
   */
  static getBOS(candles: Candle[]): 'up' | 'down' | null {
    if (candles.length < 50) return null;

    const closes = candles.map(c => c.close);
    const sma50 = this.sma(closes, 50);
    const prevSMA50 = sma50[sma50.length - 2];
    const lastSMA50 = sma50[sma50.length - 1];

    if (lastSMA50 > prevSMA50) return 'up';
    if (lastSMA50 < prevSMA50) return 'down';
    return null;
  }

  /**
   * Detect Pullback in Trend
   * A pullback is when price retraces against the trend direction
   * Best entries happen at 0.382, 0.5, or 0.618 fib levels
   */
  static detectPullback(candles: Candle[], trend: 'up' | 'down' | null): PullbackInfo {
    const defaultResult: PullbackInfo = {
      isPullback: false,
      pullbackDepth: 0,
      pullbackBars: 0,
      swingHigh: 0,
      swingLow: 0,
      fibLevel: 'none'
    };

    if (!trend || candles.length < 30) return defaultResult;

    const recent = candles.slice(-30);
    const currentPrice = recent[recent.length - 1].close;

    // Find the swing points in the last 30 candles
    let swingHigh = -Infinity;
    let swingLow = Infinity;
    let swingHighIdx = 0;
    let swingLowIdx = 0;

    for (let i = 0; i < recent.length; i++) {
      if (recent[i].high > swingHigh) {
        swingHigh = recent[i].high;
        swingHighIdx = i;
      }
      if (recent[i].low < swingLow) {
        swingLow = recent[i].low;
        swingLowIdx = i;
      }
    }

    const range = swingHigh - swingLow;
    if (range <= 0) return defaultResult;

    // Calculate retracement depth
    let pullbackDepth = 0;
    let isPullback = false;
    let pullbackBars = 0;

    if (trend === 'up') {
      // In uptrend, pullback = price dropping from swing high
      // We want swing high to be BEFORE current price (impulse up, then pullback)
      if (swingHighIdx < recent.length - 3 && swingHighIdx > swingLowIdx) {
        // Price made a high, then pulled back
        pullbackDepth = (swingHigh - currentPrice) / range;
        pullbackBars = recent.length - 1 - swingHighIdx;
        isPullback = pullbackDepth > 0.2 && pullbackDepth < 0.8; // Valid pullback range
      }
    } else {
      // In downtrend, pullback = price rising from swing low
      if (swingLowIdx < recent.length - 3 && swingLowIdx > swingHighIdx) {
        // Price made a low, then pulled back up
        pullbackDepth = (currentPrice - swingLow) / range;
        pullbackBars = recent.length - 1 - swingLowIdx;
        isPullback = pullbackDepth > 0.2 && pullbackDepth < 0.8;
      }
    }

    // Determine which fib level price is near
    let fibLevel: '0.382' | '0.5' | '0.618' | '0.786' | 'none' = 'none';
    if (isPullback) {
      if (pullbackDepth >= 0.35 && pullbackDepth <= 0.42) fibLevel = '0.382';
      else if (pullbackDepth >= 0.47 && pullbackDepth <= 0.53) fibLevel = '0.5';
      else if (pullbackDepth >= 0.58 && pullbackDepth <= 0.65) fibLevel = '0.618';
      else if (pullbackDepth >= 0.75 && pullbackDepth <= 0.82) fibLevel = '0.786';
    }

    return {
      isPullback,
      pullbackDepth,
      pullbackBars,
      swingHigh,
      swingLow,
      fibLevel
    };
  }

  /**
   * Detect return-to-zone signals
   * These are the actual ENTRY signals - when price is returning to a valid zone
   */
  static detectReturnSignals(
    candles: Candle[],
    orderBlocks: OrderBlock[],
    fvgs: FVG[],
    trend: 'up' | 'down' | null
  ): SMCReturnSignal[] {
    const signals: SMCReturnSignal[] = [];
    const currentPrice = candles[candles.length - 1].close;
    const currentLow = candles[candles.length - 1].low;
    const currentHigh = candles[candles.length - 1].high;

    // Check Order Block returns
    for (const ob of orderBlocks) {
      const zoneHigh = Math.max(ob.open, ob.close);
      const zoneLow = Math.min(ob.open, ob.close);

      // Skip if OB is too old (> 100 bars)
      if (ob.barsAgo > 100) continue;

      // ═══════════════════════════════════════════════════════════════
      // OB STATE-BASED SIGNAL GENERATION
      // "An order block is only tradable after mitigation, not when it first forms"
      // ═══════════════════════════════════════════════════════════════

      // Skip NEW_OB - never trade at formation
      if (ob.state === 'NEW_OB') continue;

      // Check if current price is IN or NEAR the zone
      const inZone = currentLow <= zoneHigh && currentHigh >= zoneLow;
      const distToZone = ob.type === 'bull'
        ? (currentPrice - zoneHigh) / currentPrice
        : (zoneLow - currentPrice) / currentPrice;

      // Signal if price is in zone or within 1% of zone
      if (inZone || Math.abs(distToZone) < 0.01) {
        let strength = 0.3;  // Base strength

        // ═══════════════════════════════════════════════════════════════
        // STATE-BASED SCORING (THE KEY TO PROPER SMC!)
        // ═══════════════════════════════════════════════════════════════

        if (ob.state === 'CONFIRMED_MITIGATED') {
          // BEST: Price returned, rejected, and confirmed - ENTRY ALLOWED
          strength += 0.5;
        } else if (ob.state === 'IN_MITIGATION') {
          // GOOD: Price is testing zone - watch for confirmation
          strength += 0.25;
        } else if (ob.state === 'WAITING_FOR_MITIGATION') {
          // NEUTRAL: First touch incoming - this IS the first return (best!)
          if (ob.testCount === 0) {
            strength += 0.4;  // First-time return to fresh OB
          } else {
            strength += 0.1;  // Already tested before
          }
        }

        // PENALTY: Multiple tests weaken the zone (mitigation exhaustion)
        if (ob.testCount > 1) {
          strength -= 0.15 * (ob.testCount - 1);  // -15% per extra test
        }

        // OB QUALITY BONUS (from formation analysis)
        if (ob.causedBOS) strength += 0.1;
        if (ob.hasDisplacement) strength += 0.1;
        if (ob.hasFVG) strength += 0.05;
        if (ob.impulseSize > 2) strength += 0.1;  // Strong impulse

        // Age bonus (fresher OBs are stronger)
        if (ob.barsAgo < 20) strength += 0.1;
        else if (ob.barsAgo < 50) strength += 0.05;

        // Trend alignment
        if ((ob.type === 'bull' && trend === 'up') ||
            (ob.type === 'bear' && trend === 'down')) {
          strength += 0.15;
        }

        // Clamp strength to 0-1
        strength = Math.max(0, Math.min(1, strength));

        signals.push({
          type: 'ob_return',
          zone: ob,
          direction: ob.type === 'bull' ? 'long' : 'short',
          strength: Math.min(1, strength),
          priceInZone: inZone,
          distanceToZone: distToZone
        });
      }
    }

    // Check FVG fills
    for (const fvg of fvgs) {
      // Skip if FVG is too old or already mostly filled
      if (fvg.barsAgo > 100 || fvg.fillPercent > 0.8) continue;

      // Check if current price is entering the gap
      const inGap = currentLow <= fvg.to && currentHigh >= fvg.from;
      const distToGap = fvg.type === 'bull'
        ? (currentPrice - fvg.to) / currentPrice   // Distance to top of gap (we enter from top)
        : (fvg.from - currentPrice) / currentPrice; // Distance to bottom of gap (we enter from bottom)

      // Signal if price is in gap or within 0.5% of entering
      if (inGap || Math.abs(distToGap) < 0.005) {
        let strength = 0.5;

        // Age bonus
        if (fvg.barsAgo < 20) strength += 0.2;
        else if (fvg.barsAgo < 50) strength += 0.1;

        // Size bonus (bigger gaps = more significant)
        const gapPercent = fvg.size / currentPrice;
        if (gapPercent > 0.02) strength += 0.15;
        else if (gapPercent > 0.01) strength += 0.1;

        // Trend alignment
        if ((fvg.type === 'bull' && trend === 'up') ||
            (fvg.type === 'bear' && trend === 'down')) {
          strength += 0.25;
        }

        // Fresh gaps (not partially filled) are stronger
        if (fvg.fillPercent < 0.2) strength += 0.1;

        signals.push({
          type: 'fvg_fill',
          zone: fvg,
          direction: fvg.type === 'bull' ? 'long' : 'short',
          strength: Math.min(1, strength),
          priceInZone: inGap,
          distanceToZone: distToGap
        });
      }
    }

    // Sort by strength (best signals first)
    signals.sort((a, b) => b.strength - a.strength);

    return signals;
  }

  /**
   * Perform full SMC analysis on candles
   */
  static analyze(candles: Candle[]): SMCAnalysis {
    const closes = candles.map(c => c.close);

    const ema50Arr = this.ema(closes, 50);
    const ema200Arr = this.ema(closes, 200);
    const rsiArr = this.rsi(candles, 14);
    const atrArr = this.atr(candles, 14);
    const vwapArr = this.vwap(candles);

    const trend = this.getTrend(candles);
    const bos = this.getBOS(candles);
    const orderBlocks = this.detectOrderBlocks(candles);
    const fvg = this.detectFVG(candles);
    const liquidityZones = this.detectLiquidity(candles);

    // Detect Change of Character (CHoCH) - trend reversal signal
    const choch = bos !== null && trend !== null && bos !== trend;

    // Get return signals (the actual entry opportunities)
    const returnSignals = this.detectReturnSignals(candles, orderBlocks, fvg, trend);

    // Generate tiered signals
    const signals: SMCSignal[] = [];
    const currentPrice = candles[candles.length - 1].close;

    // TIER 1 (90%): HTF OB/FVG confirmed + structure alignment
    for (const ob of orderBlocks) {
      if (ob.state === 'CONFIRMED_MITIGATED' && ob.causedBOS) {
        const direction = ob.type === 'bull' ? 'long' : 'short';
        if ((direction === 'long' && trend === 'up') || (direction === 'short' && trend === 'down')) {
          signals.push({
            tier: 1,
            type: 'ob_return',
            direction,
            strength: 0.9,
            zone: ob,
            confluence: ['CONFIRMED_MITIGATED', 'causedBOS', 'trend_aligned']
          });
        }
      }
    }

    // TIER 2 (70%): Strong FVG + rejection
    for (const gap of fvg) {
      if (gap.state === 'STRONG_FVG' && gap.rejectedAt50) {
        signals.push({
          tier: 2,
          type: 'fvg_fill',
          direction: gap.type === 'bull' ? 'long' : 'short',
          strength: 0.7,
          zone: gap,
          confluence: ['STRONG_FVG', 'rejected_at_50']
        });
      }
    }

    // TIER 3 (50%): OB in mitigation with displacement
    for (const ob of orderBlocks) {
      if (ob.state === 'IN_MITIGATION' && ob.hasDisplacement) {
        signals.push({
          tier: 3,
          type: 'ob_return',
          direction: ob.type === 'bull' ? 'long' : 'short',
          strength: 0.5,
          zone: ob,
          confluence: ['IN_MITIGATION', 'has_displacement']
        });
      }
    }

    // Sort signals by tier (best first)
    signals.sort((a, b) => a.tier - b.tier || b.strength - a.strength);

    return {
      trend,
      bos,
      choch,
      ema50: ema50Arr[ema50Arr.length - 1] || null,
      ema200: ema200Arr[ema200Arr.length - 1] || null,
      rsi: rsiArr[rsiArr.length - 1] || null,
      orderBlocks,
      fvg,
      liquidityZones,
      breakerBlocks: [],  // TODO: Implement breaker detection
      atr: atrArr[atrArr.length - 1] || null,
      vwap: vwapArr[vwapArr.length - 1] || null,
      pullback: this.detectPullback(candles, trend),
      signals,
      bestSignal: signals.length > 0 ? signals[0] : null,
      returnSignals
    };
  }

  /**
   * Score a setup based on SMC factors
   */
  static scoreSetup(
    analysis: SMCAnalysis,
    currentPrice: number,
    weights: Record<string, number>
  ): { score: number; breakdown: Record<string, number>; confluence: string[] } {
    const breakdown: Record<string, number> = {};
    const confluence: string[] = [];
    let totalScore = 0;

    // Trend Structure (0-40 points)
    if (analysis.trend && analysis.bos && analysis.trend === analysis.bos) {
      breakdown.trend_structure = weights.trend_structure;
      confluence.push(`${analysis.trend} trend + BOS aligned`);
    } else if (analysis.trend) {
      breakdown.trend_structure = weights.trend_structure * 0.5;
      confluence.push(`${analysis.trend} trend`);
    } else {
      breakdown.trend_structure = 0;
    }
    totalScore += breakdown.trend_structure || 0;

    // Order Blocks (0-30 points)
    const relevantOBs = analysis.orderBlocks.filter(ob => {
      const obHigh = Math.max(ob.open, ob.close);
      const obLow = Math.min(ob.open, ob.close);
      return currentPrice >= obLow * 0.95 && currentPrice <= obHigh * 1.05;
    });

    if (relevantOBs.length > 0) {
      const bullOBs = relevantOBs.filter(ob => ob.type === 'bull');
      const bearOBs = relevantOBs.filter(ob => ob.type === 'bear');
      
      if (analysis.trend === 'up' && bullOBs.length > 0) {
        breakdown.order_blocks = weights.order_blocks;
        confluence.push(`${bullOBs.length} bullish order block(s)`);
      } else if (analysis.trend === 'down' && bearOBs.length > 0) {
        breakdown.order_blocks = weights.order_blocks;
        confluence.push(`${bearOBs.length} bearish order block(s)`);
      } else {
        breakdown.order_blocks = weights.order_blocks * 0.3;
        confluence.push(`${relevantOBs.length} order block(s) (mixed/aligned)`);
      }
    } else {
      breakdown.order_blocks = 0;
    }
    totalScore += breakdown.order_blocks || 0;

    // FVGs (0-20 points)
    const relevantFVGs = analysis.fvg.filter(fvg => {
      return currentPrice >= fvg.from * 0.98 && currentPrice <= fvg.to * 1.02;
    });

    if (relevantFVGs.length > 0) {
      breakdown.fvgs = weights.fvgs;
      confluence.push(`${relevantFVGs.length} FVG(s)`);
    } else {
      breakdown.fvgs = 0;
    }
    totalScore += breakdown.fvgs || 0;

    // EMA Alignment (0-15 points)
    if (analysis.ema50 && analysis.ema200) {
      const emaTrend = analysis.ema50 > analysis.ema200 ? 'up' : 'down';
      if (analysis.trend === emaTrend) {
        breakdown.ema_alignment = weights.ema_alignment;
        confluence.push(`EMA aligned ${emaTrend}`);
      } else {
        breakdown.ema_alignment = 0;
      }
    } else {
      breakdown.ema_alignment = 0;
    }
    totalScore += breakdown.ema_alignment || 0;

    // Liquidity (0-10 points)
    const hasLiquidity = analysis.liquidityZones.highs.length > 0 || 
                         analysis.liquidityZones.lows.length > 0;
    if (hasLiquidity) {
      breakdown.liquidity = weights.liquidity;
      confluence.push(`Liquidity zones present`);
    } else {
      breakdown.liquidity = 0;
    }
    totalScore += breakdown.liquidity || 0;

    // MTF Bonus (0-35 points)
    breakdown.mtf_bonus = weights.mtf_bonus;
    totalScore += breakdown.mtf_bonus;

    // RSI Penalty (-15 to 0)
    if (analysis.rsi) {
      if (analysis.rsi > 70) {
        breakdown.rsi_penalty = weights.rsi_penalty;
        confluence.push(`RSI overbought (${analysis.rsi.toFixed(1)})`);
      } else if (analysis.rsi < 30) {
        breakdown.rsi_penalty = weights.rsi_penalty * 0.5;
        confluence.push(`RSI oversold (${analysis.rsi.toFixed(1)})`);
      } else {
        breakdown.rsi_penalty = 0;
      }
    } else {
      breakdown.rsi_penalty = 0;
    }
    totalScore += breakdown.rsi_penalty || 0;

    return { score: Math.max(0, totalScore), breakdown, confluence };
  }
}