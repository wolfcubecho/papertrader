/**
 * ICT (Inner Circle Trader) Indicators
 *
 * Proper institutional methodology for SMC trading:
 * - OTE Zones (61.8-79% Fibonacci)
 * - Displacement Detection (3+ candles >1.5 ATR)
 * - Liquidity Grab Detection
 * - Judas Swing (Asian fakeout)
 * - Killzone Timing
 * - HTF Cascade Validation
 */

import { Candle, SMCIndicators, OrderBlock, FVG, SMCAnalysis } from './smc-indicators.js';

// ═══════════════════════════════════════════════════════════════
// INTERFACES
// ═══════════════════════════════════════════════════════════════

export interface OTEZone {
  top: number;           // 61.8% level
  bottom: number;        // 79% level (deeper retracement)
  impulseHigh: number;   // Top of impulse leg
  impulseLow: number;    // Bottom of impulse leg
  direction: 'bullish' | 'bearish';
  valid: boolean;
}

export interface Displacement {
  detected: boolean;
  direction: 'up' | 'down';
  candleCount: number;      // How many displacement candles
  totalMove: number;        // Total price move
  avgCandleSize: number;    // Average body size
  atrMultiple: number;      // Move as multiple of ATR
  startIndex: number;
  endIndex: number;
}

export interface LiquidityGrab {
  detected: boolean;
  type: 'buy_side' | 'sell_side';  // Buy-side = grabbed longs' stops, Sell-side = grabbed shorts' stops
  grabLevel: number;        // Price level that was grabbed
  wickSize: number;         // Size of the wick
  volumeSpike: number;      // Volume relative to average
  candleIndex: number;
}

export interface JudasSwing {
  detected: boolean;
  direction: 'bullish' | 'bearish';  // The REAL direction after the fake
  fakeoutLevel: number;     // Where price faked to
  asianHigh: number;
  asianLow: number;
  manipulationComplete: boolean;
}

export interface Killzone {
  active: boolean;
  name: 'london' | 'new_york' | 'asian' | 'none';
  hoursRemaining: number;
}

export interface MarketStructure {
  trend: 'bullish' | 'bearish' | 'ranging';
  lastBOS: { price: number; index: number; direction: 'up' | 'down' } | null;
  lastCHoCH: { price: number; index: number; direction: 'up' | 'down' } | null;
  swingHigh: number;
  swingLow: number;
  equilibrium: number;      // (swingHigh + swingLow) / 2
  inDiscount: boolean;      // price < equilibrium
  inPremium: boolean;       // price > equilibrium
}

export interface HTFCascade {
  aligned: boolean;
  htfTrend: 'bullish' | 'bearish' | 'ranging';
  mtfTrend: 'bullish' | 'bearish' | 'ranging';
  ltfTrend: 'bullish' | 'bearish' | 'ranging';
  alignmentScore: number;   // 0-1, how well timeframes align
}

export interface EntryChecklist {
  // The 12-Condition Entry Checklist
  conditions: {
    htfPOIAlignment: boolean;      // 1. HTF POI (OB/FVG) alignment
    displacementPresent: boolean;  // 2. Displacement confirmed
    inOTEZone: boolean;            // 3. Price in OTE zone
    obFvgConfluence: boolean;      // 4. OB or FVG confluence
    killzoneActive: boolean;       // 5. In killzone
    liquiditySwept: boolean;       // 6. Liquidity has been swept
    bosConfirmed: boolean;         // 7. BOS confirmed
    noCHoCH: boolean;              // 8. No Change of Character
    properSession: boolean;        // 9. Good trading session
    trendAligned: boolean;         // 10. Market structure matches direction
    discountPremium: boolean;      // 11. Buying discount / selling premium
    cleanPriceAction: boolean;     // 12. Clean, readable price action
  };
  passedCount: number;
  totalConditions: number;
  passRate: number;
  entryGrade: 'A' | 'B' | 'C' | 'D' | 'F';
}

export interface ICTAnalysis {
  structure: MarketStructure;
  ote: OTEZone;
  displacement: Displacement;
  liquidityGrab: LiquidityGrab;
  judasSwing: JudasSwing;
  killzone: Killzone;

  // Zone confluence
  obInOTE: OrderBlock[];    // Order blocks that overlap OTE
  fvgInOTE: FVG[];          // FVGs that overlap OTE

  // HTF Cascade (multi-timeframe alignment)
  htfCascade: HTFCascade;

  // 12-Condition Entry Checklist
  entryChecklist: EntryChecklist;

  // Entry validity
  entryValid: boolean;
  entryScore: number;       // 0-100
  entryReasons: string[];
  noEntryReasons: string[];
}

// ═══════════════════════════════════════════════════════════════
// ICT INDICATORS CLASS
// ═══════════════════════════════════════════════════════════════

export class ICTIndicators {

  /**
   * Calculate ATR for a candle array
   */
  static calculateATR(candles: Candle[], period: number = 14): number {
    if (candles.length < period) return candles[candles.length - 1].high - candles[candles.length - 1].low;

    const atrValues = SMCIndicators.atr(candles, period);
    return atrValues[atrValues.length - 1] || 0;
  }

  /**
   * Detect Market Structure (BOS, CHoCH, Swing Points)
   * OPTIMIZED: Uses 3-bar pivot instead of 5-bar for faster computation
   */
  static detectMarketStructure(candles: Candle[], lookback: number = 30): MarketStructure {
    if (candles.length < lookback) {
      return {
        trend: 'ranging',
        lastBOS: null,
        lastCHoCH: null,
        swingHigh: candles[candles.length - 1].high,
        swingLow: candles[candles.length - 1].low,
        equilibrium: (candles[candles.length - 1].high + candles[candles.length - 1].low) / 2,
        inDiscount: false,
        inPremium: false
      };
    }

    const recent = candles.slice(-lookback);
    const currentPrice = recent[recent.length - 1].close;

    // Find swing highs and lows (using 3-bar pivot for speed)
    const swingHighs: { price: number; index: number }[] = [];
    const swingLows: { price: number; index: number }[] = [];

    for (let i = 3; i < recent.length - 3; i++) {
      // Simple 3-bar pivot check (faster than slice)
      const isSwingHigh = recent[i].high >= recent[i-1].high &&
                          recent[i].high >= recent[i-2].high &&
                          recent[i].high >= recent[i-3].high &&
                          recent[i].high >= recent[i+1].high &&
                          recent[i].high >= recent[i+2].high &&
                          recent[i].high >= recent[i+3].high;
      const isSwingLow = recent[i].low <= recent[i-1].low &&
                         recent[i].low <= recent[i-2].low &&
                         recent[i].low <= recent[i-3].low &&
                         recent[i].low <= recent[i+1].low &&
                         recent[i].low <= recent[i+2].low &&
                         recent[i].low <= recent[i+3].low;

      if (isSwingHigh) swingHighs.push({ price: recent[i].high, index: i });
      if (isSwingLow) swingLows.push({ price: recent[i].low, index: i });
    }

    // Determine trend from swing sequence
    let trend: 'bullish' | 'bearish' | 'ranging' = 'ranging';
    let lastBOS: { price: number; index: number; direction: 'up' | 'down' } | null = null;
    let lastCHoCH: { price: number; index: number; direction: 'up' | 'down' } | null = null;

    if (swingHighs.length >= 2 && swingLows.length >= 2) {
      const lastTwoHighs = swingHighs.slice(-2);
      const lastTwoLows = swingLows.slice(-2);

      // Higher highs and higher lows = bullish
      if (lastTwoHighs[1].price > lastTwoHighs[0].price &&
          lastTwoLows[1].price > lastTwoLows[0].price) {
        trend = 'bullish';
      }
      // Lower highs and lower lows = bearish
      else if (lastTwoHighs[1].price < lastTwoHighs[0].price &&
               lastTwoLows[1].price < lastTwoLows[0].price) {
        trend = 'bearish';
      }

      // Detect BOS (Break of Structure)
      // Bullish BOS = close above previous swing high
      // Bearish BOS = close below previous swing low
      for (let i = recent.length - 1; i >= recent.length - 10 && i >= 0; i--) {
        const candle = recent[i];

        // Check for bullish BOS
        const prevSwingHigh = swingHighs.filter(sh => sh.index < i).slice(-1)[0];
        if (prevSwingHigh && candle.close > prevSwingHigh.price) {
          lastBOS = { price: prevSwingHigh.price, index: i, direction: 'up' };
          break;
        }

        // Check for bearish BOS
        const prevSwingLow = swingLows.filter(sl => sl.index < i).slice(-1)[0];
        if (prevSwingLow && candle.close < prevSwingLow.price) {
          lastBOS = { price: prevSwingLow.price, index: i, direction: 'down' };
          break;
        }
      }

      // Detect CHoCH (Change of Character) - break against trend
      if (trend === 'bullish') {
        // CHoCH in bullish trend = break below swing low
        const recentSwingLow = swingLows.slice(-1)[0];
        if (recentSwingLow && currentPrice < recentSwingLow.price) {
          lastCHoCH = { price: recentSwingLow.price, index: recent.length - 1, direction: 'down' };
        }
      } else if (trend === 'bearish') {
        // CHoCH in bearish trend = break above swing high
        const recentSwingHigh = swingHighs.slice(-1)[0];
        if (recentSwingHigh && currentPrice > recentSwingHigh.price) {
          lastCHoCH = { price: recentSwingHigh.price, index: recent.length - 1, direction: 'up' };
        }
      }
    }

    const swingHigh = swingHighs.length > 0 ? swingHighs[swingHighs.length - 1].price : recent[recent.length - 1].high;
    const swingLow = swingLows.length > 0 ? swingLows[swingLows.length - 1].price : recent[recent.length - 1].low;
    const equilibrium = (swingHigh + swingLow) / 2;

    return {
      trend,
      lastBOS,
      lastCHoCH,
      swingHigh,
      swingLow,
      equilibrium,
      inDiscount: currentPrice < equilibrium * 0.995,
      inPremium: currentPrice > equilibrium * 1.005
    };
  }

  /**
   * Detect OTE Zone (Optimal Trade Entry)
   * 61.8% - 79% Fibonacci retracement of last impulse
   */
  static detectOTE(candles: Candle[], structure: MarketStructure): OTEZone {
    const invalidOTE: OTEZone = {
      top: 0, bottom: 0, impulseHigh: 0, impulseLow: 0,
      direction: 'bullish', valid: false
    };

    if (candles.length < 20) return invalidOTE;

    const recent = candles.slice(-50);
    const atr = this.calculateATR(candles);

    // Find the last impulse leg (displacement move)
    const displacement = this.detectDisplacement(candles);
    if (!displacement.detected) return invalidOTE;

    let impulseHigh: number;
    let impulseLow: number;
    let direction: 'bullish' | 'bearish';

    if (displacement.direction === 'up') {
      // Bullish impulse - look for OTE below current price
      impulseLow = Math.min(...recent.slice(displacement.startIndex, displacement.endIndex + 1).map(c => c.low));
      impulseHigh = Math.max(...recent.slice(displacement.startIndex, displacement.endIndex + 1).map(c => c.high));
      direction = 'bullish';
    } else {
      // Bearish impulse - look for OTE above current price
      impulseHigh = Math.max(...recent.slice(displacement.startIndex, displacement.endIndex + 1).map(c => c.high));
      impulseLow = Math.min(...recent.slice(displacement.startIndex, displacement.endIndex + 1).map(c => c.low));
      direction = 'bearish';
    }

    const range = impulseHigh - impulseLow;
    if (range <= 0) return invalidOTE;

    // Calculate Fibonacci levels
    // For bullish: OTE is 61.8%-79% retracement FROM the high (pullback into zone)
    // For bearish: OTE is 61.8%-79% retracement FROM the low (pullback into zone)
    let oteTop: number;
    let oteBottom: number;

    if (direction === 'bullish') {
      // Bullish OTE: price should retrace down into this zone
      oteTop = impulseHigh - (range * 0.618);     // 61.8% retracement
      oteBottom = impulseHigh - (range * 0.79);   // 79% retracement
    } else {
      // Bearish OTE: price should retrace up into this zone
      oteBottom = impulseLow + (range * 0.618);   // 61.8% retracement
      oteTop = impulseLow + (range * 0.79);       // 79% retracement
    }

    return {
      top: oteTop,
      bottom: oteBottom,
      impulseHigh,
      impulseLow,
      direction,
      valid: true
    };
  }

  /**
   * Calculate adaptive displacement threshold based on volatility
   * In high volatility, we need larger moves to confirm displacement
   */
  static getAdaptiveDisplacementThreshold(candles: Candle[]): number {
    const atr = this.calculateATR(candles);
    const price = candles[candles.length - 1].close;
    const volatility = atr / price;

    // Higher volatility = lower threshold (1.2x), lower volatility = higher threshold (1.8x)
    return Math.max(1.2, 1.8 - volatility * 100);
  }

  /**
   * Detect Displacement using simplified scoring (OPTIMIZED for speed)
   * Checks last 15 candles for 3+ consecutive strong moves
   */
  static detectDisplacement(candles: Candle[], minCandles: number = 3): Displacement {
    const noDisplacement: Displacement = {
      detected: false, direction: 'up', candleCount: 0,
      totalMove: 0, avgCandleSize: 0, atrMultiple: 0,
      startIndex: 0, endIndex: 0
    };

    if (candles.length < minCandles + 14) return noDisplacement;

    const atr = this.calculateATR(candles);
    const threshold = 1.2; // Simplified threshold
    const recent = candles.slice(-15); // Only check last 15 candles

    let consecutiveUp = 0;
    let consecutiveDown = 0;
    let totalMoveUp = 0;
    let totalMoveDown = 0;
    let startIdx = 0;

    // Single pass through recent candles
    for (let i = 0; i < recent.length; i++) {
      const candle = recent[i];
      const range = candle.high - candle.low;
      const body = Math.abs(candle.close - candle.open);

      // Is this a strong candle? (range >= threshold*ATR AND body >= 50% of range)
      const isStrong = range >= threshold * atr && body >= 0.5 * range;

      if (!isStrong) {
        // Reset counters
        if (consecutiveUp >= minCandles) break; // Already found
        if (consecutiveDown >= minCandles) break;
        consecutiveUp = 0;
        consecutiveDown = 0;
        totalMoveUp = 0;
        totalMoveDown = 0;
        startIdx = i + 1;
        continue;
      }

      if (candle.close > candle.open) {
        if (consecutiveDown > 0) {
          // Direction change - reset
          consecutiveDown = 0;
          totalMoveDown = 0;
          startIdx = i;
        }
        consecutiveUp++;
        totalMoveUp += body;
      } else {
        if (consecutiveUp > 0) {
          // Direction change - reset
          consecutiveUp = 0;
          totalMoveUp = 0;
          startIdx = i;
        }
        consecutiveDown++;
        totalMoveDown += body;
      }
    }

    // Check if we found displacement
    if (consecutiveUp >= minCandles) {
      return {
        detected: true,
        direction: 'up',
        candleCount: consecutiveUp,
        totalMove: totalMoveUp,
        avgCandleSize: totalMoveUp / consecutiveUp,
        atrMultiple: totalMoveUp / atr,
        startIndex: startIdx,
        endIndex: recent.length - 1
      };
    }
    if (consecutiveDown >= minCandles) {
      return {
        detected: true,
        direction: 'down',
        candleCount: consecutiveDown,
        totalMove: totalMoveDown,
        avgCandleSize: totalMoveDown / consecutiveDown,
        atrMultiple: totalMoveDown / atr,
        startIndex: startIdx,
        endIndex: recent.length - 1
      };
    }

    return noDisplacement;
  }

  /**
   * Calculate Efficiency Ratio for regime detection
   * Higher ratio = more trending, lower ratio = more ranging
   */
  static calculateEfficiencyRatio(candles: Candle[], periods: number = 30): number {
    if (candles.length < periods + 2) return 0;

    const recent = candles.slice(-periods - 1);
    const startPrice = recent[0].close;
    const endPrice = recent[recent.length - 1].close;
    const netMove = Math.abs(endPrice - startPrice);

    let totalMove = 0;
    for (let i = 1; i < recent.length; i++) {
      totalMove += Math.abs(recent[i].close - recent[i - 1].close);
    }

    return totalMove > 0 ? netMove / totalMove : 0;
  }

  /**
   * Check if market is trending (for BOS trades)
   */
  static isTrendingMarket(candles: Candle[]): boolean {
    const efficiencyRatio = this.calculateEfficiencyRatio(candles, 30);
    return efficiencyRatio >= 0.30; // 30% or higher = trending
  }

  /**
   * Detect Liquidity Grab
   * Wick penetrates level by >0.5 ATR, close returns inside, volume spike
   */
  static detectLiquidityGrab(candles: Candle[], structure: MarketStructure): LiquidityGrab {
    const noGrab: LiquidityGrab = {
      detected: false, type: 'buy_side', grabLevel: 0,
      wickSize: 0, volumeSpike: 0, candleIndex: 0
    };

    if (candles.length < 20) return noGrab;

    const atr = this.calculateATR(candles);
    const recent = candles.slice(-10);
    const avgVolume = candles.slice(-20).reduce((sum, c) => sum + c.volume, 0) / 20;

    // Look for liquidity grab in recent candles
    for (let i = recent.length - 1; i >= recent.length - 5 && i >= 0; i--) {
      const candle = recent[i];
      const volumeSpike = candle.volume / avgVolume;

      // Buy-side liquidity grab (stop hunt below lows, then close above)
      // This is bullish - grabbed sell stops, now going up
      const lowerWick = Math.min(candle.open, candle.close) - candle.low;
      if (lowerWick > atr * 0.5 &&
          candle.low < structure.swingLow &&
          candle.close > structure.swingLow &&
          volumeSpike > 1.5) {
        return {
          detected: true,
          type: 'sell_side',  // Grabbed sell-side liquidity (stops below)
          grabLevel: structure.swingLow,
          wickSize: lowerWick,
          volumeSpike,
          candleIndex: candles.length - recent.length + i
        };
      }

      // Sell-side liquidity grab (stop hunt above highs, then close below)
      // This is bearish - grabbed buy stops, now going down
      const upperWick = candle.high - Math.max(candle.open, candle.close);
      if (upperWick > atr * 0.5 &&
          candle.high > structure.swingHigh &&
          candle.close < structure.swingHigh &&
          volumeSpike > 1.5) {
        return {
          detected: true,
          type: 'buy_side',  // Grabbed buy-side liquidity (stops above)
          grabLevel: structure.swingHigh,
          wickSize: upperWick,
          volumeSpike,
          candleIndex: candles.length - recent.length + i
        };
      }
    }

    return noGrab;
  }

  /**
   * Detect Judas Swing (Asian session fakeout)
   */
  static detectJudasSwing(candles: Candle[]): JudasSwing {
    const noJudas: JudasSwing = {
      detected: false, direction: 'bullish',
      fakeoutLevel: 0, asianHigh: 0, asianLow: 0,
      manipulationComplete: false
    };

    if (candles.length < 24) return noJudas;

    // Find Asian session candles (roughly last 8-12 candles if hourly)
    // Asian session: 8PM - 4AM EST = 1:00 - 9:00 UTC
    const asianCandles: Candle[] = [];

    for (let i = candles.length - 1; i >= Math.max(0, candles.length - 24); i--) {
      const hour = new Date(candles[i].timestamp).getUTCHours();
      if (hour >= 1 && hour < 9) {
        asianCandles.unshift(candles[i]);
      }
    }

    if (asianCandles.length < 4) return noJudas;

    const asianHigh = Math.max(...asianCandles.map(c => c.high));
    const asianLow = Math.min(...asianCandles.map(c => c.low));
    const asianRange = asianHigh - asianLow;

    // Look for fakeout after Asian session
    const postAsianCandles = candles.slice(-5);
    const lastCandle = postAsianCandles[postAsianCandles.length - 1];

    // Bullish Judas: Faked below Asian low, now closed above it
    for (const candle of postAsianCandles) {
      if (candle.low < asianLow - (asianRange * 0.1) && // Wick below
          candle.close > asianLow) {                     // Closed inside
        return {
          detected: true,
          direction: 'bullish',  // Real move will be up
          fakeoutLevel: candle.low,
          asianHigh,
          asianLow,
          manipulationComplete: lastCandle.close > asianLow
        };
      }
    }

    // Bearish Judas: Faked above Asian high, now closed below it
    for (const candle of postAsianCandles) {
      if (candle.high > asianHigh + (asianRange * 0.1) && // Wick above
          candle.close < asianHigh) {                      // Closed inside
        return {
          detected: true,
          direction: 'bearish',  // Real move will be down
          fakeoutLevel: candle.high,
          asianHigh,
          asianLow,
          manipulationComplete: lastCandle.close < asianHigh
        };
      }
    }

    return noJudas;
  }

  /**
   * Check if current time is in a Killzone
   * London: 2-5 AM EST (7-10 UTC)
   * New York: 8-11 AM EST (13-16 UTC)
   * Asian: 8-11 PM EST (1-4 UTC)
   */
  static getKillzone(timestamp?: number): Killzone {
    const now = timestamp ? new Date(timestamp) : new Date();
    const hour = now.getUTCHours();

    // London Killzone: 7-10 UTC (2-5 AM EST)
    if (hour >= 7 && hour < 10) {
      return { active: true, name: 'london', hoursRemaining: 10 - hour };
    }

    // New York Killzone: 13-16 UTC (8-11 AM EST)
    if (hour >= 13 && hour < 16) {
      return { active: true, name: 'new_york', hoursRemaining: 16 - hour };
    }

    // Asian Killzone: 1-4 UTC (8-11 PM EST)
    if (hour >= 1 && hour < 4) {
      return { active: true, name: 'asian', hoursRemaining: 4 - hour };
    }

    return { active: false, name: 'none', hoursRemaining: 0 };
  }

  /**
   * Find OBs and FVGs that overlap with OTE zone
   */
  static findZonesInOTE(
    orderBlocks: OrderBlock[],
    fvgs: FVG[],
    ote: OTEZone
  ): { obInOTE: OrderBlock[]; fvgInOTE: FVG[] } {
    if (!ote.valid) return { obInOTE: [], fvgInOTE: [] };

    // Find OBs that overlap OTE
    const obInOTE = orderBlocks.filter(ob => {
      const obTop = Math.max(ob.open, ob.close);
      const obBottom = Math.min(ob.open, ob.close);
      // Check for overlap
      return !(obTop < ote.bottom || obBottom > ote.top);
    });

    // Find FVGs that overlap OTE
    const fvgInOTE = fvgs.filter(fvg => {
      // Check for overlap
      return !(fvg.to < ote.bottom || fvg.from > ote.top);
    });

    return { obInOTE, fvgInOTE };
  }

  /**
   * HTF Cascade Validation (OPTIMIZED)
   * Simulates multi-timeframe analysis using different lookback periods
   * HTF = 100 bars, MTF = 30 bars, LTF = 10 bars
   */
  static detectHTFCascade(candles: Candle[]): HTFCascade {
    const htfTrend = this.getTrendFromPeriod(candles, 100);
    const mtfTrend = this.getTrendFromPeriod(candles, 30);
    const ltfTrend = this.getTrendFromPeriod(candles, 10);

    // Calculate alignment score
    let alignmentScore = 0;
    const trends = [htfTrend, mtfTrend, ltfTrend];

    // All bullish or all bearish = perfect alignment
    const bullishCount = trends.filter(t => t === 'bullish').length;
    const bearishCount = trends.filter(t => t === 'bearish').length;

    if (bullishCount === 3 || bearishCount === 3) {
      alignmentScore = 1.0;
    } else if (bullishCount === 2 || bearishCount === 2) {
      alignmentScore = 0.66;
    } else if (bullishCount === 1 || bearishCount === 1) {
      alignmentScore = 0.33;
    }

    // HTF must align with MTF for cascade to be valid
    const aligned = htfTrend === mtfTrend && htfTrend !== 'ranging';

    return {
      aligned,
      htfTrend,
      mtfTrend,
      ltfTrend,
      alignmentScore
    };
  }

  /**
   * Get trend direction from a specific lookback period
   */
  private static getTrendFromPeriod(candles: Candle[], period: number): 'bullish' | 'bearish' | 'ranging' {
    if (candles.length < period) return 'ranging';

    const recent = candles.slice(-period);
    const firstQuarter = recent.slice(0, Math.floor(period / 4));
    const lastQuarter = recent.slice(-Math.floor(period / 4));

    const startPrice = firstQuarter.reduce((sum, c) => sum + c.close, 0) / firstQuarter.length;
    const endPrice = lastQuarter.reduce((sum, c) => sum + c.close, 0) / lastQuarter.length;

    const percentChange = ((endPrice - startPrice) / startPrice) * 100;

    // Thresholds adjusted for multi-timeframe
    if (percentChange > 2) return 'bullish';
    if (percentChange < -2) return 'bearish';
    return 'ranging';
  }

  /**
   * Build the 12-Condition Entry Checklist
   */
  static buildEntryChecklist(
    candles: Candle[],
    structure: MarketStructure,
    ote: OTEZone,
    displacement: Displacement,
    liquidityGrab: LiquidityGrab,
    killzone: Killzone,
    obInOTE: OrderBlock[],
    fvgInOTE: FVG[],
    htfCascade: HTFCascade
  ): EntryChecklist {
    const currentPrice = candles[candles.length - 1].close;

    const conditions = {
      // 1. HTF POI alignment - Higher timeframe trend aligns
      htfPOIAlignment: htfCascade.aligned && htfCascade.alignmentScore >= 0.66,

      // 2. Displacement present
      displacementPresent: displacement.detected && displacement.atrMultiple >= 1.5,

      // 3. Price in OTE zone
      inOTEZone: ote.valid && currentPrice >= ote.bottom && currentPrice <= ote.top,

      // 4. OB or FVG confluence in OTE
      obFvgConfluence: obInOTE.length > 0 || fvgInOTE.length > 0,

      // 5. In killzone
      killzoneActive: killzone.active,

      // 6. Liquidity has been swept
      liquiditySwept: liquidityGrab.detected,

      // 7. BOS confirmed
      bosConfirmed: structure.lastBOS !== null,

      // 8. No CHoCH (trend not reversing)
      noCHoCH: structure.lastCHoCH === null,

      // 9. Proper session (London/NY preferred)
      properSession: killzone.name === 'london' || killzone.name === 'new_york',

      // 10. Market structure matches intended direction
      trendAligned: structure.trend !== 'ranging',

      // 11. Buying in discount OR selling in premium
      discountPremium: (structure.trend === 'bullish' && structure.inDiscount) ||
                       (structure.trend === 'bearish' && structure.inPremium),

      // 12. Clean price action (no excessive wicks, clear structure)
      cleanPriceAction: this.isCleanPriceAction(candles.slice(-10))
    };

    const passedCount = Object.values(conditions).filter(Boolean).length;
    const totalConditions = 12;
    const passRate = passedCount / totalConditions;

    // Grade based on pass rate
    let entryGrade: 'A' | 'B' | 'C' | 'D' | 'F';
    if (passRate >= 0.9) entryGrade = 'A';
    else if (passRate >= 0.75) entryGrade = 'B';
    else if (passRate >= 0.6) entryGrade = 'C';
    else if (passRate >= 0.45) entryGrade = 'D';
    else entryGrade = 'F';

    return {
      conditions,
      passedCount,
      totalConditions,
      passRate,
      entryGrade
    };
  }

  /**
   * Check if recent price action is "clean" (readable, not choppy)
   */
  private static isCleanPriceAction(candles: Candle[]): boolean {
    if (candles.length < 5) return true;

    let excessiveWicks = 0;
    let dojiCount = 0;

    for (const candle of candles) {
      const body = Math.abs(candle.close - candle.open);
      const range = candle.high - candle.low;

      // Excessive wick = body is less than 30% of range
      if (range > 0 && body / range < 0.3) {
        excessiveWicks++;
      }

      // Doji = body is less than 10% of range
      if (range > 0 && body / range < 0.1) {
        dojiCount++;
      }
    }

    // Clean if less than 30% are excessive wicks and less than 20% are dojis
    return excessiveWicks / candles.length < 0.3 && dojiCount / candles.length < 0.2;
  }

  /**
   * Fast ICT Analysis (optimized for backtesting)
   * Skips expensive computations, focuses on key signals
   */
  static analyzeFast(candles: Candle[], smcAnalysis: SMCAnalysis): ICTAnalysis {
    const lastCandle = candles[candles.length - 1];

    // Simplified structure from existing SMC analysis
    const structure: MarketStructure = {
      trend: smcAnalysis.trend === 'up' ? 'bullish' : smcAnalysis.trend === 'down' ? 'bearish' : 'ranging',
      lastBOS: smcAnalysis.bos ? { price: lastCandle.close, index: candles.length - 1, direction: smcAnalysis.bos as 'up' | 'down' } : null,
      lastCHoCH: null,
      swingHigh: Math.max(...candles.slice(-20).map(c => c.high)),
      swingLow: Math.min(...candles.slice(-20).map(c => c.low)),
      equilibrium: 0,
      inDiscount: false,
      inPremium: false
    };
    structure.equilibrium = (structure.swingHigh + structure.swingLow) / 2;
    structure.inDiscount = lastCandle.close < structure.equilibrium * 0.995;
    structure.inPremium = lastCandle.close > structure.equilibrium * 1.005;

    // Quick displacement check
    const atr = smcAnalysis.atr || (lastCandle.high - lastCandle.low);
    const recentCandles = candles.slice(-5);
    let upCount = 0, downCount = 0;
    for (const c of recentCandles) {
      const body = Math.abs(c.close - c.open);
      if (body > atr * 0.8) {
        if (c.close > c.open) upCount++; else downCount++;
      }
    }
    const displacement: Displacement = {
      detected: upCount >= 3 || downCount >= 3,
      direction: upCount >= 3 ? 'up' : 'down',
      candleCount: Math.max(upCount, downCount),
      totalMove: 0,
      avgCandleSize: atr,
      atrMultiple: 1.5,
      startIndex: candles.length - 5,
      endIndex: candles.length - 1
    };

    // Simple OTE from range
    const range = structure.swingHigh - structure.swingLow;
    const ote: OTEZone = {
      valid: range > 0,
      direction: structure.trend === 'bullish' ? 'bullish' : 'bearish',
      top: structure.swingHigh - range * 0.618,
      bottom: structure.swingHigh - range * 0.79,
      impulseHigh: structure.swingHigh,
      impulseLow: structure.swingLow
    };
    const inOTE = lastCandle.close >= ote.bottom && lastCandle.close <= ote.top;

    // Quick killzone
    const killzone = this.getKillzone(lastCandle.timestamp);

    // Simplified liquidity grab and judas swing
    const liquidityGrab: LiquidityGrab = { detected: false, type: 'buy_side', grabLevel: 0, wickSize: 0, volumeSpike: 1, candleIndex: 0 };
    const judasSwing: JudasSwing = { detected: false, direction: 'bullish', fakeoutLevel: 0, asianHigh: 0, asianLow: 0, manipulationComplete: false };

    // Quick zone overlap check
    const obInOTE = smcAnalysis.orderBlocks.filter(ob => {
      const mid = (ob.open + ob.close) / 2;
      return mid >= ote.bottom && mid <= ote.top;
    });
    const fvgInOTE = smcAnalysis.fvg.filter(fvg => {
      const mid = (fvg.from + fvg.to) / 2;
      return mid >= ote.bottom && mid <= ote.top;
    });

    // Simplified HTF cascade (use pullback info if available)
    const htfCascade: HTFCascade = {
      aligned: structure.trend !== 'ranging',
      htfTrend: structure.trend,
      mtfTrend: structure.trend,
      ltfTrend: structure.trend,
      alignmentScore: structure.trend !== 'ranging' ? 0.8 : 0.3
    };

    // Quick entry checklist
    const entryChecklist: EntryChecklist = {
      conditions: {
        htfPOIAlignment: htfCascade.aligned,
        displacementPresent: displacement.detected,
        inOTEZone: inOTE,
        obFvgConfluence: obInOTE.length > 0 || fvgInOTE.length > 0,
        killzoneActive: killzone.active,
        liquiditySwept: false,
        bosConfirmed: structure.lastBOS !== null,
        noCHoCH: true,
        properSession: killzone.name === 'london' || killzone.name === 'new_york',
        trendAligned: structure.trend !== 'ranging',
        discountPremium: (structure.trend === 'bullish' && structure.inDiscount) || (structure.trend === 'bearish' && structure.inPremium),
        cleanPriceAction: true
      },
      passedCount: 0,
      totalConditions: 12,
      passRate: 0,
      entryGrade: 'C'
    };
    entryChecklist.passedCount = Object.values(entryChecklist.conditions).filter(Boolean).length;
    entryChecklist.passRate = entryChecklist.passedCount / 12;
    entryChecklist.entryGrade = entryChecklist.passRate >= 0.75 ? 'B' : entryChecklist.passRate >= 0.5 ? 'C' : 'D';

    // Calculate entry score
    let entryScore = 0;
    const entryReasons: string[] = [];
    const noEntryReasons: string[] = [];

    if (structure.trend !== 'ranging') { entryScore += 15; entryReasons.push(`${structure.trend} trend`); }
    if (displacement.detected) { entryScore += 15; entryReasons.push('Displacement'); }
    if (inOTE) { entryScore += 20; entryReasons.push('In OTE'); }
    if (obInOTE.length > 0 || fvgInOTE.length > 0) { entryScore += 15; entryReasons.push('OB/FVG confluence'); }
    if (killzone.active) { entryScore += 10; entryReasons.push(`${killzone.name} killzone`); }
    if (structure.lastBOS) { entryScore += 10; entryReasons.push('BOS confirmed'); }

    return {
      structure,
      ote,
      displacement,
      liquidityGrab,
      judasSwing,
      killzone,
      obInOTE,
      fvgInOTE,
      htfCascade,
      entryChecklist,
      entryValid: entryScore >= 50,
      entryScore: Math.min(100, entryScore),
      entryReasons,
      noEntryReasons
    };
  }

  /**
   * Full ICT Analysis (more thorough but slower)
   */
  static analyze(candles: Candle[], smcAnalysis: SMCAnalysis): ICTAnalysis {
    const structure = this.detectMarketStructure(candles);
    const ote = this.detectOTE(candles, structure);
    const displacement = this.detectDisplacement(candles);
    const liquidityGrab = this.detectLiquidityGrab(candles, structure);
    const judasSwing = this.detectJudasSwing(candles);
    const killzone = this.getKillzone(candles[candles.length - 1].timestamp);

    const { obInOTE, fvgInOTE } = this.findZonesInOTE(
      smcAnalysis.orderBlocks,
      smcAnalysis.fvg,
      ote
    );

    // HTF Cascade validation
    const htfCascade = this.detectHTFCascade(candles);

    // Build 12-condition entry checklist
    const entryChecklist = this.buildEntryChecklist(
      candles,
      structure,
      ote,
      displacement,
      liquidityGrab,
      killzone,
      obInOTE,
      fvgInOTE,
      htfCascade
    );

    // Calculate entry validity
    const entryReasons: string[] = [];
    const noEntryReasons: string[] = [];
    let entryScore = 0;

    // Condition checks
    if (structure.trend !== 'ranging') {
      entryReasons.push(`Clear ${structure.trend} trend`);
      entryScore += 10;
    } else {
      noEntryReasons.push('No clear trend');
    }

    if (structure.lastBOS) {
      entryReasons.push(`BOS confirmed at ${structure.lastBOS.price.toFixed(2)}`);
      entryScore += 10;
    } else {
      noEntryReasons.push('No BOS');
    }

    if (structure.lastCHoCH) {
      noEntryReasons.push(`CHoCH detected - trend may be reversing`);
      entryScore -= 20;
    }

    if (displacement.detected) {
      entryReasons.push(`Displacement: ${displacement.candleCount} candles, ${displacement.atrMultiple.toFixed(1)}x ATR`);
      entryScore += 10;
    } else {
      noEntryReasons.push('No displacement');
    }

    if (ote.valid) {
      const currentPrice = candles[candles.length - 1].close;
      const inOTE = currentPrice >= ote.bottom && currentPrice <= ote.top;
      if (inOTE) {
        entryReasons.push(`Price IN OTE zone (${ote.direction})`);
        entryScore += 15;
      } else {
        noEntryReasons.push('Price not in OTE zone');
      }
    } else {
      noEntryReasons.push('No valid OTE zone');
    }

    if (obInOTE.length > 0 || fvgInOTE.length > 0) {
      entryReasons.push(`${obInOTE.length} OBs and ${fvgInOTE.length} FVGs in OTE`);
      entryScore += 10;
    } else {
      noEntryReasons.push('No OB/FVG confluence in OTE');
    }

    if (liquidityGrab.detected) {
      entryReasons.push(`Liquidity grab: ${liquidityGrab.type} at ${liquidityGrab.grabLevel.toFixed(2)}`);
      entryScore += 10;
    }

    if (judasSwing.manipulationComplete) {
      entryReasons.push(`Judas swing complete: ${judasSwing.direction}`);
      entryScore += 10;
    }

    if (killzone.active) {
      entryReasons.push(`${killzone.name} killzone active`);
      entryScore += 5;
    } else {
      noEntryReasons.push('Not in killzone');
    }

    // Discount/Premium check
    if (structure.trend === 'bullish' && structure.inDiscount) {
      entryReasons.push('Buying in discount zone');
      entryScore += 5;
    } else if (structure.trend === 'bearish' && structure.inPremium) {
      entryReasons.push('Selling in premium zone');
      entryScore += 5;
    }

    // HTF Cascade bonus
    if (htfCascade.aligned) {
      entryReasons.push(`HTF cascade aligned (${htfCascade.htfTrend})`);
      entryScore += 10;
    } else {
      noEntryReasons.push('HTF cascade not aligned');
    }

    // Use entry checklist pass rate to adjust score
    const checklistBonus = Math.floor(entryChecklist.passRate * 15);
    entryScore += checklistBonus;
    if (entryChecklist.passRate >= 0.75) {
      entryReasons.push(`Entry checklist: ${entryChecklist.entryGrade} grade (${entryChecklist.passedCount}/${entryChecklist.totalConditions})`);
    }

    // Entry is valid if:
    // - Score >= 50
    // - Clear trend
    // - No CHoCH
    // - At least 6/12 checklist conditions pass
    const entryValid = entryScore >= 50 &&
                       structure.trend !== 'ranging' &&
                       !structure.lastCHoCH &&
                       entryChecklist.passedCount >= 6;

    return {
      structure,
      ote,
      displacement,
      liquidityGrab,
      judasSwing,
      killzone,
      obInOTE,
      fvgInOTE,
      htfCascade,
      entryChecklist,
      entryValid,
      entryScore: Math.max(0, Math.min(100, entryScore)),
      entryReasons,
      noEntryReasons
    };
  }
}
