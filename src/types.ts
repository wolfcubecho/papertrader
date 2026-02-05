// Core types for the learning and evolution system

export interface SMCWeights {
  trend_structure: number;
  order_blocks: number;
  fvgs: number;
  ema_alignment: number;
  liquidity: number;
  mtf_bonus: number;
  rsi_penalty: number;
  [key: string]: number;
}

export interface Strategy {
  id: string;
  name: string;
  mode: 'live' | 'paper';
  version: string;
  created: string;
  weights: SMCWeights;
  min_score?: number;
  max_positions?: number;
  // Performance tracking
  total_trades: number;
  win_rate: number | null;
  profit_factor: number | null;
  max_drawdown?: number;
  // Evolution tracking
  parent_strategy?: string;
  parent_version?: string;
  mutation_type?: string;
  phase?: 'pre-training' | 'hybrid' | 'live_dominant' | 'mature';
  // Weekly stats
  weekly_stats?: StrategyStats;
  // Promotion tracking
  previous_version?: string;
  promoted_from?: string;
  promoted_date?: string;
}

export interface StrategyStats {
  strategy_id: string;
  period_start: string;
  period_end: string;
  trades: number;
  wins: number;
  losses: number;
  win_rate: number;
  total_pnl: number;
  profit_factor: number;
  max_drawdown: number;
  avg_trade: number;
  avg_win: number;
  avg_loss: number;
  best_score: number;
  worst_score: number;
  // Confidence metrics
  confidence_level: 'low' | 'medium' | 'high';
  margin_of_error: number;
  // Data sources
  backtest_trades?: number;
  live_trades?: number;
  paper_trades?: number;
}

export interface Trade {
  trade_id: string;
  timestamp: string;
  symbol: string;
  direction: 'long' | 'short';
  exchange: 'binance' | 'bitget';
  strategy_id: string;
  mode: 'live' | 'paper';
  // Macro context
  regime: string;
  macro_score: number;
  // SMC scoring
  smc_score: number;
  score_breakdown: SMCWeights;
  confluence_factors: string[];
  // Trade details
  entry_price: number;
  stop_loss: number;
  take_profits: number[];
  risk_usd: number;
  risk_percent: number;
  position_size: number;
  order_id?: string;
  // Outcome
  outcome?: 'WIN' | 'LOSS' | 'PENDING' | 'CANCELLED';
  exit_price?: number;
  exit_reason?: string;
  pnl?: number;
  holding_time_hours?: number;
}

export interface PaperTrade extends Trade {
  paper_outcome: {
    result: 'WIN' | 'LOSS' | 'BREAKEVEN';
    exit_price: number;
    exit_reason: string;
    pnl: number;
    holding_time_hours: number;
  };
}

export interface EvolutionConfig {
  triggers: {
    min_trades: number;
    min_days: number;
    max_days_without_evolution: number;
  };
  max_experimental_versions: number;
  min_trades_before_promotion: number;
  min_win_rate_improvement: number;
  min_profit_factor: number;
  max_consecutive_failures: number;
  regime_change_required: boolean;
}

export interface PhaseConfig {
  live_trades_min: number;
  live_trades_max?: number;
  backtest_weight: number;
  live_weight: number;
  min_trades_for_evolution: number;
  evolution_frequency: 'weekly' | 'weekly_or_20_trades' | 'weekly_or_50_trades' | 'weekly_or_100_trades';
  min_improvement: number;
  backtest_period: '6_months' | '3_months' | 'regime_specific' | 'rare_only';
  backtest_symbols: string[];
}

export interface EvolutionConfigWithPhases {
  evolution: EvolutionConfig;
  phases: {
    pre_training: PhaseConfig;
    hybrid: PhaseConfig;
    live_dominant: PhaseConfig;
    mature: PhaseConfig;
  };
}

export interface Mutation {
  name: string;
  weights: SMCWeights;
  min_score?: number;
  max_positions?: number;
  mutation_reason: string;
  mutation_type: 'tune_winner' | 'explore_direction' | 'regime_specific' | 'hybrid_best';
}

export interface BacktestConfig {
  period: '6_months' | '3_months' | '1_month';
  symbols: string[];
  regime?: string;
}

export interface BacktestResult {
  trades: number;
  wins: number;
  losses: number;
  win_rate: number;
  total_pnl: number;
  profit_factor: number;
  max_drawdown: number;
  by_score_tier: Record<string, {
    trades: number;
    win_rate: number;
    avg_pnl: number;
  }>;
}

export interface PromotionEligibility {
  eligible: boolean;
  reason: string;
  actual_improvement?: number;
  required_improvement?: number;
  confidence?: string;
  required_trades?: number;
}

export interface Evaluation {
  production: StrategyStats;
  experimental: StrategyStats;
  winner: string;
  recommendation: 'promote' | 'continue' | 'archive';
  reasoning: string;
  statistical_significance: {
    p_value?: number;
    confidence: number;
    sample_sizes: {
      production: number;
      experimental: number;
    };
  };
}