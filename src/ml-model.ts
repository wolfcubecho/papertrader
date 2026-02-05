/**
 * Machine Learning Model for Trading
 *
 * Uses gradient descent to learn optimal feature weights.
 * Actually learns and improves over iterations.
 */

import { TradeFeatures } from './trade-features.js';

export interface Prediction {
  winProbability: number;
  confidence: number;
  keyFeatures: string[];
  reason: string;
}

interface FeatureStats {
  mean: number;
  std: number;
  min: number;
  max: number;
}

// Numerical features to use
const NUMERIC_FEATURES = [
  'trend_strength', 'ob_distance', 'ob_size', 'ob_age',
  'fvg_nearest_distance', 'fvg_size', 'fvg_count', 'volatility',
  'rsi_value', 'atr_value', 'price_position', 'distance_to_high',
  'distance_to_low', 'volume_ratio', 'confluence_score', 'potential_rr'
];

// Categorical features (one-hot encoded)
const CATEGORICAL_FEATURES = [
  'trend_direction', 'ob_type', 'fvg_type', 'ema_trend',
  'rsi_state', 'direction', 'session'
];

export class TradingMLModel {
  private weights: Map<string, number> = new Map();
  private bias: number = 0;
  private featureStats: Map<string, FeatureStats> = new Map();
  private categoricalValues: Map<string, Set<string>> = new Map();
  private trained = false;

  // Training hyperparameters
  private learningRate = 0.01;
  private l2Lambda = 0.001;  // L2 regularization
  private maxEpochs = 100;
  private patience = 10;  // Early stopping patience

  // Training history
  private trainingHistory: { epoch: number; trainLoss: number; valLoss: number; accuracy: number }[] = [];
  private bestWeights: Map<string, number> = new Map();
  private bestBias: number = 0;
  private bestValLoss: number = Infinity;

  /**
   * Train the model using gradient descent
   */
  train(trades: TradeFeatures[]): void {
    if (trades.length < 100) {
      console.log(`[ML] Not enough trades (${trades.length}). Need 100+.`);
      return;
    }

    console.log(`\n=== Training ML Model (Gradient Descent) ===`);
    console.log(`Trades: ${trades.length}`);

    // Split into train/validation (80/20)
    const shuffled = [...trades].sort(() => Math.random() - 0.5);
    const splitIdx = Math.floor(shuffled.length * 0.8);
    const trainSet = shuffled.slice(0, splitIdx);
    const valSet = shuffled.slice(splitIdx);

    console.log(`Train: ${trainSet.length}, Validation: ${valSet.length}`);

    // Initialize feature stats from training data
    this.computeFeatureStats(trainSet);

    // Initialize weights randomly (small values)
    this.initializeWeights();

    // Training loop with early stopping
    let epochsWithoutImprovement = 0;

    for (let epoch = 1; epoch <= this.maxEpochs; epoch++) {
      // Train one epoch
      const trainLoss = this.trainEpoch(trainSet);

      // Evaluate on validation
      const { loss: valLoss, accuracy } = this.evaluate(valSet);

      this.trainingHistory.push({ epoch, trainLoss, valLoss, accuracy });

      // Check for improvement
      if (valLoss < this.bestValLoss) {
        this.bestValLoss = valLoss;
        this.saveBestWeights();
        epochsWithoutImprovement = 0;

        if (epoch % 10 === 0 || epoch === 1) {
          console.log(`  Epoch ${epoch}: loss=${valLoss.toFixed(4)}, acc=${(accuracy * 100).toFixed(1)}% [BEST]`);
        }
      } else {
        epochsWithoutImprovement++;
        if (epoch % 20 === 0) {
          console.log(`  Epoch ${epoch}: loss=${valLoss.toFixed(4)}, acc=${(accuracy * 100).toFixed(1)}%`);
        }
      }

      // Early stopping
      if (epochsWithoutImprovement >= this.patience) {
        console.log(`  Early stopping at epoch ${epoch} (no improvement for ${this.patience} epochs)`);
        break;
      }
    }

    // Restore best weights
    this.restoreBestWeights();

    // Final evaluation
    const { accuracy: finalAcc } = this.evaluate(valSet);
    console.log(`\n✓ Training complete. Best accuracy: ${(finalAcc * 100).toFixed(1)}%`);

    this.trained = true;
    this.printTopFeatures();
  }

  /**
   * Train one epoch using mini-batch gradient descent
   */
  private trainEpoch(trades: TradeFeatures[]): number {
    const batchSize = 32;
    let totalLoss = 0;

    // Shuffle for each epoch
    const shuffled = [...trades].sort(() => Math.random() - 0.5);

    for (let i = 0; i < shuffled.length; i += batchSize) {
      const batch = shuffled.slice(i, i + batchSize);

      // Compute gradients
      const gradients = new Map<string, number>();
      let biasGrad = 0;
      let batchLoss = 0;

      for (const trade of batch) {
        const features = this.extractNormalizedFeatures(trade);
        const target = trade.outcome === 'WIN' ? 1 : 0;

        // Forward pass (sigmoid)
        const logit = this.computeLogit(features);
        const pred = this.sigmoid(logit);

        // Binary cross-entropy loss
        const epsilon = 1e-7;
        batchLoss += -target * Math.log(pred + epsilon) - (1 - target) * Math.log(1 - pred + epsilon);

        // Backward pass (gradient of BCE with sigmoid)
        const error = pred - target;

        for (const [name, value] of features.entries()) {
          const grad = (gradients.get(name) || 0) + error * value;
          gradients.set(name, grad);
        }
        biasGrad += error;
      }

      // Update weights with L2 regularization
      for (const [name, grad] of gradients.entries()) {
        const currentWeight = this.weights.get(name) || 0;
        const avgGrad = grad / batch.length;
        const l2Term = this.l2Lambda * currentWeight;
        const newWeight = currentWeight - this.learningRate * (avgGrad + l2Term);
        this.weights.set(name, newWeight);
      }

      // Update bias (no regularization on bias)
      this.bias -= this.learningRate * (biasGrad / batch.length);

      totalLoss += batchLoss;
    }

    return totalLoss / trades.length;
  }

  /**
   * Evaluate model on a dataset
   */
  private evaluate(trades: TradeFeatures[]): { loss: number; accuracy: number } {
    let totalLoss = 0;
    let correct = 0;

    for (const trade of trades) {
      const features = this.extractNormalizedFeatures(trade);
      const target = trade.outcome === 'WIN' ? 1 : 0;

      const logit = this.computeLogit(features);
      const pred = this.sigmoid(logit);

      // Loss
      const epsilon = 1e-7;
      totalLoss += -target * Math.log(pred + epsilon) - (1 - target) * Math.log(1 - pred + epsilon);

      // Accuracy
      const predicted = pred >= 0.5 ? 1 : 0;
      if (predicted === target) correct++;
    }

    return {
      loss: totalLoss / trades.length,
      accuracy: correct / trades.length
    };
  }

  /**
   * Compute feature statistics for normalization
   */
  private computeFeatureStats(trades: TradeFeatures[]): void {
    this.featureStats.clear();
    this.categoricalValues.clear();

    // Numeric features
    for (const name of NUMERIC_FEATURES) {
      const values = trades.map(t => t[name as keyof TradeFeatures] as number).filter(v => !isNaN(v));
      if (values.length === 0) continue;

      const mean = values.reduce((a, b) => a + b, 0) / values.length;
      const variance = values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / values.length;
      const std = Math.sqrt(variance) || 1;

      this.featureStats.set(name, {
        mean,
        std,
        min: Math.min(...values),
        max: Math.max(...values)
      });
    }

    // Categorical features - collect unique values
    for (const name of CATEGORICAL_FEATURES) {
      const values = new Set<string>();
      for (const trade of trades) {
        const val = trade[name as keyof TradeFeatures];
        if (val !== undefined && val !== null) {
          values.add(String(val));
        }
      }
      this.categoricalValues.set(name, values);
    }
  }

  /**
   * Initialize weights with small random values
   */
  private initializeWeights(): void {
    this.weights.clear();
    this.bias = 0;

    // Numeric features
    for (const name of NUMERIC_FEATURES) {
      if (this.featureStats.has(name)) {
        this.weights.set(name, (Math.random() - 0.5) * 0.1);
      }
    }

    // Categorical features (one-hot)
    for (const [catName, values] of this.categoricalValues.entries()) {
      for (const val of values) {
        const key = `${catName}_${val}`;
        this.weights.set(key, (Math.random() - 0.5) * 0.1);
      }
    }

    console.log(`Initialized ${this.weights.size} weights`);
  }

  /**
   * Extract and normalize features from a trade
   */
  private extractNormalizedFeatures(trade: TradeFeatures): Map<string, number> {
    const features = new Map<string, number>();

    // Numeric features (z-score normalization)
    for (const name of NUMERIC_FEATURES) {
      const stats = this.featureStats.get(name);
      if (!stats) continue;

      const rawValue = trade[name as keyof TradeFeatures] as number;
      if (isNaN(rawValue)) continue;

      const normalized = (rawValue - stats.mean) / stats.std;
      features.set(name, normalized);
    }

    // Categorical features (one-hot encoding)
    for (const [catName, possibleValues] of this.categoricalValues.entries()) {
      const actualValue = String(trade[catName as keyof TradeFeatures] || '');

      for (const val of possibleValues) {
        const key = `${catName}_${val}`;
        features.set(key, actualValue === val ? 1 : 0);
      }
    }

    return features;
  }

  /**
   * Compute logit (weighted sum)
   */
  private computeLogit(features: Map<string, number>): number {
    let logit = this.bias;

    for (const [name, value] of features.entries()) {
      const weight = this.weights.get(name) || 0;
      logit += weight * value;
    }

    return logit;
  }

  /**
   * Sigmoid activation
   */
  private sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x))));
  }

  /**
   * Save current weights as best
   */
  private saveBestWeights(): void {
    this.bestWeights = new Map(this.weights);
    this.bestBias = this.bias;
  }

  /**
   * Restore best weights
   */
  private restoreBestWeights(): void {
    this.weights = new Map(this.bestWeights);
    this.bias = this.bestBias;
  }

  /**
   * Predict win probability
   */
  predict(features: TradeFeatures): Prediction {
    if (!this.trained) {
      return {
        winProbability: 0.5,
        confidence: 0,
        keyFeatures: [],
        reason: 'Model not trained'
      };
    }

    const normalizedFeatures = this.extractNormalizedFeatures(features);
    const logit = this.computeLogit(normalizedFeatures);
    const winProbability = this.sigmoid(logit);

    // Find top contributing features
    const contributions: { name: string; contribution: number }[] = [];

    for (const [name, value] of normalizedFeatures.entries()) {
      const weight = this.weights.get(name) || 0;
      contributions.push({ name, contribution: Math.abs(weight * value) });
    }

    contributions.sort((a, b) => b.contribution - a.contribution);
    const topFeatures = contributions.slice(0, 5).map(c => c.name);

    // Confidence based on how far from 0.5
    const confidence = Math.abs(winProbability - 0.5) * 2;

    const direction = winProbability > 0.5 ? 'bullish' : 'bearish';
    const strength = winProbability > 0.65 || winProbability < 0.35 ? 'strong' : 'weak';

    return {
      winProbability,
      confidence,
      keyFeatures: topFeatures,
      reason: `${strength} ${direction} signal (${(winProbability * 100).toFixed(0)}% win prob)`
    };
  }

  /**
   * Print top features by weight magnitude
   */
  private printTopFeatures(): void {
    const sorted = Array.from(this.weights.entries())
      .map(([name, weight]) => ({ name, weight, absWeight: Math.abs(weight) }))
      .sort((a, b) => b.absWeight - a.absWeight)
      .slice(0, 15);

    console.log(`\n📊 Top Features by Weight:`);
    for (const { name, weight } of sorted) {
      const direction = weight > 0 ? '↑' : '↓';
      const impact = weight > 0 ? 'increases' : 'decreases';
      console.log(`  ${direction} ${name.padEnd(30)} ${weight > 0 ? '+' : ''}${weight.toFixed(4)} (${impact} win prob)`);
    }
  }

  /**
   * Get training history
   */
  getHistory(): typeof this.trainingHistory {
    return this.trainingHistory;
  }

  /**
   * Get model statistics
   */
  getStats(): any {
    const lastHistory = this.trainingHistory[this.trainingHistory.length - 1];
    return {
      trained: this.trained,
      numWeights: this.weights.size,
      bestValLoss: this.bestValLoss,
      finalAccuracy: lastHistory?.accuracy || 0,
      epochs: this.trainingHistory.length
    };
  }

  /**
   * Export weights for persistence
   */
  exportWeights(): { weights: Record<string, number>; bias: number; featureStats: Record<string, FeatureStats>; categoricalValues: Record<string, string[]> } {
    return {
      weights: Object.fromEntries(this.weights),
      bias: this.bias,
      featureStats: Object.fromEntries(this.featureStats),
      categoricalValues: Object.fromEntries(
        Array.from(this.categoricalValues.entries()).map(([k, v]) => [k, Array.from(v)])
      )
    };
  }

  /**
   * Import weights from persistence
   */
  importWeights(data: ReturnType<typeof this.exportWeights>): void {
    this.weights = new Map(Object.entries(data.weights));
    this.bias = data.bias;
    this.featureStats = new Map(Object.entries(data.featureStats));
    this.categoricalValues = new Map(
      Object.entries(data.categoricalValues).map(([k, v]) => [k, new Set(v)])
    );
    this.trained = true;
  }
}
