/**
 * LightGBM Predictor
 *
 * Uses the trained LightGBM model (72.8% accuracy) for predictions.
 * Calls Python subprocess for prediction since LightGBM is Python-native.
 */

import { execSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import { TradeFeatures } from './trade-features.js';

export interface LightGBMPrediction {
  winProbability: number;
  confidence: number;
  keyFeatures: string[];
  reason: string;
  shouldTakeTrade: boolean;  // Based on walk-forward optimal threshold
  threshold: number;         // The optimal threshold used
}

export class LightGBMPredictor {
  private static hasLoggedLoad = false;  // Only log model load once across all instances
  private modelPath: string;
  private metadataPath: string;
  private metadata: any = null;
  private loaded = false;
  private lastModified: number = 0;
  private lastMetadataModified: number = 0;

  constructor(modelDir: string = path.join(process.cwd(), 'data', 'models')) {
    this.modelPath = path.join(modelDir, 'lightgbm_model.txt');
    this.metadataPath = path.join(modelDir, 'lightgbm_metadata.json');
  }

  /**
   * Load model metadata
   */
  load(): boolean {
    try {
      if (!fs.existsSync(this.modelPath)) {
        console.log('⚠️  LightGBM model not found, falling back to GD model');
        return false;
      }

      if (!fs.existsSync(this.metadataPath)) {
        console.log('⚠️  LightGBM metadata not found, falling back to GD model');
        return false;
      }

      this.metadata = JSON.parse(fs.readFileSync(this.metadataPath, 'utf-8'));
      this.lastModified = fs.statSync(this.modelPath).mtimeMs;
      this.lastMetadataModified = fs.statSync(this.metadataPath).mtimeMs;
      this.loaded = true;

      // Show walk-forward metrics if available (only log once across all instances)
      if (!LightGBMPredictor.hasLoggedLoad) {
        LightGBMPredictor.hasLoggedLoad = true;
        const wf = this.metadata?.walk_forward;
        if (wf) {
          console.log(`✓ LightGBM walk-forward model loaded`);
          console.log(`  Threshold: ${(wf.optimal_threshold * 100).toFixed(0)}%, ` +
                      `PnL: $${wf.total_filtered_pnl?.toFixed(0) || 0}, ` +
                      `Win: ${((wf.avg_filtered_win_rate || 0) * 100).toFixed(1)}%`);
        } else {
          const acc = this.metadata?.accuracy || 0;
          console.log(`✓ LightGBM model loaded (${(acc * 100).toFixed(1)}% accuracy)`);
        }
      }
      return true;
    } catch (e) {
      console.error('⚠️  Failed to load LightGBM model:', e);
      return false;
    }
  }

  /**
   * Check for model updates and reload if needed
   */
  checkForUpdates(): boolean {
    if (!this.loaded) {
      return this.load();
    }

    try {
      const modelModified = fs.existsSync(this.modelPath) 
        ? fs.statSync(this.modelPath).mtimeMs 
        : 0;
      const metadataModified = fs.existsSync(this.metadataPath)
        ? fs.statSync(this.metadataPath).mtimeMs
        : 0;

      // Check if either file was modified
      if (modelModified > this.lastModified || 
          metadataModified > this.lastMetadataModified) {
        console.log('🔄 LightGBM model updated! Reloading...');
        this.loaded = false;
        return this.load();
      }

      return false;
    } catch (e) {
      console.error('⚠️  Failed to check for LightGBM updates:', e);
      return false;
    }
  }

  /**
   * Get optimal threshold from walk-forward metadata
   */
  private getOptimalThreshold(): number {
    // Try walk-forward threshold first (from new training)
    const wfThreshold = this.metadata?.walk_forward?.optimal_threshold;
    if (typeof wfThreshold === 'number' && wfThreshold > 0 && wfThreshold < 1) {
      return wfThreshold;
    }
    // Default to 0.5 if no walk-forward data
    return 0.5;
  }

  /**
   * Predict win probability using LightGBM
   */
  predict(features: TradeFeatures): LightGBMPrediction {
    const threshold = this.getOptimalThreshold();

    if (!this.loaded) {
      return {
        winProbability: 0.5,
        confidence: 0,
        keyFeatures: [],
        reason: 'LightGBM model not loaded',
        shouldTakeTrade: false,
        threshold
      };
    }

    try {
      // Create temporary input file
      const inputPath = path.join(process.cwd(), 'data', 'temp_predict_input.json');
      const outputPath = path.join(process.cwd(), 'data', 'temp_predict_output.json');

      // Prepare feature vector (convert to format expected by model)
      const featureVector = this.featuresToVector(features);

      // Write input
      fs.writeFileSync(inputPath, JSON.stringify({
        features: featureVector,
        feature_names: Object.keys(featureVector)
      }, null, 2));

      // Call Python predictor
      const pythonScript = path.join(process.cwd(), 'scripts', 'lightgbm_predict.py');
      
      try {
        execSync(
          `python "${pythonScript}" --model "${this.modelPath}" --input "${inputPath}" --output "${outputPath}"`,
          { 
            stdio: 'pipe',
            timeout: 5000 // 5 second timeout
          }
        );

        // Read prediction
        const prediction = JSON.parse(fs.readFileSync(outputPath, 'utf-8'));
        
        // Cleanup
        try {
          fs.unlinkSync(inputPath);
          fs.unlinkSync(outputPath);
        } catch {}

        const prob = prediction.probability || 0.5;
        const shouldTake = prob >= threshold;
        return {
          winProbability: prob,
          confidence: Math.abs(prob - 0.5) * 2,
          keyFeatures: prediction.top_features || [],
          reason: shouldTake
            ? `TAKE: ${(prob * 100).toFixed(0)}% >= ${(threshold * 100).toFixed(0)}% threshold`
            : `SKIP: ${(prob * 100).toFixed(0)}% < ${(threshold * 100).toFixed(0)}% threshold`,
          shouldTakeTrade: shouldTake,
          threshold
        };
      } catch (e) {
        // Python predictor not available, use metadata-based heuristic
        return this.fallbackPredict(features);
      }

    } catch (e) {
      console.error('LightGBM prediction failed:', e);
      return this.fallbackPredict(features);
    }
  }

  /**
   * Fallback prediction using feature weights from metadata
   */
  private fallbackPredict(features: TradeFeatures): LightGBMPrediction {
    const threshold = this.getOptimalThreshold();

    if (!this.metadata || !this.metadata.featureImportance) {
      return {
        winProbability: 0.5,
        confidence: 0,
        keyFeatures: [],
        reason: 'Prediction unavailable',
        shouldTakeTrade: false,
        threshold
      };
    }

    // Simple weighted sum using feature importance as proxy
    const importance = this.metadata.featureImportance;
    let weightedScore = 0;
    let totalWeight = 0;

    for (const [feature, value] of Object.entries(features)) {
      if (typeof value === 'number' && !isNaN(value)) {
        const imp = importance[feature] || 0;
        // Normalize value between 0-1
        const normalized = Math.min(1, Math.max(0, value));
        weightedScore += normalized * imp;
        totalWeight += imp;
      }
    }

    const winProbability = totalWeight > 0
      ? Math.min(0.95, Math.max(0.05, weightedScore / totalWeight))
      : 0.5;

    const shouldTake = winProbability >= threshold;

    return {
      winProbability,
      confidence: Math.abs(winProbability - 0.5) * 2,
      keyFeatures: Object.keys(importance).slice(0, 5),
      reason: shouldTake
        ? `Fallback TAKE: ${(winProbability * 100).toFixed(0)}% >= ${(threshold * 100).toFixed(0)}%`
        : `Fallback SKIP: ${(winProbability * 100).toFixed(0)}% < ${(threshold * 100).toFixed(0)}%`,
      shouldTakeTrade: shouldTake,
      threshold
    };
  }

  /**
   * Convert TradeFeatures to feature vector for LightGBM
   */
  private featuresToVector(features: TradeFeatures): Record<string, any> {
    // Convert to flat structure expected by LightGBM model
    const vector: Record<string, any> = {};

    // Copy all numeric and categorical features
    for (const [key, value] of Object.entries(features)) {
      if (typeof value === 'number' && !isNaN(value)) {
        vector[key] = value;
      } else if (typeof value === 'string' || typeof value === 'boolean') {
        vector[key] = value;
      }
    }

    return vector;
  }

  /**
   * Get model metadata
   */
  getMetadata(): any {
    return this.metadata;
  }

  /**
   * Check if model is loaded
   */
  isLoaded(): boolean {
    return this.loaded;
  }
}