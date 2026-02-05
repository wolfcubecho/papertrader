#!/usr/bin/env python3
"""
LightGBM Prediction Script

Loads a trained LightGBM model and makes predictions on input features.
"""

import argparse
import json
import numpy as np
import lightgbm as lgb


def load_model(model_path):
    """Load LightGBM model from file"""
    model = lgb.Booster(model_file=model_path)
    return model


def predict(model, features, feature_names):
    """
    Make prediction using LightGBM model
    
    Args:
        model: LightGBM Booster
        features: dict of feature values
        feature_names: list of feature names in order
    
    Returns:
        dict with prediction probability and top features
    """
    # Create feature array in correct order
    feature_array = []
    for name in feature_names:
        feature_array.append(features.get(name, 0))
    
    # Reshape for single prediction
    feature_matrix = np.array([feature_array])
    
    # Predict
    prediction = model.predict(feature_matrix)[0]
    probability = float(prediction)
    
    # Get feature importance
    importance = dict(zip(feature_names, model.feature_importance(importance_type='split')))
    
    # Sort by importance
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
    top_feature_names = [name for name, _ in top_features]
    
    return {
        "probability": probability,
        "top_features": top_feature_names,
        "feature_importance": importance
    }


def main():
    parser = argparse.ArgumentParser(description='LightGBM Prediction')
    parser.add_argument('--model', required=True, help='Path to LightGBM model file')
    parser.add_argument('--input', required=True, help='Path to input JSON')
    parser.add_argument('--output', required=True, help='Path to output JSON')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model}")
    model = load_model(args.model)
    
    # Load input
    print(f"Loading input from {args.input}")
    with open(args.input, 'r') as f:
        input_data = json.load(f)
    
    features = input_data.get('features', {})
    feature_names = input_data.get('feature_names', list(features.keys()))
    
    # Predict
    print("Making prediction...")
    result = predict(model, features, feature_names)
    
    # Save output
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Prediction saved to {args.output}")
    print(f"Probability: {result['probability']:.4f}")
    print(f"Top features: {result['top_features']}")


if __name__ == '__main__':
    main()