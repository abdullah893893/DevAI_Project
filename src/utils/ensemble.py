import numpy as np
import tensorflow as tf

def soft_voting_ensemble(models, x_test):
    """Soft Voting Ensemble"""
    predictions = []
    for model in models:
        pred = model.predict(x_test, verbose=0)
        predictions.append(pred)
    
    avg_predictions = np.mean(predictions, axis=0)
    final_predictions = np.argmax(avg_predictions, axis=1)
    
    return final_predictions, avg_predictions

def weighted_ensemble(models, weights, x_test):
    """Ağırlıklı Ensemble"""
    if weights is None:
        # Varsayılan ağırlıklar
        weights = [0.15, 0.15, 0.20, 0.20, 0.15, 0.15]
    
    predictions = []
    for i, model in enumerate(models):
        pred = model.predict(x_test, verbose=0)
        weighted_pred = pred * weights[i]
        predictions.append(weighted_pred)
    
    final_predictions = np.sum(predictions, axis=0)
    final_classes = np.argmax(final_predictions, axis=1)
    
    return final_classes, final_predictions