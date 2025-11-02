"""
Model Comparison Script
Compare PyCaret traditional ML models with Deep Neural Network

This script loads results from both modeling approaches and provides
a comprehensive comparison to determine which performs better.
"""

import pandas as pd
import numpy as np
import json
import os


def load_pycaret_results(target):
    """Load PyCaret model results"""
    pred_file = f"test_predictions_{target}.csv"
    if os.path.exists(pred_file):
        return pd.read_csv(pred_file)
    return None


def load_dnn_results(target):
    """Load DNN model results"""
    pred_file = f"dnn_test_predictions_{target}.csv"
    if os.path.exists(pred_file):
        return pd.read_csv(pred_file)
    return None


def calculate_metrics(y_true, y_pred, y_proba):
    """Calculate classification metrics"""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix
    )
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_proba),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }


def compare_models(target):
    """Compare PyCaret and DNN models for a given target"""
    print(f"\n{'='*80}")
    print(f"MODEL COMPARISON: {target}")
    print(f"{'='*80}")
    
    # Load results
    pycaret_results = load_pycaret_results(target)
    dnn_results = load_dnn_results(target)
    
    if pycaret_results is None or dnn_results is None:
        print("Error: Missing results files. Please run both training scripts first.")
        return
    
    # Calculate metrics for both
    pycaret_metrics = calculate_metrics(
        pycaret_results[target],
        pycaret_results['prediction_label'],
        pycaret_results['prediction_score']
    )
    
    dnn_metrics = calculate_metrics(
        dnn_results[target],
        dnn_results['prediction_label'],
        dnn_results['prediction_score']
    )
    
    # Display comparison
    print("\n" + "="*80)
    print(f"{'Metric':<20} {'PyCaret (ML)':<20} {'Deep Neural Net':<20} {'Winner':<15}")
    print("="*80)
    
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    winners = {'pycaret': 0, 'dnn': 0, 'tie': 0}
    
    for metric in metrics_to_compare:
        pycaret_val = pycaret_metrics[metric]
        dnn_val = dnn_metrics[metric]
        
        if abs(pycaret_val - dnn_val) < 0.001:
            winner = "TIE"
            winners['tie'] += 1
        elif pycaret_val > dnn_val:
            winner = "PyCaret"
            winners['pycaret'] += 1
        else:
            winner = "DNN"
            winners['dnn'] += 1
        
        print(f"{metric.upper():<20} {pycaret_val:<20.4f} {dnn_val:<20.4f} {winner:<15}")
    
    print("="*80)
    
    # Confusion matrices
    print("\nConfusion Matrix - PyCaret:")
    cm_py = pycaret_metrics['confusion_matrix']
    print(f"  TN: {cm_py[0][0]:6d}  |  FP: {cm_py[0][1]:6d}")
    print(f"  FN: {cm_py[1][0]:6d}  |  TP: {cm_py[1][1]:6d}")
    
    print("\nConfusion Matrix - Deep Neural Network:")
    cm_dnn = dnn_metrics['confusion_matrix']
    print(f"  TN: {cm_dnn[0][0]:6d}  |  FP: {cm_dnn[0][1]:6d}")
    print(f"  FN: {cm_dnn[1][0]:6d}  |  TP: {cm_dnn[1][1]:6d}")
    
    # Overall winner
    print("\n" + "="*80)
    print("OVERALL WINNER:")
    if winners['pycaret'] > winners['dnn']:
        print(f"  🏆 PyCaret (wins {winners['pycaret']}/{len(metrics_to_compare)} metrics)")
    elif winners['dnn'] > winners['pycaret']:
        print(f"  🏆 Deep Neural Network (wins {winners['dnn']}/{len(metrics_to_compare)} metrics)")
    else:
        print(f"  🤝 TIE ({winners['tie']} ties, {winners['pycaret']} PyCaret, {winners['dnn']} DNN)")
    print("="*80)
    
    return {
        'pycaret_metrics': pycaret_metrics,
        'dnn_metrics': dnn_metrics,
        'winners': winners
    }


def main():
    """Main comparison pipeline"""
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("PyCaret Traditional ML vs Deep Neural Network")
    print("="*80)
    
    targets = ["y_oud", "will_get_opioid_rx"]
    
    all_results = {}
    for target in targets:
        results = compare_models(target)
        if results:
            all_results[target] = results
    
    # Summary across both targets
    if len(all_results) == 2:
        print("\n" + "="*80)
        print("SUMMARY ACROSS BOTH TARGETS")
        print("="*80)
        
        total_pycaret = sum(r['winners']['pycaret'] for r in all_results.values())
        total_dnn = sum(r['winners']['dnn'] for r in all_results.values())
        total_tie = sum(r['winners']['tie'] for r in all_results.values())
        
        print(f"\nTotal wins across both targets:")
        print(f"  PyCaret ML:        {total_pycaret}")
        print(f"  Deep Neural Net:   {total_dnn}")
        print(f"  Ties:              {total_tie}")
        
        if total_pycaret > total_dnn:
            print(f"\n🏆 Overall winner: PyCaret Traditional ML")
        elif total_dnn > total_pycaret:
            print(f"\n🏆 Overall winner: Deep Neural Network")
        else:
            print(f"\n🤝 Overall: Very close performance between both approaches")
        
        print("="*80)


if __name__ == "__main__":
    main()
