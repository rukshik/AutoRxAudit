"""
Compare All Models - PyCaret vs Deep Neural Networks
====================================================

Compares performance of PyCaret and DNN models for both:
1. Eligibility Model
2. OUD Risk Model

Provides comprehensive comparison to select the best model for each task.

Usage:
    python compare_all_models.py --results-dir ./results
"""

import pandas as pd
import numpy as np
import json
import os
import argparse
from tabulate import tabulate


def load_metrics(results_dir, model_type, model_name):
    """Load metrics for a specific model"""
    metrics_path = os.path.join(results_dir, f"{model_type}_{model_name}_metrics.json")
    
    if not os.path.exists(metrics_path):
        return None
    
    with open(metrics_path, 'r') as f:
        return json.load(f)


def load_predictions(results_dir, model_type, model_name):
    """Load predictions for a specific model"""
    pred_path = os.path.join(results_dir, f"{model_type}_{model_name}_predictions.csv")
    
    if not os.path.exists(pred_path):
        return None
    
    return pd.read_csv(pred_path)


def compare_single_task(results_dir, model_name, task_name):
    """Compare PyCaret vs DNN for a single task"""
    
    print(f"\n{'='*80}")
    print(f"{task_name.upper()}")
    print(f"{'='*80}")
    
    # Load metrics
    pycaret_metrics = load_metrics(results_dir, 'pycaret', model_name)
    dnn_metrics = load_metrics(results_dir, 'dnn', model_name)
    
    if pycaret_metrics is None or dnn_metrics is None:
        print("⚠ Missing results - please run both training scripts first")
        return None
    
    # Prepare comparison table
    metrics_list = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    table_data = []
    winners = {'pycaret': 0, 'dnn': 0, 'tie': 0}
    
    for metric in metrics_list:
        pycaret_val = pycaret_metrics.get(metric, 0)
        dnn_val = dnn_metrics.get(metric, 0)
        
        # Determine winner
        if abs(pycaret_val - dnn_val) < 0.001:
            winner = "TIE"
            winner_symbol = "="
            winners['tie'] += 1
        elif pycaret_val > dnn_val:
            winner = "PyCaret"
            winner_symbol = "←"
            winners['pycaret'] += 1
        else:
            winner = "DNN"
            winner_symbol = "→"
            winners['dnn'] += 1
        
        table_data.append([
            metric.upper(),
            f"{pycaret_val:.4f}",
            f"{dnn_val:.4f}",
            f"{winner} {winner_symbol}"
        ])
    
    # Print comparison table
    headers = ['Metric', 'PyCaret', 'DNN', 'Winner']
    print("\n" + tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # Summary
    print(f"\nScore Summary:")
    print(f"  PyCaret wins: {winners['pycaret']}")
    print(f"  DNN wins:     {winners['dnn']}")
    print(f"  Ties:         {winners['tie']}")
    
    # Determine overall winner
    if winners['pycaret'] > winners['dnn']:
        overall_winner = 'PyCaret'
    elif winners['dnn'] > winners['pycaret']:
        overall_winner = 'DNN'
    else:
        # Use AUC as tiebreaker
        if pycaret_metrics['auc'] > dnn_metrics['auc']:
            overall_winner = 'PyCaret (AUC tiebreaker)'
        else:
            overall_winner = 'DNN (AUC tiebreaker)'
    
    print(f"\n✓ Overall winner: {overall_winner}")
    
    return {
        'task': task_name,
        'pycaret_metrics': pycaret_metrics,
        'dnn_metrics': dnn_metrics,
        'winners': winners,
        'overall_winner': overall_winner
    }


def analyze_confusion_matrices(results_dir, model_name, target_col):
    """Analyze confusion matrices for both models"""
    
    pycaret_preds = load_predictions(results_dir, 'pycaret', model_name)
    dnn_preds = load_predictions(results_dir, 'dnn', model_name)
    
    if pycaret_preds is None or dnn_preds is None:
        return
    
    from sklearn.metrics import confusion_matrix
    
    # PyCaret confusion matrix
    pycaret_cm = confusion_matrix(
        pycaret_preds[target_col],
        pycaret_preds['prediction_label']
    )
    
    # DNN confusion matrix
    dnn_cm = confusion_matrix(
        dnn_preds[target_col],
        dnn_preds['prediction_label']
    )
    
    print(f"\nConfusion Matrices:")
    print(f"\n  PyCaret:")
    print(f"    TN: {pycaret_cm[0][0]:6d}  |  FP: {pycaret_cm[0][1]:6d}")
    print(f"    FN: {pycaret_cm[1][0]:6d}  |  TP: {pycaret_cm[1][1]:6d}")
    
    print(f"\n  DNN:")
    print(f"    TN: {dnn_cm[0][0]:6d}  |  FP: {dnn_cm[0][1]:6d}")
    print(f"    FN: {dnn_cm[1][0]:6d}  |  TP: {dnn_cm[1][1]:6d}")


def create_summary_report(results_dir, comparisons):
    """Create a summary report"""
    
    report_path = os.path.join(results_dir, 'model_comparison_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TWO-MODEL OPIOID AUDIT SYSTEM - MODEL COMPARISON REPORT\n")
        f.write("="*80 + "\n\n")
        
        for comp in comparisons:
            if comp is None:
                continue
            
            f.write(f"\n{comp['task'].upper()}\n")
            f.write("-"*80 + "\n")
            f.write(f"Overall Winner: {comp['overall_winner']}\n")
            f.write(f"PyCaret wins: {comp['winners']['pycaret']}, ")
            f.write(f"DNN wins: {comp['winners']['dnn']}, ")
            f.write(f"Ties: {comp['winners']['tie']}\n")
            
            f.write("\nPyCaret Metrics:\n")
            for k, v in comp['pycaret_metrics'].items():
                f.write(f"  {k}: {v:.4f}\n")
            
            f.write("\nDNN Metrics:\n")
            for k, v in comp['dnn_metrics'].items():
                f.write(f"  {k}: {v:.4f}\n")
            
            f.write("\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("RECOMMENDATION\n")
        f.write("="*80 + "\n\n")
        
        f.write("Based on the comparison, use the following models for deployment:\n\n")
        
        for comp in comparisons:
            if comp is None:
                continue
            f.write(f"- {comp['task']}: {comp['overall_winner']}\n")
        
        f.write("\n")
    
    print(f"\n✓ Summary report saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare PyCaret and DNN models"
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='./results',
        help='Directory containing model results (default: ./results)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory not found: {args.results_dir}")
        print("Please run training scripts first:")
        print("  1. python train_two_models_pycaret.py")
        print("  2. python train_two_models_dnn.py")
        return
    
    print("\n" + "="*80)
    print("TWO-MODEL OPIOID AUDIT SYSTEM - MODEL COMPARISON")
    print("="*80)
    print("\nComparing PyCaret (traditional ML) vs Deep Neural Networks")
    
    # Compare Eligibility Model
    eligibility_comp = compare_single_task(
        args.results_dir,
        'eligibility_model',
        'Eligibility Model'
    )
    
    if eligibility_comp:
        analyze_confusion_matrices(
            args.results_dir,
            'eligibility_model',
            'opioid_eligibility'
        )
    
    # Compare OUD Risk Model
    oud_comp = compare_single_task(
        args.results_dir,
        'oud_risk_model',
        'OUD Risk Model'
    )
    
    if oud_comp:
        analyze_confusion_matrices(
            args.results_dir,
            'oud_risk_model',
            'y_oud'
        )
    
    # Final summary
    print("\n\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    if eligibility_comp:
        print(f"\n1. Eligibility Model: {eligibility_comp['overall_winner']}")
        print(f"   Best AUC: {max(eligibility_comp['pycaret_metrics']['auc'], eligibility_comp['dnn_metrics']['auc']):.4f}")
    
    if oud_comp:
        print(f"\n2. OUD Risk Model: {oud_comp['overall_winner']}")
        print(f"   Best AUC: {max(oud_comp['pycaret_metrics']['auc'], oud_comp['dnn_metrics']['auc']):.4f}")
    
    # Create summary report
    comparisons = [eligibility_comp, oud_comp]
    create_summary_report(args.results_dir, comparisons)
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Review the comparison report in ./results/model_comparison_report.txt")
    print("2. Select the best model for each task based on metrics")
    print("3. Implement audit system using selected models")
    print("4. Test audit system on larger dataset (scale to 100K patients)")


if __name__ == "__main__":
    main()
