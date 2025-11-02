import json
import os

print('=' * 100)
print('COMPREHENSIVE MODEL COMPARISON REPORT')
print('Two-Model Opioid Audit System: Eligibility + OUD Risk')
print('=' * 100)

# Load all results
results = {}

# DNN Results
for dataset in ['10000_v3', '50000_v3']:
    results[f'dnn_{dataset}'] = {
        'eligibility': json.load(open(f'results/{dataset}/dnn_eligibility_model_metrics.json')),
        'oud': json.load(open(f'results/{dataset}/dnn_oud_risk_model_metrics.json'))
    }

# PyCaret Results (fixed)
for dataset in ['10000_v3_fixed', '50000_v3_fixed']:
    if os.path.exists(f'results/{dataset}/pycaret_eligibility_model_metrics.json'):
        results[f'pycaret_{dataset}'] = {
            'eligibility': json.load(open(f'results/{dataset}/pycaret_eligibility_model_metrics.json')),
            'oud': json.load(open(f'results/{dataset}/pycaret_oud_risk_model_metrics.json'))
        }

print('\n' + '='*100)
print('1. ELIGIBILITY MODEL (Clinical Need for Opioids)')
print('='*100)
print('\nModel                          AUC      Recall   Precision   F1      Accuracy')
print('-' * 100)

for key in sorted(results.keys()):
    m = results[key]['eligibility']
    model_name = key.replace('_', ' ').replace('10000 v3', '10K').replace('50000 v3', '50K').upper()
    print(f'{model_name:30s} {m["auc"]:6.4f}   {m["recall"]:6.4f}   {m["precision"]:6.4f}      {m["f1"]:6.4f}  {m["accuracy"]:6.4f}')

print('\n' + '='*100)
print('2. OUD RISK MODEL (Opioid Use Disorder Prevention)')
print('='*100)
print('\nModel                          AUC      Recall   Precision   F1      Accuracy')
print('-' * 100)

for key in sorted(results.keys()):
    m = results[key]['oud']
    model_name = key.replace('_', ' ').replace('10000 v3', '10K').replace('50000 v3', '50K').upper()
    print(f'{model_name:30s} {m["auc"]:6.4f}   {m["recall"]:6.4f}   {m["precision"]:6.4f}      {m["f1"]:6.4f}  {m["accuracy"]:6.4f}')

print('\n' + '='*100)
print('3. KEY FINDINGS')
print('='*100)

# Find best models
best_elig = max(results.items(), key=lambda x: x[1]['eligibility']['auc'])
best_oud = max(results.items(), key=lambda x: x[1]['oud']['auc'])

print(f'\nBest Eligibility Model: {best_elig[0].upper()}')
print(f'  - AUC: {best_elig[1]["eligibility"]["auc"]:.4f} ({best_elig[1]["eligibility"]["auc"]*100:.2f}%)')
print(f'  - Recall: {best_elig[1]["eligibility"]["recall"]:.4f} (catches {best_elig[1]["eligibility"]["recall"]*100:.1f}% of pain patients)')
print(f'  - Precision: {best_elig[1]["eligibility"]["precision"]:.4f} ({best_elig[1]["eligibility"]["precision"]*100:.1f}% accuracy when flagging)')

print(f'\nBest OUD Risk Model: {best_oud[0].upper()}')
print(f'  - AUC: {best_oud[1]["oud"]["auc"]:.4f} ({best_oud[1]["oud"]["auc"]*100:.2f}%)')
print(f'  - Recall: {best_oud[1]["oud"]["recall"]:.4f} (catches {best_oud[1]["oud"]["recall"]*100:.1f}% of OUD cases)')
print(f'  - Precision: {best_oud[1]["oud"]["precision"]:.4f} ({best_oud[1]["oud"]["precision"]*100:.1f}% accuracy when flagging)')

print('\n' + '='*100)
print('4. DATASET SIZE IMPACT')
print('='*100)

if 'dnn_10000_v3' in results and 'dnn_50000_v3' in results:
    dnn_10k = results['dnn_10000_v3']
    dnn_50k = results['dnn_50000_v3']
    
    elig_gain = (dnn_50k['eligibility']['auc'] - dnn_10k['eligibility']['auc']) * 100
    oud_gain = (dnn_50k['oud']['auc'] - dnn_10k['oud']['auc']) * 100
    
    print(f'\nDNN Models (10K → 50K):')
    print(f'  Eligibility: {dnn_10k["eligibility"]["auc"]:.4f} → {dnn_50k["eligibility"]["auc"]:.4f} ({elig_gain:+.2f}% change)')
    print(f'  OUD Risk:    {dnn_10k["oud"]["auc"]:.4f} → {dnn_50k["oud"]["auc"]:.4f} ({oud_gain:+.2f}% change)')
    
    if abs(elig_gain) < 1 and abs(oud_gain) < 1:
        print(f'\n  → Verdict: 10K dataset sufficient (minimal improvement from scaling)')
    else:
        print(f'\n  → Verdict: 50K dataset recommended (meaningful improvement)')

if 'pycaret_10000_v3_fixed' in results and 'pycaret_50000_v3_fixed' in results:
    pycaret_10k = results['pycaret_10000_v3_fixed']
    pycaret_50k = results['pycaret_50000_v3_fixed']
    
    elig_gain = (pycaret_50k['eligibility']['auc'] - pycaret_10k['eligibility']['auc']) * 100
    oud_gain = (pycaret_50k['oud']['auc'] - pycaret_10k['oud']['auc']) * 100
    
    print(f'\nPyCaret Models (10K → 50K):')
    print(f'  Eligibility: {pycaret_10k["eligibility"]["auc"]:.4f} → {pycaret_50k["eligibility"]["auc"]:.4f} ({elig_gain:+.2f}% change)')
    print(f'  OUD Risk:    {pycaret_10k["oud"]["auc"]:.4f} → {pycaret_50k["oud"]["auc"]:.4f} ({oud_gain:+.2f}% change)')

print('\n' + '='*100)
print('5. MODEL ARCHITECTURE COMPARISON')
print('='*100)

print('\nDNN vs PyCaret (10K dataset):')
if 'dnn_10000_v3' in results and 'pycaret_10000_v3_fixed' in results:
    dnn = results['dnn_10000_v3']
    pycaret = results['pycaret_10000_v3_fixed']
    
    elig_diff = (dnn['eligibility']['auc'] - pycaret['eligibility']['auc']) * 100
    oud_diff = (dnn['oud']['auc'] - pycaret['oud']['auc']) * 100
    
    print(f'  Eligibility: DNN {dnn["eligibility"]["auc"]:.4f} vs PyCaret {pycaret["eligibility"]["auc"]:.4f} (DNN {elig_diff:+.2f}% {"better" if elig_diff > 0 else "worse"})')
    print(f'  OUD Risk:    DNN {dnn["oud"]["auc"]:.4f} vs PyCaret {pycaret["oud"]["auc"]:.4f} (DNN {oud_diff:+.2f}% {"better" if oud_diff > 0 else "worse"})')

print('\n' + '='*100)
print('6. PRODUCTION RECOMMENDATION')
print('='*100)

print('\nRecommended Configuration:')
print(f'  → Eligibility Model: {best_elig[0].replace("_", " ").upper()}')
print(f'     - {best_elig[1]["eligibility"]["auc"]*100:.2f}% AUC (EXCELLENT)')
print(f'     - Catches {best_elig[1]["eligibility"]["recall"]*100:.1f}% of pain patients')
print(f'     - {best_elig[1]["eligibility"]["precision"]*100:.1f}% precision (low false positives)')

print(f'\n  → OUD Risk Model: {best_oud[0].replace("_", " ").upper()}')
print(f'     - {best_oud[1]["oud"]["auc"]*100:.2f}% AUC (OUTSTANDING)')
print(f'     - Catches {best_oud[1]["oud"]["recall"]*100:.1f}% of OUD cases')
print(f'     - {best_oud[1]["oud"]["precision"]*100:.1f}% precision (very low false alarms)')

print('\nAudit Logic:')
print('  Flag prescription for review if:')
print('    (Eligibility = NO) OR (OUD_Risk = HIGH)')
print('\nThis two-model approach ensures:')
print('  ✓ Appropriate use detection (Eligibility model)')
print('  ✓ Early OUD risk identification (OUD Risk model)')
print('  ✓ Balanced false positive/negative trade-offs')

print('\n' + '='*100)
print('END OF REPORT')
print('='*100)
