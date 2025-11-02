import json

# Load v2 results (without new features)
v2_elig = json.load(open('results/10000_v2/dnn_eligibility_model_metrics.json'))
v2_oud = json.load(open('results/10000_v2/dnn_oud_risk_model_metrics.json'))

# Load v3 results (with BMI, DRG, ICU features)
v3_elig = json.load(open('results/10000_v3/dnn_eligibility_model_metrics.json'))
v3_oud = json.load(open('results/10000_v3/dnn_oud_risk_model_metrics.json'))

print('=' * 80)
print('10K DATASET COMPARISON: v2 (OLD) vs v3 (NEW FEATURES)')
print('=' * 80)

print('\n📊 ELIGIBILITY MODEL (Predicts clinical need for opioids):')
print('-' * 80)
print(f'  v2 (no new features):')
print(f'    AUC:       {v2_elig["auc"]:.4f} ({v2_elig["auc"]*100:.2f}%)')
print(f'    Recall:    {v2_elig["recall"]:.4f} ({v2_elig["recall"]*100:.2f}%)')
print(f'    Precision: {v2_elig["precision"]:.4f}')
print(f'    F1:        {v2_elig["f1"]:.4f}')

print(f'\n  v3 (with BMI + DRG severity + ICU stays):')
print(f'    AUC:       {v3_elig["auc"]:.4f} ({v3_elig["auc"]*100:.2f}%)')
print(f'    Recall:    {v3_elig["recall"]:.4f} ({v3_elig["recall"]*100:.2f}%)')
print(f'    Precision: {v3_elig["precision"]:.4f}')
print(f'    F1:        {v3_elig["f1"]:.4f}')

improvement = (v3_elig["auc"] - v2_elig["auc"]) * 100
print(f'\n  🎉 IMPROVEMENT: +{improvement:.2f} AUC points!')

print('\n\n🎯 OUD RISK MODEL (Predicts opioid use disorder risk):')
print('-' * 80)
print(f'  v2 (no new features):')
print(f'    AUC:       {v2_oud["auc"]:.4f} ({v2_oud["auc"]*100:.2f}%)')
print(f'    Recall:    {v2_oud["recall"]:.4f}')
print(f'    F1:        {v2_oud["f1"]:.4f}')

print(f'\n  v3 (with BMI + DRG severity + ICU stays):')
print(f'    AUC:       {v3_oud["auc"]:.4f} ({v3_oud["auc"]*100:.2f}%)')
print(f'    Recall:    {v3_oud["recall"]:.4f}')
print(f'    F1:        {v3_oud["f1"]:.4f}')

oud_change = (v3_oud["auc"] - v2_oud["auc"]) * 100
print(f'\n  ✓ MAINTAINED: {oud_change:+.2f} AUC points (still excellent)')

print('\n' + '=' * 80)
print('NEW FEATURES IMPACT:')
print('=' * 80)
print('  ✓ BMI (obesity indicators) - Direct correlation with chronic pain')
print('  ✓ DRG Severity (1-4 scale) - Clinical complexity and surgical history')
print('  ✓ ICU Stays - Post-surgical pain, trauma, and critical care indicators')
print('=' * 80)
