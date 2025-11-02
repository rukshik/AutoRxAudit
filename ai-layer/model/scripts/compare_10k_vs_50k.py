import json

# Load 10K v3 results
dnn_elig_10k = json.load(open('results/10000_v3/dnn_eligibility_model_metrics.json'))
dnn_oud_10k = json.load(open('results/10000_v3/dnn_oud_risk_model_metrics.json'))

# Load 50K v3 results
dnn_elig_50k = json.load(open('results/50000_v3/dnn_eligibility_model_metrics.json'))
dnn_oud_50k = json.load(open('results/50000_v3/dnn_oud_risk_model_metrics.json'))

print('=' * 80)
print('SCALING IMPACT: 10K vs 50K Dataset Comparison (v3 with new features)')
print('=' * 80)

print('\n📊 ELIGIBILITY MODEL (Clinical Need for Opioids):')
print('-' * 80)
print('Metric         10K           50K           Change')
print('-' * 80)
print(f'AUC:           {dnn_elig_10k["auc"]:.4f}        {dnn_elig_50k["auc"]:.4f}        {dnn_elig_50k["auc"] - dnn_elig_10k["auc"]:+.4f}')
print(f'Recall:        {dnn_elig_10k["recall"]:.4f}        {dnn_elig_50k["recall"]:.4f}        {dnn_elig_50k["recall"] - dnn_elig_10k["recall"]:+.4f}')
print(f'Precision:     {dnn_elig_10k["precision"]:.4f}        {dnn_elig_50k["precision"]:.4f}        {dnn_elig_50k["precision"] - dnn_elig_10k["precision"]:+.4f}')
print(f'F1 Score:      {dnn_elig_10k["f1"]:.4f}        {dnn_elig_50k["f1"]:.4f}        {dnn_elig_50k["f1"] - dnn_elig_10k["f1"]:+.4f}')
print(f'Accuracy:      {dnn_elig_10k["accuracy"]:.4f}        {dnn_elig_50k["accuracy"]:.4f}        {dnn_elig_50k["accuracy"] - dnn_elig_10k["accuracy"]:+.4f}')

auc_gain_elig = (dnn_elig_50k["auc"] - dnn_elig_10k["auc"]) * 100
if auc_gain_elig > 2:
    verdict_elig = f'🟢 GOOD GAIN: +{auc_gain_elig:.2f} AUC points'
elif auc_gain_elig > 0:
    verdict_elig = f'🟡 MARGINAL GAIN: +{auc_gain_elig:.2f} AUC points'
else:
    verdict_elig = f'🔴 NO IMPROVEMENT: {auc_gain_elig:+.2f} AUC points'

print(f'\n{verdict_elig}')

print('\n\n🎯 OUD RISK MODEL (Opioid Use Disorder Prevention):')
print('-' * 80)
print('Metric         10K           50K           Change')
print('-' * 80)
print(f'AUC:           {dnn_oud_10k["auc"]:.4f}        {dnn_oud_50k["auc"]:.4f}        {dnn_oud_50k["auc"] - dnn_oud_10k["auc"]:+.4f}')
print(f'Recall:        {dnn_oud_10k["recall"]:.4f}        {dnn_oud_50k["recall"]:.4f}        {dnn_oud_50k["recall"] - dnn_oud_10k["recall"]:+.4f}')
print(f'Precision:     {dnn_oud_10k["precision"]:.4f}        {dnn_oud_50k["precision"]:.4f}        {dnn_oud_50k["precision"] - dnn_oud_10k["precision"]:+.4f}')
print(f'F1 Score:      {dnn_oud_10k["f1"]:.4f}        {dnn_oud_50k["f1"]:.4f}        {dnn_oud_50k["f1"] - dnn_oud_10k["f1"]:+.4f}')
print(f'Accuracy:      {dnn_oud_10k["accuracy"]:.4f}        {dnn_oud_50k["accuracy"]:.4f}        {dnn_oud_50k["accuracy"] - dnn_oud_10k["accuracy"]:+.4f}')

auc_gain_oud = (dnn_oud_50k["auc"] - dnn_oud_10k["auc"]) * 100
if auc_gain_oud > 0.5:
    verdict_oud = f'🟢 IMPROVEMENT: +{auc_gain_oud:.2f} AUC points'
elif auc_gain_oud > 0:
    verdict_oud = f'🟡 MINIMAL GAIN: +{auc_gain_oud:.2f} AUC points'
else:
    verdict_oud = f'🔵 MAINTAINED: {auc_gain_oud:+.2f} AUC points (already near-perfect)'

print(f'\n{verdict_oud}')

print('\n' + '=' * 80)
print('OVERALL ASSESSMENT:')
print('=' * 80)

total_gain = auc_gain_elig + auc_gain_oud
if total_gain > 2:
    print('✅ WORTH IT: Significant improvement from 50K dataset')
    print(f'   Combined AUC gain: +{total_gain:.2f} points')
elif total_gain > 0:
    print('⚠️ MARGINAL: Small improvement, may not justify extra training time')
    print(f'   Combined AUC gain: +{total_gain:.2f} points')
else:
    print('❌ NOT WORTH IT: No meaningful improvement from 50K')
    print(f'   Combined AUC change: {total_gain:+.2f} points')

print('\nRecommendation:')
if auc_gain_elig > 2 or total_gain > 3:
    print('  → Use 50K dataset for production (better performance)')
elif dnn_elig_50k["auc"] > 0.82 and auc_gain_elig > 0:
    print('  → Consider 50K if training time acceptable (83%+ AUC achieved)')
else:
    print('  → Stick with 10K dataset (sufficient performance, faster training)')

print('\nKey Findings:')
print(f'  • Eligibility improvement: {auc_gain_elig:+.2f} AUC points')
print(f'  • OUD Risk: {dnn_oud_50k["auc"]*100:.2f}% AUC ({"maintained" if abs(auc_gain_oud) < 0.1 else "improved"})')
print(f'  • Both models: {"EXCELLENT" if dnn_elig_50k["auc"] > 0.80 and dnn_oud_50k["auc"] > 0.95 else "GOOD"} performance')
print('=' * 80)
