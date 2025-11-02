import json

# Load DNN results
dnn_elig = json.load(open('results/10000_v3/dnn_eligibility_model_metrics.json'))
dnn_oud = json.load(open('results/10000_v3/dnn_oud_risk_model_metrics.json'))

# Load PyCaret results
pycaret_elig = json.load(open('results/10000_v3/pycaret_eligibility_model_metrics.json'))
pycaret_oud = json.load(open('results/10000_v3/pycaret_oud_risk_model_metrics.json'))

print('=' * 80)
print('DNN vs PYCARET COMPARISON - 10K Dataset with New Features (v3)')
print('=' * 80)

print('\n📊 ELIGIBILITY MODEL (Clinical Need for Opioids):')
print('-' * 80)
print('Metric         DNN           PyCaret       Winner')
print('-' * 80)
print(f'AUC:           {dnn_elig["auc"]:.4f}        {pycaret_elig["auc"]:.4f}        {"DNN" if dnn_elig["auc"] > pycaret_elig["auc"] else "PyCaret"}')
print(f'Recall:        {dnn_elig["recall"]:.4f}        {pycaret_elig["recall"]:.4f}        {"DNN" if dnn_elig["recall"] > pycaret_elig["recall"] else "PyCaret"}')
print(f'Precision:     {dnn_elig["precision"]:.4f}        {pycaret_elig["precision"]:.4f}        {"DNN" if dnn_elig["precision"] > pycaret_elig["precision"] else "PyCaret"}')
print(f'F1 Score:      {dnn_elig["f1"]:.4f}        {pycaret_elig["f1"]:.4f}        {"DNN" if dnn_elig["f1"] > pycaret_elig["f1"] else "PyCaret"}')
print(f'Accuracy:      {dnn_elig["accuracy"]:.4f}        {pycaret_elig["accuracy"]:.4f}        {"DNN" if dnn_elig["accuracy"] > pycaret_elig["accuracy"] else "PyCaret"}')

elig_winner = "DNN" if dnn_elig["auc"] > pycaret_elig["auc"] else "PyCaret"
print(f'\n🏆 ELIGIBILITY WINNER: {elig_winner}')

print('\n\n🎯 OUD RISK MODEL (Opioid Use Disorder Prevention):')
print('-' * 80)
print('Metric         DNN           PyCaret       Winner')
print('-' * 80)
print(f'AUC:           {dnn_oud["auc"]:.4f}        {pycaret_oud["auc"]:.4f}        {"DNN" if dnn_oud["auc"] > pycaret_oud["auc"] else "PyCaret"}')
print(f'Recall:        {dnn_oud["recall"]:.4f}        {pycaret_oud["recall"]:.4f}        {"DNN" if dnn_oud["recall"] > pycaret_oud["recall"] else "PyCaret"}')
print(f'Precision:     {dnn_oud["precision"]:.4f}        {pycaret_oud["precision"]:.4f}        {"DNN" if dnn_oud["precision"] > pycaret_oud["precision"] else "PyCaret"}')
print(f'F1 Score:      {dnn_oud["f1"]:.4f}        {pycaret_oud["f1"]:.4f}        {"DNN" if dnn_oud["f1"] > pycaret_oud["f1"] else "PyCaret"}')
print(f'Accuracy:      {dnn_oud["accuracy"]:.4f}        {pycaret_oud["accuracy"]:.4f}        {"DNN" if dnn_oud["accuracy"] > pycaret_oud["accuracy"] else "PyCaret"}')

oud_winner = "DNN" if dnn_oud["auc"] > pycaret_oud["auc"] else "PyCaret"
print(f'\n🏆 OUD RISK WINNER: {oud_winner}')

print('\n' + '=' * 80)
print('RECOMMENDATION:')
print('=' * 80)

# Determine overall winner
dnn_wins = sum([
    dnn_elig["auc"] > pycaret_elig["auc"],
    dnn_oud["auc"] > pycaret_oud["auc"]
])

if dnn_wins == 2:
    print('✅ Use DNN for BOTH models (better AUC on both tasks)')
elif dnn_wins == 0:
    print('✅ Use PyCaret for BOTH models (better AUC on both tasks)')
else:
    print(f'✅ HYBRID: Use {elig_winner} for Eligibility, {oud_winner} for OUD Risk')

print('\nKey Considerations:')
print('  • AUC is the primary metric (discrimination ability)')
print('  • Recall important for Eligibility (catch pain patients)')
print('  • Precision important for OUD (avoid false alarms)')
print('  • DNN: Better for complex patterns, slower training')
print('  • PyCaret: AutoML convenience, faster deployment')
print('=' * 80)
