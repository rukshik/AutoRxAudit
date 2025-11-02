import pandas as pd

nn_train = pd.read_csv('neural_network_training_data.csv')
nn_test = pd.read_csv('neural_network_final_test_data.csv')
feature_importance = pd.read_csv('feature_importance_results.csv')

print("=" * 60)
print("FINAL RESULTS SUMMARY")
print("=" * 60)

print("\nNeural Network Training Data:")
print(f"  Shape: {nn_train.shape}")
print(f"  Records: {len(nn_train):,}")
print(f"  Features: {len(nn_train.columns) - 3}")  # Excluding subject_id and 2 targets
print(f"  OUD positive cases: {nn_train['y_oud'].sum():,} ({nn_train['y_oud'].mean()*100:.1f}%)")
print(f"  Opioid Rx cases: {nn_train['will_get_opioid_rx'].sum():,} ({nn_train['will_get_opioid_rx'].mean()*100:.1f}%)")

print("\nNeural Network Final Test Data:")
print(f"  Shape: {nn_test.shape}")
print(f"  Records: {len(nn_test):,}")
print(f"  OUD positive cases: {nn_test['y_oud'].sum():,} ({nn_test['y_oud'].mean()*100:.1f}%)")
print(f"  Opioid Rx cases: {nn_test['will_get_opioid_rx'].sum():,} ({nn_test['will_get_opioid_rx'].mean()*100:.1f}%)")

print("\nSelected Features:")
print(f"  Total: {len(feature_importance)}")
features_list = [col for col in nn_train.columns if col not in ['subject_id', 'y_oud', 'will_get_opioid_rx']]
for i, feat in enumerate(features_list, 1):
    print(f"  {i:2d}. {feat}")

print("\n" + "=" * 60)
print("READY FOR NEURAL NETWORK TRAINING!")
print("=" * 60)
print("\nNext steps:")
print("  1. Use 'neural_network_training_data.csv' for training your model")
print("  2. Hold out 'neural_network_final_test_data.csv' for final evaluation")
print("  3. All features are pre-selected based on importance analysis")
print("  4. No data leakage - feature selection used separate subset")