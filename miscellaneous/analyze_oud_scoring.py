import pandas as pd

# Load predictions
pred = pd.read_csv('ai-layer/model/results/10000_v3/dnn_oud_risk_model_predictions.csv')

print("=" * 80)
print("OUD Model Scoring Analysis")
print("=" * 80)

print(f"\nTotal predictions: {len(pred)}")
print(f"y_oud=1 (actual HIGH OUD risk): {(pred['y_oud']==1).sum()}")
print(f"y_oud=0 (actual LOW OUD risk): {(pred['y_oud']==0).sum()}")

print(f"\nPrediction label=1 (predicted HIGH risk): {(pred['prediction_label']==1).sum()}")
print(f"Prediction label=0 (predicted LOW risk): {(pred['prediction_label']==0).sum()}")

print(f"\nAverage prediction_score when y_oud=1 (true HIGH risk): {pred[pred['y_oud']==1]['prediction_score'].mean():.3f}")
print(f"Average prediction_score when y_oud=0 (true LOW risk): {pred[pred['y_oud']==0]['prediction_score'].mean():.3f}")

print("\n" + "=" * 80)
print("INTERPRETATION:")
print("=" * 80)
print("HIGH prediction_score = HIGH OUD RISK")
print("LOW prediction_score = LOW OUD RISK")
print()
print("For patient 20000199:")
print("  - True label: y_oud=1 (HIGH RISK)")
print("  - API returned score: 0.084 (8.4% - LOW RISK)")
print("  - This is a FALSE NEGATIVE!")
print()
print("The model is NOT detecting this high-risk patient correctly in production.")
