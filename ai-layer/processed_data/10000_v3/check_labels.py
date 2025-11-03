import pandas as pd

df = pd.read_csv('full_data_selected_features.csv')

print(f"Total patients: {len(df)}")
print(f"\nEligibility Distribution:")
print(f"  Eligible (has pain): {df['opioid_eligibility'].sum()} ({df['opioid_eligibility'].mean()*100:.1f}%)")
print(f"  Not eligible: {(df['opioid_eligibility']==0).sum()} ({(df['opioid_eligibility']==0).mean()*100:.1f}%)")

print(f"\nOUD Distribution:")
print(f"  OUD=1: {df['y_oud'].sum()} ({df['y_oud'].mean()*100:.1f}%)")
print(f"  OUD=0: {(df['y_oud']==0).sum()} ({(df['y_oud']==0).mean()*100:.1f}%)")

print(f"\nPatients with both eligible AND no OUD:")
eligible_no_oud = df[(df['opioid_eligibility']==1) & (df['y_oud']==0)]
print(f"  Count: {len(eligible_no_oud)} ({len(eligible_no_oud)/len(df)*100:.1f}%)")
print(f"  These should pass both models (good candidates)")

print(f"\nPatients NOT eligible but no OUD:")
not_eligible_no_oud = df[(df['opioid_eligibility']==0) & (df['y_oud']==0)]
print(f"  Count: {len(not_eligible_no_oud)} ({len(not_eligible_no_oud)/len(df)*100:.1f}%)")
print(f"  These will be flagged for lack of clinical need")

# Sample some eligible patients with no OUD
print(f"\nSample patient IDs (eligible + no OUD):")
if len(eligible_no_oud) > 0:
    sample_ids = eligible_no_oud['subject_id'].head(10).tolist()
    print(f"  {sample_ids}")
