"""
Deep dive analysis of MIMIC-IV data for pain-related features
"""
import pandas as pd
import numpy as np
import os

data_dir = "data/mimic-clinical-iv-demo/hosp"

print("=" * 80)
print("DEEP DIVE: PAIN-RELATED FEATURES IN MIMIC-IV")
print("=" * 80)

# 1. PROCEDURES - Find pain-related surgical codes
print("\n🔍 ANALYZING SURGICAL PROCEDURES...")
try:
    procedures = pd.read_csv(os.path.join(data_dir, "procedures_icd.csv.gz"))
    print(f"✓ Loaded {len(procedures):,} procedure records")
    print(f"  Unique patients: {procedures['subject_id'].nunique()}")
    print(f"  Unique procedures: {procedures['icd_code'].nunique()}")
    
    # Load procedure dictionary
    d_procedures = pd.read_csv(os.path.join(data_dir, "d_icd_procedures.csv.gz"))
    
    # Merge to get descriptions
    procs_with_desc = procedures.merge(d_procedures, on=['icd_code', 'icd_version'], how='left')
    
    # Search for pain-related procedures
    pain_keywords = ['spine', 'spinal', 'lumbar', 'back', 'fusion', 'laminectomy', 
                     'joint', 'knee', 'hip', 'replacement', 'arthroplasty',
                     'cancer', 'tumor', 'oncology', 'mastectomy', 'amputation',
                     'fracture', 'trauma', 'orthopedic']
    
    print("\n  Pain-related procedure descriptions:")
    for keyword in pain_keywords:
        matching = procs_with_desc[procs_with_desc['long_title'].str.contains(keyword, case=False, na=False)]
        if len(matching) > 0:
            print(f"    '{keyword}': {len(matching)} procedures, {matching['subject_id'].nunique()} patients")
            print(f"      Examples: {matching['long_title'].value_counts().head(2).to_dict()}")
    
    # Most common procedures
    print("\n  Top 20 most common procedures:")
    top_procs = procs_with_desc['long_title'].value_counts().head(20)
    for proc, count in top_procs.items():
        print(f"    {count:3d} - {proc[:70]}")
    
except Exception as e:
    print(f"❌ Error: {e}")

# 2. OMR - Check for pain scores and BMI
print("\n" + "=" * 80)
print("🔍 ANALYZING OMR (ORGAN MEASUREMENTS)...")
try:
    omr = pd.read_csv(os.path.join(data_dir, "omr.csv.gz"))
    print(f"✓ Loaded {len(omr):,} OMR records")
    print(f"  Unique patients: {omr['subject_id'].nunique()}")
    
    print("\n  All measurement types:")
    measurements = omr['result_name'].value_counts()
    for name, count in measurements.items():
        print(f"    {count:4d} - {name}")
    
    # Check for pain scores
    pain_related = omr[omr['result_name'].str.contains('pain|comfort|VAS|NRS', case=False, na=False)]
    if len(pain_related) > 0:
        print(f"\n  ✓ FOUND PAIN SCORES: {len(pain_related)} records")
        print(f"    Pain measurements: {pain_related['result_name'].value_counts().to_dict()}")
    else:
        print("\n  ❌ No direct pain scores found")
    
    # BMI analysis
    bmi_data = omr[omr['result_name'] == 'BMI (kg/m2)']
    if len(bmi_data) > 0:
        print(f"\n  ✓ BMI DATA: {len(bmi_data)} records, {bmi_data['subject_id'].nunique()} patients")
        bmi_values = pd.to_numeric(bmi_data['result_value'], errors='coerce')
        print(f"    BMI range: {bmi_values.min():.1f} - {bmi_values.max():.1f}")
        print(f"    BMI mean: {bmi_values.mean():.1f}")
        print(f"    Obese (BMI>30): {(bmi_values > 30).sum()} records ({(bmi_values > 30).sum() / len(bmi_values) * 100:.1f}%)")
        print(f"    Morbidly obese (BMI>40): {(bmi_values > 40).sum()} records")
    
except Exception as e:
    print(f"❌ Error: {e}")

# 3. SERVICES - Check for pain/ortho services
print("\n" + "=" * 80)
print("🔍 ANALYZING CLINICAL SERVICES...")
try:
    services = pd.read_csv(os.path.join(data_dir, "services.csv.gz"))
    print(f"✓ Loaded {len(services):,} service transfer records")
    print(f"  Unique patients: {services['subject_id'].nunique()}")
    
    print("\n  All services (curr_service):")
    all_services = services['curr_service'].value_counts()
    for service, count in all_services.items():
        print(f"    {count:4d} - {service}")
    
    # Look for pain-related services
    pain_services = ['ORTH', 'ORTHO', 'NSURG', 'TSURG', 'CSURG', 'ONC', 'PAIN']
    for svc in pain_services:
        matches = services[services['curr_service'].str.contains(svc, case=False, na=False)]
        if len(matches) > 0:
            print(f"\n  ✓ Found '{svc}': {len(matches)} transfers, {matches['subject_id'].nunique()} patients")
    
except Exception as e:
    print(f"❌ Error: {e}")

# 4. DRG CODES - Diagnosis Related Groups
print("\n" + "=" * 80)
print("🔍 ANALYZING DRG CODES...")
try:
    drg = pd.read_csv(os.path.join(data_dir, "drgcodes.csv.gz"))
    print(f"✓ Loaded {len(drg):,} DRG records")
    print(f"  Unique patients: {drg['subject_id'].nunique()}")
    
    # Check severity and mortality scores
    if 'drg_severity' in drg.columns:
        print(f"\n  DRG Severity distribution:")
        print(drg['drg_severity'].value_counts().to_dict())
    
    if 'drg_mortality' in drg.columns:
        print(f"\n  DRG Mortality distribution:")
        print(drg['drg_mortality'].value_counts().to_dict())
    
    # Look for pain-related DRGs
    pain_drg_keywords = ['pain', 'spine', 'back', 'joint', 'orthopedic', 'fracture']
    print("\n  Pain-related DRG descriptions:")
    for keyword in pain_drg_keywords:
        matching = drg[drg['description'].str.contains(keyword, case=False, na=False)]
        if len(matching) > 0:
            print(f"    '{keyword}': {len(matching)} cases")
            print(f"      Examples: {matching['description'].value_counts().head(2).to_dict()}")
    
except Exception as e:
    print(f"❌ Error: {e}")

# 5. LAB EVENTS - Inflammatory markers
print("\n" + "=" * 80)
print("🔍 ANALYZING LAB EVENTS (INFLAMMATORY MARKERS)...")
try:
    # Load lab dictionary first
    d_lab = pd.read_csv(os.path.join(data_dir, "d_labitems.csv.gz"))
    
    # Find inflammatory markers
    inflammatory_keywords = ['CRP', 'C-Reactive', 'ESR', 'Sedimentation', 'WBC', 'White Blood']
    
    print(f"✓ Loaded {len(d_lab)} lab test definitions")
    print("\n  Inflammatory marker tests:")
    for keyword in inflammatory_keywords:
        matching = d_lab[d_lab['label'].str.contains(keyword, case=False, na=False)]
        if len(matching) > 0:
            print(f"    '{keyword}':")
            for _, row in matching.iterrows():
                print(f"      itemid={row['itemid']}: {row['label']}")
    
    # Sample lab events
    lab_events = pd.read_csv(os.path.join(data_dir, "labevents.csv.gz"), nrows=10000)
    print(f"\n  Loaded {len(lab_events):,} lab event records (sample)")
    print(f"  Unique patients: {lab_events['subject_id'].nunique()}")
    print(f"  Unique tests: {lab_events['itemid'].nunique()}")
    
except Exception as e:
    print(f"❌ Error: {e}")

# 6. TRANSFERS - ICU stays (severe cases)
print("\n" + "=" * 80)
print("🔍 ANALYZING TRANSFERS (ICU INDICATORS)...")
try:
    transfers = pd.read_csv(os.path.join(data_dir, "transfers.csv.gz"))
    print(f"✓ Loaded {len(transfers):,} transfer records")
    print(f"  Unique patients: {transfers['subject_id'].nunique()}")
    
    print("\n  Care units:")
    care_units = transfers['careunit'].value_counts()
    for unit, count in care_units.head(15).items():
        print(f"    {count:4d} - {unit}")
    
    # ICU stays indicate severe conditions (often with pain)
    icu_keywords = ['ICU', 'SICU', 'MICU', 'CCU', 'CVICU', 'TSICU']
    icu_transfers = transfers[transfers['careunit'].str.contains('|'.join(icu_keywords), case=False, na=False)]
    print(f"\n  ✓ ICU transfers: {len(icu_transfers)} records, {icu_transfers['subject_id'].nunique()} patients")
    
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 80)
print("SUMMARY: BEST FEATURES TO ADD")
print("=" * 80)
print("""
🎯 RECOMMENDED FEATURES (in priority order):

1. **BMI from omr.csv.gz** ⭐⭐⭐⭐⭐
   - Available for many patients
   - Strong correlation: Obesity → back pain, knee pain
   - Features: bmi, obesity_flag (>30), morbid_obesity_flag (>40)

2. **Surgical procedures from procedures_icd.csv.gz** ⭐⭐⭐⭐⭐
   - Direct indicator of traumatic/chronic pain conditions
   - Features: has_spine_surgery, has_joint_replacement, has_cancer_surgery, 
              has_fracture_repair, total_surgeries, surgery_count

3. **DRG severity/mortality from drgcodes.csv.gz** ⭐⭐⭐⭐
   - Indicates condition severity
   - Features: drg_severity_score, drg_mortality_score, high_severity_flag

4. **ICU stays from transfers.csv.gz** ⭐⭐⭐⭐
   - Severe illness indicator (trauma, post-surgical, cancer)
   - Features: icu_stay_flag, icu_days, sicu_flag (surgical ICU)

5. **Clinical services from services.csv.gz** ⭐⭐⭐
   - Orthopedic, surgical, oncology services → pain likely
   - Features: orthopedic_service_flag, surgical_service_flag

6. **Inflammatory markers from labevents.csv.gz** ⭐⭐
   - CRP, ESR elevation → inflammation/pain
   - Features: elevated_crp_flag, elevated_esr_flag
   - (Complex to implement, lower priority)

EXPECTED IMPACT ON ELIGIBILITY MODEL:
- Current: 64% AUC (8 features, no pain indicators)
- With BMI + Procedures: 72-78% AUC (estimated)
- With all features: 75-82% AUC (estimated)
""")
