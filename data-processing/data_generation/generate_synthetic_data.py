"""
Synthetic Data Generator for MIMIC-IV Demo Expansion
Expands the 100-record demo dataset to N records while preserving
statistical distributions and medical data relationships.

Key Logic:
1. Patient-level pain flag: 70% of patients have chronic pain conditions (matches MIMIC data)
2. Admission-level pain diagnosis: If patient has pain, 60% of their admissions show pain diagnosis
3. ICD codes: Mix of OUD (0.5% admissions), Pain (from real MIMIC codes), and Common conditions
4. Opioid prescriptions:
   - With pain diagnosis: 40% chance (appropriate use - not all pain needs opioids)
   - Without pain diagnosis: 2% chance (inappropriate/potential abuse)
   This creates realistic variation for eligibility model training

Usage:
  python generate_synthetic_data.py --patients 1000
  python generate_synthetic_data.py --patients 100000 --output-dir ../../synthetic_data_100k/mimic-clinical-iv-demo/hosp
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from collections import defaultdict
import gzip
import json
import argparse

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Default configuration
DEFAULT_SOURCE_DIR = os.path.join("..", "..", "data", "mimic-clinical-iv-demo", "hosp")
DEFAULT_OUTPUT_DIR = os.path.join("..", "..", "synthetic_data", "mimic-clinical-iv-demo", "hosp")
DEFAULT_TARGET_PATIENTS = 1000
TARGET_ADMISSIONS_PER_PATIENT_RANGE = (1, 8)  # 1-8 admissions per patient
TARGET_PRESCRIPTIONS_PER_ADMISSION_RANGE = (2, 15)  # 2-15 prescriptions per admission
TARGET_DIAGNOSES_PER_ADMISSION_RANGE = (1, 8)  # 1-8 diagnoses per admission

class SyntheticDataGenerator:
    def __init__(self, target_patients, source_dir, output_dir):
        self.target_patients = target_patients
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.next_subject_id = 20000000  # Start from a safe range
        self.next_hadm_id = 30000000
        self.next_prescription_id = 40000000
        
        # Load original data for sampling distributions
        print("Loading original data...")
        self.load_original_data()
        
    def load_original_data(self):
        """Load original sample data to understand distributions"""
        self.orig_patients = pd.read_csv(os.path.join(self.source_dir, "patients.csv.gz"))
        self.orig_admissions = pd.read_csv(os.path.join(self.source_dir, "admissions.csv.gz"))
        self.orig_diagnoses = pd.read_csv(os.path.join(self.source_dir, "diagnoses_icd.csv.gz"))
        self.orig_prescriptions = pd.read_csv(os.path.join(self.source_dir, "prescriptions.csv.gz"))
        
        # Load pain diagnosis codes
        with open('pain_diagnosis_codes.json', 'r') as f:
            pain_codes = json.load(f)
            self.pain_icd9_codes = pain_codes['icd9']
            self.pain_icd10_codes = pain_codes['icd10']
            print(f"Loaded {len(self.pain_icd9_codes)} ICD-9 and {len(self.pain_icd10_codes)} ICD-10 pain codes")
        
        print(f"Original data shapes:")
        print(f"  Patients: {self.orig_patients.shape}")
        print(f"  Admissions: {self.orig_admissions.shape}")
        print(f"  Diagnoses: {self.orig_diagnoses.shape}")
        print(f"  Prescriptions: {self.orig_prescriptions.shape}")
        
    def generate_patients(self):
        """Generate synthetic patient records"""
        print(f"Generating {self.target_patients} synthetic patients...")
        
        # Analyze original patient distributions
        gender_dist = self.orig_patients['gender'].value_counts(normalize=True)
        age_stats = self.orig_patients['anchor_age'].describe()
        year_stats = self.orig_patients['anchor_year'].describe()
        
        # Track which patients have chronic pain conditions (70% based on MIMIC data)
        self.pain_patients = set()
        
        # Track which patients have OUD (2.5% - realistic prevalence)
        self.oud_patients = set()
        
        patients = []
        for i in range(self.target_patients):
            if (i + 1) % 10000 == 0:
                print(f"  Generated {i + 1:,} patients...")
            
            # Generate subject_id
            subject_id = self.next_subject_id + i
            
            # 2.5% of patients have OUD (realistic prevalence)
            has_oud = np.random.random() < 0.025
            if has_oud:
                self.oud_patients.add(subject_id)
                # OUD patients ALWAYS have chronic pain (95%+ comorbidity)
                self.pain_patients.add(subject_id)
            else:
                # 70% of non-OUD patients have pain conditions (matches MIMIC data)
                if np.random.random() < 0.70:
                    self.pain_patients.add(subject_id)
            
            # Sample gender based on original distribution
            gender = np.random.choice(gender_dist.index, p=gender_dist.values)
            
            # Generate age with realistic distribution
            # OUD patients tend to be younger (mean ~45 vs ~65 for general population)
            if has_oud:
                age = int(np.clip(np.random.normal(45, 12), 25, 75))
            else:
                age = int(np.clip(np.random.normal(age_stats['mean'], age_stats['std']), 18, 89))
            
            # Generate anchor_year around the original range
            anchor_year = int(np.random.uniform(year_stats['min'], year_stats['max']))
            
            # Anchor year group (simplified)
            anchor_year_group = "2011 - 2013"
            
            # Death date (10% chance of having death date)
            dod = None
            if np.random.random() < 0.10:
                # Generate death date 1-5 years after anchor year
                death_year = anchor_year + np.random.randint(1, 6)
                death_month = np.random.randint(1, 13)
                death_day = np.random.randint(1, 29)
                dod = f"{death_year}-{death_month:02d}-{death_day:02d}"
            
            patients.append({
                'subject_id': subject_id,
                'gender': gender,
                'anchor_age': age,
                'anchor_year': anchor_year,
                'anchor_year_group': anchor_year_group,
                'dod': dod
            })
            
        self.synthetic_patients = pd.DataFrame(patients)
        print(f"Generated patients: {self.synthetic_patients.shape}")
        
    def generate_admissions(self):
        """Generate synthetic admission records"""
        print("Generating synthetic admissions...")
        
        # Analyze original admission patterns
        insurance_dist = self.orig_admissions['insurance'].value_counts(normalize=True)
        race_dist = self.orig_admissions['race'].value_counts(normalize=True)
        admission_type_dist = self.orig_admissions['admission_type'].value_counts(normalize=True)
        
        admissions = []
        hadm_id = self.next_hadm_id
        patient_count = 0
        
        for _, patient in self.synthetic_patients.iterrows():
            patient_count += 1
            if patient_count % 10000 == 0:
                print(f"  Processing patient {patient_count:,} / {self.target_patients:,}...")
            
            # Number of admissions for this patient
            # OUD patients have more frequent hospitalizations (1.5x more)
            subject_id = patient['subject_id']
            if subject_id in self.oud_patients:
                n_admissions = np.random.randint(3, 10)  # 3-10 admissions for OUD patients
            else:
                n_admissions = np.random.randint(*TARGET_ADMISSIONS_PER_PATIENT_RANGE)
            
            # Generate admissions for this patient
            base_date = datetime(patient['anchor_year'], 1, 1)
            
            for adm_num in range(n_admissions):
                # Generate admission date (spread across multiple years)
                days_offset = np.random.randint(0, 365 * 3)  # Up to 3 years
                admit_datetime = base_date + timedelta(days=days_offset)
                
                # Length of stay (1-30 days, with realistic distribution)
                los_days = max(1, int(np.random.exponential(4)))  # Exponential distribution
                los_days = min(los_days, 30)  # Cap at 30 days
                
                discharge_datetime = admit_datetime + timedelta(days=los_days)
                
                # Sample other admission characteristics
                insurance = np.random.choice(insurance_dist.index, p=insurance_dist.values)
                race = np.random.choice(race_dist.index, p=race_dist.values)
                admission_type = np.random.choice(admission_type_dist.index, p=admission_type_dist.values)
                
                # Other fields with realistic defaults
                admit_provider_id = f"prov_{np.random.randint(1000, 9999)}"
                admission_location = np.random.choice([
                    'EMERGENCY ROOM', 'PHYSICIAN REFERRAL', 'CLINIC REFERRAL', 
                    'TRANSFER FROM HOSPITAL', 'AMBULATORY SURGERY'
                ], p=[0.4, 0.25, 0.15, 0.15, 0.05])
                
                discharge_location = np.random.choice([
                    'HOME', 'HOME HEALTH CARE', 'SKILLED NURSING FACILITY',
                    'REHAB', 'HOSPICE', 'EXPIRED'
                ], p=[0.6, 0.15, 0.1, 0.08, 0.04, 0.03])
                
                # Death during admission (2% chance)
                deathtime = None
                hospital_expire_flag = 0
                if np.random.random() < 0.02:
                    hospital_expire_flag = 1
                    death_offset_hours = np.random.randint(1, los_days * 24)
                    deathtime = admit_datetime + timedelta(hours=death_offset_hours)
                    discharge_location = 'EXPIRED'
                
                admissions.append({
                    'subject_id': patient['subject_id'],
                    'hadm_id': hadm_id,
                    'admittime': admit_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                    'dischtime': discharge_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                    'deathtime': deathtime.strftime('%Y-%m-%d %H:%M:%S') if deathtime else None,
                    'admission_type': admission_type,
                    'admit_provider_id': admit_provider_id,
                    'admission_location': admission_location,
                    'discharge_location': discharge_location,
                    'insurance': insurance,
                    'language': 'ENGLISH',  # Simplified
                    'marital_status': np.random.choice(['MARRIED', 'SINGLE', 'DIVORCED', 'WIDOWED']),
                    'race': race,
                    'edregtime': None,  # Simplified
                    'edouttime': None,  # Simplified
                    'hospital_expire_flag': hospital_expire_flag
                })
                
                hadm_id += 1
                
        self.synthetic_admissions = pd.DataFrame(admissions)
        print(f"Generated admissions: {self.synthetic_admissions.shape}")
        
    def generate_diagnoses(self):
        """Generate synthetic diagnosis records - OPTIMIZED VERSION"""
        print("Generating synthetic diagnoses...")
        
        # Pre-sample common codes for speed
        common_icd9_codes = ['41401', '4280', '25000', '2724', '4019', '5849', '78650', '2859', '5990', '42731']
        common_icd10_codes = ['I25.10', 'I50.9', 'E11.9', 'E78.5', 'I10', 'N18.6', 'R50.9', 'D64.9', 'N39.0', 'I48.91']
        
        # OUD-related codes
        oud_icd9_codes = ['30400', '30401', '30402', '30403', '30470', '30471', '30472', '30473']
        oud_icd10_codes = ['F11.10', 'F11.11', 'F11.12', 'F11.20', 'F11.21', 'F11.22']
        
        # Sample subset of pain codes for faster selection
        pain_icd9_sample = np.random.choice(self.pain_icd9_codes, min(500, len(self.pain_icd9_codes)), replace=False).tolist()
        pain_icd10_sample = np.random.choice(self.pain_icd10_codes, min(500, len(self.pain_icd10_codes)), replace=False).tolist()
        
        # Process in batches for speed
        batch_size = 50000
        all_diagnoses = []
        total_admissions = len(self.synthetic_admissions)
        
        # Convert to numpy arrays for faster access
        admission_ids = self.synthetic_admissions['hadm_id'].values
        subject_ids = self.synthetic_admissions['subject_id'].values
        
        for batch_start in range(0, total_admissions, batch_size):
            batch_end = min(batch_start + batch_size, total_admissions)
            print(f"  Processing admissions {batch_start:,} to {batch_end:,}...")
            
            batch_diagnoses = []
            
            for idx in range(batch_start, batch_end):
                hadm_id = admission_ids[idx]
                subject_id = subject_ids[idx]
                
                # Number of diagnoses for this admission
                n_diagnoses = np.random.randint(*TARGET_DIAGNOSES_PER_ADMISSION_RANGE)
                
                # Check if this is an OUD patient (patient-level, not random per admission)
                has_oud = subject_id in self.oud_patients
                
                # OUD patients: 30% chance OUD diagnosis appears in THIS admission
                # (not every admission has OUD coded, even for OUD patients)
                has_oud_this_admission = has_oud and (np.random.random() < 0.30)
                
                # Check if this patient has pain conditions (patient-level, not admission-level)
                has_pain = subject_id in self.pain_patients
                
                # If patient has pain, 60% chance this specific admission involves pain diagnosis
                if has_pain:
                    has_pain_this_admission = np.random.random() < 0.60
                else:
                    has_pain_this_admission = False
                
                for diag_num in range(n_diagnoses):
                    # Choose ICD version (90% ICD-10, 10% ICD-9)
                    icd_version = 10 if np.random.random() < 0.9 else 9
                    
                    # Select ICD code with priority logic
                    if has_oud_this_admission and diag_num == 0:  # First diagnosis is OUD
                        icd_code = np.random.choice(oud_icd10_codes if icd_version == 10 else oud_icd9_codes)
                    elif has_pain_this_admission and diag_num == 0:  # First diagnosis can be pain-related
                        icd_code = np.random.choice(pain_icd10_sample if icd_version == 10 else pain_icd9_sample)
                    else:
                        # Regular diagnosis
                        icd_code = np.random.choice(common_icd10_codes if icd_version == 10 else common_icd9_codes)
                    
                    batch_diagnoses.append({
                        'subject_id': subject_id,
                        'hadm_id': hadm_id,
                        'seq_num': diag_num + 1,
                        'icd_code': icd_code,
                        'icd_version': icd_version
                    })
            
            all_diagnoses.extend(batch_diagnoses)
        
        self.synthetic_diagnoses = pd.DataFrame(all_diagnoses)
        print(f"Generated diagnoses: {self.synthetic_diagnoses.shape}")
        
        # Track pain-related diagnoses
        pain_mask = (
            self.synthetic_diagnoses['icd_code'].isin(pain_icd9_sample) |
            self.synthetic_diagnoses['icd_code'].isin(pain_icd10_sample)
        )
        pain_diagnoses = self.synthetic_diagnoses[pain_mask]
        pain_patients = pain_diagnoses['subject_id'].nunique()
        print(f"  Pain-related diagnoses: {len(pain_diagnoses)} records")
        print(f"  Patients with pain: {pain_patients} ({pain_patients/self.target_patients*100:.1f}%)")
        
        # Create set of admissions with pain diagnoses (for prescription logic)
        self.pain_admissions = set(pain_diagnoses['hadm_id'].unique())
        
    def generate_prescriptions(self):
        """Generate synthetic prescription records - OPTIMIZED VERSION"""
        print("Generating synthetic prescriptions...")
        
        # Define realistic drug categories with examples
        drug_categories = {
            'opioids': ['Morphine', 'Oxycodone', 'Hydromorphone', 'Fentanyl', 'Codeine', 
                       'Tramadol', 'Hydrocodone', 'Oxymorphone'],
            'benzos': ['Lorazepam', 'Diazepam', 'Alprazolam', 'Clonazepam', 'Midazolam'],
            'antibiotics': ['Vancomycin', 'Cefazolin', 'Ciprofloxacin', 'Azithromycin', 
                           'Penicillin', 'Amoxicillin'],
            'cardiovascular': ['Metoprolol', 'Lisinopril', 'Atorvastatin', 'Amlodipine', 
                             'Furosemide', 'Carvedilol'],
            'diabetes': ['Insulin', 'Metformin', 'Glipizide', 'Lantus'],
            'psychiatric': ['Sertraline', 'Risperidone', 'Quetiapine', 'Haloperidol'],
            'other': ['Acetaminophen', 'Ibuprofen', 'Prednisone', 'Omeprazole', 
                     'Simvastatin', 'Warfarin', 'Heparin', 'Albuterol']
        }
        
        # Process in batches for speed
        batch_size = 50000
        all_prescriptions = []
        prescription_id = self.next_prescription_id
        total_admissions = len(self.synthetic_admissions)
        
        # Convert to numpy arrays for faster access
        admission_ids = self.synthetic_admissions['hadm_id'].values
        subject_ids = self.synthetic_admissions['subject_id'].values
        admit_times = self.synthetic_admissions['admittime'].values
        discharge_times = self.synthetic_admissions['dischtime'].values
        
        for batch_start in range(0, total_admissions, batch_size):
            batch_end = min(batch_start + batch_size, total_admissions)
            print(f"  Processing admissions {batch_start:,} to {batch_end:,}...")
            
            batch_prescriptions = []
            
            for idx in range(batch_start, batch_end):
                hadm_id = admission_ids[idx]
                subject_id = subject_ids[idx]
                admit_time_str = admit_times[idx]
                discharge_time_str = discharge_times[idx]
                
                # Check if this admission has pain diagnosis
                has_pain_diagnosis = hadm_id in self.pain_admissions
                
                # Number of prescriptions for this admission
                n_prescriptions = np.random.randint(*TARGET_PRESCRIPTIONS_PER_ADMISSION_RANGE)
                
                admit_date = datetime.strptime(admit_time_str, '%Y-%m-%d %H:%M:%S')
                discharge_date = datetime.strptime(discharge_time_str, '%Y-%m-%d %H:%M:%S')
                
                # Track opioid prescriptions for this admission
                opioid_count = 0
                
                # Check if this is an OUD patient
                is_oud_patient = subject_id in self.oud_patients
                
                # Decide how many opioids for this admission
                # OUD patients: multiple opioids per admission (80% chance, 1-3 opioids)
                # Pain diagnosis (non-OUD): moderate use (40% chance, 1 opioid)
                # No pain (non-OUD): rare (3% chance, 1 opioid)
                max_opioids = 0
                if is_oud_patient and np.random.random() < 0.80:  # 80% of OUD admissions get opioids
                    max_opioids = np.random.randint(1, 4)  # 1-3 opioids per admission
                elif has_pain_diagnosis and np.random.random() < 0.40:  # 40% with pain
                    max_opioids = 1
                elif not has_pain_diagnosis and np.random.random() < 0.03:  # 3% inappropriate
                    max_opioids = 1
                
                for rx_num in range(n_prescriptions):
                    # Prescribe opioids up to max_opioids limit
                    if opioid_count < max_opioids:
                        category = 'opioids'
                        opioid_count += 1
                    else:
                        # Select non-opioid drug categories
                        available_categories = [k for k in drug_categories.keys() if k != 'opioids']
                        category = np.random.choice(available_categories)
                    
                    drug = np.random.choice(drug_categories[category])
                    
                    # Generate realistic dosing
                    if category == 'opioids':
                        dose_val_rx = f"{np.random.choice([2, 5, 10, 15, 30])} mg"
                        route = np.random.choice(['PO', 'IV', 'SL'])
                    elif category == 'antibiotics':
                        dose_val_rx = f"{np.random.choice([250, 500, 1000])} mg"
                        route = np.random.choice(['PO', 'IV'])
                    else:
                        dose_val_rx = f"{np.random.choice([5, 10, 25, 50, 100])} mg"
                        route = np.random.choice(['PO', 'IV', 'SL', 'TOP'])
                    
                    # Prescription timing within admission
                    start_offset_hours = np.random.randint(0, 24)
                    duration_hours = np.random.randint(1, max(1, int((discharge_date - admit_date).total_seconds() // 3600)))
                    
                    starttime = admit_date + timedelta(hours=start_offset_hours)
                    stoptime = starttime + timedelta(hours=duration_hours)
                    
                    # Ensure stoptime doesn't exceed discharge
                    if stoptime > discharge_date:
                        stoptime = discharge_date
                    
                    batch_prescriptions.append({
                        'subject_id': subject_id,
                        'hadm_id': hadm_id,
                        'pharmacy_id': prescription_id,
                        'poe_id': f"poe_{prescription_id}",
                        'poe_seq': rx_num + 1,
                        'order_provider_id': f"provider_{np.random.randint(1000, 9999)}",
                        'starttime': starttime.strftime('%Y-%m-%d %H:%M:%S'),
                        'stoptime': stoptime.strftime('%Y-%m-%d %H:%M:%S'),
                        'drug_type': 'MAIN',
                        'drug': drug,
                        'formulary_drug_cd': f"FORM{prescription_id % 10000}",
                        'gsn': f"{np.random.randint(100000, 999999)}",
                        'ndc': f"{np.random.randint(10000, 99999)}-{np.random.randint(100, 999)}-{np.random.randint(10, 99)}",
                        'prod_strength': dose_val_rx,
                        'form_rx': np.random.choice(['TAB', 'CAP', 'INJ', 'SUSP']),
                        'dose_val_rx': dose_val_rx,
                        'dose_unit_rx': 'mg',
                        'form_val_disp': np.random.randint(1, 100),
                        'form_unit_disp': 'each',
                        'doses_per_24_hrs': np.random.choice([1, 2, 3, 4, 6]),
                        'route': route
                    })
                    
                    prescription_id += 1
            
            all_prescriptions.extend(batch_prescriptions)
                
        self.synthetic_prescriptions = pd.DataFrame(all_prescriptions)
        print(f"Generated prescriptions: {self.synthetic_prescriptions.shape}")
    
    def generate_omr(self):
        """Generate OMR (Organ Measurement Records) - BMI, weight, height"""
        print("Generating OMR records (BMI, weight, height)...")
        
        omr_records = []
        seq_num = 1
        
        for _, patient in self.synthetic_patients.iterrows():
            subject_id = patient['subject_id']
            age = patient['anchor_age']
            gender = patient['gender']
            has_pain = subject_id in self.pain_patients
            
            # ~78% of patients have BMI recorded (based on MIMIC analysis)
            if np.random.random() < 0.78:
                # Generate realistic BMI
                # Pain patients tend to have higher BMI (obesity → back pain, knee pain)
                if has_pain:
                    # Pain patients: higher BMI (mean ~31, std 7)
                    bmi = np.clip(np.random.normal(31, 7), 16, 50)
                else:
                    # Non-pain patients: normal BMI (mean ~27, std 5)
                    bmi = np.clip(np.random.normal(27, 5), 16, 45)
                
                # Generate height (in inches) - realistic for gender
                if gender == 'M':
                    height = np.clip(np.random.normal(69, 3), 60, 78)  # Male: 5'9" avg
                else:
                    height = np.clip(np.random.normal(64, 3), 55, 73)  # Female: 5'4" avg
                
                # Calculate weight from BMI: weight(kg) = BMI * height(m)^2
                height_m = height * 0.0254  # inches to meters
                weight_kg = bmi * (height_m ** 2)
                weight_lbs = weight_kg * 2.20462  # kg to lbs
                
                # Generate chartdate (during one of their admissions)
                patient_admissions = self.synthetic_admissions[
                    self.synthetic_admissions['subject_id'] == subject_id
                ]
                if len(patient_admissions) > 0:
                    # Pick random admission
                    admission = patient_admissions.sample(n=1).iloc[0]
                    admit_date = pd.to_datetime(admission['admittime']).date()
                    
                    # Add BMI record
                    omr_records.append({
                        'subject_id': subject_id,
                        'chartdate': admit_date,
                        'seq_num': seq_num,
                        'result_name': 'BMI (kg/m2)',
                        'result_value': f"{bmi:.1f}"
                    })
                    seq_num += 1
                    
                    # Add weight record
                    omr_records.append({
                        'subject_id': subject_id,
                        'chartdate': admit_date,
                        'seq_num': seq_num,
                        'result_name': 'Weight (Lbs)',
                        'result_value': f"{weight_lbs:.1f}"
                    })
                    seq_num += 1
                    
                    # Add height record
                    omr_records.append({
                        'subject_id': subject_id,
                        'chartdate': admit_date,
                        'seq_num': seq_num,
                        'result_name': 'Height (Inches)',
                        'result_value': f"{height:.1f}"
                    })
                    seq_num += 1
        
        self.synthetic_omr = pd.DataFrame(omr_records)
        print(f"✓ Generated OMR records: {len(omr_records):,} measurements, {len(set(r['subject_id'] for r in omr_records)):,} patients")
    
    def generate_drgcodes(self):
        """Generate DRG (Diagnosis Related Group) codes with severity/mortality"""
        print("Generating DRG codes...")
        
        drg_records = []
        
        for _, admission in self.synthetic_admissions.iterrows():
            subject_id = admission['subject_id']
            hadm_id = admission['hadm_id']
            has_pain = subject_id in self.pain_patients
            has_oud = subject_id in self.oud_patients
            
            # DRG severity (1=Minor, 2=Moderate, 3=Major, 4=Extreme)
            # Pain/OUD patients have higher severity
            if has_oud:
                # OUD patients: higher severity (mean 3.2)
                severity = np.random.choice([2, 3, 4], p=[0.2, 0.5, 0.3])
            elif has_pain:
                # Pain patients: moderate-high severity (mean 2.8)
                severity = np.random.choice([1, 2, 3, 4], p=[0.1, 0.3, 0.4, 0.2])
            else:
                # Non-pain: lower severity (mean 2.2)
                severity = np.random.choice([1, 2, 3, 4], p=[0.2, 0.5, 0.2, 0.1])
            
            # DRG mortality risk (1=Minor, 2=Moderate, 3=Major, 4=Extreme)
            # Correlated with severity but with variation
            mortality = np.clip(severity + np.random.choice([-1, 0, 1], p=[0.2, 0.6, 0.2]), 1, 4)
            
            # Sample DRG code and description based on condition
            if has_pain:
                drg_options = [
                    (460, 'SPINAL FUSION EXCEPT CERVICAL W/O MCC'),
                    (461, 'BILATERAL OR MULTIPLE MAJOR JOINT PROCS OF LOWER EXTREMITY W/O MCC'),
                    (552, 'MEDICAL BACK PROBLEMS W/O MCC'),
                    (741, 'CHRONIC PAIN SYNDROME'),
                ]
            else:
                drg_options = [
                    (871, 'SEPTICEMIA OR SEVERE SEPSIS W/O MV >96 HOURS W MCC'),
                    (189, 'PULMONARY EDEMA & RESPIRATORY FAILURE'),
                    (291, 'HEART FAILURE & SHOCK W MCC'),
                ]
            
            drg_code, description = random.choice(drg_options)
            
            drg_records.append({
                'subject_id': subject_id,
                'hadm_id': hadm_id,
                'drg_type': 'MS',  # Medicare Severity
                'drg_code': str(drg_code),
                'description': description,
                'drg_severity': float(severity),
                'drg_mortality': float(mortality)
            })
        
        self.synthetic_drgcodes = pd.DataFrame(drg_records)
        avg_severity = self.synthetic_drgcodes['drg_severity'].mean()
        print(f"✓ Generated DRG codes: {len(drg_records):,} admissions, avg severity {avg_severity:.1f}")
    
    def generate_transfers(self):
        """Generate transfer records including ICU stays"""
        print("Generating transfer records (ICU indicators)...")
        
        transfer_records = []
        transfer_id = 1
        icu_count = 0
        
        for _, admission in self.synthetic_admissions.iterrows():
            subject_id = admission['subject_id']
            hadm_id = admission['hadm_id']
            admittime = pd.to_datetime(admission['admittime'])
            dischtime = pd.to_datetime(admission['dischtime'])
            has_pain = subject_id in self.pain_patients
            has_oud = subject_id in self.oud_patients
            
            # ICU admission rate: higher for pain/OUD patients
            if has_oud:
                icu_rate = 0.50  # 50% of OUD admissions → ICU (overdose, complications)
            elif has_pain:
                icu_rate = 0.35  # 35% of pain admissions → ICU (post-surgical, trauma)
            else:
                icu_rate = 0.20  # 20% of other admissions → ICU
            
            goes_to_icu = np.random.random() < icu_rate
            
            if goes_to_icu:
                icu_count += 1
                # Determine ICU type
                icu_types = [
                    'Medical Intensive Care Unit (MICU)',
                    'Surgical Intensive Care Unit (SICU)',
                    'Cardiac Vascular Intensive Care Unit (CVICU)',
                    'Medical/Surgical Intensive Care Unit (MICU/SICU)'
                ]
                
                # Pain/surgical patients more likely to go to SICU
                if has_pain:
                    icu_unit = np.random.choice(icu_types, p=[0.2, 0.5, 0.2, 0.1])
                else:
                    icu_unit = np.random.choice(icu_types, p=[0.4, 0.2, 0.3, 0.1])
                
                # ICU stay duration (1-7 days typically)
                icu_duration = np.random.randint(1, 8)
                icu_intime = admittime + timedelta(hours=np.random.randint(1, 12))
                icu_outtime = icu_intime + timedelta(days=icu_duration)
                
                # Ensure ICU outtime doesn't exceed discharge time
                if icu_outtime > dischtime:
                    icu_outtime = dischtime
                
                # Add ICU transfer record
                transfer_records.append({
                    'subject_id': subject_id,
                    'hadm_id': hadm_id,
                    'transfer_id': transfer_id,
                    'eventtype': 'admit',
                    'careunit': icu_unit,
                    'intime': icu_intime.strftime('%Y-%m-%d %H:%M:%S'),
                    'outtime': icu_outtime.strftime('%Y-%m-%d %H:%M:%S')
                })
                transfer_id += 1
                
                # After ICU, transfer to regular ward
                ward_intime = icu_outtime
                ward_outtime = dischtime
                ward_unit = np.random.choice(['Medicine', 'Surgery', 'Med/Surg'])
                
                transfer_records.append({
                    'subject_id': subject_id,
                    'hadm_id': hadm_id,
                    'transfer_id': transfer_id,
                    'eventtype': 'transfer',
                    'careunit': ward_unit,
                    'intime': ward_intime.strftime('%Y-%m-%d %H:%M:%S'),
                    'outtime': ward_outtime.strftime('%Y-%m-%d %H:%M:%S')
                })
                transfer_id += 1
            else:
                # No ICU - direct to regular ward
                ward_unit = np.random.choice([
                    'Medicine', 'Surgery', 'Med/Surg', 'Emergency Department', 'Observation'
                ], p=[0.3, 0.2, 0.2, 0.2, 0.1])
                
                transfer_records.append({
                    'subject_id': subject_id,
                    'hadm_id': hadm_id,
                    'transfer_id': transfer_id,
                    'eventtype': 'admit',
                    'careunit': ward_unit,
                    'intime': admittime.strftime('%Y-%m-%d %H:%M:%S'),
                    'outtime': dischtime.strftime('%Y-%m-%d %H:%M:%S')
                })
                transfer_id += 1
        
        self.synthetic_transfers = pd.DataFrame(transfer_records)
        icu_rate = (icu_count / len(self.synthetic_admissions)) * 100
        print(f"✓ Generated transfer records: {len(transfer_records):,} transfers, {icu_count:,} ICU stays ({icu_rate:.1f}%)")
        
    def save_data(self):
        """Save all synthetic data to compressed CSV files"""
        print("\nSaving synthetic data...")
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save each dataset
        datasets = {
            'patients.csv.gz': self.synthetic_patients,
            'admissions.csv.gz': self.synthetic_admissions,
            'diagnoses_icd.csv.gz': self.synthetic_diagnoses,
            'prescriptions.csv.gz': self.synthetic_prescriptions,
            'omr.csv.gz': self.synthetic_omr,
            'drgcodes.csv.gz': self.synthetic_drgcodes,
            'transfers.csv.gz': self.synthetic_transfers
        }
        
        for filename, df in datasets.items():
            filepath = os.path.join(self.output_dir, filename)
            df.to_csv(filepath, index=False, compression='gzip')
            print(f"  ✓ Saved {filename}: {df.shape[0]:,} rows × {df.shape[1]} cols")
        
        print(f"\n✓ All synthetic data saved to: {self.output_dir}")
        
    def generate_all(self):
        """Generate complete synthetic dataset"""
        print("=" * 60)
        print("SYNTHETIC DATA GENERATION")
        print("=" * 60)
        
        self.generate_patients()
        self.generate_admissions()
        self.generate_diagnoses()
        self.generate_prescriptions()
        self.generate_omr()
        self.generate_drgcodes()
        self.generate_transfers()
        self.save_data()
        
        print("\n" + "=" * 60)
        print("GENERATION COMPLETE")
        print("=" * 60)
        print(f"Generated {self.target_patients:,} patients with realistic clinical data")
        print(f"Total admissions: {len(self.synthetic_admissions):,}")
        print(f"Total diagnoses: {len(self.synthetic_diagnoses):,}")
        print(f"Total prescriptions: {len(self.synthetic_prescriptions):,}")
        print(f"Total OMR records: {len(self.synthetic_omr):,}")
        print(f"Total DRG codes: {len(self.synthetic_drgcodes):,}")
        print(f"Total transfers: {len(self.synthetic_transfers):,}")
        
        # Calculate some statistics
        oud_diagnoses = self.synthetic_diagnoses[
            (self.synthetic_diagnoses['icd_code'].str.startswith('304', na=False)) |
            (self.synthetic_diagnoses['icd_code'].str.startswith('305', na=False)) |
            (self.synthetic_diagnoses['icd_code'].str.startswith('F11', na=False))
        ]
        print(f"OUD diagnoses: {len(oud_diagnoses)} ({len(oud_diagnoses)/len(self.synthetic_diagnoses)*100:.1f}%)")
        
        # Calculate opioid prescriptions
        opioid_prescriptions = self.synthetic_prescriptions[
            self.synthetic_prescriptions['drug'].str.contains(
                'Morphine|Oxycodone|Hydromorphone|Fentanyl|Codeine|Tramadol|Hydrocodone|Oxymorphone',
                case=False, na=False
            )
        ]
        print(f"Opioid prescriptions: {len(opioid_prescriptions)} ({len(opioid_prescriptions)/len(self.synthetic_prescriptions)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic MIMIC-IV clinical data for opioid audit system',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 1000 patients (fast for development)
  python generate_synthetic_data.py --patients 1000
  
  # Generate 100K patients with custom output directory
  python generate_synthetic_data.py --patients 100000 --output-dir ../../synthetic_data_100k/mimic-clinical-iv-demo/hosp
  
  # Use custom source data location
  python generate_synthetic_data.py --patients 5000 --source-dir /path/to/mimic/data
        """
    )
    
    parser.add_argument(
        '--patients', '-n',
        type=int,
        default=DEFAULT_TARGET_PATIENTS,
        help=f'Number of patients to generate (default: {DEFAULT_TARGET_PATIENTS})'
    )
    
    parser.add_argument(
        '--source-dir', '-s',
        type=str,
        default=DEFAULT_SOURCE_DIR,
        help=f'Directory containing original MIMIC-IV demo data (default: {DEFAULT_SOURCE_DIR})'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f'Directory to save generated synthetic data (default: {DEFAULT_OUTPUT_DIR})'
    )
    
    args = parser.parse_args()
    
    print(f"Configuration:")
    print(f"  Target patients: {args.patients:,}")
    print(f"  Source directory: {args.source_dir}")
    print(f"  Output directory: {args.output_dir}")
    print()
    
    generator = SyntheticDataGenerator(
        target_patients=args.patients,
        source_dir=args.source_dir,
        output_dir=args.output_dir
    )
    generator.generate_all()


if __name__ == "__main__":
    main()