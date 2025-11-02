"""
Synthetic Data Generator for MIMIC-IV Demo Expansion
Expands the 100-record demo dataset to 100,000 records while preserving
statistical distributions and medical data relationships.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from collections import defaultdict
import gzip

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Configuration
SOURCE_DIR = os.path.join("..", "..", "data", "mimic-clinical-iv-demo", "hosp")
OUTPUT_DIR = os.path.join("..", "..", "synthetic_data", "mimic-clinical-iv-demo", "hosp")
TARGET_PATIENTS = 100000
TARGET_ADMISSIONS_PER_PATIENT_RANGE = (1, 8)  # 1-8 admissions per patient
TARGET_PRESCRIPTIONS_PER_ADMISSION_RANGE = (2, 15)  # 2-15 prescriptions per admission
TARGET_DIAGNOSES_PER_ADMISSION_RANGE = (1, 8)  # 1-8 diagnoses per admission

class SyntheticDataGenerator:
    def __init__(self):
        self.next_subject_id = 20000000  # Start from a safe range
        self.next_hadm_id = 30000000
        self.next_prescription_id = 40000000
        
        # Load original data for sampling distributions
        print("Loading original data...")
        self.load_original_data()
        
    def load_original_data(self):
        """Load original sample data to understand distributions"""
        self.orig_patients = pd.read_csv(os.path.join(SOURCE_DIR, "patients.csv.gz"))
        self.orig_admissions = pd.read_csv(os.path.join(SOURCE_DIR, "admissions.csv.gz"))
        self.orig_diagnoses = pd.read_csv(os.path.join(SOURCE_DIR, "diagnoses_icd.csv.gz"))
        self.orig_prescriptions = pd.read_csv(os.path.join(SOURCE_DIR, "prescriptions.csv.gz"))
        
        print(f"Original data shapes:")
        print(f"  Patients: {self.orig_patients.shape}")
        print(f"  Admissions: {self.orig_admissions.shape}")
        print(f"  Diagnoses: {self.orig_diagnoses.shape}")
        print(f"  Prescriptions: {self.orig_prescriptions.shape}")
        
    def generate_patients(self):
        """Generate synthetic patient records"""
        print(f"Generating {TARGET_PATIENTS} synthetic patients...")
        
        # Analyze original patient distributions
        gender_dist = self.orig_patients['gender'].value_counts(normalize=True)
        age_stats = self.orig_patients['anchor_age'].describe()
        year_stats = self.orig_patients['anchor_year'].describe()
        
        patients = []
        for i in range(TARGET_PATIENTS):
            # Generate subject_id
            subject_id = self.next_subject_id + i
            
            # Sample gender based on original distribution
            gender = np.random.choice(gender_dist.index, p=gender_dist.values)
            
            # Generate age with realistic distribution (normal with clipping)
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
        
        for _, patient in self.synthetic_patients.iterrows():
            # Number of admissions for this patient
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
        """Generate synthetic diagnosis records"""
        print("Generating synthetic diagnoses...")
        
        # Analyze original diagnosis patterns
        icd_codes = self.orig_diagnoses['icd_code'].value_counts()
        icd_versions = self.orig_diagnoses['icd_version'].value_counts(normalize=True)
        
        # Create lists of common ICD codes by category
        common_icd9_codes = ['41401', '4280', '25000', '2724', '4019', '5849', '78650', '2859', '5990', '42731']
        common_icd10_codes = ['I25.10', 'I50.9', 'E11.9', 'E78.5', 'I10', 'N18.6', 'R50.9', 'D64.9', 'N39.0', 'I48.91']
        
        # OUD-related codes (for creating positive cases)
        oud_icd9_codes = ['30400', '30401', '30402', '30403', '30470', '30471', '30472', '30473', 
                         '30550', '30551', '30552', '30553']
        oud_icd10_codes = ['F11.10', 'F11.11', 'F11.12', 'F11.13', 'F11.14', 'F11.15', 
                          'F11.16', 'F11.17', 'F11.18', 'F11.19', 'F11.20', 'F11.21', 'F11.22']
        
        diagnoses = []
        
        for _, admission in self.synthetic_admissions.iterrows():
            # Number of diagnoses for this admission
            n_diagnoses = np.random.randint(*TARGET_DIAGNOSES_PER_ADMISSION_RANGE)
            
            # 5% chance this admission will have OUD diagnosis
            has_oud = np.random.random() < 0.05
            
            for diag_num in range(n_diagnoses):
                # Choose ICD version
                icd_version = np.random.choice(icd_versions.index, p=icd_versions.values)
                
                # Select ICD code
                if has_oud and diag_num == 0:  # First diagnosis can be OUD
                    if icd_version == 9:
                        icd_code = np.random.choice(oud_icd9_codes)
                    else:
                        icd_code = np.random.choice(oud_icd10_codes)
                else:
                    # Regular diagnosis
                    if icd_version == 9:
                        icd_code = np.random.choice(common_icd9_codes)
                    else:
                        icd_code = np.random.choice(common_icd10_codes)
                
                diagnoses.append({
                    'subject_id': admission['subject_id'],
                    'hadm_id': admission['hadm_id'],
                    'seq_num': diag_num + 1,
                    'icd_code': icd_code,
                    'icd_version': icd_version
                })
                
        self.synthetic_diagnoses = pd.DataFrame(diagnoses)
        print(f"Generated diagnoses: {self.synthetic_diagnoses.shape}")
        
    def generate_prescriptions(self):
        """Generate synthetic prescription records"""
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
        
        prescriptions = []
        prescription_id = self.next_prescription_id
        
        for _, admission in self.synthetic_admissions.iterrows():
            # Number of prescriptions for this admission
            n_prescriptions = np.random.randint(*TARGET_PRESCRIPTIONS_PER_ADMISSION_RANGE)
            
            admit_date = datetime.strptime(admission['admittime'], '%Y-%m-%d %H:%M:%S')
            discharge_date = datetime.strptime(admission['dischtime'], '%Y-%m-%d %H:%M:%S')
            
            for rx_num in range(n_prescriptions):
                # Select drug category and specific drug
                category = np.random.choice(list(drug_categories.keys()))
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
                start_offset_hours = np.random.randint(0, 24)  # Start within first day
                duration_hours = np.random.randint(1, int((discharge_date - admit_date).total_seconds() // 3600))
                
                starttime = admit_date + timedelta(hours=start_offset_hours)
                stoptime = starttime + timedelta(hours=duration_hours)
                
                # Ensure stoptime doesn't exceed discharge
                if stoptime > discharge_date:
                    stoptime = discharge_date
                
                prescriptions.append({
                    'subject_id': admission['subject_id'],
                    'hadm_id': admission['hadm_id'],
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
                
        self.synthetic_prescriptions = pd.DataFrame(prescriptions)
        print(f"Generated prescriptions: {self.synthetic_prescriptions.shape}")
        
    def save_data(self):
        """Save all synthetic data to compressed CSV files"""
        print("Saving synthetic data...")
        
        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Save each dataset
        datasets = {
            'patients.csv.gz': self.synthetic_patients,
            'admissions.csv.gz': self.synthetic_admissions,
            'diagnoses_icd.csv.gz': self.synthetic_diagnoses,
            'prescriptions.csv.gz': self.synthetic_prescriptions
        }
        
        for filename, df in datasets.items():
            filepath = os.path.join(OUTPUT_DIR, filename)
            df.to_csv(filepath, index=False, compression='gzip')
            print(f"  Saved {filename}: {df.shape}")
        
        print(f"\nAll synthetic data saved to: {OUTPUT_DIR}")
        
    def generate_all(self):
        """Generate complete synthetic dataset"""
        print("=" * 60)
        print("SYNTHETIC DATA GENERATION")
        print("=" * 60)
        
        self.generate_patients()
        self.generate_admissions()
        self.generate_diagnoses()
        self.generate_prescriptions()
        self.save_data()
        
        print("\n" + "=" * 60)
        print("GENERATION COMPLETE")
        print("=" * 60)
        print(f"Generated {TARGET_PATIENTS:,} patients with realistic clinical data")
        print(f"Total admissions: {len(self.synthetic_admissions):,}")
        print(f"Total diagnoses: {len(self.synthetic_diagnoses):,}")
        print(f"Total prescriptions: {len(self.synthetic_prescriptions):,}")
        
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


if __name__ == "__main__":
    generator = SyntheticDataGenerator()
    generator.generate_all()