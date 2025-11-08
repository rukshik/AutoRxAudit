"""
Feature Extract - Extract features from raw EHR data.
24 features required by Eligibility (16) and OUD Risk (19) models.
"""

import psycopg2
from typing import Dict
import pandas as pd
from datetime import datetime

# Feature lists for each model
ELIGIBILITY_FEATURES = [
    'avg_drg_severity', 'bmi', 'avg_drg_mortality', 'n_icu_admissions',
    'n_icu_stays', 'max_drg_severity', 'high_severity_flag', 'total_icu_hours',
    'obesity_flag', 'total_icu_days', 'atc_A_rx_count', 'n_admissions_with_drg',
    'n_hospital_admits', 'avg_los_days', 'total_los_days', 'has_bmi'
]

OUD_FEATURES = [
    'opioid_rx_count', 'distinct_opioids', 'opioid_exposure_days', 'opioid_hadms',
    'total_los_days', 'max_drg_severity', 'age_at_first_admit', 'any_opioid_flag',
    'atc_C_rx_count', 'atc_B_rx_count', 'avg_los_days', 'n_hospital_admits',
    'total_icu_days', 'n_icu_admissions', 'bmi', 'avg_drg_severity',
    'high_severity_flag', 'n_icu_stays', 'has_bmi'
]

# Extractor class
class FeatureExtractor:
    
    def __init__(self, db_config):
        self.db_config = db_config
        self._opioid_drugs = [
            'fentanyl', 'morphine', 'oxycodone', 'hydrocodone', 'hydromorphone',
            'oxymorphone', 'codeine', 'tramadol', 'methadone', 'buprenorphine',
            'meperidine', 'tapentadol'
        ]
    
    def extract_eligibility_features(self, patient_id):
        conn = psycopg2.connect(**self.db_config)
        
        try:
            features = {}
            
            # Calculate all eligibility features
            features['avg_drg_severity'] = self._get_avg_drg_severity(conn, patient_id)
            features['bmi'] = self._get_bmi(conn, patient_id)
            features['avg_drg_mortality'] = self._get_avg_drg_mortality(conn, patient_id)
            features['n_icu_admissions'] = self._get_n_icu_admissions(conn, patient_id)
            features['n_icu_stays'] = self._get_n_icu_stays(conn, patient_id)
            features['max_drg_severity'] = self._get_max_drg_severity(conn, patient_id)
            features['high_severity_flag'] = self._get_high_severity_flag(conn, patient_id)
            features['total_icu_hours'] = self._get_total_icu_hours(conn, patient_id)
            features['obesity_flag'] = self._get_obesity_flag(conn, patient_id)
            features['total_icu_days'] = self._get_total_icu_days(conn, patient_id)
            features['atc_A_rx_count'] = self._get_atc_rx_count(conn, patient_id, 'A')
            features['n_admissions_with_drg'] = self._get_n_admissions_with_drg(conn, patient_id)
            features['n_hospital_admits'] = self._get_n_hospital_admits(conn, patient_id)
            features['avg_los_days'] = self._get_avg_los_days(conn, patient_id)
            features['total_los_days'] = self._get_total_los_days(conn, patient_id)
            features['has_bmi'] = self._get_has_bmi(conn, patient_id)
            
            return features
        
        finally:
            conn.close()
    
    def extract_oud_features(self, patient_id):

        conn = psycopg2.connect(**self.db_config)
        
        try:
            features = {}
            
            # Opioid-specific features (1-4, 11)
            features['opioid_rx_count'] = self._get_opioid_rx_count(conn, patient_id)
            features['distinct_opioids'] = self._get_distinct_opioids(conn, patient_id)
            features['opioid_hadms'] = self._get_opioid_hadms(conn, patient_id)
            features['opioid_exposure_days'] = self._get_opioid_exposure_days(conn, patient_id)
            features['any_opioid_flag'] = self._get_any_opioid_flag(conn, patient_id)
            
            # ICU features (5, 6, 8, 9)
            features['n_icu_stays'] = self._get_n_icu_stays(conn, patient_id)
            features['total_icu_hours'] = self._get_total_icu_hours(conn, patient_id)
            features['n_icu_admissions'] = self._get_n_icu_admissions(conn, patient_id)
            features['total_icu_days'] = self._get_total_icu_days(conn, patient_id)
            
            # Demographics (7)
            features['age_at_first_admit'] = self._get_age_at_first_admit(conn, patient_id)
            
            # DRG/Severity features (10, 14, 16)
            features['avg_drg_mortality'] = self._get_avg_drg_mortality(conn, patient_id)
            features['avg_drg_severity'] = self._get_avg_drg_severity(conn, patient_id)
            features['max_drg_mortality'] = self._get_max_drg_mortality(conn, patient_id)
            
            # ATC medication features (12, 17, 18, 19)
            features['atc_J_rx_count'] = self._get_atc_rx_count(conn, patient_id, 'J')
            features['atc_Other_rx_count'] = self._get_atc_other_rx_count(conn, patient_id)
            features['atc_C_rx_count'] = self._get_atc_rx_count(conn, patient_id, 'C')
            features['atc_N_rx_count'] = self._get_atc_rx_count(conn, patient_id, 'N')
            
            # BMI (13)
            features['bmi'] = self._get_bmi(conn, patient_id)
            
            # Length of stay (15)
            features['avg_los_days'] = self._get_avg_los_days(conn, patient_id)
            
            return features
        
        finally:
            conn.close()
    
    
    # caculuation
    def _get_avg_drg_severity(self, conn, patient_id):
        cursor = conn.cursor()
        cursor.execute("""
            SELECT AVG(drg_severity::float) 
            FROM drgcodes 
            WHERE patient_id = %s AND drg_severity IS NOT NULL
        """, (patient_id,))
        result = cursor.fetchone()[0]
        cursor.close()
        return float(result) if result else 0.0
    
    def _get_bmi(self, conn, patient_id):
        cursor = conn.cursor()
        cursor.execute("""
            SELECT result_value 
            FROM omr 
            WHERE patient_id = %s AND result_name = 'BMI'
            ORDER BY chart_time DESC
            LIMIT 1
        """, (patient_id,))
        result = cursor.fetchone()
        cursor.close()
        return float(result[0]) if result and result[0] else 0.0
    
    def _get_avg_drg_mortality(self, conn, patient_id):
        cursor = conn.cursor()
        cursor.execute("""
            SELECT AVG(drg_mortality::float) 
            FROM drgcodes 
            WHERE patient_id = %s AND drg_mortality IS NOT NULL
        """, (patient_id,))
        result = cursor.fetchone()[0]
        cursor.close()
        return float(result) if result else 0.0
    
    def _get_max_drg_mortality(self, conn, patient_id):
        cursor = conn.cursor()
        cursor.execute("""
            SELECT MAX(drg_mortality::float) 
            FROM drgcodes 
            WHERE patient_id = %s AND drg_mortality IS NOT NULL
        """, (patient_id,))
        result = cursor.fetchone()
        cursor.close()
        return float(result[0]) if result and result[0] else 0.0
    
    def _get_n_icu_admissions(self, conn, patient_id):
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(DISTINCT admission_id)
            FROM transfers
            WHERE patient_id = %s AND careunit_type = 'ICU'
        """, (patient_id,))
        result = cursor.fetchone()[0]
        cursor.close()
        return float(result) if result else 0.0
    
    def _get_n_icu_stays(self, conn, patient_id):
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*)
            FROM transfers
            WHERE patient_id = %s AND careunit_type = 'ICU'
        """, (patient_id,))
        result = cursor.fetchone()[0]
        cursor.close()
        return float(result) if result else 0.0
    
    def _get_max_drg_severity(self, conn, patient_id):
        cursor = conn.cursor()
        cursor.execute("""
            SELECT MAX(drg_severity)
            FROM drgcodes
            WHERE patient_id = %s
        """, (patient_id,))
        result = cursor.fetchone()[0]
        cursor.close()
        return float(result) if result else 0.0
    
    def _get_high_severity_flag(self, conn, patient_id):
        max_severity = self._get_max_drg_severity(conn, patient_id)
        return 1.0 if max_severity >= 3 else 0.0
    
    def _get_total_icu_hours(self, conn, patient_id):
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COALESCE(SUM(los_hours), 0)
            FROM transfers
            WHERE patient_id = %s AND careunit_type = 'ICU'
        """, (patient_id,))
        result = cursor.fetchone()[0]
        cursor.close()
        return float(result) if result else 0.0
    
    def _get_obesity_flag(self, conn, patient_id):
        bmi = self._get_bmi(conn, patient_id)
        return 1.0 if bmi >= 30 else 0.0
    
    def _get_total_icu_days(self, conn, patient_id):
        total_hours = self._get_total_icu_hours(conn, patient_id)
        return total_hours / 24.0
    
    def _get_atc_rx_count(self, conn, patient_id, atc_category):

        # ATC mapping patterns (same as training)
        atc_patterns = {
            'A': ['insulin', 'metformin', 'glipizide', 'lantus', 'proton pump inhibitor', 'prazole', 'omeprazole'],
            'B': ['anticoagulant', 'heparin', 'warfarin', 'antiplatelet'],
            'C': ['antihypertensive', 'beta blocker', 'metoprolol', 'carvedilol', 'ace inhibitor', 
                  'lisinopril', 'statin', 'atorvastatin', 'simvastatin', 'amlodipine', 'furosemide'],
            'H': ['thyroid', 'levothyroxine', 'glucocorticoid', 'prednisone'],
            'J': ['antibiotic', 'penicillin', 'cephalosporin', 'vancomycin', 'ciprofloxacin', 
                  'azithromycin', 'amoxicillin'],
            'N': ['antidepressant', 'ssri', 'snri', 'sertraline', 'antipsychotic', 'risperidone', 
                  'quetiapine', 'haloperidol', 'anticonvulsant'],
            'R': ['bronchodilator', 'albuterol', 'inhaler']
        }
        
        if atc_category not in atc_patterns:
            return 0.0
        
        patterns = atc_patterns[atc_category]
        
        # Get all non-opioid, non-benzo prescriptions
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT drug_name
                FROM prescriptions
                WHERE patient_id = %s
                AND drug_name IS NOT NULL
                AND LOWER(drug_name) NOT LIKE '%morphine%'
                AND LOWER(drug_name) NOT LIKE '%hydromorphone%'
                AND LOWER(drug_name) NOT LIKE '%oxycodone%'
                AND LOWER(drug_name) NOT LIKE '%hydrocodone%'
                AND LOWER(drug_name) NOT LIKE '%fentanyl%'
                AND LOWER(drug_name) NOT LIKE '%codeine%'
                AND LOWER(drug_name) NOT LIKE '%tramadol%'
                AND LOWER(drug_name) NOT LIKE '%oxymorphone%'
                AND LOWER(drug_name) NOT LIKE '%tapentadol%'
                AND LOWER(drug_name) NOT LIKE '%methadone%'
                AND LOWER(drug_name) NOT LIKE '%buprenorphine%'
                AND LOWER(drug_name) NOT LIKE '%diazepam%'
                AND LOWER(drug_name) NOT LIKE '%lorazepam%'
                AND LOWER(drug_name) NOT LIKE '%alprazolam%'
                AND LOWER(drug_name) NOT LIKE '%clonazepam%'
            """, (patient_id,))
            
            results = cursor.fetchall()
            drugs = [row[0] for row in results if row and len(row) > 0 and row[0] is not None]
        except Exception as e:
            print(f"Error in _get_atc_rx_count for category {atc_category}: {e}")
            drugs = []
        finally:
            cursor.close()
        
        # Count drugs matching this ATC category
        count = 0
        for drug in drugs:
            if drug:  # Extra safety check
                drug_lower = drug.lower()
                if any(pattern in drug_lower for pattern in patterns):
                    count += 1
        
        return float(count)
    
    def _get_atc_other_rx_count(self, conn, patient_id):
          
        # ATC mapping patterns (same as training)
        atc_patterns = {
            'A': ['insulin', 'metformin', 'glipizide', 'lantus', 'proton pump inhibitor', 'prazole', 'omeprazole'],
            'B': ['anticoagulant', 'heparin', 'warfarin', 'antiplatelet'],
            'C': ['antihypertensive', 'beta blocker', 'metoprolol', 'carvedilol', 'ace inhibitor', 
                  'lisinopril', 'statin', 'atorvastatin', 'simvastatin', 'amlodipine', 'furosemide'],
            'H': ['thyroid', 'levothyroxine', 'glucocorticoid', 'prednisone'],
            'J': ['antibiotic', 'penicillin', 'cephalosporin', 'vancomycin', 'ciprofloxacin', 
                  'azithromycin', 'amoxicillin'],
            'N': ['antidepressant', 'ssri', 'snri', 'sertraline', 'antipsychotic', 'risperidone', 
                  'quetiapine', 'haloperidol', 'anticonvulsant'],
            'R': ['bronchodilator', 'albuterol', 'inhaler']
        }
        
        # Get all non-opioid, non-benzo prescriptions
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT drug_name
                FROM prescriptions
                WHERE patient_id = %s
                AND drug_name IS NOT NULL
                AND LOWER(drug_name) NOT LIKE '%morphine%'
                AND LOWER(drug_name) NOT LIKE '%hydromorphone%'
                AND LOWER(drug_name) NOT LIKE '%oxycodone%'
                AND LOWER(drug_name) NOT LIKE '%hydrocodone%'
                AND LOWER(drug_name) NOT LIKE '%fentanyl%'
                AND LOWER(drug_name) NOT LIKE '%codeine%'
                AND LOWER(drug_name) NOT LIKE '%tramadol%'
                AND LOWER(drug_name) NOT LIKE '%oxymorphone%'
                AND LOWER(drug_name) NOT LIKE '%tapentadol%'
                AND LOWER(drug_name) NOT LIKE '%methadone%'
                AND LOWER(drug_name) NOT LIKE '%buprenorphine%'
                AND LOWER(drug_name) NOT LIKE '%diazepam%'
                AND LOWER(drug_name) NOT LIKE '%lorazepam%'
                AND LOWER(drug_name) NOT LIKE '%alprazolam%'
                AND LOWER(drug_name) NOT LIKE '%clonazepam%'
            """, (patient_id,))
            
            results = cursor.fetchall()
            drugs = [row[0] for row in results if row and len(row) > 0 and row[0] is not None]
        except Exception as e:
            print(f"Error in _get_atc_other_rx_count: {e}")
            drugs = []
        finally:
            cursor.close()
        
        # Count drugs that don't match any ATC category
        other_count = 0
        for drug in drugs:
            if drug:  # Extra safety check
                drug_lower = drug.lower()
                is_categorized = False
                for patterns in atc_patterns.values():
                    if any(pattern in drug_lower for pattern in patterns):
                        is_categorized = True
                        break
                if not is_categorized:
                    other_count += 1
        
        return float(other_count)
    
    def _get_n_admissions_with_drg(self, conn, patient_id):
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(DISTINCT admission_id)
            FROM drgcodes
            WHERE patient_id = %s
        """, (patient_id,))
        result = cursor.fetchone()[0]
        cursor.close()
        return float(result) if result else 0.0
    
    def _get_n_hospital_admits(self, conn, patient_id):
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*)
            FROM admissions
            WHERE patient_id = %s
        """, (patient_id,))
        result = cursor.fetchone()[0]
        cursor.close()
        return float(result) if result else 0.0
    
    def _get_avg_los_days(self, conn, patient_id):
        cursor = conn.cursor()
        cursor.execute("""
            SELECT AVG(EXTRACT(EPOCH FROM (discharge_time - admit_time)) / 86400)
            FROM admissions
            WHERE patient_id = %s AND discharge_time IS NOT NULL
        """, (patient_id,))
        result = cursor.fetchone()[0]
        cursor.close()
        return float(result) if result else 0.0
    
    def _get_total_los_days(self, conn, patient_id):
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COALESCE(SUM(EXTRACT(EPOCH FROM (discharge_time - admit_time)) / 86400), 0)
            FROM admissions
            WHERE patient_id = %s AND discharge_time IS NOT NULL
        """, (patient_id,))
        result = cursor.fetchone()[0]
        cursor.close()
        return float(result) if result else 0.0
    
    def _get_has_bmi(self, conn, patient_id):
        bmi = self._get_bmi(conn, patient_id)
        return 1.0 if bmi > 0 else 0.0
    
    # OUD-specific features
    
    def _get_opioid_rx_count(self, conn, patient_id):
        cursor = conn.cursor()
        opioid_pattern = '|'.join(self._opioid_drugs)
        cursor.execute("""
            SELECT COUNT(*)
            FROM prescriptions
            WHERE patient_id = %s 
            AND (drug_name ~* %s OR generic_name ~* %s)
        """, (patient_id, opioid_pattern, opioid_pattern))
        result = cursor.fetchone()[0]
        cursor.close()
        return float(result) if result else 0.0
    
    def _get_distinct_opioids(self, conn, patient_id):
        cursor = conn.cursor()
        opioid_pattern = '|'.join(self._opioid_drugs)
        cursor.execute("""
            SELECT COUNT(DISTINCT LOWER(drug_name))
            FROM prescriptions
            WHERE patient_id = %s 
            AND (drug_name ~* %s OR generic_name ~* %s)
        """, (patient_id, opioid_pattern, opioid_pattern))
        result = cursor.fetchone()[0]
        cursor.close()
        return float(result) if result else 0.0
    
    def _get_opioid_exposure_days(self, conn, patient_id):
        cursor = conn.cursor()
        opioid_pattern = '|'.join(self._opioid_drugs)
        cursor.execute("""
            SELECT COALESCE(SUM(EXTRACT(EPOCH FROM (stop_time - start_time)) / 86400), 0)
            FROM prescriptions
            WHERE patient_id = %s 
            AND (drug_name ~* %s OR generic_name ~* %s)
            AND stop_time IS NOT NULL
        """, (patient_id, opioid_pattern, opioid_pattern))
        result = cursor.fetchone()[0]
        cursor.close()
        return float(result) if result else 0.0
    
    def _get_opioid_hadms(self, conn, patient_id):
        cursor = conn.cursor()
        opioid_pattern = '|'.join(self._opioid_drugs)
        cursor.execute("""
            SELECT COUNT(DISTINCT admission_id)
            FROM prescriptions
            WHERE patient_id = %s 
            AND admission_id IS NOT NULL
            AND (drug_name ~* %s OR generic_name ~* %s)
        """, (patient_id, opioid_pattern, opioid_pattern))
        result = cursor.fetchone()[0]
        cursor.close()
        return float(result) if result else 0.0
    
    def _get_any_opioid_flag(self, conn, patient_id):
        opioid_count = self._get_opioid_rx_count(conn, patient_id)
        return 1.0 if opioid_count > 0 else 0.0
    
    def _get_age_at_first_admit(self, conn, patient_id):
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                EXTRACT(YEAR FROM MIN(a.admit_time)) - EXTRACT(YEAR FROM p.date_of_birth) AS age
            FROM admissions a
            JOIN patients p ON a.patient_id = p.patient_id
            WHERE a.patient_id = %s
            GROUP BY p.date_of_birth
        """, (patient_id,))
        result = cursor.fetchone()
        cursor.close()
        return float(result[0]) if result and result[0] else 0.0