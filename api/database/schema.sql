-- AutoRxAudit Database Schema
-- Exported from working database
-- ============================================================================

-- ADMISSIONS TABLE
CREATE TABLE IF NOT EXISTS admissions (admission_id character varying(50) NOT NULL, patient_id character varying(50) NOT NULL, admit_time timestamp without time zone NOT NULL, discharge_time timestamp without time zone, admission_type character varying(50), admission_location character varying(100), discharge_location character varying(100), insurance character varying(50), language character varying(50), marital_status character varying(50), hospital_expire_flag boolean DEFAULT false);

CREATE INDEX idx_admissions_patient ON public.admissions USING btree (patient_id);
CREATE INDEX idx_admissions_dates ON public.admissions USING btree (admit_time, discharge_time);

-- AUDIT_LOGS TABLE
CREATE TABLE IF NOT EXISTS audit_logs (audit_id integer NOT NULL DEFAULT nextval('audit_logs_audit_id_seq'::regclass), prescription_id integer, patient_id character varying(50) NOT NULL, drug_name character varying(255) NOT NULL, eligibility_score double precision NOT NULL, eligibility_prediction integer NOT NULL, oud_risk_score double precision NOT NULL, oud_risk_prediction integer NOT NULL, flagged boolean NOT NULL, flag_reason text, recommendation text, reviewed_by integer, action character varying(50), action_reason text, reviewed_at timestamp without time zone, audited_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP);

CREATE INDEX idx_audit_logs_prescription ON public.audit_logs USING btree (prescription_id);
CREATE INDEX idx_audit_logs_patient ON public.audit_logs USING btree (patient_id);
CREATE INDEX idx_audit_logs_flagged ON public.audit_logs USING btree (flagged);
CREATE INDEX idx_audit_logs_action ON public.audit_logs USING btree (action);
CREATE INDEX idx_audit_logs_reviewed_by ON public.audit_logs USING btree (reviewed_by);
CREATE INDEX idx_audit_logs_audited ON public.audit_logs USING btree (audited_at DESC);
CREATE INDEX idx_audit_logs_reviewed ON public.audit_logs USING btree (reviewed_at DESC);

-- DIAGNOSES TABLE
CREATE TABLE IF NOT EXISTS diagnoses (diagnosis_id integer NOT NULL DEFAULT nextval('diagnoses_diagnosis_id_seq'::regclass), patient_id character varying(50) NOT NULL, admission_id character varying(50), icd_code character varying(20) NOT NULL, icd_version integer DEFAULT 10, seq_num integer);

CREATE INDEX idx_diagnoses_patient ON public.diagnoses USING btree (patient_id);
CREATE INDEX idx_diagnoses_admission ON public.diagnoses USING btree (admission_id);
CREATE INDEX idx_diagnoses_icd ON public.diagnoses USING btree (icd_code);
CREATE INDEX idx_diagnoses_primary ON public.diagnoses USING btree (seq_num) WHERE (seq_num = 1);

-- DRGCODES TABLE
CREATE TABLE IF NOT EXISTS drgcodes (drg_id integer NOT NULL DEFAULT nextval('drgcodes_drg_id_seq'::regclass), patient_id character varying(50) NOT NULL, admission_id character varying(50) NOT NULL, drg_code character varying(20) NOT NULL, drg_type character varying(50), drg_severity integer, drg_mortality integer);

CREATE INDEX idx_drg_patient ON public.drgcodes USING btree (patient_id);
CREATE INDEX idx_drg_admission ON public.drgcodes USING btree (admission_id);
CREATE INDEX idx_drg_code ON public.drgcodes USING btree (drg_code);

-- OMR TABLE
CREATE TABLE IF NOT EXISTS omr (omr_id integer NOT NULL DEFAULT nextval('omr_omr_id_seq'::regclass), patient_id character varying(50) NOT NULL, chart_time timestamp without time zone NOT NULL, result_name character varying(100) NOT NULL, result_type character varying(50), result_value double precision, result_unit character varying(20), normal_range_low double precision, normal_range_high double precision, is_abnormal boolean);

CREATE INDEX idx_omr_patient ON public.omr USING btree (patient_id);
CREATE INDEX idx_omr_result ON public.omr USING btree (result_name);
CREATE INDEX idx_omr_time ON public.omr USING btree (chart_time);
CREATE INDEX idx_omr_type ON public.omr USING btree (result_type);

-- PATIENTS TABLE
CREATE TABLE IF NOT EXISTS patients (patient_id character varying(50) NOT NULL, date_of_birth date NOT NULL, gender character varying(10), race character varying(50), ethnicity character varying(50), created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP, updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP);

-- PRESCRIPTION_REQUESTS TABLE
CREATE TABLE IF NOT EXISTS prescription_requests (prescription_id integer NOT NULL DEFAULT nextval('prescription_requests_prescription_id_seq'::regclass), patient_id character varying(50) NOT NULL, drug_name character varying(255) NOT NULL, prescriber_id integer, prescribed_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP, status character varying(50) DEFAULT 'PENDING'::character varying);

CREATE INDEX idx_prescription_requests_patient ON public.prescription_requests USING btree (patient_id);
CREATE INDEX idx_prescription_requests_status ON public.prescription_requests USING btree (status);
CREATE INDEX idx_prescription_requests_created ON public.prescription_requests USING btree (prescribed_at DESC);

-- PRESCRIPTIONS TABLE
CREATE TABLE IF NOT EXISTS prescriptions (prescription_id integer NOT NULL DEFAULT nextval('prescriptions_prescription_id_seq'::regclass), subject_id character varying(50) NOT NULL, hadm_id character varying(50), pharmacy_id character varying(50), starttime timestamp without time zone, stoptime timestamp without time zone, drug_type character varying(50), drug text, drug_name text, generic_name text, formulary_drug_cd character varying(50), gsn text, ndc character varying(50), prod_strength text, form_rx character varying(50), dose_val_rx character varying(100), dose_unit_rx character varying(50), form_val_disp character varying(100), form_unit_disp character varying(50), doses_per_24_hrs double precision, route character varying(50), patient_id character varying(50), admission_id character varying(50), start_time timestamp without time zone, stop_time timestamp without time zone);

CREATE INDEX idx_prescriptions_subject ON public.prescriptions USING btree (subject_id);
CREATE INDEX idx_prescriptions_patient ON public.prescriptions USING btree (patient_id);
CREATE INDEX idx_prescriptions_hadm ON public.prescriptions USING btree (hadm_id);
CREATE INDEX idx_prescriptions_admission ON public.prescriptions USING btree (admission_id);
CREATE INDEX idx_prescriptions_drug ON public.prescriptions USING btree (drug);
CREATE INDEX idx_prescriptions_drug_name ON public.prescriptions USING btree (drug_name);
CREATE INDEX idx_prescriptions_generic ON public.prescriptions USING btree (generic_name);
CREATE INDEX idx_prescriptions_dates ON public.prescriptions USING btree (starttime, stoptime);
CREATE INDEX idx_prescriptions_dates_compat ON public.prescriptions USING btree (start_time, stop_time);

-- REF_ATC_CODES TABLE
CREATE TABLE IF NOT EXISTS ref_atc_codes (atc_code character varying(20) NOT NULL, atc_level integer, description text, parent_code character varying(20), is_opioid boolean DEFAULT false);

CREATE INDEX idx_atc_level ON public.ref_atc_codes USING btree (atc_level);
CREATE INDEX idx_atc_parent ON public.ref_atc_codes USING btree (parent_code);

-- REF_DRG_CODES TABLE
CREATE TABLE IF NOT EXISTS ref_drg_codes (drg_code character varying(20) NOT NULL, description text, mdc_code character varying(10), severity_level integer, mortality_risk integer);

-- REF_ICD_CODES TABLE
CREATE TABLE IF NOT EXISTS ref_icd_codes (icd_code character varying(20) NOT NULL, description text, category character varying(100), is_pain_related boolean DEFAULT false, is_oud_related boolean DEFAULT false);

CREATE INDEX idx_icd_category ON public.ref_icd_codes USING btree (category);

-- REF_OPIOID_DRUGS TABLE
CREATE TABLE IF NOT EXISTS ref_opioid_drugs (drug_name character varying(255) NOT NULL, generic_name character varying(255), atc_code character varying(20), strength_unit character varying(50), is_long_acting boolean DEFAULT false, mme_conversion_factor double precision);

-- TRANSFERS TABLE
CREATE TABLE IF NOT EXISTS transfers (transfer_id integer NOT NULL DEFAULT nextval('transfers_transfer_id_seq'::regclass), patient_id character varying(50) NOT NULL, admission_id character varying(50) NOT NULL, in_time timestamp without time zone NOT NULL, out_time timestamp without time zone, care_unit character varying(100), careunit_type character varying(50), los_hours double precision);

CREATE INDEX idx_transfer_patient ON public.transfers USING btree (patient_id);
CREATE INDEX idx_transfer_admission ON public.transfers USING btree (admission_id);
CREATE INDEX idx_transfer_unit ON public.transfers USING btree (care_unit);
CREATE INDEX idx_transfer_type ON public.transfers USING btree (careunit_type);
CREATE INDEX idx_transfer_times ON public.transfers USING btree (in_time, out_time);

-- USERS TABLE
CREATE TABLE IF NOT EXISTS users (user_id integer NOT NULL DEFAULT nextval('users_user_id_seq'::regclass), email character varying(255) NOT NULL, password character varying(255) NOT NULL, full_name character varying(255), role character varying(50) DEFAULT 'clinician'::character varying, is_active boolean DEFAULT true, created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP, last_login timestamp without time zone);

CREATE UNIQUE INDEX users_email_key ON public.users USING btree (email);
CREATE INDEX idx_users_email ON public.users USING btree (email);
