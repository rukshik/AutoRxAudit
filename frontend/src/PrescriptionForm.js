import React, { useState, useEffect } from 'react';
import './PrescriptionForm.css';

function PrescriptionForm({ user, onLogout, onNavigateToHistory, selectedAudit, onAuditActionComplete }) {
  const [patients, setPatients] = useState([]);
  const [drugs, setDrugs] = useState({ opioids: [], non_opioids: [] });
  const [selectedPatient, setSelectedPatient] = useState('');
  const [selectedDrug, setSelectedDrug] = useState('');
  const [loading, setLoading] = useState(false);
  const [auditResult, setAuditResult] = useState(null);
  const [actionReason, setActionReason] = useState('');

  useEffect(() => {
    loadPatientsAndDrugs();
  }, []);

  useEffect(() => {
    // If an audit is selected from history, show it as the audit result
    if (selectedAudit) {
      setAuditResult(selectedAudit);
      setSelectedPatient(selectedAudit.patient_id);
      setSelectedDrug(selectedAudit.drug_name);
    }
  }, [selectedAudit]);

  const loadPatientsAndDrugs = async () => {
    try {
      const [patientsRes, drugsRes] = await Promise.all([
        fetch('http://localhost:8000/api/patients'),
        fetch('http://localhost:8000/api/drugs')
      ]);
      
      if (patientsRes.ok && drugsRes.ok) {
        setPatients(await patientsRes.json());
        setDrugs(await drugsRes.json());
      }
    } catch (err) {
      console.error('Error loading data:', err);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setAuditResult(null);

    try {
      const response = await fetch('http://localhost:8000/audit-prescription', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          patient_id: selectedPatient,
          drug_name: selectedDrug,
        }),
      });

      const data = await response.json();
      setAuditResult(data);
    } catch (err) {
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleAction = async (action) => {
    try {
      const response = await fetch('http://localhost:8000/api/audit-action', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          audit_id: auditResult.audit_id,
          action: action,
          action_reason: actionReason || null,
        }),
      });

      if (response.ok) {
        alert(`Prescription ${action.toLowerCase()} successfully!`);
        setAuditResult(null);
        setSelectedPatient('');
        setSelectedDrug('');
        setActionReason('');
        
        // If this was from audit history, navigate back to history
        if (selectedAudit && onAuditActionComplete) {
          setTimeout(() => {
            onAuditActionComplete();
          }, 500);
        }
      }
    } catch (err) {
      console.error('Error recording action:', err);
    }
  };

  const getScoreClass = (score) => {
    if (score >= 0.7) return 'score-high';
    if (score >= 0.4) return 'score-medium';
    return 'score-low';
  };

  return (
    <div className="prescription-container">
      <header className="app-header">
        <div>
          <h1>AutoRxAudit</h1>
          <p className="user-info">Logged in as: {user.full_name} ({user.role})</p>
        </div>
        <div>
          <button onClick={onNavigateToHistory} className="nav-button">
            View Audit History
          </button>
          <button onClick={onLogout} className="logout-button">
            Logout
          </button>
        </div>
      </header>

      <div className="main-content">
        <div className="form-section">
          {selectedAudit ? (
            <div className="audit-review-header">
              <h2>Review Pending Audit</h2>
              <button 
                onClick={onNavigateToHistory} 
                className="back-button"
                type="button"
              >
                ← Back to History
              </button>
            </div>
          ) : (
            <h2>Prescription Audit Form</h2>
          )}
          
          {!selectedAudit && (
          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label htmlFor="patient">Patient ID</label>
              <select
                id="patient"
                value={selectedPatient}
                onChange={(e) => setSelectedPatient(e.target.value)}
                required
                disabled={loading}
              >
                <option value="">Select a patient...</option>
                {patients.map((p) => (
                  <option key={p.patient_id} value={p.patient_id}>
                    {p.patient_id} - {p.gender}, DOB: {p.date_of_birth}
                  </option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label htmlFor="drug">Drug</label>
              <select
                id="drug"
                value={selectedDrug}
                onChange={(e) => setSelectedDrug(e.target.value)}
                required
                disabled={loading}
              >
                <option value="">Select a drug...</option>
                <optgroup label="Opioids">
                  {drugs.opioids.map((drug) => (
                    <option key={drug} value={drug}>
                      {drug}
                    </option>
                  ))}
                </optgroup>
                <optgroup label="Non-Opioids">
                  {drugs.non_opioids.map((drug) => (
                    <option key={drug} value={drug}>
                      {drug}
                    </option>
                  ))}
                </optgroup>
              </select>
            </div>

            <button type="submit" className="submit-button" disabled={loading || !selectedPatient || !selectedDrug}>
              {loading ? 'Auditing...' : 'Audit Prescription'}
            </button>
          </form>
          )}
        </div>

        {auditResult && (
          <div className="result-section">
            <h2>Audit Result</h2>
            
            <div className={`result-banner ${auditResult.flagged ? 'flagged' : 'approved'}`}>
              {auditResult.flagged ? '🚩 FLAGGED FOR REVIEW' : '✅ NO CONCERNS IDENTIFIED'}
            </div>

            <div className="result-details">
              <div className="score-card">
                <h3>Eligibility Score</h3>
                <div className={`score ${getScoreClass(auditResult.eligibility_score)}`}>
                  {(auditResult.eligibility_score * 100).toFixed(1)}%
                </div>
                <p className="score-label">
                  {auditResult.eligibility_prediction === 1 ? 'ELIGIBLE' : 'NOT ELIGIBLE'}
                </p>
                <p className="score-explanation">
                  Higher score = Patient has clinical need for opioids (pain diagnosis)
                </p>
              </div>

              <div className="score-card">
                <h3>OUD Risk Score</h3>
                <div className={`score ${getScoreClass(1 - auditResult.oud_risk_score)}`}>
                  {(auditResult.oud_risk_score * 100).toFixed(1)}%
                </div>
                <p className="score-label">
                  {auditResult.oud_risk_prediction === 1 ? 'HIGH RISK' : 'LOW RISK'}
                </p>
                <p className="score-explanation">
                  Lower score = Lower risk of opioid use disorder
                </p>
              </div>
            </div>

            <div className="recommendation-box">
              <h3>Recommendation</h3>
              <p><strong>Reason:</strong> {auditResult.flag_reason}</p>
              <p><strong>Action:</strong> {auditResult.recommendation}</p>
            </div>

            <div className="action-section">
              <h3>Pharmacist Decision</h3>
              
              <textarea
                className="action-reason"
                placeholder="Enter reason for your decision (optional)"
                value={actionReason}
                onChange={(e) => setActionReason(e.target.value)}
                rows="3"
              />

              <div className="action-buttons">
                {auditResult.flagged ? (
                  <>
                    <button 
                      className="action-button deny"
                      onClick={() => handleAction('DENIED')}
                    >
                      Deny Prescription
                    </button>
                    <button 
                      className="action-button override"
                      onClick={() => handleAction('OVERRIDE_APPROVE')}
                    >
                      Override & Approve
                    </button>
                  </>
                ) : (
                  <>
                    <button 
                      className="action-button approve"
                      onClick={() => handleAction('APPROVED')}
                    >
                      Approve Prescription
                    </button>
                    <button 
                      className="action-button override-deny"
                      onClick={() => handleAction('OVERRIDE_DENY')}
                    >
                      Override & Deny
                    </button>
                  </>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default PrescriptionForm;
