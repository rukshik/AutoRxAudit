import React, { useState, useEffect } from 'react';
import './AuditHistory.css';

function AuditHistory({ user, onLogout, onNavigateToForm }) {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/audit-history');
      if (response.ok) {
        const data = await response.json();
        setHistory(data);
      } else {
        setError('Failed to load audit history');
      }
    } catch (err) {
      console.error('Error loading history:', err);
      setError('Error connecting to server');
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const getActionClass = (action) => {
    switch (action) {
      case 'APPROVED':
        return 'action-approved';
      case 'DENIED':
        return 'action-denied';
      case 'OVERRIDE_APPROVE':
        return 'action-override';
      case 'OVERRIDE_DENY':
        return 'action-override-deny';
      default:
        return '';
    }
  };

  const getActionLabel = (action) => {
    switch (action) {
      case 'APPROVED':
        return '✅ Approved';
      case 'DENIED':
        return '❌ Denied';
      case 'OVERRIDE_APPROVE':
        return '⚠️ Override Approve';
      case 'OVERRIDE_DENY':
        return '⚠️ Override Deny';
      default:
        return action;
    }
  };

  return (
    <div className="history-container">
      <header className="app-header">
        <div>
          <h1>AutoRxAudit - Audit History</h1>
          <p className="user-info">Logged in as: {user.full_name} ({user.role})</p>
        </div>
        <div>
          <button onClick={onNavigateToForm} className="nav-button">
            New Audit
          </button>
          <button onClick={onLogout} className="logout-button">
            Logout
          </button>
        </div>
      </header>

      <div className="history-content">
        <div className="history-card">
          <h2>Prescription Audit Records</h2>
          
          {loading && (
            <div className="loading-message">Loading audit history...</div>
          )}

          {error && (
            <div className="error-message">{error}</div>
          )}

          {!loading && !error && history.length === 0 && (
            <div className="empty-message">No audit records found</div>
          )}

          {!loading && !error && history.length > 0 && (
            <div className="table-wrapper">
              <table className="history-table">
                <thead>
                  <tr>
                    <th>Date & Time</th>
                    <th>Patient ID</th>
                    <th>Drug</th>
                    <th>Eligibility</th>
                    <th>OUD Risk</th>
                    <th>Flagged</th>
                    <th>Action</th>
                    <th>Clinician</th>
                    <th>Reason</th>
                  </tr>
                </thead>
                <tbody>
                  {history.map((record) => (
                    <tr key={record.audit_id}>
                      <td className="date-cell">{formatDate(record.created_at)}</td>
                      <td className="patient-cell">{record.patient_id}</td>
                      <td className="drug-cell">{record.drug_name}</td>
                      <td className="score-cell">
                        <div className={`score-badge ${record.eligibility_prediction === 1 ? 'positive' : 'negative'}`}>
                          {(record.eligibility_score * 100).toFixed(0)}%
                        </div>
                      </td>
                      <td className="score-cell">
                        <div className={`score-badge ${record.oud_risk_prediction === 0 ? 'positive' : 'negative'}`}>
                          {(record.oud_risk_score * 100).toFixed(0)}%
                        </div>
                      </td>
                      <td className="flagged-cell">
                        {record.flagged ? (
                          <span className="flag-yes">🚩 Yes</span>
                        ) : (
                          <span className="flag-no">✅ No</span>
                        )}
                      </td>
                      <td className="action-cell">
                        <span className={`action-badge ${getActionClass(record.action)}`}>
                          {getActionLabel(record.action)}
                        </span>
                      </td>
                      <td className="clinician-cell">{record.clinician_name}</td>
                      <td className="reason-cell">
                        {record.action_reason || <em>No reason provided</em>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default AuditHistory;
