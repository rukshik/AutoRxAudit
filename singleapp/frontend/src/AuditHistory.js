import React, { useState, useEffect } from 'react';
import './AuditHistory.css';

function AuditHistory({ user, onLogout, onNavigateToForm, onNavigateToBlockchain, onNavigateToAudit }) {
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

  const handleRowClick = (audit) => {
    // Navigate to audit action page with the audit data
    if (onNavigateToAudit) {
      onNavigateToAudit(audit);
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
          <button onClick={onNavigateToBlockchain} className="nav-button">
            🔗 Blockchain Audit Trail
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
                    <th>Audit ID</th>
                    <th>Rx ID</th>
                    <th>Date & Time</th>
                    <th>Patient ID</th>
                    <th>Drug</th>
                    <th>Eligibility</th>
                    <th>OUD Risk</th>
                    <th>Flagged</th>
                    <th>Flag Reason</th>
                    <th>Action</th>
                    <th>Reviewed By</th>
                    <th>Reviewed At</th>
                    <th>Action Reason</th>
                  </tr>
                </thead>
                <tbody>
                  {history.map((record) => (
                    <tr 
                      key={record.audit_id}
                      onClick={() => !record.action && handleRowClick(record)}
                      className={!record.action ? 'pending-clickable' : ''}
                      style={{cursor: !record.action ? 'pointer' : 'default'}}
                    >
                      <td className="audit-id-cell">{record.audit_id}</td>
                      <td className="prescription-id-cell">{record.prescription_id}</td>
                      <td className="date-cell">{formatDate(record.audited_at)}</td>
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
                      <td className="flag-reason-cell">{record.flag_reason || '-'}</td>
                      <td className="action-cell">
                        {record.action ? (
                          <span className={`action-badge ${getActionClass(record.action)}`}>
                            {getActionLabel(record.action)}
                          </span>
                        ) : (
                          <span className="pending-badge">⏳ Pending - Click to Review</span>
                        )}
                      </td>
                      <td className="clinician-cell">
                        {record.clinician_name || <em>Not reviewed</em>}
                      </td>
                      <td className="reviewed-date-cell">
                        {record.reviewed_at ? formatDate(record.reviewed_at) : '-'}
                      </td>
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
