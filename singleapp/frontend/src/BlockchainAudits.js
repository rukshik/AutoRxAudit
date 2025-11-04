import React, { useState, useEffect } from 'react';
import './BlockchainAudits.css';

function BlockchainAudits({ user, onLogout, onNavigateToForm, onNavigateToHistory }) {
  const [audits, setAudits] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedAudit, setSelectedAudit] = useState(null);
  const [showModal, setShowModal] = useState(false);

  useEffect(() => {
    fetchBlockchainAudits();
  }, []);

  const fetchBlockchainAudits = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://localhost:8000/api/blockchain-audits?limit=100');
      
      if (response.status === 503) {
        setError('Blockchain service is not running. Please start the blockchain service.');
        setLoading(false);
        return;
      }
      
      if (!response.ok) {
        throw new Error('Failed to fetch blockchain audits');
      }
      
      const data = await response.json();
      setAudits(data.audit_records || []);
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const viewDetails = async (blockchainId) => {
    try {
      const response = await fetch(`http://localhost:8000/api/blockchain-audit/${blockchainId}`);
      if (!response.ok) {
        throw new Error('Failed to fetch audit details');
      }
      const data = await response.json();
      setSelectedAudit(data);
      setShowModal(true);
    } catch (err) {
      alert(`Error: ${err.message}`);
    }
  };

  const closeModal = () => {
    setShowModal(false);
    setSelectedAudit(null);
  };

  const formatTimestamp = (timestamp) => {
    if (!timestamp || timestamp === 0) return 'N/A';
    return new Date(timestamp * 1000).toLocaleString();
  };

  if (loading) {
    return (
      <div className="blockchain-audits">
        <h2>🔗 Blockchain Audit Trail</h2>
        <div className="loading">Loading blockchain records...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="blockchain-audits">
        <h2>🔗 Blockchain Audit Trail</h2>
        <div className="error-message">
          <p>⚠️ {error}</p>
          <button onClick={fetchBlockchainAudits} className="retry-button">
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="blockchain-audits">
      <div className="header">
        <div className="header-left">
          <h2>🔗 Blockchain Audit Trail</h2>
          <p className="subtitle">Immutable records of all prescription audits</p>
          {user && <p className="user-info">Logged in as: {user.full_name} ({user.role})</p>}
        </div>
        <div className="header-actions">
          {onNavigateToForm && (
            <button onClick={onNavigateToForm} className="nav-button">
              New Audit
            </button>
          )}
          {onNavigateToHistory && (
            <button onClick={onNavigateToHistory} className="nav-button">
              Audit History
            </button>
          )}
          <button onClick={fetchBlockchainAudits} className="refresh-button">
            🔄 Refresh
          </button>
          {onLogout && (
            <button onClick={onLogout} className="logout-button">
              Logout
            </button>
          )}
        </div>
      </div>

      {audits.length === 0 ? (
        <div className="no-data">
          <p>No blockchain records found.</p>
          <p className="hint">Audit records will appear here once prescriptions are audited.</p>
        </div>
      ) : (
        <>
          <div className="stats">
            <div className="stat-card">
              <div className="stat-value">{audits.length}</div>
              <div className="stat-label">Total Records</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">
                {audits.filter(a => a.flagged).length}
              </div>
              <div className="stat-label">Flagged</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">
                {audits.filter(a => a.action).length}
              </div>
              <div className="stat-label">Reviewed</div>
            </div>
          </div>

          <div className="table-container">
            <table className="audit-table">
              <thead>
                <tr>
                  <th>Blockchain ID</th>
                  <th>Audit ID</th>
                  <th>Prescription ID</th>
                  <th>Patient ID</th>
                  <th>Drug Name</th>
                  <th>Flagged</th>
                  <th>Action</th>
                  <th>Reviewed By</th>
                  <th>Audited At</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {audits.map((audit) => (
                  <tr key={audit.blockchain_id} className={audit.flagged ? 'flagged-row' : ''}>
                    <td>
                      <span className="blockchain-id">#{audit.blockchain_id}</span>
                    </td>
                    <td>{audit.audit_id}</td>
                    <td>{audit.prescription_id}</td>
                    <td>{audit.patient_id}</td>
                    <td>{audit.drug_name}</td>
                    <td>
                      <span className={`flag-badge ${audit.flagged ? 'flagged' : 'approved'}`}>
                        {audit.flagged ? '⚠️ YES' : '✅ NO'}
                      </span>
                    </td>
                    <td>
                      {audit.action ? (
                        <span className={`action-badge ${audit.action.toLowerCase()}`}>
                          {audit.action}
                        </span>
                      ) : (
                        <span className="action-badge pending">PENDING</span>
                      )}
                    </td>
                    <td>{audit.reviewed_by_name || audit.reviewed_by || '-'}</td>
                    <td>{formatTimestamp(audit.audited_at)}</td>
                    <td>
                      <button 
                        onClick={() => viewDetails(audit.blockchain_id)}
                        className="view-button"
                      >
                        View Details
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {/* Modal for detailed view */}
      {showModal && selectedAudit && (
        <div className="modal-overlay" onClick={closeModal}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>🔗 Blockchain Audit Record #{selectedAudit.blockchain_id}</h3>
              <button onClick={closeModal} className="close-button">×</button>
            </div>
            
            <div className="modal-body">
              <div className="detail-section">
                <h4>Prescription Information</h4>
                <div className="detail-grid">
                  <div className="detail-item">
                    <span className="label">Blockchain ID:</span>
                    <span className="value">#{selectedAudit.blockchain_id}</span>
                  </div>
                  <div className="detail-item">
                    <span className="label">Audit ID:</span>
                    <span className="value">{selectedAudit.audit_id}</span>
                  </div>
                  <div className="detail-item">
                    <span className="label">Prescription ID:</span>
                    <span className="value">{selectedAudit.prescription_id}</span>
                  </div>
                  <div className="detail-item">
                    <span className="label">Patient ID:</span>
                    <span className="value">{selectedAudit.patient_id}</span>
                  </div>
                  <div className="detail-item">
                    <span className="label">Drug Name:</span>
                    <span className="value">{selectedAudit.drug_name}</span>
                  </div>
                </div>
              </div>

              <div className="detail-section">
                <h4>Risk Assessment</h4>
                <div className="detail-grid">
                  <div className="detail-item">
                    <span className="label">Eligibility Score:</span>
                    <span className="value">{selectedAudit.eligibility_score}%</span>
                  </div>
                  <div className="detail-item">
                    <span className="label">Eligibility Prediction:</span>
                    <span className="value">
                      {selectedAudit.eligibility_prediction === 1 ? 'Eligible' : 'Not Eligible'}
                    </span>
                  </div>
                  <div className="detail-item">
                    <span className="label">OUD Risk Score:</span>
                    <span className={`value ${selectedAudit.oud_risk_score >= 70 ? 'high-risk' : 'low-risk'}`}>
                      {selectedAudit.oud_risk_score}%
                    </span>
                  </div>
                  <div className="detail-item">
                    <span className="label">OUD Risk Prediction:</span>
                    <span className="value">
                      {selectedAudit.oud_risk_prediction === 1 ? 'High Risk' : 'Low Risk'}
                    </span>
                  </div>
                  <div className="detail-item full-width">
                    <span className="label">Flagged:</span>
                    <span className={`value ${selectedAudit.flagged ? 'flagged' : 'approved'}`}>
                      {selectedAudit.flagged ? '⚠️ YES' : '✅ NO'}
                    </span>
                  </div>
                </div>
              </div>

              <div className="detail-section">
                <h4>AI Assessment</h4>
                <div className="detail-item full-width">
                  <span className="label">Flag Reason:</span>
                  <p className="value text">{selectedAudit.flag_reason}</p>
                </div>
                <div className="detail-item full-width">
                  <span className="label">Recommendation:</span>
                  <p className="value text">{selectedAudit.recommendation}</p>
                </div>
              </div>

              <div className="detail-section">
                <h4>Review Information</h4>
                <div className="detail-grid">
                  <div className="detail-item">
                    <span className="label">Action:</span>
                    <span className="value">
                      {selectedAudit.action || 'PENDING'}
                    </span>
                  </div>
                  <div className="detail-item">
                    <span className="label">Reviewed By:</span>
                    <span className="value">
                      {selectedAudit.reviewed_by_name || selectedAudit.reviewed_by || 'Not reviewed'}
                    </span>
                  </div>
                  {selectedAudit.reviewed_by_email && (
                    <div className="detail-item">
                      <span className="label">Reviewer Email:</span>
                      <span className="value">{selectedAudit.reviewed_by_email}</span>
                    </div>
                  )}
                  {selectedAudit.action_reason && (
                    <div className="detail-item full-width">
                      <span className="label">Action Reason:</span>
                      <p className="value text">{selectedAudit.action_reason}</p>
                    </div>
                  )}
                </div>
              </div>

              <div className="detail-section">
                <h4>Timestamps</h4>
                <div className="detail-grid">
                  <div className="detail-item">
                    <span className="label">Audited At:</span>
                    <span className="value">{formatTimestamp(selectedAudit.audited_at)}</span>
                  </div>
                  <div className="detail-item">
                    <span className="label">Reviewed At:</span>
                    <span className="value">
                      {formatTimestamp(selectedAudit.reviewed_at)}
                    </span>
                  </div>
                </div>
              </div>

              <div className="immutability-notice">
                <strong>🔒 Immutable Record:</strong> This audit record is stored on the blockchain 
                and cannot be modified or deleted, ensuring data integrity and auditability.
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default BlockchainAudits;
