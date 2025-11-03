import React, { useState, useEffect } from 'react';
import Login from './Login';
import PrescriptionForm from './PrescriptionForm';
import AuditHistory from './AuditHistory';
import BlockchainAudits from './BlockchainAudits';
import './App.css';

function App() {
  const [user, setUser] = useState(null);
  const [currentView, setCurrentView] = useState('form'); // 'form', 'history', or 'blockchain'
  const [selectedAudit, setSelectedAudit] = useState(null);

  useEffect(() => {
    // Check if user is already logged in
    const savedUser = localStorage.getItem('user');
    if (savedUser) {
      try {
        setUser(JSON.parse(savedUser));
      } catch (err) {
        console.error('Error parsing saved user:', err);
        localStorage.removeItem('user');
      }
    }
  }, []);

  const handleLogin = (userData) => {
    setUser(userData);
    localStorage.setItem('user', JSON.stringify(userData));
  };

  const handleLogout = () => {
    setUser(null);
    localStorage.removeItem('user');
    setCurrentView('form');
  };

  const navigateToForm = () => {
    setSelectedAudit(null);
    setCurrentView('form');
  };

  const navigateToHistory = () => {
    setSelectedAudit(null);
    setCurrentView('history');
  };

  const navigateToBlockchain = () => {
    setSelectedAudit(null);
    setCurrentView('blockchain');
  };

  const navigateToAudit = (audit) => {
    setSelectedAudit(audit);
    setCurrentView('form');
  };

  // If not logged in, show login page
  if (!user) {
    return <Login onLogin={handleLogin} />;
  }

  // If logged in, show the appropriate view
  return (
    <div className="App">
      {currentView === 'form' ? (
        <PrescriptionForm 
          user={user} 
          onLogout={handleLogout} 
          onNavigateToHistory={navigateToHistory}
          onNavigateToBlockchain={navigateToBlockchain}
          selectedAudit={selectedAudit}
          onAuditActionComplete={navigateToHistory}
        />
      ) : currentView === 'history' ? (
        <AuditHistory 
          user={user} 
          onLogout={handleLogout} 
          onNavigateToForm={navigateToForm}
          onNavigateToBlockchain={navigateToBlockchain}
          onNavigateToAudit={navigateToAudit}
        />
      ) : (
        <BlockchainAudits 
          user={user} 
          onLogout={handleLogout} 
          onNavigateToForm={navigateToForm}
          onNavigateToHistory={navigateToHistory}
        />
      )}
    </div>
  );
}

export default App;
