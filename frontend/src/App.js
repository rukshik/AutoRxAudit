import React, { useState, useEffect } from 'react';
import Login from './Login';
import PrescriptionForm from './PrescriptionForm';
import AuditHistory from './AuditHistory';
import './App.css';

function App() {
  const [user, setUser] = useState(null);
  const [currentView, setCurrentView] = useState('form'); // 'form' or 'history'

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
    setCurrentView('form');
  };

  const navigateToHistory = () => {
    setCurrentView('history');
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
        />
      ) : (
        <AuditHistory 
          user={user} 
          onLogout={handleLogout} 
          onNavigateToForm={navigateToForm}
        />
      )}
    </div>
  );
}

export default App;
