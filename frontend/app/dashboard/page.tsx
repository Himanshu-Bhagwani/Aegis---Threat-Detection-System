"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import {
  getHealth,
  computeUnifiedRisk,
  scoreLoginEvent,
  scorePassword,
  scoreFraudTransaction,
  UnifiedRiskResponse,
  LoginAnomalyResponse,
  PasswordScoreResponse,
  FraudScoreResponse,
  getRiskLevelColor,
  getRiskLevelEmoji,
  formatProbability,
  getThreatCategoryLabel
} from "@/lib/api";

// Types for demo data
interface DemoState {
  unifiedRisk: UnifiedRiskResponse | null;
  loginRisk: LoginAnomalyResponse | null;
  passwordRisk: PasswordScoreResponse | null;
  fraudRisk: FraudScoreResponse | null;
  loading: boolean;
  error: string | null;
}

export default function DashboardPage() {
  const [healthStatus, setHealthStatus] = useState<string>("checking");
  const [components, setComponents] = useState<Record<string, string>>({});
  const [demo, setDemo] = useState<DemoState>({
    unifiedRisk: null,
    loginRisk: null,
    passwordRisk: null,
    fraudRisk: null,
    loading: false,
    error: null
  });

  // Form states
  const [loginForm, setLoginForm] = useState({
    hour_of_day: 3,
    failed_10min: 2,
    is_new_comp: 1,
    comp_deg: 150
  });
  const [passwordInput, setPasswordInput] = useState("");
  const [fraudForm, setFraudForm] = useState({
    amount: "1000",
    is_international: false,
    hour: "3",
    tx_count_1h: "5"
  });

  useEffect(() => {
    checkHealth();
  }, []);

  const checkHealth = async () => {
    try {
      const health = await getHealth();
      setHealthStatus(health.status);
      setComponents(health.components);
    } catch (err) {
      setHealthStatus("offline");
      console.error("Health check failed:", err);
    }
  };

  const runDemoAnalysis = async () => {
    setDemo(prev => ({ ...prev, loading: true, error: null }));
    
    try {
      // Run unified risk analysis with demo data
      const unifiedResult = await computeUnifiedRisk({
        user_id: "demo_user_001",
        event_id: "demo_evt_001",
        gps_data: {
          spoof_probability: 0.35,
          confidence: 0.8
        },
        login_data: {
          ...loginForm,
          user_deg: 5,
          time_since_user_last: 7200
        },
        password_score: 0.45,
        transaction_data: {
          fraud_probability: 0.25
        },
        fusion_strategy: "weighted_average"
      });

      setDemo(prev => ({
        ...prev,
        unifiedRisk: unifiedResult,
        loading: false
      }));
    } catch (err) {
      setDemo(prev => ({
        ...prev,
        error: err instanceof Error ? err.message : "Analysis failed",
        loading: false
      }));
    }
  };

  const testLoginAnomaly = async () => {
    setDemo(prev => ({ ...prev, loading: true, error: null }));
    
    try {
      const result = await scoreLoginEvent({
        ...loginForm,
        user_deg: 5,
        time_since_user_last: 3600,
        time_since_comp_last: 300,
        is_new_user: 0,
        user_id: "test_user"
      });
      
      setDemo(prev => ({ ...prev, loginRisk: result, loading: false }));
    } catch (err) {
      setDemo(prev => ({
        ...prev,
        error: err instanceof Error ? err.message : "Login analysis failed",
        loading: false
      }));
    }
  };

  const testPassword = async () => {
    if (!passwordInput.trim()) return;
    
    setDemo(prev => ({ ...prev, loading: true, error: null }));
    
    try {
      const result = await scorePassword(passwordInput);
      setDemo(prev => ({ ...prev, passwordRisk: result, loading: false }));
    } catch (err) {
      setDemo(prev => ({
        ...prev,
        error: err instanceof Error ? err.message : "Password analysis failed",
        loading: false
      }));
    }
  };

  const testFraud = async () => {
    setDemo(prev => ({ ...prev, loading: true, error: null }));
    
    try {
      const payload = {
        amount: parseFloat(fraudForm.amount),
        is_international: fraudForm.is_international ? 1 : 0,
        hour: parseInt(fraudForm.hour),
        tx_count_1h: parseInt(fraudForm.tx_count_1h),
        time_since_last_tx: 300,
        amount_ratio: 2.5,
        merchant_freq_user: 1
      };
      
      const result = await scoreFraudTransaction(payload);
      setDemo(prev => ({ ...prev, fraudRisk: result, loading: false }));
    } catch (err) {
      setDemo(prev => ({
        ...prev,
        error: err instanceof Error ? err.message : "Fraud analysis failed",
        loading: false
      }));
    }
  };

  return (
    <main className="dashboard-page">
      {/* Hero Section */}
      <header className="dashboard-hero">
        <div>
          <span className="dashboard-kicker">Aegis Command Center</span>
          <h1>Unified Threat Detection</h1>
          <p>
            Multi-layer threat intelligence combining GPS spoofing, login anomalies,
            password integrity, and fraud detection into actionable risk scores.
          </p>
        </div>
        <div className="hero-actions">
          <span className={`status-badge status-${healthStatus}`}>
            {healthStatus === "healthy" ? "🟢" : healthStatus === "degraded" ? "🟡" : "🔴"}
            {" "}API {healthStatus}
          </span>
          <Link href="/" className="hero-cta hero-cta-ghost">
            Back to landing
          </Link>
        </div>
      </header>

      {/* Quick Stats */}
      <section className="stats-grid">
        <div className="stat-card">
          <span className="stat-icon">🛰️</span>
          <div className="stat-info">
            <span className="stat-label">GPS Detection</span>
            <span className={`stat-status ${components.gps === "healthy" ? "active" : "inactive"}`}>
              {components.gps === "healthy" ? "Active" : "Offline"}
            </span>
          </div>
        </div>
        <div className="stat-card">
          <span className="stat-icon">🔐</span>
          <div className="stat-info">
            <span className="stat-label">Login Analysis</span>
            <span className={`stat-status ${components.login === "healthy" ? "active" : "inactive"}`}>
              {components.login === "healthy" ? "Active" : "Offline"}
            </span>
          </div>
        </div>
        <div className="stat-card">
          <span className="stat-icon">🔑</span>
          <div className="stat-info">
            <span className="stat-label">Password Check</span>
            <span className={`stat-status ${components.password === "healthy" ? "active" : "inactive"}`}>
              {components.password === "healthy" ? "Active" : "Offline"}
            </span>
          </div>
        </div>
        <div className="stat-card">
          <span className="stat-icon">💳</span>
          <div className="stat-info">
            <span className="stat-label">Fraud Detection</span>
            <span className={`stat-status ${components.fraud === "healthy" ? "active" : "inactive"}`}>
              {components.fraud === "healthy" ? "Active" : "Offline"}
            </span>
          </div>
        </div>
      </section>

      {/* Main Dashboard Panels */}
      <section className="dashboard-panels">
        {/* Unified Risk Panel */}
        <article className="dashboard-panel panel-wide">
          <header>
            <h2>⚡ Unified Risk Assessment</h2>
            <button 
              className="panel-action-btn"
              onClick={runDemoAnalysis}
              disabled={demo.loading}
            >
              {demo.loading ? "Analyzing..." : "Run Demo Analysis"}
            </button>
          </header>
          
          {demo.error && (
            <div className="error-banner">{demo.error}</div>
          )}
          
          {demo.unifiedRisk ? (
            <div className="risk-result">
              <div className="risk-score-display">
                <div 
                  className="risk-score-circle"
                  style={{ 
                    borderColor: getRiskLevelColor(demo.unifiedRisk.risk_level),
                    boxShadow: `0 0 30px ${getRiskLevelColor(demo.unifiedRisk.risk_level)}40`
                  }}
                >
                  <span className="risk-score-value">
                    {formatProbability(demo.unifiedRisk.unified_score)}
                  </span>
                  <span className="risk-score-label">
                    {getRiskLevelEmoji(demo.unifiedRisk.risk_level)} {demo.unifiedRisk.risk_level.toUpperCase()}
                  </span>
                </div>
              </div>
              
              <div className="risk-breakdown">
                <h3>Component Risks</h3>
                <div className="risk-bars">
                  <RiskBar label="GPS Spoofing" value={demo.unifiedRisk.gps_risk} />
                  <RiskBar label="Login Anomaly" value={demo.unifiedRisk.login_risk} />
                  <RiskBar label="Password Risk" value={demo.unifiedRisk.password_risk} />
                  <RiskBar label="Fraud Risk" value={demo.unifiedRisk.fraud_risk} />
                </div>
              </div>
              
              {demo.unifiedRisk.primary_threats.length > 0 && (
                <div className="threats-list">
                  <h3>⚠️ Primary Threats</h3>
                  <div className="threat-tags">
                    {demo.unifiedRisk.primary_threats.map((threat, i) => (
                      <span key={i} className="threat-tag">
                        {getThreatCategoryLabel(threat)}
                      </span>
                    ))}
                  </div>
                </div>
              )}
              
              <div className="recommendations">
                <h3>📋 Recommended Actions</h3>
                <ul>
                  {demo.unifiedRisk.recommended_actions.slice(0, 3).map((action, i) => (
                    <li key={i}>{action}</li>
                  ))}
                </ul>
              </div>
            </div>
          ) : (
            <div className="panel-placeholder">
              <p>Click "Run Demo Analysis" to see unified risk scoring in action</p>
            </div>
          )}
        </article>

        {/* Login Anomaly Panel */}
        <article className="dashboard-panel">
          <header>
            <h2>🔐 Login Anomaly Test</h2>
            {demo.loginRisk && (
              <span 
                className="badge"
                style={{ backgroundColor: getRiskLevelColor(demo.loginRisk.risk_level) }}
              >
                {demo.loginRisk.risk_level}
              </span>
            )}
          </header>
          
          <div className="form-group">
            <label>Hour of Day (0-23)</label>
            <input
              type="number"
              min="0"
              max="23"
              value={loginForm.hour_of_day}
              onChange={(e) => setLoginForm(prev => ({ ...prev, hour_of_day: parseInt(e.target.value) || 0 }))}
            />
          </div>
          
          <div className="form-group">
            <label>Failed Attempts (last 10 min)</label>
            <input
              type="number"
              min="0"
              value={loginForm.failed_10min}
              onChange={(e) => setLoginForm(prev => ({ ...prev, failed_10min: parseInt(e.target.value) || 0 }))}
            />
          </div>
          
          <div className="form-group">
            <label>
              <input
                type="checkbox"
                checked={loginForm.is_new_comp === 1}
                onChange={(e) => setLoginForm(prev => ({ ...prev, is_new_comp: e.target.checked ? 1 : 0 }))}
              />
              {" "}New Device
            </label>
          </div>
          
          <button 
            className="test-btn"
            onClick={testLoginAnomaly}
            disabled={demo.loading}
          >
            Analyze Login
          </button>
          
          {demo.loginRisk && (
            <div className="result-mini">
              <div className="result-stat">
                <span>Anomaly Score</span>
                <strong style={{ color: getRiskLevelColor(demo.loginRisk.risk_level) }}>
                  {formatProbability(demo.loginRisk.anomaly_probability)}
                </strong>
              </div>
              <div className="result-stat">
                <span>Confidence</span>
                <strong>{formatProbability(demo.loginRisk.confidence)}</strong>
              </div>
            </div>
          )}
        </article>

        {/* Password Risk Panel */}
        <article className="dashboard-panel">
          <header>
            <h2>🔑 Password Strength</h2>
            {demo.passwordRisk && (
              <span 
                className="badge"
                style={{ backgroundColor: getRiskLevelColor(demo.passwordRisk.risk_level) }}
              >
                {demo.passwordRisk.risk_level}
              </span>
            )}
          </header>
          
          <div className="form-group">
            <label>Test Password</label>
            <input
              type="text"
              placeholder="Enter password to test..."
              value={passwordInput}
              onChange={(e) => setPasswordInput(e.target.value)}
            />
          </div>
          
          <button 
            className="test-btn"
            onClick={testPassword}
            disabled={demo.loading || !passwordInput.trim()}
          >
            Check Password
          </button>
          
          {demo.passwordRisk && (
            <div className="result-mini">
              <div className="result-stat">
                <span>Breach Risk</span>
                <strong style={{ color: getRiskLevelColor(demo.passwordRisk.risk_level) }}>
                  {formatProbability(demo.passwordRisk.breach_probability)}
                </strong>
              </div>
              <div className="result-stat">
                <span>Strength</span>
                <strong>{formatProbability(demo.passwordRisk.strength_score)}</strong>
              </div>
              <div className="result-stat">
                <span>Entropy</span>
                <strong>{demo.passwordRisk.entropy_bits.toFixed(1)} bits</strong>
              </div>
              {demo.passwordRisk.recommendations.length > 0 && (
                <div className="recommendations-mini">
                  <strong>Tips:</strong>
                  <ul>
                    {demo.passwordRisk.recommendations.slice(0, 2).map((rec, i) => (
                      <li key={i}>{rec}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </article>

        {/* Fraud Detection Panel */}
        <article className="dashboard-panel">
          <header>
            <h2>💳 Fraud Detection Test</h2>
            {demo.fraudRisk && (
              <span 
                className="badge"
                style={{ 
                  backgroundColor: demo.fraudRisk.fraud_probability > 0.5 
                    ? getRiskLevelColor("high") 
                    : demo.fraudRisk.fraud_probability > 0.3
                    ? getRiskLevelColor("medium")
                    : getRiskLevelColor("low") 
                }}
              >
                {demo.fraudRisk.fraud_probability > 0.5 ? "High Risk" : 
                 demo.fraudRisk.fraud_probability > 0.3 ? "Medium Risk" : "Low Risk"}
              </span>
            )}
          </header>
          
          <div className="form-group">
            <label>Transaction Amount ($)</label>
            <input
              type="text"
              placeholder="1000"
              value={fraudForm.amount}
              onChange={(e) => setFraudForm(prev => ({ ...prev, amount: e.target.value }))}
            />
          </div>
          
          <div className="form-group">
            <label>Hour of Day (0-23)</label>
            <input
              type="number"
              min="0"
              max="23"
              value={fraudForm.hour}
              onChange={(e) => setFraudForm(prev => ({ ...prev, hour: e.target.value }))}
            />
          </div>
          
          <div className="form-group">
            <label>Transactions Last Hour</label>
            <input
              type="number"
              min="0"
              value={fraudForm.tx_count_1h}
              onChange={(e) => setFraudForm(prev => ({ ...prev, tx_count_1h: e.target.value }))}
            />
          </div>
          
          <div className="form-group">
            <label>
              <input
                type="checkbox"
                checked={fraudForm.is_international}
                onChange={(e) => setFraudForm(prev => ({ ...prev, is_international: e.target.checked }))}
              />
              {" "}International Transaction
            </label>
          </div>
          
          <button 
            className="test-btn"
            onClick={testFraud}
            disabled={demo.loading}
          >
            Analyze Transaction
          </button>
          
          {demo.fraudRisk && (
            <div className="result-mini">
              <div className="result-stat">
                <span>Fraud Probability</span>
                <strong style={{ 
                  color: demo.fraudRisk.fraud_probability > 0.5 
                    ? getRiskLevelColor("high") 
                    : demo.fraudRisk.fraud_probability > 0.3
                    ? getRiskLevelColor("medium")
                    : getRiskLevelColor("low") 
                }}>
                  {formatProbability(demo.fraudRisk.fraud_probability)}
                </strong>
              </div>
              {demo.fraudRisk.method && (
                <div className="result-stat">
                  <span>Method</span>
                  <strong style={{ fontSize: "0.85rem" }}>
                    {demo.fraudRisk.method === "ml_model" ? "ML Model" : "Rule-Based"}
                  </strong>
                </div>
              )}
            </div>
          )}
        </article>

        {/* API Info Panel */}
        <article className="dashboard-panel">
          <header>
            <h2>📡 API Endpoints</h2>
            <span className="badge badge-info">REST</span>
          </header>
          
          <div className="endpoint-list">
            <div className="endpoint">
              <code>POST /risk/unified</code>
              <span>Combined risk score</span>
            </div>
            <div className="endpoint">
              <code>POST /gps/score</code>
              <span>GPS spoofing detection</span>
            </div>
            <div className="endpoint">
              <code>POST /login/score</code>
              <span>Login anomaly detection</span>
            </div>
            <div className="endpoint">
              <code>POST /password/score</code>
              <span>Password risk assessment</span>
            </div>
            <div className="endpoint">
              <code>POST /fraud/score</code>
              <span>Transaction fraud detection</span>
            </div>
          </div>
          
          <a 
            href="http://localhost:8000/docs" 
            target="_blank" 
            rel="noopener noreferrer"
            className="docs-link"
          >
            View Full API Docs →
          </a>
        </article>
      </section>
    </main>
  );
}

// Risk Bar Component
function RiskBar({ label, value }: { label: string; value: number }) {
  const percentage = Math.round(value * 100);
  const color = value >= 0.7 ? '#ef4444' : value >= 0.4 ? '#f97316' : '#22c55e';
  
  return (
    <div className="risk-bar">
      <div className="risk-bar-header">
        <span>{label}</span>
        <span>{percentage}%</span>
      </div>
      <div className="risk-bar-track">
        <div 
          className="risk-bar-fill" 
          style={{ width: `${percentage}%`, backgroundColor: color }}
        />
      </div>
    </div>
  );
}
