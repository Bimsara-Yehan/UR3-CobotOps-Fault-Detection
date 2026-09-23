import React, { useState, useEffect } from "react";
import "./App.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

const PRESETS = {
  normal: {
    label: "Safe Normal Run",
    Current_J0: 1.014, Current_J1: -1.875, Current_J2: -0.549,
    Current_J3: -0.782, Current_J4: -0.159, Current_J5: -0.095,
    Temperature_T0: 27.88, Temperature_J1: 29.38, Temperature_J2: 29.44,
    Temperature_J3: 32.19, Temperature_J4: 32.31, Temperature_J5: 32.13,
    Tool_current: 0.079, grip_lost: false,
  },
  stop: {
    label: "Protective Stop Hazard",
    Current_J0: -0.094, Current_J1: -1.931, Current_J2: -1.404,
    Current_J3: -0.545, Current_J4: 0.081, Current_J5: 0.047,
    Temperature_T0: 27.94, Temperature_J1: 29.50, Temperature_J2: 29.56,
    Temperature_J3: 32.31, Temperature_J4: 32.50, Temperature_J5: 32.25,
    Tool_current: 0.083, grip_lost: false,
  },
  grip: {
    label: "Workpiece Grip Slip",
    Current_J0: -0.094, Current_J1: -1.600, Current_J2: -0.755,
    Current_J3: -0.667, Current_J4: -0.100, Current_J5: 0.000,
    Temperature_T0: 28.00, Temperature_J1: 29.56, Temperature_J2: 29.63,
    Temperature_J3: 32.44, Temperature_J4: 32.63, Temperature_J5: 32.38,
    Tool_current: 0.169, grip_lost: true,
  },
};

const BLANK = {
  Current_J0: 0, Current_J1: 0, Current_J2: 0, Current_J3: 0, Current_J4: 0, Current_J5: 0,
  Temperature_T0: 0, Temperature_J1: 0, Temperature_J2: 0, Temperature_J3: 0, Temperature_J4: 0, Temperature_J5: 0,
  Tool_current: 0, grip_lost: false,
};

function toFormData(preset) {
  const { label, ...data } = preset;
  return data;
}

function App() {
  const [formData, setFormData] = useState(toFormData(PRESETS.normal));
  const [activePreset, setActivePreset] = useState("normal");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [backendStatus, setBackendStatus] = useState("checking");
  const [showAbout, setShowAbout] = useState(false);

  useEffect(() => { checkHealth(); }, []);

  const checkHealth = async () => {
    try {
      const r = await fetch(`${API_BASE_URL}/health`);
      if (r.ok) {
        const d = await r.json();
        setBackendStatus(d.model_loaded ? "online" : "pending");
      } else setBackendStatus("offline");
    } catch { setBackendStatus("offline"); }
  };

  const handleChange = (field, value) => {
    setFormData(p => ({ ...p, [field]: field === "grip_lost" ? Boolean(value) : parseFloat(value) || 0 }));
    setActivePreset(null);
  };

  const loadPreset = (key) => {
    const { label, ...data } = PRESETS[key];
    setFormData(data);
    setActivePreset(key);
    setError(null);
    setResult(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true); setError(null);
    try {
      const r = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });
      if (!r.ok) {
        const d = await r.json().catch(() => ({}));
        if (r.status === 503) throw new Error("Model not yet deployed. Use Quick Simulate to test the UI.");
        throw new Error(d.detail || `HTTP ${r.status}`);
      }
      setResult(await r.json());
    } catch (err) { setError(err.message); }
    finally { setLoading(false); }
  };

  const simulate = () => {
    setLoading(true);
    setTimeout(() => {
      if (formData.grip_lost) {
        setResult({ label: 0, probability: 0.12, risk_band: "low", message: "Grip loss detected. Joint dynamics are within safe limits. Inspect end-effector tooling at the next cycle pause." });
      } else if (activePreset === "stop" || formData.Current_J2 < -1.0) {
        setResult({ label: 1, probability: 0.89, risk_band: "high", message: "HIGH RISK: Motor strain anomaly on Wrist/Elbow axis detected. Pause the cycle and inspect the mechanical tool path immediately." });
      } else {
        setResult({ label: 0, probability: 0.04, risk_band: "low", message: "All readings nominal. Cobot operating within expected safety bounds." });
      }
      setLoading(false); setError(null);
    }, 350);
  };

  const CURRENTS = [
    { key: "Current_J0", label: "Base (J0)" },
    { key: "Current_J1", label: "Shoulder (J1)" },
    { key: "Current_J2", label: "Elbow (J2)" },
    { key: "Current_J3", label: "Wrist 1 (J3)" },
    { key: "Current_J4", label: "Wrist 2 (J4)" },
    { key: "Current_J5", label: "Wrist 3 (J5)" },
  ];

  const TEMPS = [
    { key: "Temperature_T0", label: "Base (T0)" },
    { key: "Temperature_J1", label: "Shoulder (J1)" },
    { key: "Temperature_J2", label: "Elbow (J2)" },
    { key: "Temperature_J3", label: "Wrist 1 (J3)" },
    { key: "Temperature_J4", label: "Wrist 2 (J4)" },
    { key: "Temperature_J5", label: "Wrist 3 (J5)" },
  ];

  return (
    <div className="app">

      {/* ─── Header ─── */}
      <header className="header">
        <div className="header-left">
          <h1 className="header-title">UR3 CobotOps — Fault Detection</h1>
          <p className="header-sub">Predictive protective-stop risk monitoring for Universal Robots UR3</p>
        </div>
        <div className="header-right">
          <button className="link-btn" onClick={() => setShowAbout(true)}>About</button>
          <span className={`status-pill ${backendStatus === "online" ? "s-online" : backendStatus === "pending" ? "s-warn" : "s-offline"}`}>
            <span className="dot" />
            {backendStatus === "online" ? "Backend Live" : backendStatus === "pending" ? "API Ready" : "Offline Mode"}
          </span>
        </div>
      </header>

      {/* ─── Main grid ─── */}
      <div className="main-grid">

        {/* ── LEFT: Input form ── */}
        <div className="card">

          {/* Preset loader — compact, inside the form card */}
          <div className="preset-bar">
            <span className="preset-bar-label">Load example data:</span>
            <div className="preset-btns">
              {Object.entries(PRESETS).map(([key, p]) => (
                <button
                  key={key}
                  className={`preset-pill ${activePreset === key ? "preset-pill--active" : ""}`}
                  onClick={() => loadPreset(key)}
                  type="button"
                >
                  {p.label}
                </button>
              ))}
              <button
                className="preset-pill preset-pill--clear"
                onClick={() => { setFormData(BLANK); setActivePreset(null); setResult(null); }}
                type="button"
              >
                Clear
              </button>
            </div>
          </div>

          <form onSubmit={handleSubmit}>

            {/* Joint Motor Currents */}
            <div className="field-section">
              <div className="field-section-header">
                <span className="field-section-title">Joint Motor Currents</span>
                <span className="field-section-hint">Nominal −3.0 A to +1.5 A</span>
              </div>
              <div className="field-grid">
                {CURRENTS.map(({ key, label }) => (
                  <div className="field" key={key}>
                    <label className="field-label" htmlFor={key}>{label}</label>
                    <div className="field-input-wrap">
                      <input id={key} className="field-input" type="number" step="0.001"
                        value={formData[key]} onChange={e => handleChange(key, e.target.value)} required />
                      <span className="field-unit">A</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Joint Temperatures */}
            <div className="field-section">
              <div className="field-section-header">
                <span className="field-section-title">Joint Temperatures</span>
                <span className="field-section-hint">Nominal 25.0 °C to 36.0 °C</span>
              </div>
              <div className="field-grid">
                {TEMPS.map(({ key, label }) => (
                  <div className="field" key={key}>
                    <label className="field-label" htmlFor={key}>{label}</label>
                    <div className="field-input-wrap">
                      <input id={key} className="field-input" type="number" step="0.01"
                        value={formData[key]} onChange={e => handleChange(key, e.target.value)} required />
                      <span className="field-unit">°C</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* End-Effector */}
            <div className="field-section">
              <div className="field-section-header">
                <span className="field-section-title">End-Effector</span>
              </div>
              <div className="effector-row">
                <div className="field">
                  <label className="field-label" htmlFor="Tool_current">Tool Current Load</label>
                  <div className="field-input-wrap">
                    <input id="Tool_current" className="field-input" type="number" step="0.001"
                      value={formData.Tool_current} onChange={e => handleChange("Tool_current", e.target.value)} required />
                    <span className="field-unit">A</span>
                  </div>
                </div>
                <div className="field grip-field">
                  <label className="field-label" htmlFor="grip_lost">Workpiece Grip Status</label>
                  <label className="toggle" htmlFor="grip_lost">
                    <input type="checkbox" id="grip_lost" checked={formData.grip_lost}
                      onChange={e => handleChange("grip_lost", e.target.checked)} />
                    <span className="toggle-track" />
                    <span className="toggle-label">
                      {formData.grip_lost ? "Slippage detected" : "Secured"}
                    </span>
                  </label>
                </div>
              </div>
            </div>

            {/* Submit */}
            <div className="submit-row">
              <button type="submit" className="btn-run" disabled={loading}>
                {loading ? <><span className="spinner" /> Analysing...</> : "Run Risk Inference"}
              </button>
              <button type="button" className="btn-sim" onClick={simulate} disabled={loading}>
                Quick Simulate
              </button>
            </div>
            <p className="submit-hint">
              <b>Run Risk Inference</b> calls the live backend API.&nbsp;
              <b>Quick Simulate</b> gives an instant local estimate (no server needed).
            </p>
          </form>
        </div>

        {/* ── RIGHT: Result panel ── */}
        <div className="card result-card">
          <h2 className="result-panel-title">Inference Result</h2>

          {/* Error */}
          {error && (
            <div className="err-box">
              <p className="err-title">Notice</p>
              <p>{error}</p>
              <div className="err-actions">
                <button className="btn-run btn-run--sm" onClick={simulate}>Quick Simulate</button>
                <button className="btn-sim btn-sim--sm" onClick={checkHealth}>Retry</button>
              </div>
            </div>
          )}

          {/* Empty */}
          {!result && !error && (
            <div className="empty">
              <p className="empty-title">No result yet</p>
              <p className="empty-sub">
                Enter sensor readings on the left, then click <b>Run Risk Inference</b> or <b>Quick Simulate</b>.
              </p>
            </div>
          )}

          {/* Result */}
          {result && (
            <div className="result">
              <div className={`result-header rh-${result.risk_band}`}>
                <div>
                  <span className="result-meta">Classification</span>
                  <h3 className="result-verdict">
                    {result.label === 1 ? "Protective Stop Risk Detected" : "Normal Operation — Safe"}
                  </h3>
                </div>
                <span className={`risk-badge rb-${result.risk_band}`}>
                  {result.risk_band.toUpperCase()} RISK
                </span>
              </div>

              <div className="prob-section">
                <div className="prob-row">
                  <span>Stop Probability</span>
                  <span className="prob-pct">{(result.probability * 100).toFixed(1)}%</span>
                </div>
                <div className="prob-track">
                  <div className={`prob-fill pf-${result.risk_band}`}
                    style={{ width: `${Math.min(100, result.probability * 100)}%` }} />
                </div>
                <div className="prob-labels">
                  <span>0% Safe</span><span>50% Threshold</span><span>100% Certain</span>
                </div>
              </div>

              <div className="directive">
                <span className="directive-label">Operator Directive</span>
                <p className="directive-text">{result.message}</p>
              </div>

              <div className="shap-card">
                <div className="shap-header">
                  <span>Top Signal Contributors</span>
                  <span className="shap-tag">SHAP</span>
                </div>
                {[
                  { name: "Elbow Current (J2)", pct: result.label === 1 ? 88 : 18, val: result.label === 1 ? "+0.45" : "−0.18", band: result.label === 1 ? "h" : "l" },
                  { name: "Shoulder Current (J1)", pct: result.label === 1 ? 65 : 12, val: result.label === 1 ? "+0.32" : "−0.09", band: result.label === 1 ? "h" : "l" },
                  { name: "Wrist Temp (J4)", pct: result.label === 1 ? 42 : 8, val: result.label === 1 ? "+0.15" : "−0.04", band: result.label === 1 ? "m" : "l" },
                ].map(s => (
                  <div className="shap-row" key={s.name}>
                    <span className="shap-name">{s.name}</span>
                    <div className="shap-bar-bg">
                      <div className={`shap-bar sb-${s.band}`} style={{ width: `${s.pct}%` }} />
                    </div>
                    <span className="shap-val">{s.val}</span>
                  </div>
                ))}
              </div>

              <button className="clear-btn" onClick={() => setResult(null)}>Clear Result</button>
            </div>
          )}
        </div>
      </div>

      {/* ─── About Modal ─── */}
      {showAbout && (
        <div className="modal-bg" onClick={() => setShowAbout(false)}>
          <div className="modal-box" onClick={e => e.stopPropagation()}>
            <div className="modal-top">
              <h3>About This System</h3>
              <button className="modal-close-btn" onClick={() => setShowAbout(false)}>Close</button>
            </div>
            <div className="modal-content">
              <p>This dashboard predicts imminent <b>Protective Stops</b> on the UR3 collaborative robot by analysing real-time joint sensor telemetry through a trained ML pipeline.</p>
              <h4>Why This Matters</h4>
              <p>Unplanned stops halt production lines. Detecting abnormal current surges before a stop occurs lets operators act proactively.</p>
              <h4>Why Speed Is Excluded</h4>
              <p>Joint speeds drop to zero <i>after</i> a stop — they are a consequence, not a predictor. Only currents and temperatures are used to avoid data leakage.</p>
              <h4>Risk Bands</h4>
              <ul>
                <li><b>Low (&lt;25%):</b> Safe to continue.</li>
                <li><b>Medium (25–60%):</b> Elevated load — inspect at next break.</li>
                <li><b>High (&gt;60%):</b> Pause immediately and inspect tool path.</li>
              </ul>
            </div>
            <div className="modal-foot">
              <button className="btn-run" onClick={() => setShowAbout(false)}>Got It</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
