import React, { useState, useEffect } from "react";
import "./App.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

const JOINTS = [
  { key: "Current_J0", label: "Base (J0)" },
  { key: "Current_J1", label: "Shoulder (J1)" },
  { key: "Current_J2", label: "Elbow (J2)" },
  { key: "Current_J3", label: "Wrist 1 (J3)" },
  { key: "Current_J4", label: "Wrist 2 (J4)" },
  { key: "Current_J5", label: "Wrist 3 (J5)" },
];

// Smallest and largest value of each input in the training data (amps). Used only to warn, never to block.
const TRAIN_RANGE = {
  Current_J0: [-6.25, 6.47],
  Current_J1: [-5.81, 1.08],
  Current_J2: [-4.07, 2.30],
  Current_J3: [-2.84, 2.20],
  Current_J4: [-4.04, 4.09],
  Current_J5: [-0.48, 0.40],
  Tool_current: [0.07, 0.60],
};

// Readable names for what the API reports as top factors.
const FACTOR_NAMES = {
  ...Object.fromEntries(JOINTS.map(({ key, label }) => [key, `${label} current`])),
  Tool_current: "Tool current",
  total_joint_current: "Total joint current",
  total_joint_current_delta3: "Total-current change (3 readings)",
};

// Real readings from the held-out test split (rounded to 3 decimals): each has the reading to score and the
// joint currents from 3 readings earlier in the same cycle.
const SAMPLES = {
  normal: {
    label: "Normal reading",
    current: { Current_J0: 0.084, Current_J1: -1.692, Current_J2: -0.724, Current_J3: -0.694, Current_J4: -0.031, Current_J5: 0.068, Tool_current: 0.082 },
    earlier: { Current_J0: 0.129, Current_J1: -1.975, Current_J2: -0.999, Current_J3: -0.574, Current_J4: -0.004, Current_J5: -0.01 },
  },
  borderline: {
    label: "Borderline reading",
    current: { Current_J0: 0.19, Current_J1: -2.885, Current_J2: -2.074, Current_J3: -1.685, Current_J4: 0.864, Current_J5: 0.093, Tool_current: 0.083 },
    earlier: { Current_J0: 0.139, Current_J1: -3.548, Current_J2: -2.173, Current_J3: -0.742, Current_J4: 0.265, Current_J5: -0.026 },
  },
  stop: {
    label: "Protective stop",
    current: { Current_J0: -0.122, Current_J1: -3.712, Current_J2: -2.185, Current_J3: -0.851, Current_J4: 0.747, Current_J5: -0.033, Tool_current: 0.079 },
    earlier: { Current_J0: 0.162, Current_J1: -2.162, Current_J2: -1.3, Current_J3: -0.793, Current_J4: -0.036, Current_J5: -0.139 },
  },
};

const VERDICTS = {
  high: "Protective-stop pattern detected",
  medium: "Borderline: keep watching",
  low: "Normal operation",
};

// Form values are kept as text so an empty box stays empty instead of turning into 0.
const toText = (values) => Object.fromEntries(Object.entries(values).map(([k, v]) => [k, String(v)]));
const toForm = (sample) => ({ current: toText(sample.current), earlier: toText(sample.earlier) });
const BLANK = {
  current: Object.fromEntries([...JOINTS.map((j) => j.key), "Tool_current"].map((k) => [k, ""])),
  earlier: Object.fromEntries(JOINTS.map((j) => [j.key, ""])),
};

const toNumbers = (values) => Object.fromEntries(Object.entries(values).map(([k, v]) => [k, Number(v)]));

const pct = (x) => `${Math.round(x * 100)}%`;

function describeError(status, detail) {
  if (status === 503) return "The server has no model loaded. Start the backend with models/final_pipeline.joblib in place.";
  if (Array.isArray(detail) && detail.length) {
    return detail.map((d) => `${d.loc.slice(1).join(" / ")}: ${d.msg}`).join("; ");
  }
  return typeof detail === "string" ? detail : `The server returned an error (HTTP ${status}).`;
}

function App() {
  const [form, setForm] = useState(toForm(SAMPLES.normal));
  const [activeSample, setActiveSample] = useState("normal");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [backendStatus, setBackendStatus] = useState("checking");
  const [thresholds, setThresholds] = useState(null);
  const [showAbout, setShowAbout] = useState(false);

  useEffect(() => { checkHealth(); }, []);

  const checkHealth = async () => {
    try {
      const r = await fetch(`${API_BASE_URL}/health`);
      if (!r.ok) { setBackendStatus("offline"); return; }
      const d = await r.json();
      setBackendStatus(d.model_loaded ? "online" : "pending");
      if (d.model_loaded) {
        const info = await fetch(`${API_BASE_URL}/model-info`);
        if (info.ok) setThresholds((await info.json()).thresholds);
      }
    } catch { setBackendStatus("offline"); }
  };

  const handleChange = (section, field, value) => {
    setForm((f) => ({ ...f, [section]: { ...f[section], [field]: value } }));
    setActiveSample(null);
  };

  const loadSample = (key) => {
    setForm(toForm(SAMPLES[key]));
    setActiveSample(key);
    setError(null);
    setResult(null);
  };

  const outOfRange = Object.entries(form.current).concat(Object.entries(form.earlier).map(([k, v]) => [`earlier ${k}`, v]))
    .filter(([k, v]) => {
      const range = TRAIN_RANGE[k.replace("earlier ", "")];
      return v !== "" && range && (Number(v) < range[0] || Number(v) > range[1]);
    })
    .map(([k]) => k);

  const isOut = (section, key) => outOfRange.includes(section === "earlier" ? `earlier ${key}` : key);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true); setError(null);
    try {
      const r = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ current: toNumbers(form.current), earlier: toNumbers(form.earlier) }),
      });
      if (!r.ok) {
        const d = await r.json().catch(() => ({}));
        throw new Error(describeError(r.status, d.detail));
      }
      setResult(await r.json());
      if (backendStatus !== "online") checkHealth();
    } catch (err) {
      setResult(null);
      setError(err instanceof TypeError ? "Cannot reach the backend. Start it with: uvicorn backend.main:app" : err.message);
    } finally { setLoading(false); }
  };

  const field = (section, key, label, step = "0.001") => (
    <div className="field" key={`${section}-${key}`}>
      <label className="field-label" htmlFor={`${section}-${key}`}>{label}</label>
      <div className="field-input-wrap">
        <input id={`${section}-${key}`} className={`field-input ${isOut(section, key) ? "field-input--out" : ""}`}
          type="number" step={step} value={form[section][key]} required
          onChange={(e) => handleChange(section, key, e.target.value)} />
        <span className="field-unit">A</span>
      </div>
      {TRAIN_RANGE[key] && <span className="field-range">seen: {TRAIN_RANGE[key][0]} to {TRAIN_RANGE[key][1]}</span>}
    </div>
  );

  const maxContribution = result ? Math.max(...result.top_factors.map((f) => Math.abs(f.contribution)), 1e-9) : 1;

  return (
    <div className="app">

      {/* ─── Header ─── */}
      <header className="header">
        <div className="header-left">
          <h1 className="header-title">UR3 CobotOps — Fault Detection</h1>
          <p className="header-sub">Protective-stop risk from the robot's motor currents, for the Universal Robots UR3</p>
        </div>
        <div className="header-right">
          <button className="link-btn" onClick={() => setShowAbout(true)}>About</button>
          <span className={`status-pill ${backendStatus === "online" ? "s-online" : backendStatus === "pending" ? "s-warn" : "s-offline"}`}>
            <span className="dot" />
            {backendStatus === "online" ? "Backend Live" : backendStatus === "pending" ? "Model Not Loaded" : backendStatus === "checking" ? "Connecting…" : "Backend Offline"}
          </span>
        </div>
      </header>

      {/* ─── Main grid ─── */}
      <div className="main-grid">

        {/* ── LEFT: Input form ── */}
        <div className="card">

          <div className="preset-bar">
            <span className="preset-bar-label">Load a real example:</span>
            <div className="preset-btns">
              {Object.entries(SAMPLES).map(([key, s]) => (
                <button key={key} type="button"
                  className={`preset-pill ${activeSample === key ? "preset-pill--active" : ""}`}
                  onClick={() => loadSample(key)}>
                  {s.label}
                </button>
              ))}
              <button type="button" className="preset-pill preset-pill--clear"
                onClick={() => { setForm(BLANK); setActiveSample(null); setResult(null); setError(null); }}>
                Clear
              </button>
            </div>
          </div>

          <form onSubmit={handleSubmit}>

            <div className="field-section">
              <div className="field-section-header">
                <span className="field-section-title">Latest reading: joint motor currents</span>
                <span className="field-section-hint">amps</span>
              </div>
              <div className="field-grid">
                {JOINTS.map(({ key, label }) => field("current", key, label))}
              </div>
            </div>

            <div className="field-section">
              <div className="field-section-header">
                <span className="field-section-title">Latest reading: tool</span>
              </div>
              <div className="field-grid">
                {field("current", "Tool_current", "Tool current load")}
              </div>
            </div>

            <div className="field-section">
              <div className="field-section-header">
                <span className="field-section-title">3 seconds earlier: joint motor currents</span>
                <span className="field-section-hint">same working cycle</span>
              </div>
              <div className="field-grid">
                {JOINTS.map(({ key, label }) => field("earlier", key, label))}
              </div>
              <p className="field-note">
                The model looks at how the total joint current has changed over the last 3 readings, so it needs the joint currents
                from 3 readings (about 3 seconds) before the latest one, taken in the same working cycle.
              </p>
            </div>

            {outOfRange.length > 0 && (
              <p className="range-note">
                {outOfRange.length === 1 ? "One value is" : `${outOfRange.length} values are`} outside the range seen when the model
                was trained (outlined in amber). The prediction may be unreliable.
              </p>
            )}

            <div className="submit-row">
              <button type="submit" className="btn-run" disabled={loading}>
                {loading ? <><span className="spinner" /> Analysing...</> : "Run Risk Inference"}
              </button>
            </div>
            <p className="submit-hint">
              Examples are real readings from the held-out test data. Editing any value switches to your own readings.
            </p>
          </form>
        </div>

        {/* ── RIGHT: Result panel ── */}
        <div className="card result-card">
          <h2 className="result-panel-title">Inference Result</h2>

          {error && (
            <div className="err-box">
              <p className="err-title">Could not get a prediction</p>
              <p>{error}</p>
              <div className="err-actions">
                <button className="btn-sim btn-sim--sm" type="button" onClick={checkHealth}>Check connection</button>
              </div>
            </div>
          )}

          {!result && !error && (
            <div className="empty">
              <p className="empty-title">No result yet</p>
              <p className="empty-sub">
                Load an example or enter readings on the left, then click <b>Run Risk Inference</b>.
              </p>
            </div>
          )}

          {result && (
            <div className="result">
              <div className={`result-header rh-${result.risk_band}`}>
                <div>
                  <span className="result-meta">Assessment</span>
                  <h3 className="result-verdict">{VERDICTS[result.risk_band]}</h3>
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
                <div className="prob-bar">
                  <div className="prob-track">
                    <div className={`prob-fill pf-${result.risk_band}`}
                      style={{ width: `${Math.min(100, result.probability * 100)}%` }} />
                  </div>
                  <span className="prob-tick" style={{ left: `${result.thresholds.watch * 100}%` }} />
                  <span className="prob-tick" style={{ left: `${result.thresholds.alert * 100}%` }} />
                </div>
                <div className="prob-scale">
                  <span style={{ left: `${result.thresholds.watch * 100}%` }}>Watch {pct(result.thresholds.watch)}</span>
                  <span style={{ left: `${result.thresholds.alert * 100}%` }}>Alert {pct(result.thresholds.alert)}</span>
                </div>
              </div>

              <div className="directive">
                <span className="directive-label">What this means</span>
                <p className="directive-text">{result.message}</p>
              </div>

              <div className="shap-card">
                <div className="shap-header">
                  <span>What drove this result</span>
                  <span className="shap-tag" title="SHAP values from the trained model, for this reading">SHAP</span>
                </div>
                <p className="shap-note">
                  The three inputs that moved this prediction most. Red pushed it toward a stop, green pushed it away.
                </p>
                {result.top_factors.map((f) => (
                  <div className="shap-row" key={f.feature}>
                    <span className="shap-name">
                      {FACTOR_NAMES[f.feature] ?? f.feature}
                      <span className="shap-input">= {f.value.toFixed(3)} A</span>
                    </span>
                    <div className="shap-bar-bg">
                      <div className={`shap-bar ${f.effect === "raises risk" ? "sb-h" : "sb-l"}`}
                        style={{ width: `${(Math.abs(f.contribution) / maxContribution) * 100}%` }} />
                    </div>
                    <span className="shap-val" title="SHAP value in log-odds of a protective stop">
                      {f.contribution > 0 ? "+" : "−"}{Math.abs(f.contribution).toFixed(2)}
                    </span>
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
          <div className="modal-box" onClick={(e) => e.stopPropagation()}>
            <div className="modal-top">
              <h3>About This System</h3>
              <button className="modal-close-btn" onClick={() => setShowAbout(false)}>Close</button>
            </div>
            <div className="modal-content">
              <p>This tool estimates the chance that a UR3 collaborative robot is in, or entering, a <b>protective stop</b>, from its motor currents. It was trained on one robot's readings from a single day.</p>
              <h4>What it can and cannot do</h4>
              <p>It recognises the current pattern that goes with protective stops, mostly as a stop begins or is under way. It gives little advance warning, and it does not replace the robot's own safety system.</p>
              <h4>Why speed and temperature are not used</h4>
              <p>Joint speeds drop to zero <i>after</i> a stop, so they only confirm a stop that has already happened. Temperatures were tested and left out: in this data they mostly track the time of day and added nothing on later cycles.</p>
              <h4>Why an earlier reading is needed</h4>
              <p>One input is how much the total joint current changed over the last 3 readings. Leaving it out lowers accuracy, and guessing it would mislead the model.</p>
              <h4>Risk bands</h4>
              <ul>
                <li><b>Low{thresholds ? ` (under ${pct(thresholds.watch)})` : ""}:</b> readings look like normal operation.</li>
                <li><b>Medium{thresholds ? ` (${pct(thresholds.watch)} to ${pct(thresholds.alert)})` : ""}:</b> drifting toward the stop pattern. Keep watching.</li>
                <li><b>High{thresholds ? ` (${pct(thresholds.alert)} and above)` : ""}:</b> matches the stop pattern. Check the robot and work cell.</li>
              </ul>
              <p>In testing on later cycles, the alert level flagged about 84% of stop episodes, and raised roughly 27 false-alarm runs per hour of readings.</p>
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
