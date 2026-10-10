import React, { useState, useEffect, useRef } from "react";
import CobotScene, { MODES } from "./components/CobotScene.jsx";
import RiskGauge from "./components/RiskGauge.jsx";
import "./App.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

const JOINTS = [
  { key: "Current_J0", id: "J0", label: "Base" },
  { key: "Current_J1", id: "J1", label: "Shoulder" },
  { key: "Current_J2", id: "J2", label: "Elbow" },
  { key: "Current_J3", id: "J3", label: "Wrist 1" },
  { key: "Current_J4", id: "J4", label: "Wrist 2" },
  { key: "Current_J5", id: "J5", label: "Wrist 3" },
];

// Smallest and largest value of each input in the training split (amps, rounded outward to 3 decimals).
// Used only to warn, never to block.
const TRAIN_RANGE = {
  Current_J0: [-6.248, 6.461],
  Current_J1: [-5.809, 1.075],
  Current_J2: [-4.067, 2.293],
  Current_J3: [-2.832, 2.198],
  Current_J4: [-4.036, 4.090],
  Current_J5: [-0.475, 0.393],
  Tool_current: [0.070, 0.594],
};

// Readable names for what the API reports as top factors.
const FACTOR_NAMES = {
  ...Object.fromEntries(JOINTS.map(({ key, id, label }) => [key, `${label} (${id}) current`])),
  Tool_current: "Tool current",
  total_joint_current: "Total joint current",
  total_joint_current_delta3: "Total-current change (3 readings)",
};

// Real readings from the held-out test split (rounded to 3 decimals): each has the reading to score and the
// joint currents from 3 readings earlier in the same cycle. `row` is the dataset's Num column.
const SAMPLES = {
  normal: {
    label: "Normal operation",
    blurb: "Steady motion mid-cycle",
    recorded: "normal", row: 5943, cycle: 231,
    current: { Current_J0: 0.084, Current_J1: -1.692, Current_J2: -0.724, Current_J3: -0.694, Current_J4: -0.031, Current_J5: 0.068, Tool_current: 0.082 },
    earlier: { Current_J0: 0.129, Current_J1: -1.975, Current_J2: -0.999, Current_J3: -0.574, Current_J4: -0.004, Current_J5: -0.01 },
  },
  borderline: {
    label: "Borderline",
    blurb: "Load shifting, no stop recorded",
    recorded: "normal", row: 5553, cycle: 218,
    current: { Current_J0: 0.19, Current_J1: -2.885, Current_J2: -2.074, Current_J3: -1.685, Current_J4: 0.864, Current_J5: 0.093, Tool_current: 0.083 },
    earlier: { Current_J0: 0.139, Current_J1: -3.548, Current_J2: -2.173, Current_J3: -0.742, Current_J4: 0.265, Current_J5: -0.026 },
  },
  stop: {
    label: "Protective stop",
    blurb: "Recorded stop as it begins",
    recorded: "stop", row: 5523, cycle: 217,
    current: { Current_J0: -0.122, Current_J1: -3.712, Current_J2: -2.185, Current_J3: -0.851, Current_J4: 0.747, Current_J5: -0.033, Tool_current: 0.079 },
    earlier: { Current_J0: 0.162, Current_J1: -2.162, Current_J2: -1.3, Current_J3: -0.793, Current_J4: -0.036, Current_J5: -0.139 },
  },
};

const VERDICTS = {
  high: "Protective-stop pattern detected",
  medium: "Borderline — keep watching",
  low: "Normal operation",
};

const BAND_TO_MODE = { low: "run", medium: "watch", high: "stop" };

// Held-out results from notebook 07 (later cycles, evaluated once).
const MODEL_CARD = [
  { value: "84.3", unit: "%", label: "Stop episodes flagged", note: "at least once, 51 episodes" },
  { value: "0.648", unit: "", label: "Recall", note: "stop rows at the alert threshold" },
  { value: "0.519", unit: "", label: "Precision", note: "alerts that were real stops" },
  { value: "26.7", unit: "/h", label: "False-alarm runs", note: "per hour of readings" },
];

// Form values are kept as text so an empty box stays empty instead of turning into 0.
const toText = (values) => Object.fromEntries(Object.entries(values).map(([k, v]) => [k, String(v)]));
const toForm = (sample) => ({ current: toText(sample.current), earlier: toText(sample.earlier) });
const BLANK = {
  current: Object.fromEntries([...JOINTS.map((j) => j.key), "Tool_current"].map((k) => [k, ""])),
  earlier: Object.fromEntries(JOINTS.map((j) => [j.key, ""])),
};

const toNumbers = (values) => Object.fromEntries(Object.entries(values).map(([k, v]) => [k, Number(v)]));

const pct = (x) => `${Math.round(x * 100)}%`;
const isNum = (v) => v !== "" && Number.isFinite(Number(v));
const fmt = (x, d = 3) => (x === null ? "—" : `${x > 0 ? "+" : x < 0 ? "−" : ""}${Math.abs(x).toFixed(d)}`);
const clock = () => new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });

// Same definition as src/features.py: sum of absolute joint currents.
const totalCurrent = (values) =>
  JOINTS.every(({ key }) => isNum(values[key])) ? JOINTS.reduce((s, { key }) => s + Math.abs(Number(values[key])), 0) : null;

function describeError(status, detail) {
  if (status === 503) return "The server has no model loaded. Start the backend with models/final_pipeline.joblib in place.";
  if (Array.isArray(detail) && detail.length) {
    return detail.map((d) => `${d.loc.slice(1).join(" / ")}: ${d.msg}`).join("; ");
  }
  return typeof detail === "string" ? detail : `The server returned an error (HTTP ${status}).`;
}

function RangeMeter({ value, range, out }) {
  if (!range || !isNum(value)) return <span className="meter meter--empty" />;
  const [lo, hi] = range;
  const pos = Math.min(1, Math.max(0, (Number(value) - lo) / (hi - lo)));
  const zero = lo < 0 && hi > 0 ? (0 - lo) / (hi - lo) : null;
  return (
    <span className={`meter ${out ? "meter--out" : ""}`} title={`Training range ${lo} to ${hi} A`}>
      {zero !== null && <span className="meter-zero" style={{ left: `${zero * 100}%` }} />}
      <span className="meter-mark" style={{ left: `${pos * 100}%` }} />
    </span>
  );
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
  const [log, setLog] = useState([]);
  const [focus, setFocus] = useState(null);
  const resultRef = useRef(null);
  const stageRef = useRef(null);

  useEffect(() => { checkHealth(); }, []);

  // Panels glow under the pointer: one listener feeds the cursor position to whichever panel it is over.
  useEffect(() => {
    const onMove = (e) => {
      const el = e.target.closest?.(".spot");
      if (!el) return;
      const r = el.getBoundingClientRect();
      el.style.setProperty("--mx", `${e.clientX - r.left}px`);
      el.style.setProperty("--my", `${e.clientY - r.top}px`);
    };
    window.addEventListener("pointermove", onMove);
    return () => window.removeEventListener("pointermove", onMove);
  }, []);

  useEffect(() => {
    if (!showAbout) return undefined;
    const onKey = (e) => e.key === "Escape" && setShowAbout(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [showAbout]);

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

  const outOfRange = Object.entries(form.current).concat(Object.entries(form.earlier).map(([k, v]) => [`earlier ${k}`, v]))
    .filter(([k, v]) => {
      const range = TRAIN_RANGE[k.replace("earlier ", "")];
      return v !== "" && range && (Number(v) < range[0] || Number(v) > range[1]);
    })
    .map(([k]) => k);

  const isOut = (section, key) => outOfRange.includes(section === "earlier" ? `earlier ${key}` : key);

  const predict = async (values, sampleKey) => {
    setLoading(true); setError(null);
    try {
      const r = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ current: toNumbers(values.current), earlier: toNumbers(values.earlier) }),
      });
      if (!r.ok) {
        const d = await r.json().catch(() => ({}));
        throw new Error(describeError(r.status, d.detail));
      }
      const data = await r.json();
      const at = clock();
      setResult({ ...data, at, sampleKey });
      setLog((l) => [{ at, source: sampleKey ? SAMPLES[sampleKey].label : "Manual entry", ...data }, ...l].slice(0, 5));
      if (backendStatus !== "online") checkHealth();
    } catch (err) {
      setResult(null);
      setError(err instanceof TypeError ? "Cannot reach the backend. Start it with: uvicorn backend.main:app" : err.message);
    } finally { setLoading(false); }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    predict(form, activeSample);
  };

  // Scenario cards load a real reading and score it in one click.
  const runScenario = (key) => {
    const values = toForm(SAMPLES[key]);
    setForm(values);
    setActiveSample(key);
    predict(values, key);
    if (window.innerWidth < 1000) resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  // 1 / 2 / 3 run the scenarios, unless the user is typing in a field.
  useEffect(() => {
    const onKey = (e) => {
      if (e.metaKey || e.ctrlKey || e.altKey || showAbout || loading) return;
      if (["INPUT", "TEXTAREA"].includes(document.activeElement?.tagName)) return;
      const key = Object.keys(SAMPLES)[Number(e.key) - 1];
      if (key) runScenario(key);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  // The twin tilts a little toward the pointer.
  const tilt = (e) => {
    const el = stageRef.current;
    if (!el || window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) return;
    const r = el.getBoundingClientRect();
    const x = (e.clientX - r.left) / r.width - 0.5;
    const y = (e.clientY - r.top) / r.height - 0.5;
    el.style.setProperty("--rx", `${(-y * 5).toFixed(2)}deg`);
    el.style.setProperty("--ry", `${(x * 7).toFixed(2)}deg`);
  };
  const untilt = () => {
    stageRef.current?.style.setProperty("--rx", "0deg");
    stageRef.current?.style.setProperty("--ry", "0deg");
  };

  // Pointing at a row (or editing one of its boxes) marks that joint on the twin.
  const focusProps = (feature) => ({
    onMouseEnter: () => setFocus(feature),
    onMouseLeave: () => setFocus((f) => (f === feature ? null : f)),
    onFocus: () => setFocus(feature),
    onBlur: () => setFocus((f) => (f === feature ? null : f)),
  });

  const input = (section, key, label) => (
    <div className="cell-input">
      <div className="input-wrap">
        <input id={`${section}-${key}`} aria-label={label}
          className={`num-input ${isOut(section, key) ? "num-input--out" : ""}`}
          type="number" step="0.001" value={form[section][key]} required
          onChange={(e) => handleChange(section, key, e.target.value)} />
        <span className="unit">A</span>
      </div>
      <RangeMeter value={form[section][key]} range={TRAIN_RANGE[key]} out={isOut(section, key)} />
    </div>
  );

  const totalNow = totalCurrent(form.current);
  const totalEarlier = totalCurrent(form.earlier);
  const delta3 = totalNow !== null && totalEarlier !== null ? totalNow - totalEarlier : null;

  const mode = result ? BAND_TO_MODE[result.risk_band] : backendStatus === "offline" ? "offline" : "run";
  const maxContribution = result ? Math.max(...result.top_factors.map((f) => Math.abs(f.contribution)), 1e-9) : 1;
  const statusText = { online: "Model online", pending: "Model not loaded", checking: "Connecting…", offline: "Backend offline" }[backendStatus];
  const recorded = result?.sampleKey ? SAMPLES[result.sampleKey] : null;

  return (
    <>
      {/* ─── Top bar ─── */}
      <header className="topbar-wrap">
        <div className="topbar">
          <div className="brand">
            <svg className="brand-mark" viewBox="0 0 32 32" aria-hidden="true">
              <rect width="32" height="32" rx="8" />
              <path d="M9 25 L12 15 L22 10 L24 17" />
              <circle cx="12" cy="15" r="2.6" /><circle cx="22" cy="10" r="2.6" />
            </svg>
            <span className="brand-text">
              <span className="brand-name">Cobot<span>Ops</span></span>
              <span className="brand-sub">UR3 fault detection</span>
            </span>
          </div>
          <nav className="topbar-actions">
            <a className="nav-link" href="#workbench">Workbench</a>
            <a className="nav-link" href="#model">Model</a>
            <button className="nav-link" onClick={() => setShowAbout(true)}>About</button>
            <button className={`status status--${backendStatus}`} onClick={checkHealth} title="Re-check connection">
              <span className="status-dot" />
              {statusText}
            </button>
          </nav>
        </div>
      </header>
      <div className="shell">


        {/* ─── Hero: scenarios + digital twin ─── */}
        <section className="hero">
          <div className="hero-copy">
            <p className="eyebrow"><span className="eyebrow-dot" />Protective-stop detection · UR3 cobot</p>
            <h1 className="wordmark" aria-label="UR3 CobotOps">
              <span className="wordmark-pre">UR3</span>
              <span className="wordmark-main">Cobot<span>Ops</span></span>
            </h1>
            <p className="tagline">Know when your cobot <em>stops</em> — from its motor currents alone.</p>
            <p className="hero-sub">
              A model trained on real UR3 readings scores the latest joint currents and shows which joints drove the call.
              Pick a recorded scenario to see it work.
            </p>

            <div className="scenarios" role="group" aria-label="Recorded scenarios">
              {Object.entries(SAMPLES).map(([key, s]) => (
                <button key={key} type="button" disabled={loading}
                  className={`scenario spot scenario--${key} ${activeSample === key && result?.sampleKey === key ? "scenario--active" : ""}`}
                  onClick={() => runScenario(key)}>
                  <span className="scenario-dot" />
                  <span className="scenario-text">
                    <span className="scenario-title">{s.label}</span>
                    <span className="scenario-blurb">{s.blurb}</span>
                  </span>
                  <span className="scenario-meta">row {s.row}<kbd>{Object.keys(SAMPLES).indexOf(key) + 1}</kbd></span>
                  <span className="scenario-go" aria-hidden="true">
                    {loading && activeSample === key ? <span className="spinner spinner--light" /> : "→"}
                  </span>
                </button>
              ))}
            </div>
            <p className="hero-foot">
              Real readings from held-out test cycles the model never trained on.
              <span className="kbd-hint"> Press <kbd>1</kbd> <kbd>2</kbd> <kbd>3</kbd> to run them.</span>
            </p>
          </div>

          <div className={`stage stage--${mode}`} ref={stageRef} onPointerMove={tilt} onPointerLeave={untilt}>
            <div className="stage-hud">
              <span className={`mode-chip mode-chip--${mode}`}><span className="mode-dot" />{MODES[mode].label}</span>
              {result && (
                <span className={`hud-prob hud-prob--${result.risk_band}`}>
                  <small>Stop probability</small>
                  {(result.probability * 100).toFixed(1)}%
                </span>
              )}
            </div>
            <CobotScene mode={mode} highlights={result?.top_factors ?? []} focus={focus} />
            <div className="stage-foot">
              <span><i className="lg lg--green" />Running</span>
              <span><i className="lg lg--amber" />Watch</span>
              <span><i className="lg lg--red" />Stop</span>
              <span className="stage-foot-right"><i className="lg-ring" />Joint that drove the result · hover a row to locate a joint</span>
            </div>
          </div>
        </section>

        {/* ─── Workbench ─── */}
        <main className="workbench" id="workbench">

          <section className="panel console spot">
            <div className="panel-head">
              <div>
                <p className="kicker">01 · Input</p>
                <h2 className="panel-title">Sensor readings</h2>
              </div>
              <button type="button" className="text-btn"
                onClick={() => { setForm(BLANK); setActiveSample(null); setResult(null); setError(null); }}>
                Clear all
              </button>
            </div>

            <form onSubmit={handleSubmit}>
              <div className="io">
                <div className="io-row io-row--head">
                  <span>Axis</span>
                  <span>Latest <em>t</em></span>
                  <span>Earlier <em>t−3</em></span>
                  <span className="io-delta-col">Δ |I|</span>
                </div>

                {JOINTS.map(({ key, id, label }) => {
                  const d = isNum(form.current[key]) && isNum(form.earlier[key])
                    ? Math.abs(Number(form.current[key])) - Math.abs(Number(form.earlier[key])) : null;
                  const hot = result?.top_factors.some((f) => f.feature === key);
                  return (
                    <div className={`io-row ${hot ? "io-row--hot" : ""} ${focus === key ? "io-row--focus" : ""}`} key={key} {...focusProps(key)}>
                      <span className="axis">
                        <span className="axis-id">{id}</span>
                        <span className="axis-name">{label}</span>
                      </span>
                      {input("current", key, `${label} current, latest reading`)}
                      {input("earlier", key, `${label} current, 3 readings earlier`)}
                      <span className={`io-delta-col delta ${d > 0 ? "delta--up" : d < 0 ? "delta--down" : ""}`}>{fmt(d, 2)}</span>
                    </div>
                  );
                })}

                <div className={`io-row ${result?.top_factors.some((f) => f.feature === "Tool_current") ? "io-row--hot" : ""} ${focus === "Tool_current" ? "io-row--focus" : ""}`}
                  {...focusProps("Tool_current")}>
                  <span className="axis">
                    <span className="axis-id axis-id--tool">TL</span>
                    <span className="axis-name">Tool</span>
                  </span>
                  {input("current", "Tool_current", "Tool current, latest reading")}
                  <span className="io-na">not needed</span>
                  <span className="io-delta-col" />
                </div>

                <div className={`io-row io-row--total ${focus === "total_joint_current_delta3" ? "io-row--focus" : ""}`}
                  {...focusProps("total_joint_current_delta3")}>
                  <span className="axis">
                    <span className="axis-id axis-id--sum">Σ</span>
                    <span className="axis-name">Total |I|</span>
                  </span>
                  <span className="total-val">{totalNow === null ? "—" : totalNow.toFixed(3)}</span>
                  <span className="total-val">{totalEarlier === null ? "—" : totalEarlier.toFixed(3)}</span>
                  <span className={`io-delta-col delta delta--strong ${delta3 > 0 ? "delta--up" : delta3 < 0 ? "delta--down" : ""}`}>{fmt(delta3, 2)}</span>
                </div>
              </div>

              <p className="hint">
                Currents in amps, both readings from the same working cycle, about 3 s apart. The model uses the change in
                total current (Σ row), so the earlier reading is required. The bar under each box shows where it sits in the training range.
              </p>

              {outOfRange.length > 0 && (
                <p className="warn">
                  {outOfRange.length === 1 ? "One value is" : `${outOfRange.length} values are`} outside the training range
                  (outlined in amber). The prediction may be unreliable.
                </p>
              )}

              <div className="actions">
                <button type="submit" className="run-btn" disabled={loading}>
                  {loading ? <><span className="spinner" /> Scoring…</> : <>Run inference <kbd>↵</kbd></>}
                </button>
                <span className="actions-note">
                  {activeSample ? `${SAMPLES[activeSample].label} · test row ${SAMPLES[activeSample].row}, cycle ${SAMPLES[activeSample].cycle}` : "Manual entry"}
                </span>
              </div>
            </form>
          </section>

          <section className="panel result spot" aria-live="polite" ref={resultRef}>
            <div className="panel-head">
              <div>
                <p className="kicker">02 · Assessment</p>
                <h2 className="panel-title">{result ? VERDICTS[result.risk_band] : "Awaiting a reading"}</h2>
              </div>
              {result && <span className={`band band--${result.risk_band}`}>{result.risk_band} risk</span>}
            </div>

            {error && (
              <div className="alarm">
                <p className="alarm-title">Could not get a prediction</p>
                <p>{error}</p>
                <button className="text-btn text-btn--boxed" type="button" onClick={checkHealth}>Check connection</button>
              </div>
            )}

            {!result && !error && (
              <div className="empty">
                <svg viewBox="0 0 48 48" aria-hidden="true"><circle cx="24" cy="24" r="20" /><path d="M24 14v11l7 4" /></svg>
                <p>Pick a scenario above, or enter readings and run inference.</p>
              </div>
            )}

            {result && (
              <div className="result-body" key={result.at}>
                <RiskGauge probability={result.probability} watch={result.thresholds.watch}
                  alert={result.thresholds.alert} band={result.risk_band} />

                <div className={`verdict verdict--${result.risk_band}`}>
                  <p>{result.message}</p>
                  {recorded && (
                    <p className="verdict-truth">
                      Recorded in the data: <b>{recorded.recorded === "stop" ? "protective stop" : "no stop"}</b>
                      <span> · row {recorded.row}, cycle {recorded.cycle}</span>
                    </p>
                  )}
                </div>

                <div className="drivers">
                  <div className="drivers-head">
                    <span>What drove this result</span>
                    <span className="tag" title="SHAP values from the trained model, for this reading">SHAP</span>
                  </div>
                  <div className="drivers-axis" aria-hidden="true">
                    <span />
                    <span className="drivers-axis-labels"><span>← lowers risk</span><span>raises risk →</span></span>
                    <span />
                  </div>
                  {result.top_factors.map((f, i) => {
                    const w = (Math.abs(f.contribution) / maxContribution) * 50;
                    const up = f.effect === "raises risk";
                    return (
                      <div className={`driver ${focus === f.feature ? "driver--focus" : ""}`} key={f.feature}
                        style={{ animationDelay: `${i * 80}ms` }} {...focusProps(f.feature)}>
                        <div className="driver-label">
                          <span>{FACTOR_NAMES[f.feature] ?? f.feature}</span>
                          <span className="driver-input">{f.value.toFixed(3)} A</span>
                        </div>
                        <div className="driver-track">
                          <span className="driver-mid" />
                          <span className={`driver-bar ${up ? "driver-bar--up" : "driver-bar--down"}`}
                            style={up ? { left: "50%", width: `${w}%` } : { right: "50%", width: `${w}%` }} />
                        </div>
                        <span className={`driver-val ${up ? "delta--up" : "delta--down"}`} title="SHAP value in log-odds of a protective stop">
                          {fmt(f.contribution, 2)}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {log.length > 0 && (
              <div className="log">
                <div className="log-head">
                  <span>Recent runs</span>
                  <button className="text-btn" onClick={() => setLog([])}>Clear</button>
                </div>
                <ol className="log-list">
                  {log.map((e, i) => (
                    <li key={`${e.at}-${i}`} className={`log-item log-item--${e.risk_band}`}>
                      <span className="log-time">{e.at}</span>
                      <span className="log-src">{e.source}</span>
                      <span className="log-p">{(e.probability * 100).toFixed(1)}%</span>
                    </li>
                  ))}
                </ol>
              </div>
            )}
          </section>
        </main>

        {/* ─── Model card ─── */}
        <section className="model spot" id="model">
          <div className="model-copy">
            <p className="kicker">03 · Model</p>
            <h2>How reliable is it?</h2>
            <p>
              XGBoost on 9 current-based features, alert threshold {thresholds ? thresholds.alert.toFixed(2) : "0.66"}. Scored once on
              the last 20% of working cycles, which it never saw during training.
            </p>
          </div>
          <dl className="kpis">
            {MODEL_CARD.map((k) => (
              <div className="kpi" key={k.label}>
                <dt>{k.label}</dt>
                <dd>{k.value}<small>{k.unit}</small></dd>
                <p>{k.note}</p>
              </div>
            ))}
          </dl>
        </section>

        <footer className="footer">
          <p>
            <b>Decision support only.</b> Trained on one UR3 and about two hours of readings (UCI CobotOps). It detects stops as they
            begin and does not replace the robot's certified safety functions.
          </p>
          <p className="footer-meta">IT3051 group project · API {API_BASE_URL.replace(/^https?:\/\//, "")}</p>
        </footer>

        {/* ─── About modal ─── */}
        {showAbout && (
          <div className="modal-bg" onClick={() => setShowAbout(false)}>
            <div className="modal" role="dialog" aria-modal="true" aria-labelledby="about-title" onClick={(e) => e.stopPropagation()}>
              <div className="modal-top">
                <h3 id="about-title">About this system</h3>
                <button className="text-btn text-btn--boxed" onClick={() => setShowAbout(false)}>Close</button>
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
                <h4>The digital twin</h4>
                <p>The animated work cell is an illustration, not a live feed. It mirrors the latest assessment: it keeps running when risk is low, slows down on watch, and halts on a detected stop. Rings mark the joints whose currents moved the prediction most.</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  );
}

export default App;
