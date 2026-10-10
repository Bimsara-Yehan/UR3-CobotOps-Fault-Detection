import React, { useEffect, useState } from "react";

// Semicircular dial: green below the watch threshold, amber up to the alert threshold, red above it.
const CX = 130, CY = 124, R = 100;

const point = (p, r = R) => {
  const a = Math.PI * (1 - p);
  return [CX + r * Math.cos(a), CY - r * Math.sin(a)];
};

const arc = (p0, p1, r = R) => {
  const [x0, y0] = point(p0, r);
  const [x1, y1] = point(p1, r);
  return `M ${x0} ${y0} A ${r} ${r} 0 ${p1 - p0 > 0.5 ? 1 : 0} 1 ${x1} ${y1}`;
};

export default function RiskGauge({ probability, watch, alert, band }) {
  const [shown, setShown] = useState(0);

  const [count, setCount] = useState(0);

  // Swing the needle and count the readout up from zero, in step with each other.
  useEffect(() => {
    const target = Math.min(1, Math.max(0, probability));
    const start = performance.now();
    let id = requestAnimationFrame(function step(now) {
      const u = Math.min(1, (now - start) / 900);
      setCount(target * (1 - Math.pow(1 - u, 3)));
      if (u < 1) id = requestAnimationFrame(step);
    });
    setShown(target);
    return () => cancelAnimationFrame(id);
  }, [probability]);

  const ticks = Array.from({ length: 21 }, (_, i) => i / 20);

  return (
    <div className="gauge">
      <svg viewBox="0 0 260 150" role="img" aria-label={`Stop probability ${(probability * 100).toFixed(1)} percent`}>
        <path d={arc(0, 1)} className="gauge-track" />
        <path d={arc(0, watch)} className="gauge-band gauge-band--low" />
        <path d={arc(watch, alert)} className="gauge-band gauge-band--medium" />
        <path d={arc(alert, 1)} className="gauge-band gauge-band--high" />

        {ticks.map((p) => {
          const major = Math.round(p * 20) % 5 === 0;
          const [x0, y0] = point(p, R - 16);
          const [x1, y1] = point(p, major ? R - 27 : R - 21);
          return <line key={p} x1={x0} y1={y0} x2={x1} y2={y1} className={`gauge-tick ${major ? "gauge-tick--major" : ""}`} />;
        })}

        {[[watch, "WATCH"], [alert, "ALERT"]].map(([p, label]) => {
          const [x0, y0] = point(p, R + 8);
          const [x1, y1] = point(p, R - 8);
          const [tx, ty] = point(p, R + 20);
          return (
            <g key={label}>
              <line x1={x0} y1={y0} x2={x1} y2={y1} className="gauge-threshold" />
              <text x={tx} y={ty} textAnchor="middle" className="gauge-threshold-label">{label}</text>
            </g>
          );
        })}

        <g className="gauge-needle" style={{ transform: `rotate(${shown * 180}deg)`, transformOrigin: `${CX}px ${CY}px` }}>
          <path d={`M ${CX} ${CY - 3.5} L ${CX - R + 18} ${CY} L ${CX} ${CY + 3.5} Z`} />
        </g>
        <circle cx={CX} cy={CY} r="9" className="gauge-hub" />
        <circle cx={CX} cy={CY} r="3.5" className={`gauge-hub-dot gauge-hub-dot--${band}`} />

        <text x={CX - R} y={CY + 18} textAnchor="middle" className="gauge-end">0</text>
        <text x={CX + R} y={CY + 18} textAnchor="middle" className="gauge-end">100</text>
      </svg>
      <div className="gauge-value">
        <span className={`gauge-pct gauge-pct--${band}`}>{(count * 100).toFixed(1)}<small>%</small></span>
        <span className="gauge-caption">stop probability</span>
      </div>
    </div>
  );
}
