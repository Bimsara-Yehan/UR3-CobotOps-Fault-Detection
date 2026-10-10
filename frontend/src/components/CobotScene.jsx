import React, { useEffect, useRef, useState } from "react";

// Animated side view of a 6-axis collaborative arm doing pick-and-place next to an operator.
// Poses come from inverse kinematics on tool-tip waypoints and are interpolated in joint space,
// so the arm swings over the top like a real one. The scene reacts to the model's verdict:
// it runs at full speed, slows down on "watch", and freezes with a red stack light on a stop.

const S = { x: 220, y: 190 };              // shoulder pivot (J1)
const L1 = 95, L2 = 84;                    // upper arm, forearm
const WRIST = 20, TOOL = 24, REACH = WRIST + TOOL;
const FLOOR = 262;
const PART = { w: 18, h: 16 };
const PICK = { x: 95, y: 224 };            // finger tips around a part resting on the conveyor
const PLACE = { x: 335, y: 228 };          // finger tips around a part resting on the table

function ik(tip, elbow) {
  const X = tip.x - S.x;
  const Y = S.y - (tip.y - REACH);
  const c = Math.max(-1, Math.min(1, (X * X + Y * Y - L1 * L1 - L2 * L2) / (2 * L1 * L2)));
  const q2 = elbow * Math.acos(c);
  const q1 = Math.atan2(Y, X) - Math.atan2(L2 * Math.sin(q2), L1 + L2 * Math.cos(q2));
  return [q1, q2];
}

const KEYFRAMES = [
  { tip: { x: 95, y: 192 }, elbow: 1, g: 0, dur: 0 },
  { tip: PICK, elbow: 1, g: 0, dur: 650 },
  { tip: PICK, elbow: 1, g: 1, dur: 320 },               // close gripper
  { tip: { x: 95, y: 186 }, elbow: 1, g: 1, dur: 600 },
  { tip: { x: 335, y: 194 }, elbow: -1, g: 1, dur: 1500 }, // swing over the top
  { tip: PLACE, elbow: -1, g: 1, dur: 650 },
  { tip: PLACE, elbow: -1, g: 0, dur: 320 },              // open gripper
  { tip: { x: 335, y: 190 }, elbow: -1, g: 0, dur: 600 },
  { tip: { x: 95, y: 192 }, elbow: 1, g: 0, dur: 1500 },
].map((k) => ({ ...k, q: ik(k.tip, k.elbow) }));

const STARTS = KEYFRAMES.reduce((acc, k, i) => [...acc, (acc[i - 1] ?? 0) + (i ? k.dur : 0)], []);
const TOTAL = STARTS[STARTS.length - 1];

const ease = (u) => (u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2);
const lerp = (a, b, u) => a + (b - a) * u;

function poseAt(t) {
  let i = 1;
  while (i < KEYFRAMES.length - 1 && t >= STARTS[i]) i++;
  const a = KEYFRAMES[i - 1], b = KEYFRAMES[i];
  const u = b.dur ? Math.min(1, Math.max(0, (t - STARTS[i - 1]) / b.dur)) : 1;
  const e = ease(u);
  const q1 = lerp(a.q[0], b.q[0], e), q2 = lerp(a.q[1], b.q[1], e);
  const E = { x: S.x + L1 * Math.cos(q1), y: S.y - L1 * Math.sin(q1) };
  const W = { x: E.x + L2 * Math.cos(q1 + q2), y: E.y - L2 * Math.sin(q1 + q2) };
  const F = { x: W.x, y: W.y + WRIST };
  return { step: i, u, E, W, F, tip: { x: F.x, y: F.y + TOOL }, g: lerp(a.g, b.g, e) };
}

export const MODES = {
  run: { speed: 1, lamp: "green", label: "Auto · running" },
  watch: { speed: 0.5, lamp: "amber", label: "Auto · watch" },
  stop: { speed: 0, lamp: "red", label: "Protective stop" },
  offline: { speed: 0, lamp: "off", label: "No link to model" },
};

function usePrefersReducedMotion() {
  const [reduced, setReduced] = useState(
    () => typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches
  );
  useEffect(() => {
    const mq = window.matchMedia?.("(prefers-reduced-motion: reduce)");
    if (!mq) return;
    const on = (e) => setReduced(e.matches);
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);
  return reduced;
}

export default function CobotScene({ mode = "run", highlights = [], focus = null }) {
  const reduced = usePrefersReducedMotion();
  const t = useRef(STARTS[3] + 400);
  const speed = useRef(MODES[mode].speed);
  const cycles = useRef(1);
  const [, setFrame] = useState(0);
  const target = MODES[mode].speed;

  useEffect(() => {
    if (reduced) return undefined;
    let raf;
    let last = performance.now();
    const tick = (now) => {
      const dt = Math.min(64, now - last);
      last = now;
      speed.current += (target - speed.current) * Math.min(1, dt / 160);
      if (speed.current > 0.003) {
        t.current += dt * speed.current;
        if (t.current >= TOTAL) { t.current -= TOTAL; cycles.current += 1; }
        setFrame((f) => (f + 1) % 1e6);
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, reduced]);

  const p = poseAt(t.current);
  const holding = p.step >= 2 && p.step <= 6;
  const half = lerp(13, 9, p.g);
  const conveyorRuns = mode === "run" || mode === "watch";

  // Where the parts are this frame.
  const parts = [];
  if (holding) {
    parts.push({ x: p.tip.x - PART.w / 2, y: p.tip.y - 12, o: 1 });
  } else if (p.step <= 1) {
    parts.push({ x: PICK.x - PART.w / 2, y: PICK.y - 12, o: 1 });
  } else {
    const k = (p.step - 7 + p.u) / 2; // 0 → 1 across the last two steps
    parts.push({ x: PLACE.x - PART.w / 2, y: PLACE.y - 12, o: 1 - Math.max(0, k * 2 - 1) });
    parts.push({ x: lerp(30, PICK.x - PART.w / 2, ease(k)), y: PICK.y - 12, o: Math.min(1, k * 3) });
  }

  const jointPoint = {
    Current_J0: { x: S.x, y: 238 },
    Current_J1: S,
    Current_J2: p.E,
    Current_J3: p.W,
    Current_J4: { x: p.W.x, y: p.W.y + 10 },
    Current_J5: p.F,
    Tool_current: { x: p.F.x, y: p.F.y + 14 },
  };
  const wholeArm = highlights.some((h) => h.feature.startsWith("total_joint_current"))
    || Boolean(focus?.startsWith("total_joint_current"));
  const focusPoint = focus ? jointPoint[focus] : null;
  const focusLabel = focus === "Tool_current" ? "TOOL" : focus?.replace("Current_", "");
  const rings = highlights.filter((h) => jointPoint[h.feature]);

  const lampOn = (c) => MODES[mode].lamp === c;

  return (
    <svg className={`scene scene--${mode}`} viewBox="0 0 440 290" role="img"
      aria-label={`Animated collaborative robot work cell. State: ${MODES[mode].label}.`}>
      <defs>
        <pattern id="dots" width="16" height="16" patternUnits="userSpaceOnUse">
          <circle cx="1" cy="1" r="0.9" className="scene-dot" />
        </pattern>
        <linearGradient id="link" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0" stopColor="#eef2f6" />
          <stop offset="1" stopColor="#aeb9c5" />
        </linearGradient>
        <linearGradient id="zone" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0" className="zone-stop-a" />
          <stop offset="1" className="zone-stop-b" />
        </linearGradient>
        <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="4" result="b" />
          <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
      </defs>

      <rect width="440" height="290" fill="url(#dots)" />

      {/* Shared (collaborative) workspace */}
      <path className="zone" d={`M 18 ${FLOOR} A 202 202 0 0 1 422 ${FLOOR} Z`} fill="url(#zone)" />
      <path className="zone-edge" d={`M 18 ${FLOOR} A 202 202 0 0 1 422 ${FLOOR}`} />
      <text className="scene-label" x="12" y="20">SHARED WORKSPACE</text>
      <text className="scene-label scene-label--dim" x="12" y="32">speed &amp; separation monitored</text>

      {/* Floor and hazard tape */}
      <line className="floor" x1="0" x2="440" y1={FLOOR} y2={FLOOR} />
      <line className="tape" x1="160" x2="282" y1={FLOOR + 6} y2={FLOOR + 6} />

      {/* Infeed conveyor */}
      <g className="conveyor">
        <rect x="44" y="244" width="6" height={FLOOR - 244} className="leg" />
        <rect x="136" y="244" width="6" height={FLOOR - 244} className="leg" />
        <rect x="26" y="228" width="128" height="14" rx="7" className="belt-body" />
        <line x1="34" x2="146" y1="228" y2="228" className={`belt ${conveyorRuns ? "belt--run" : ""}`} />
        {[40, 62, 84, 106, 128].map((x) => (
          <circle key={x} cx={x} cy="235" r="3.2" className={`roller ${conveyorRuns ? "roller--run" : ""}`} />
        ))}
        <text className="scene-label scene-label--dim" x="30" y="282">INFEED</text>
      </g>

      {/* Assembly table */}
      <g>
        <rect x="292" y="232" width="92" height="7" rx="2" className="table" />
        <rect x="300" y="239" width="5" height={FLOOR - 239} className="leg" />
        <rect x="371" y="239" width="5" height={FLOOR - 239} className="leg" />
        <text className="scene-label scene-label--dim" x="300" y="282">ASSEMBLY</text>
      </g>

      {/* Operator sharing the cell */}
      <g className="operator">
        <circle cx="414" cy="170" r="9" />
        <path className="hardhat" d="M 404.5 168 A 9.5 9.5 0 0 1 423.5 168 Z" />
        <path d="M 401 184 Q 414 179 427 184 L 425 226 L 403 226 Z" />
        <path d="M 403 190 L 386 214 L 382 222" className="operator-arm" />
        <rect x="404" y="226" width="8" height={FLOOR - 226} rx="3" />
        <rect x="416" y="226" width="8" height={FLOOR - 226} rx="3" />
      </g>

      {/* Parts */}
      {parts.map((pt, i) => (
        <g key={i} opacity={pt.o}>
          <rect x={pt.x} y={pt.y} width={PART.w} height={PART.h} rx="2.5" className="part" />
          <line x1={pt.x + 4} x2={pt.x + PART.w - 4} y1={pt.y + 5} y2={pt.y + 5} className="part-mark" />
        </g>
      ))}

      {/* Robot */}
      <g className={`arm ${wholeArm ? "arm--hot" : ""}`}>
        <rect x="194" y="250" width="52" height="12" rx="3" className="base-plate" />
        <rect x="205" y="200" width="30" height="52" rx="6" fill="url(#link)" className="link" />
        <rect x="205" y="232" width="30" height="6" className="cap-band" />

        <line x1={S.x} y1={S.y} x2={p.E.x} y2={p.E.y} className="link-line" strokeWidth="17" />
        <line x1={S.x} y1={S.y} x2={p.E.x} y2={p.E.y} className="link-shine" strokeWidth="5" />
        <line x1={p.E.x} y1={p.E.y} x2={p.W.x} y2={p.W.y} className="link-line" strokeWidth="14" />
        <line x1={p.E.x} y1={p.E.y} x2={p.W.x} y2={p.W.y} className="link-shine" strokeWidth="4" />
        <line x1={p.W.x} y1={p.W.y} x2={p.F.x} y2={p.F.y} className="link-line" strokeWidth="12" />

        <circle cx={S.x} cy={S.y} r="13" className="joint" />
        <circle cx={S.x} cy={S.y} r="7" className="joint-cap" />
        <circle cx={p.E.x} cy={p.E.y} r="11" className="joint" />
        <circle cx={p.E.x} cy={p.E.y} r="5.5" className="joint-cap" />
        <circle cx={p.W.x} cy={p.W.y} r="8.5" className="joint" />
        <circle cx={p.W.x} cy={p.W.y} r="4" className="joint-cap" />
        <circle cx={p.W.x} cy={p.W.y + 10} r="6.5" className="joint joint--minor" />

        {/* Flange + two-finger gripper */}
        <rect x={p.F.x - 9} y={p.F.y - 2} width="18" height="4" rx="1.5" className="joint" />
        <rect x={p.F.x - 12} y={p.F.y + 2} width="24" height="8" rx="2" className="gripper" />
        <rect x={p.F.x - half - 3.5} y={p.F.y + 9} width="3.5" height={TOOL - 9} rx="1" className="finger" />
        <rect x={p.F.x + half} y={p.F.y + 9} width="3.5" height={TOOL - 9} rx="1" className="finger" />
      </g>

      {/* Inputs that drove the latest prediction */}
      {rings.map((h) => {
        const pt = jointPoint[h.feature];
        return (
          <g key={h.feature} className={`ring ${h.effect === "raises risk" ? "ring--up" : "ring--down"}`}>
            <circle cx={pt.x} cy={pt.y} r="15" className="ring-pulse" />
            <circle cx={pt.x} cy={pt.y} r="15" className="ring-solid" />
          </g>
        );
      })}

      {/* Joint the user is pointing at in the input table or the drivers list */}
      {focusPoint && (
        <g className="focus">
          <circle cx={focusPoint.x} cy={focusPoint.y} r="18" className="focus-ring" />
          <rect x={focusPoint.x + 16} y={focusPoint.y - 24} width={focusLabel.length * 6.4 + 10} height="14" rx="3" className="focus-tag" />
          <text x={focusPoint.x + 21} y={focusPoint.y - 14} className="focus-text">{focusLabel}</text>
        </g>
      )}

      {/* Stack light */}
      <g className="stack">
        <rect x="252" y="214" width="3" height={FLOOR - 214} className="leg" />
        <rect x="246.5" y="178" width="14" height="12" rx="2" className={`lamp lamp--red ${lampOn("red") ? "on" : ""}`} />
        <rect x="246.5" y="191" width="14" height="12" rx="2" className={`lamp lamp--amber ${lampOn("amber") ? "on" : ""}`} />
        <rect x="246.5" y="204" width="14" height="12" rx="2" className={`lamp lamp--green ${lampOn("green") ? "on" : ""}`} />
      </g>

      <text className="scene-readout" x="430" y="20" textAnchor="end">
        CYCLE {String(cycles.current).padStart(4, "0")}
      </text>

      {mode === "stop" && (
        <g className="stop-banner">
          <rect x="140" y="12" width="160" height="26" rx="4" />
          <text x="220" y="29.5" textAnchor="middle">PROTECTIVE STOP</text>
        </g>
      )}
    </svg>
  );
}
