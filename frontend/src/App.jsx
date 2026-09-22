import { useState } from "react";
import "./App.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

function App() {
  const [result, setResult] = useState(null);

  // TODO: grouped input form (current / temperature / speed / tool) — see CLAUDE.md "System build"
  // TODO: "load sample row" button
  // TODO: POST the form values to `${API_BASE_URL}/predict`, store the response in `result`
  // TODO: result card showing label, probability, risk band, and the plain-language message
  // TODO: top-3 feature explanation (from the backend response)

  return (
    <main className="app">
      <h1>UR3 CobotOps — Protective Stop Risk</h1>
      <p>
        TODO: sensor input form + result card. Backend base URL: <code>{API_BASE_URL}</code>.
      </p>
    </main>
  );
}

export default App;
