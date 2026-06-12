import { useState, useCallback } from "react";

const API = "";

function App() {
  const [file, setFile] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) { setFile(f); setResult(null); setError(null); }
  }, []);

  const onFileChange = (e) => {
    const f = e.target.files[0];
    if (f) { setFile(f); setResult(null); setError(null); }
  };

  const analyse = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);

    const form = new FormData();
    form.append("file", file);

    try {
      const res = await fetch(`${API}/analyse`, {
        method: "POST",
        body: form,
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Analysis failed");
      }
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // ── Confidence colour helper ───────────────────────────────────────────────
  const confidenceColour = (label) => {
    if (!label) return "text-white";
    const l = label.toLowerCase();
    if (l.startsWith("high")) return "text-green-400";
    if (l.startsWith("medium")) return "text-yellow-400";
    return "text-red-400";
  };

  return (
    <div className="min-h-screen bg-[#0a0a0f]">

      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <header className="border-b border-[#00e5ff22] bg-[#0d0d1a]">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-[#00e5ff] flex items-center justify-center">
              <span className="text-black font-black text-sm">A</span>
            </div>
            <div>
              <h1 className="text-[#00e5ff] font-bold text-lg tracking-widest">ATHENA</h1>
              <p className="text-[#ffffff44] text-xs tracking-wider">VAR DECISION SUPPORT SYSTEM</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse"></div>
            <span className="text-green-400 text-xs font-mono">SYSTEM ONLINE</span>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-10">

        {/* ── Hero ───────────────────────────────────────────────────────────── */}
        <div className="text-center mb-12">
          <h2 className="text-4xl font-bold text-white mb-3">
            AI-Powered <span className="text-[#00e5ff]">VAR Analysis</span>
          </h2>
          <p className="text-[#ffffff66] text-lg max-w-2xl mx-auto">
            Upload a match clip and ATHENA will detect players, compute the offside
            line, and deliver a decision with confidence scoring.
          </p>
        </div>

        {/* ── Upload panel (shown when no result yet) ───────────────────────── */}
        {!result && (
          <div className="max-w-2xl mx-auto">

            {/* Drop Zone */}
            <div
              onDrop={onDrop}
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onClick={() => document.getElementById("fileInput").click()}
              className={`border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all duration-300 ${dragging
                  ? "border-[#00e5ff] bg-[#00e5ff11]"
                  : file
                    ? "border-green-400 bg-[#00ff8811]"
                    : "border-[#ffffff22] bg-[#0d0d1a] hover:border-[#00e5ff66] hover:bg-[#00e5ff08]"
                }`}
            >
              <input
                id="fileInput"
                type="file"
                accept=".mp4,.mov,.avi,.mkv,.webm"
                className="hidden"
                onChange={onFileChange}
              />
              <div className="text-5xl mb-4">{file ? "✅" : "⚽"}</div>
              {file ? (
                <div>
                  <p className="text-green-400 font-semibold">{file.name}</p>
                  <p className="text-[#ffffff44] text-sm mt-1">
                    {(file.size / 1024 / 1024).toFixed(1)} MB — Ready to analyse
                  </p>
                </div>
              ) : (
                <div>
                  <p className="text-white font-semibold text-lg">Drop your match clip here</p>
                  <p className="text-[#ffffff44] text-sm mt-2">
                    MP4, MOV, AVI, MKV — broadcast sideline angle works best
                  </p>
                </div>
              )}
            </div>

            {/* Error */}
            {error && (
              <div className="mt-4 bg-red-900/30 border border-red-500/50 rounded-xl p-4">
                <p className="text-red-400 text-sm">⚠ {error}</p>
              </div>
            )}

            {/* Analyse button */}
            <button
              onClick={analyse}
              disabled={!file || loading}
              className={`w-full mt-6 py-4 rounded-xl font-bold text-lg tracking-wider transition-all duration-300 ${!file || loading
                  ? "bg-[#ffffff11] text-[#ffffff33] cursor-not-allowed"
                  : "bg-[#00e5ff] text-black hover:bg-[#00b8cc] hover:shadow-[0_0_30px_#00e5ff44]"
                }`}
            >
              {loading ? (
                <span className="flex items-center justify-center gap-3">
                  <span className="w-5 h-5 border-2 border-black/30 border-t-black rounded-full animate-spin"></span>
                  ATHENA IS ANALYSING...
                </span>
              ) : (
                "⚡ ANALYSE CLIP"
              )}
            </button>

            {/* Loading steps */}
            {loading && (
              <div className="mt-6 bg-[#0d0d1a] border border-[#ffffff11] rounded-xl p-6">
                <p className="text-[#00e5ff] text-sm font-mono mb-3">Processing pipeline...</p>
                {[
                  "Extracting frames from video",
                  "Running YOLOv11 player detection",
                  "Tracking players with ByteTrack",
                  "Computing offside line geometry",
                  "Generating decision with confidence score",
                ].map((step, i) => (
                  <div key={i} className="flex items-center gap-3 py-1">
                    <div
                      className="w-1.5 h-1.5 rounded-full bg-[#00e5ff] animate-pulse"
                      style={{ animationDelay: `${i * 0.3}s` }}
                    />
                    <span className="text-[#ffffff66] text-xs font-mono">{step}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* ── Result panel ──────────────────────────────────────────────────── */}
        {result && (
          <div className="space-y-6">

            {/* Verdict banner */}
            <div className={`rounded-2xl p-6 text-center border ${result.is_offside
                ? "bg-red-900/20 border-red-500/50"
                : "bg-green-900/20 border-green-500/50"
              }`}>
              <div className={`text-5xl font-black tracking-widest mb-2 ${result.is_offside ? "text-red-400" : "text-green-400"
                }`}>
                {result.verdict}
              </div>
              <p className="text-[#ffffff66] text-sm">
                Frame {result.frame_number ?? "—"} of {result.total_frames ?? "—"} analysed
                · {result.players_detected ?? 0} players detected
              </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

              {/* ── Annotated frame ─────────────────────────────────────────── */}
              <div className="lg:col-span-2 bg-[#0d0d1a] border border-[#ffffff11] rounded-2xl overflow-hidden">
                <div className="p-4 border-b border-[#ffffff11]">
                  <h3 className="text-white font-semibold text-sm tracking-wider">
                    ANALYSIS FRAME
                  </h3>
                  <p className="text-[#ffffff44] text-xs mt-1">
                    {result.attacking_players ?? 0} attackers (red) ·{" "}
                    {result.defending_players ?? 0} defenders (blue) ·{" "}
                    direction: {result.attack_direction === "left_to_right" ? "→" : "←"}
                  </p>
                </div>

                {/* ── FIX: backend sends 'annotated_frame', not 'result_image' ── */}
                {result.annotated_frame ? (
                  <img
                    src={`data:image/jpeg;base64,${result.annotated_frame}`}
                    alt="ATHENA offside analysis"
                    className="w-full"
                  />
                ) : (
                  <div className="p-8 text-center text-[#ffffff33] text-sm">
                    No frame available
                  </div>
                )}
              </div>

              {/* ── Decision panel ──────────────────────────────────────────── */}
              <div className="space-y-4">

                {/* Confidence */}
                <div className="bg-[#0d0d1a] border border-[#ffffff11] rounded-2xl p-5">
                  <h3 className="text-[#ffffff66] text-xs uppercase tracking-wider mb-4">
                    Confidence Score
                  </h3>
                  <div className="text-5xl font-black text-[#00e5ff] mb-1">
                    {result.confidence}%
                  </div>
                  <div className={`text-sm font-semibold mb-4 ${confidenceColour(result.confidence_label)}`}>
                    {result.confidence_label}
                  </div>
                  <div className="w-full bg-[#ffffff11] rounded-full h-2">
                    <div
                      className="h-2 rounded-full bg-[#00e5ff] transition-all duration-1000"
                      style={{ width: `${result.confidence}%` }}
                    />
                  </div>
                </div>

                {/* Offside geometry */}
                <div className="bg-[#0d0d1a] border border-[#ffffff11] rounded-2xl p-5">
                  <h3 className="text-[#ffffff66] text-xs uppercase tracking-wider mb-3">
                    Offside Line
                  </h3>
                  <div className="text-2xl font-black text-white mb-1">
                    x = {result.offside_line_x ?? "—"}px
                  </div>
                  <p className="text-[#ffffff44] text-xs">
                    Frame width: {result.frame_width ?? "—"}px ·{" "}
                    {result.frame_width
                      ? `${((result.offside_line_x / result.frame_width) * 100).toFixed(1)}% across`
                      : ""}
                  </p>
                </div>

                {/* Historical cases */}
                <div className="bg-[#0d0d1a] border border-[#ffffff11] rounded-2xl p-5">
                  <h3 className="text-[#ffffff66] text-xs uppercase tracking-wider mb-3">
                    Historical Cases
                  </h3>
                  <div className="text-3xl font-black text-white mb-1">
                    {result.historical_cases ?? "—"}
                  </div>
                  <p className="text-[#ffffff44] text-xs">
                    Similar decisions in historical database
                  </p>
                </div>

                {/* Explanation */}
                <div className="bg-[#0d0d1a] border border-[#ffffff11] rounded-2xl p-5">
                  <h3 className="text-[#ffffff66] text-xs uppercase tracking-wider mb-3">
                    AI Explanation
                  </h3>
                  <p className="text-[#ffffffaa] text-xs leading-relaxed">
                    {result.explanation}
                  </p>
                </div>

              </div>
            </div>

            {/* Processing time */}
            {result.processing_time_seconds && (
              <p className="text-center text-[#ffffff33] text-xs font-mono">
                Processed in {result.processing_time_seconds}s
              </p>
            )}

            {/* Analyse another */}
            <button
              onClick={() => { setResult(null); setFile(null); setError(null); }}
              className="w-full py-3 rounded-xl border border-[#00e5ff44] text-[#00e5ff] font-semibold hover:bg-[#00e5ff11] transition-all"
            >
              ← Analyse Another Clip
            </button>

          </div>
        )}
      </main>

      {/* ── Footer ─────────────────────────────────────────────────────────── */}
      <footer className="border-t border-[#ffffff11] mt-16 py-6 text-center">
        <p className="text-[#ffffff33] text-xs font-mono">
          ATHENA VAR · YOLOv11 · ByteTrack · FastAPI · Built for intelligent referee support
        </p>
      </footer>

    </div>
  );
}

export default App;
