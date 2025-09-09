import React, { useState, useCallback } from 'react';
import axios from 'axios';
import './styles.css';

const REQUIRED_FILES = 4;
const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:5000';
const DIRECTIONS = [
  { key: 'north', label: 'North', icon: '⬆️' },
  { key: 'south', label: 'South', icon: '⬇️' },
  { key: 'west', label: 'West', icon: '⬅️' },
  { key: 'east', label: 'East', icon: '➡️' },
];

const DirectionCard = ({ label, value, icon }) => (
  <div className="direction-card" aria-label={`${label} timing`}> 
    <div className="dir-header">{icon}<span>{label}</span></div>
    <div className="dir-value">{value}<small>s</small></div>
    <div className="progress-bar"><div style={{ width: `${Math.min(100, (value/60)*100)}%` }} /></div>
  </div>
);

const FileSlot = ({ index, file, onRemove }) => (
  <div className={`file-slot ${file ? 'filled' : ''}`}> 
    {!file && <span>{DIRECTIONS[index].icon} {DIRECTIONS[index].label}</span>} 
    {file && (
      <>
        <span title={file.name}>{file.name.length > 18 ? file.name.slice(0,15)+'…' : file.name}</span>
        <button type="button" className="remove-btn" onClick={() => onRemove(index)} aria-label={`Remove file ${file.name}`}>×</button>
      </>
    )}
  </div>
);

const Loader = ({ text }) => (
  <div className="loader-wrapper" role="status">
    <div className="spinner" />
    <p>{text}</p>
  </div>
);

const ErrorBanner = ({ message, onDismiss }) => (
  <div className="error-banner" role="alert">
    <span>{message}</span>
    <button onClick={onDismiss} aria-label="Dismiss error">×</button>
  </div>
);

function App() {
  const [files, setFiles] = useState(Array(REQUIRED_FILES).fill(null));
  const [dragActive, setDragActive] = useState(false);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFiles = useCallback((incoming) => {
    setError(null);
    const newList = [...files];
    for (let f of incoming) {
      const idx = newList.findIndex(x => !x);
      if (idx === -1) break; 
      newList[idx] = f;
    }
    setFiles(newList);
  }, [files]);

  const onInputChange = (e) => {
    handleFiles(Array.from(e.target.files || []));
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragActive(false);
    handleFiles(Array.from(e.dataTransfer.files || []));
  };

  const onDragOver = (e) => { e.preventDefault(); setDragActive(true); };
  const onDragLeave = (e) => { e.preventDefault(); setDragActive(false); };

  const clearAll = () => { setFiles(Array(REQUIRED_FILES).fill(null)); setResult(null); };

  const removeFile = (idx) => {
    const updated = [...files];
    updated[idx] = null;
    setFiles(updated);
  };

  const readyToSubmit = files.every(Boolean);

  const submit = async () => {
    if (!readyToSubmit) { setError(`Please add all ${REQUIRED_FILES} videos before running.`); return; }
    setLoading(true);
    setError(null);
    setResult(null);
    const formData = new FormData();
    files.forEach(f => formData.append('videos', f));
    try {
      const { data } = await axios.post(`${API_BASE}/upload`, formData, { headers: { 'Content-Type': 'multipart/form-data' } });
      setResult(data);
    } catch (e) {
      setError(e?.response?.data?.error || 'Upload failed. Check backend.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="dashboard-root">
      <aside className="sidebar">
        <div className="brand">Dashboard v2.0</div>
        <nav>
          <a href="#upload">Upload</a>
          <a href="#results">Results</a>
          <a href="https://github.com" target="_blank" rel="noreferrer">Docs</a>
        </nav>
        <div className="footer">v2.0 Dashboard</div>
      </aside>
      <main className="main-area">
        <header className="page-header">
          <h1>AI-Based Traffic Management Dashboard</h1>
          <div className="actions">
            <button onClick={clearAll} disabled={loading}>Reset</button>
            <button className="primary" onClick={submit} disabled={!readyToSubmit || loading}>{loading ? 'Processing...' : 'Run Optimization'}</button>
          </div>
        </header>
        {error && <ErrorBanner message={error} onDismiss={() => setError(null)} />}

        <section id="upload" className="panel upload-panel">
          <h2>1. Provide Intersection Videos</h2>
          <p className="muted">Drag & drop or click to add exactly {REQUIRED_FILES} perspective videos (N, S, W, E).</p>
          <div 
            className={`dropzone ${dragActive ? 'drag' : ''}`} 
            onDrop={onDrop} 
            onDragOver={onDragOver} 
            onDragLeave={onDragLeave}
            role="button"
            tabIndex={0}
            aria-label="Video upload dropzone"
            onKeyDown={(e)=> e.key==='Enter' && document.getElementById('file-input')?.click()}
          >
            <input id="file-input" type="file" multiple accept="video/*" onChange={onInputChange} hidden />
            <div className="slots">
              {files.map((f,i)=>(<FileSlot key={i} index={i} file={f} onRemove={removeFile} />))}
            </div>
            <button type="button" className="outline" onClick={()=>document.getElementById('file-input').click()}>Select Videos</button>
          </div>
        </section>

        <section id="results" className="panel results-panel">
          <h2>2. Optimization Output</h2>
          {!result && !loading && <p className="muted">Run the optimization to see calculated green light durations.</p>}
          {loading && <Loader text="Analyzing traffic density & running genetic algorithm..." />}
          {result && !result.error && !loading && (
            <div className="directions-grid">
              {DIRECTIONS.map(dir => (
                <DirectionCard key={dir.key} label={dir.label} value={result[dir.key]} icon={dir.icon} />
              ))}
            </div>
          )}
          {result && result.error && <p className="error-text">{result.error}</p>}
        </section>

        <section className="panel info-panel">
          <h2>Methodology</h2>
            <ul className="info-list">
              <li>YOLOv4-tiny counts vehicle peaks over rolling 30s windows.</li>
              <li>Genetic Algorithm searches green time allocations within cycle constraints.</li>
              <li>Objective minimizes combined delay using calibrated fitness function.</li>
              <li>Recommended times are integers (seconds) capped at 60s per phase.</li>
            </ul>
        </section>
      </main>
    </div>
  );
}

export default App;
