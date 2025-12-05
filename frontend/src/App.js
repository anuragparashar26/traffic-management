import React, { useState, useCallback, useEffect } from 'react';
import axios from 'axios';
import './styles.css';
import { supabase } from './supabaseClient';

const REQUIRED_FILES = 4;
const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:5000';
const GA_PARAMS = {
  greenMin: 10,
  greenMax: 60,
  cycleTime: 148,
  mutationRate: 0.02,
  population: 400,
  generations: 25,
};
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

const HelmetResultCard = ({ label, value, icon }) => (
  <div className="direction-card" aria-label={`${label} count`}>
    <div className="dir-header">{icon}<span>{label}</span></div>
    <div className="dir-value">{value}</div>
  </div>
);

const CountCard = ({ label, value, icon }) => (
  <div className="direction-card" aria-label={`${label} vehicles`}>
    <div className="dir-header">{icon}<span>{label} vehicles</span></div>
    <div className="dir-value">{value?.toFixed ? value.toFixed(2) : value}</div>
  </div>
);

const resolveImage = (path) => {
  if (!path) return '';
  return path.startsWith('http') ? path : `${API_BASE}/${path}`;
};

const ViolationCard = ({ violation }) => {
  const plateLabel = violation.plate_text || violation.plate_number || 'Unknown';
  const confidence = violation.plate_confidence;
  const riderSrc = resolveImage(violation.rider_image);
  const plateSrc = resolveImage(violation.plate_image);

  return (
    <div className="violation-card">
      <div className="violation-header">
        <h3>🚨 License Plate: {plateLabel}</h3>
        <span className="confidence">Confidence: {confidence !== undefined && confidence !== null ? `${(confidence * 100).toFixed(2)}%` : '—'}</span>
      </div>
      <div className="violation-images">
        <div className="img-container">
          <img src={riderSrc} alt="Rider" />
          <label>Rider</label>
        </div>
        <div className="img-container">
          <img src={plateSrc} alt="License Plate" />
          <label>Plate</label>
        </div>
      </div>
      <div className="violation-timestamp">{violation.timestamp || 'Timestamp unavailable'}</div>
    </div>
  );
};

function App() {
  const [files, setFiles] = useState(Array(REQUIRED_FILES).fill(null));
  const [dragActive, setDragActive] = useState(false);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showStreams, setShowStreams] = useState(false);
  const [streamToken, setStreamToken] = useState(null);

  // New state for helmet detection
  const [helmetFile, setHelmetFile] = useState(null);
  const [helmetResult, setHelmetResult] = useState(null);
  const [helmetLoading, setHelmetLoading] = useState(false);
  const [helmetError, setHelmetError] = useState(null);

  // Supabase-backed violations
  const [violations, setViolations] = useState([]);
  const [violationsLoading, setViolationsLoading] = useState(false);
  const [violationsError, setViolationsError] = useState(null);

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

  const clearAll = () => {
    setFiles(Array(REQUIRED_FILES).fill(null));
    setResult(null);
    setShowStreams(false);
    setStreamToken(null);
  };

  const removeFile = (idx) => {
    const updated = [...files];
    updated[idx] = null;
    setFiles(updated);
  };

  const readyToSubmit = files.every(Boolean);

  const submit = async () => {
    if (!readyToSubmit) { setError(`Please add all ${REQUIRED_FILES} videos before running.`); return; }
    const token = Date.now();
    setShowStreams(true);
    setStreamToken(token);
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

  // New handlers for helmet detection
  const onHelmetInputChange = (e) => {
    setHelmetFile(e.target.files[0]);
    setHelmetError(null);
  };

  const submitHelmet = async () => {
    if (!helmetFile) { setHelmetError('Please select a video for helmet detection.'); return; }
    setHelmetLoading(true);
    setHelmetError(null);
    setHelmetResult(null);
    const formData = new FormData();
    formData.append('video', helmetFile);
    try {
      const { data } = await axios.post(`${API_BASE}/detect_helmets`, formData, { headers: { 'Content-Type': 'multipart/form-data' } });
      setHelmetResult(data);
    } catch (e) {
      setHelmetError(e?.response?.data?.error || 'Helmet detection failed.');
    } finally {
      setHelmetLoading(false);
    }
  };

  const clearHelmet = () => { setHelmetFile(null); setHelmetResult(null); };

  const formatTimestamp = (ts) => {
    if (!ts) return '';
    if (ts.includes('T')) {
      const [date, time] = ts.split('T');
      return date + ' ' + time.slice(0,8);
    }
    return ts;
  };
  const fetchViolations = useCallback(async () => {
    if (!supabase) { setViolationsError('Supabase environment variables are missing.'); return; }
    setViolationsLoading(true);
    setViolationsError(null);
    const { data, error } = await supabase
      .from('violations')
      .select('*')
      .order('created_at', { ascending: false });
    if (error) {
      setViolationsError(error.message);
    } else {
      const formatted = (data || []).map(v => ({ ...v, timestamp: formatTimestamp(v.timestamp) }));
      setViolations(formatted);
    }
    setViolationsLoading(false);
  }, []);

  const persistViolations = useCallback(async (list) => {
    if (!supabase || !list || list.length === 0) return;
    const pad = n => n.toString().padStart(2, '0');
    const formatTimestamp = (d) => {
      return `${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
    };
    const payload = list.map((v) => ({
      plate_text: v.plate_text || v.plate_number || 'Unknown',
      plate_confidence: v.plate_confidence ?? null,
      plate_image: resolveImage(v.plate_image),
      rider_image: resolveImage(v.rider_image),
      timestamp: v.timestamp || formatTimestamp(new Date()),
    }));
    const { error } = await supabase.from('violations').insert(payload);
    if (error) {
      setViolationsError(error.message);
    } else {
      fetchViolations();
    }
  }, [fetchViolations]);

  useEffect(() => { fetchViolations(); }, [fetchViolations]);
  useEffect(() => {
    if (helmetResult && helmetResult.violations && helmetResult.violations.length > 0) {
      persistViolations(helmetResult.violations);
    }
  }, [helmetResult, persistViolations]);

  return (
    <div className="dashboard-root">
      <aside className="sidebar">
        <div className="brand">Dashboard v3.0</div>
        <nav>
          <a href="#upload">Traffic Analysis</a>
          <a href="#helmet">Helmet Detection</a>
          <a href="#violations">Violation Records</a>
          <a href="https://github.com/anuragparashar26/traffic-management" target="_blank" rel="noreferrer">Github</a>
        </nav>
        <div className="footer">v3.0 Dashboard</div>
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
          {result && result.car_counts && !loading && (
            <div style={{ marginTop: '14px' }}>
              <h3 style={{ margin: '0 0 8px', fontSize: '.95rem', color: '#b5bcc9' }}>Input Metrics (cars detected)</h3>
              <div className="directions-grid">
                {DIRECTIONS.map(dir => (
                  <CountCard key={dir.key} label={dir.label} value={result.car_counts[dir.key]} icon={dir.icon} />
                ))}
              </div>
              <p className="muted" style={{ marginTop: '8px' }}>
                GA params: green {GA_PARAMS.greenMin}-{GA_PARAMS.greenMax}s, cycle {GA_PARAMS.cycleTime}s, pop {GA_PARAMS.population}, generations {GA_PARAMS.generations}, mutation {GA_PARAMS.mutationRate}.
              </p>
            </div>
          )}
          {result && result.error && <p className="error-text">{result.error}</p>}
        </section>

        <section className="panel stream-panel">
          <h2>3. Live Processing (4 Directions)</h2>
          <p className="muted">Streams start as soon as you run optimization. Each pane shows the annotated frames for its direction.</p>
          {!showStreams && <p className="muted">Run optimization to start live previews.</p>}
          {showStreams && (
            <div className="streams-grid">
              {DIRECTIONS.map((dir, idx) => (
                <div className="stream-card" key={dir.key}>
                  <div className="dir-header">{dir.icon}<span>{dir.label}</span></div>
                  {streamToken ? (
                    <img
                      src={`${API_BASE}/stream/${idx}?token=${streamToken}`}
                      alt={`${dir.label} live stream`}
                      className="stream-frame"
                    />
                  ) : (
                    <div className="stream-placeholder">Waiting for upload...</div>
                  )}
                </div>
              ))}
            </div>
          )}
        </section>

        <section id="helmet" className="panel helmet-panel">
          <h2>4. Helmet Detection & Violation Tracking</h2>
          <p className="muted">Upload a video to detect bike riders, helmets, and no-helmet cases for safety compliance.</p>
          <div className="actions" style={{ marginBottom: '16px' }}>
            <input id="helmet-input" type="file" accept="video/*" onChange={onHelmetInputChange} hidden />
            <button type="button" className="outline" onClick={() => document.getElementById('helmet-input').click()}>Select Video</button>
            <button onClick={clearHelmet} disabled={helmetLoading}>Clear</button>
            <button className="primary" onClick={submitHelmet} disabled={!helmetFile || helmetLoading}>{helmetLoading ? 'Detecting...' : 'Run Detection'}</button>
          </div>
          {helmetFile && <p className="muted">Selected: {helmetFile.name}</p>}
          {helmetError && <ErrorBanner message={helmetError} onDismiss={() => setHelmetError(null)} />}
          {helmetLoading && <Loader text="Analyzing video for helmets..." />}
          {helmetResult && !helmetLoading && (
            <>
              {helmetResult.violations && helmetResult.violations.length > 0 && (
                <div className="violations-section">
                  <h3>⚠️ Violations Detected ({helmetResult.violations.length})</h3>
                  <div className="violations-grid">
                    {helmetResult.violations.map((violation, idx) => (
                      <ViolationCard key={idx} violation={violation} />
                    ))}
                  </div>
                </div>
              )}
              {helmetResult.violations && helmetResult.violations.length === 0 && (
                <p className="no-violations">✅ No violations detected in this video.</p>
              )}
            </>
          )}
        </section>

        <section id="violations" className="panel violations-panel">
          <h2>Violation Records</h2>
          <p className="muted">Persisted violations stored in Supabase.</p>
          {violationsError && <ErrorBanner message={violationsError} onDismiss={() => setViolationsError(null)} />}
          {violationsLoading && <Loader text="Loading violations from Supabase..." />}
          {!violationsLoading && violations.length === 0 && !violationsError && (
            <p className="muted">No records yet. Run helmet detection to log violations.</p>
          )}
          {!violationsLoading && violations.length > 0 && (
            <div className="violations-grid">
              {violations.map((violation, idx) => (
                <ViolationCard key={violation.id || idx} violation={violation} />
              ))}
            </div>
          )}
        </section>

        <section className="panel info-panel">
          <h2>Methodology</h2>
            <ul className="info-list">
              <li>YOLOv4-tiny counts vehicle peaks over rolling 30s windows.</li>
              <li>Genetic Algorithm searches green time allocations within cycle constraints.</li>
              <li>Objective minimizes combined delay using calibrated fitness function.</li>
              <li>Recommended times are integers (seconds) capped at 60s per phase.</li>
              <li>YOLOv8 detects helmets, no-helmets, and riders for safety.</li>
            </ul>
        </section>
      </main>
    </div>
  );
}

export default App;
