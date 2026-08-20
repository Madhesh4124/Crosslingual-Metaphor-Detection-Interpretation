import React, { useState, useRef, useEffect } from 'react';
import './App.css';
import VirtualKeyboard from './VirtualKeyboard';
import History from './History';

// ── Fetch with timeout + retry ────────────────────
const fetchWithRetry = async (url, options, retries = 3) => {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 90000);
  try {
    const response = await fetch(url, { ...options, signal: controller.signal });
    clearTimeout(timeoutId);
    return response;
  } catch (error) {
    clearTimeout(timeoutId);
    if (error.name === 'AbortError') throw new Error('Request timed out. The server is taking too long to respond.');
    if (retries > 0) {
      await new Promise(r => setTimeout(r, 1000));
      return fetchWithRetry(url, options, retries - 1);
    }
    throw error;
  }
};

function App() {
  const [inputText,         setInputText]         = useState('');
  const [isRecording,       setIsRecording]       = useState(false);
  const [isLoading,         setIsLoading]         = useState(false);
  const [result,            setResult]            = useState(null);
  const [isParagraph,       setIsParagraph]       = useState(false);
  const [error,             setError]             = useState('');
  const [translation,       setTranslation]       = useState('');
  const [speechLang,        setSpeechLang]        = useState('hi-IN');
  const [interpretationLang,setInterpretationLang]= useState('english');
  const [showKeyboard,      setShowKeyboard]      = useState(false);
  const [keyboardLang,      setKeyboardLang]      = useState('hindi');
  const [showHistory,       setShowHistory]       = useState(false);
  const [theme,             setTheme]             = useState(() => localStorage.getItem('theme') || 'dark');

  const textareaRef    = useRef(null);
  const recognitionRef = useRef(null);

  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

  const SESSION_ID = (() => {
    let id = localStorage.getItem('metaphor_session_id');
    if (!id) { id = crypto.randomUUID(); localStorage.setItem('metaphor_session_id', id); }
    return id;
  })();

  // Speech recognition setup
  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) return;
    recognitionRef.current = new SpeechRecognition();
    recognitionRef.current.continuous     = true;
    recognitionRef.current.interimResults = true;
    recognitionRef.current.onstart  = () => { setIsRecording(true);  setError(''); };
    recognitionRef.current.onresult = (e) => {
      let t = '';
      for (let i = 0; i < e.results.length; i++) t += e.results[i][0].transcript;
      setInputText(t);
    };
    recognitionRef.current.onerror = (e) => setError(`Speech recognition error: ${e.error}`);
    recognitionRef.current.onend   = () => setIsRecording(false);
  }, []);

  // Theme persistence
  useEffect(() => {
    document.body.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
  }, [theme]);

  const toggleTheme    = () => setTheme(t => t === 'dark' ? 'light' : 'dark');
  const handleInputChange = (e) => { setInputText(e.target.value); setError(''); };

  // Predict
  const executePredict = async (text, lang) => {
    if (!text.trim()) return;
    setIsLoading(true); setError(''); setResult(null); setTranslation('');
    try {
      const res = await fetchWithRetry(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, interpretation_language: lang, session_id: SESSION_ID }),
      });
      if (!res.ok) { const d = await res.json(); throw new Error(d.detail || 'Prediction failed'); }
      const data = await res.json();
      setResult(data);
      setIsParagraph(data.is_paragraph || false);
      setTranslation(data.translation);
    } catch (err) {
      setError(err.message || 'An error occurred. Please try again later.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!inputText.trim()) { setError('Please enter some text or use the microphone'); return; }
    executePredict(inputText, interpretationLang);
  };

  const handleMicrophoneClick = () => {
    if (!recognitionRef.current) { setError('Speech recognition not supported. Please use Chrome, Edge, or Safari.'); return; }
    if (isRecording) {
      recognitionRef.current.stop();
    } else {
      setError('');
      recognitionRef.current.lang = speechLang;
      recognitionRef.current.start();
    }
  };

  const handleReset = () => { setInputText(''); setResult(null); setError(''); setTranslation(''); setIsParagraph(false); };

  const handleKeyboardToggle = (lang) => { setKeyboardLang(lang); if (!showKeyboard) setShowKeyboard(true); };
  const handleVirtualKeyPress = (key) => {
    if (key === 'BACKSPACE') setInputText(p => p.slice(0, -1));
    else setInputText(p => p + key);
    textareaRef.current?.focus();
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="header-content">
          <div>
            <h1>MetaphorMind</h1>
            <p className="subtitle">AI-powered metaphor detection for Hindi, Tamil, Telugu &amp; Kannada</p>
          </div>
          <div className="header-buttons">
            <button className="icon-btn theme-toggle-btn" onClick={toggleTheme} title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}>
              {theme === 'dark' ? 'Light' : 'Dark'}
            </button>
            <button className="icon-btn history-btn" onClick={() => setShowHistory(true)}>
              History
            </button>
          </div>
        </div>
      </header>

      {/* Main Card */}
      <div className="main-card">
        <form onSubmit={handleSubmit}>
          {/* Input */}
          <div className="input-section">
            <div className="input-group">
              <textarea
                ref={textareaRef}
                className="text-input"
                placeholder="Enter text in Hindi, Tamil, Telugu, or Kannada…"
                value={inputText}
                onChange={handleInputChange}
                rows={4}
                disabled={isLoading}
              />

              {/* Controls */}
              <div className="controls-row">
                <div className="select-group">
                  <label htmlFor="speech-lang">Voice</label>
                  <select id="speech-lang" value={speechLang} onChange={e => setSpeechLang(e.target.value)} disabled={isLoading || isRecording}>
                    <option value="hi-IN">Hindi</option>
                    <option value="ta-IN">Tamil</option>
                    <option value="te-IN">Telugu</option>
                    <option value="kn-IN">Kannada</option>
                  </select>
                </div>

                <div className="select-group">
                  <label htmlFor="interpretation-lang">Output</label>
                  <select id="interpretation-lang" value={interpretationLang} onChange={e => setInterpretationLang(e.target.value)} disabled={isLoading}>
                    <option value="english">English</option>
                    <option value="hindi">हिंदी</option>
                    <option value="tamil">தமிழ்</option>
                    <option value="telugu">తెలుగు</option>
                    <option value="kannada">ಕನ್ನಡ</option>
                  </select>
                </div>
              </div>

              {/* Keyboard selector */}
              <div className="keyboard-selector">
                <span className="keyboard-selector-label">Script</span>
                <div className="keyboard-buttons">
                  {[['hindi','हिंदी'],['tamil','தமிழ்'],['telugu','తెలుగు'],['kannada','ಕನ್ನಡ']].map(([lang, label]) => (
                    <button
                      key={lang}
                      type="button"
                      className={`keyboard-btn ${showKeyboard && keyboardLang === lang ? 'active' : ''}`}
                      onClick={() => handleKeyboardToggle(lang)}
                      disabled={isLoading}
                    >{label}</button>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Recording indicator */}
          {isRecording && (
            <div className="recording-indicator">
              <span className="recording-dot" />
              Recording… click Stop when done
            </div>
          )}

          {/* Error */}
          {error && <div className="error-message">⚠️ {error}</div>}

          {/* Actions */}
          <div className="input-section">
            <div className="button-group">
              <button type="button" className={`mic-button ${isRecording ? 'recording' : ''}`} onClick={handleMicrophoneClick} disabled={isLoading}>
                {isRecording ? 'Stop' : 'Record'}
              </button>
              <button type="submit" className="submit-button" disabled={isLoading || !inputText.trim()}>
                {isLoading ? (<><span className="loading-spinner" /> Analyzing…</>) : 'Analyze'}
              </button>
              <button type="button" className="reset-button" onClick={handleReset} disabled={isLoading}>
                Reset
              </button>
            </div>
          </div>
        </form>

        {/* Divider */}
        {result && <div className="section-divider" />}

        {/* Results */}
        {result && (
          <div className="result-section">
            {/* Banner */}
            <div className={`overall-result ${result.label === 'metaphor' ? 'result-metaphor' : result.label === 'neutral' ? 'result-neutral' : 'result-normal'}`}>
              <h2>
                {isParagraph ? 'Paragraph' : 'Sentence'} — {result.label === 'metaphor' ? 'Metaphor' : result.label === 'neutral' ? 'Neutral' : 'Literal'}
              </h2>
              <span className="confidence-badge">{(result.confidence * 100).toFixed(1)}% confidence</span>
            </div>

            {/* Sentence cards */}
            {result.sentences?.length > 0 && (
              <div className="sentences-container">
                {result.sentences.map((sentence, idx) => (
                  <div key={idx} className="sentence-card">
                    {isParagraph && (
                      <div className="sentence-header">
                        <h3>{sentence.sentence}</h3>
                        <div className="sentence-meta">
                          <span className={`label-badge ${sentence.label === 'metaphor' ? 'badge-metaphor' : 'badge-normal'}`}>
                            {sentence.label === 'metaphor' ? 'Metaphor' : 'Literal'}
                          </span>
                          <span className="confidence-text">{(sentence.confidence * 100).toFixed(1)}%</span>
                        </div>
                      </div>
                    )}

                    {sentence.interpretations && (
                      <div className="interpretations">
                        <h4>Interpretations</h4>

                        {/* XAI */}
                        {sentence.decision_reasoning && (
                          <div className="xai-proof-box">
                            <div className="xai-header">
                              <span className="xai-icon">◈</span>
                              <strong>Model Decision Proof</strong>
                            </div>
                            <p className="xai-reasoning">{sentence.decision_reasoning}</p>
                            {sentence.word_attributions?.length > 0 && (
                              <div className="xai-tokens-container">
                                <span className="xai-tokens-label">Token Attributions</span>
                                <div className="xai-token-badges">
                                  {sentence.word_attributions.map((token, tIdx) => (
                                    <span
                                      key={tIdx}
                                      className={`xai-token-chip ${token.is_key_trigger ? 'key-trigger' : ''}`}
                                      title={`Gradient Saliency: ${token.score}%`}
                                    >
                                      {token.word}<small>{token.score}%</small>
                                    </span>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                        )}

                        {/* Translation */}
                        <div className="interpretation-item">
                          <div className="interpretation-icon">EN</div>
                          <div className="interpretation-content">
                            <strong>Translation</strong>
                            <p>{sentence.interpretations.translation}</p>
                          </div>
                        </div>

                        {/* Literal */}
                        <div className="interpretation-item">
                          <div className="interpretation-icon">Lit</div>
                          <div className="interpretation-content">
                            <strong>Literal</strong>
                            <p>{sentence.interpretations.literal}</p>
                          </div>
                        </div>

                        {/* Emotional */}
                        <div className="interpretation-item">
                          <div className="interpretation-icon">Em</div>
                          <div className="interpretation-content">
                            <strong>Emotional</strong>
                            <p>{sentence.interpretations.emotional}</p>
                          </div>
                        </div>

                        {/* Philosophical */}
                        <div className="interpretation-item">
                          <div className="interpretation-icon">Ph</div>
                          <div className="interpretation-content">
                            <strong>Philosophical</strong>
                            <p>{sentence.interpretations.philosophical}</p>
                          </div>
                        </div>

                        {/* Cultural */}
                        <div className="interpretation-item">
                          <div className="interpretation-icon">Cx</div>
                          <div className="interpretation-content">
                            <strong>Cultural</strong>
                            <p>{sentence.interpretations.cultural}</p>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        <footer className="app-footer">
          <p>Supports Hindi · Tamil · Telugu · Kannada</p>
        </footer>
      </div>

      {/* Virtual Keyboard */}
      {showKeyboard && (
        <VirtualKeyboard language={keyboardLang} onKeyPress={handleVirtualKeyPress} onClose={() => setShowKeyboard(false)} />
      )}

      {/* History Modal */}
      {showHistory && (
        <History apiBaseUrl={API_BASE_URL} onClose={() => setShowHistory(false)} />
      )}
    </div>
  );
}

export default App;
