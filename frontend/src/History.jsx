import React, { useState, useEffect } from 'react';
import './History.css';

function History({ apiBaseUrl, onClose }) {
  const SESSION_ID = localStorage.getItem('metaphor_session_id') || 'anonymous';
  const [history,    setHistory]    = useState([]);
  const [statistics, setStatistics] = useState(null);
  const [isLoading,  setIsLoading]  = useState(true);
  const [error,      setError]      = useState('');
  const [filter,     setFilter]     = useState({ language: '', label: '' });
  const [selectedItem, setSelectedItem] = useState(null);
  const [dbUnavailable, setDbUnavailable] = useState(false);

  useEffect(() => {
    fetchHistory();
    fetchStatistics();
  }, [filter]);

  // Close on Escape
  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [onClose]);

  const fetchHistory = async () => {
    setIsLoading(true);
    setError('');
    setDbUnavailable(false);
    try {
      const params = new URLSearchParams();
      if (filter.language) params.append('language', filter.language);
      if (filter.label)    params.append('label', filter.label);
      params.append('session_id', SESSION_ID);

      // Short 8-second timeout — DB connection failures are fast-fail
      const controller = new AbortController();
      const timeoutId  = setTimeout(() => controller.abort(), 8000);

      const res = await fetch(`${apiBaseUrl}/history?${params}`, { signal: controller.signal });
      clearTimeout(timeoutId);

      if (!res.ok) throw new Error('Failed to fetch history');
      const data = await res.json();

      if (data.db_connected === false) {
        setDbUnavailable(true);
        setHistory([]);
      } else {
        setHistory(data.history || []);
      }
    } catch (err) {
      if (err.name === 'AbortError') {
        setDbUnavailable(true);  // timed out — treat as unavailable
      } else {
        setError(err.message);
      }
      setHistory([]);
    } finally {
      setIsLoading(false);
    }
  };

  const fetchStatistics = async () => {
    try {
      const controller = new AbortController();
      const timeoutId  = setTimeout(() => controller.abort(), 5000);
      const res = await fetch(`${apiBaseUrl}/statistics?session_id=${SESSION_ID}`, { signal: controller.signal });
      clearTimeout(timeoutId);
      if (res.ok) {
        const data = await res.json();
        setStatistics(data.statistics);
      }
    } catch {
      // silently ignore — DB might not be running
    }
  };

  const handleDelete = async (id) => {
    if (!window.confirm('Delete this prediction?')) return;
    try {
      const res = await fetch(`${apiBaseUrl}/history/${id}?session_id=${SESSION_ID}`, { method: 'DELETE' });
      if (!res.ok) throw new Error('Failed to delete');
      fetchHistory();
      fetchStatistics();
      setSelectedItem(null);
    } catch (err) {
      setError(err.message);
    }
  };

  const handleClearAll = async () => {
    if (!window.confirm('Clear all history? This cannot be undone.')) return;
    try {
      const res = await fetch(`${apiBaseUrl}/history`, { method: 'DELETE' });
      if (!res.ok) throw new Error('Failed to clear history');
      fetchHistory();
      fetchStatistics();
      setSelectedItem(null);
    } catch (err) {
      setError(err.message);
    }
  };

  const formatDate = (timestamp) =>
    new Date(timestamp).toLocaleString('en-IN', {
      year: 'numeric', month: 'short', day: 'numeric',
      hour: '2-digit', minute: '2-digit',
    });

  const toggleItem = (item) =>
    setSelectedItem(prev => prev?._id === item._id ? null : item);

  return (
    <div className="history-overlay" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="history-modal">

        {/* Header */}
        <div className="history-header">
          <h2>Prediction History</h2>
          <button className="close-btn" onClick={onClose} title="Close (Esc)">✕</button>
        </div>

        {/* Statistics */}
        {!dbUnavailable && statistics && (
          <div className="statistics-section">
            <div className="stat-card">
              <div className="stat-value">{statistics.total_predictions}</div>
              <div className="stat-label">Total</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{statistics.metaphor_count}</div>
              <div className="stat-label">Metaphors</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{statistics.normal_count}</div>
              <div className="stat-label">Literal</div>
            </div>
          </div>
        )}

        {/* Filters */}
        <div className="history-filters">
          <select
            value={filter.language}
            onChange={(e) => setFilter({ ...filter, language: e.target.value })}
            className="filter-select"
          >
            <option value="">All Languages</option>
            <option value="hindi">Hindi</option>
            <option value="tamil">Tamil</option>
            <option value="telugu">Telugu</option>
            <option value="kannada">Kannada</option>
          </select>

          <select
            value={filter.label}
            onChange={(e) => setFilter({ ...filter, label: e.target.value })}
            className="filter-select"
          >
            <option value="">All Types</option>
            <option value="metaphor">Metaphor</option>
            <option value="normal">Literal</option>
          </select>

          <button onClick={fetchHistory} className="refresh-btn">Refresh</button>

          {history.length > 0 && (
            <button onClick={handleClearAll} className="clear-all-btn">Clear All</button>
          )}
        </div>

        {/* Error */}
        {error && (
          <div className="history-error">{error}</div>
        )}

        {/* Content */}
        <div className="history-content">
          {isLoading ? (
            <div className="history-loading">
              <div className="loading-spinner" />
              <p>Loading…</p>
            </div>
          ) : dbUnavailable ? (
            <div className="history-empty">
              <p>History unavailable</p>
              <p className="empty-subtitle">Database is not connected. Start MongoDB to enable history.</p>
            </div>
          ) : history.length === 0 ? (
            <div className="history-empty">
              <p>No predictions yet</p>
              <p className="empty-subtitle">Analyze some text to build your history.</p>
            </div>
          ) : (
            <div className="history-list">
              {history.map((item) => (
                <div
                  key={item._id}
                  className={`history-item ${item.label}`}
                  onClick={() => toggleItem(item)}
                >
                  <div className="history-item-header">
                    <div className="history-item-text">
                      {item.text.length > 90 ? `${item.text.substring(0, 90)}…` : item.text}
                    </div>
                    <div className="history-item-meta">
                      <span className={`badge badge-${item.label}`}>
                        {item.label === 'metaphor' ? 'Metaphor' : 'Literal'}
                      </span>
                      <span className="history-item-lang">{item.language.toUpperCase()}</span>
                      {item.interpretation_language && (
                        <span className="history-item-lang-detail">
                          {item.interpretation_language.charAt(0).toUpperCase() + item.interpretation_language.slice(1)}
                        </span>
                      )}
                    </div>
                  </div>

                  <div className="history-item-footer">
                    <span className="history-item-date">{formatDate(item.timestamp)}</span>
                    <span className="history-item-confidence">
                      {(item.confidence * 100).toFixed(1)}% confidence
                    </span>
                  </div>

                  {/* Expanded */}
                  {selectedItem?._id === item._id && (
                    <div className="history-item-details">
                      <div className="detail-section">
                        <strong>Full Text</strong>
                        <p>{item.text}</p>
                      </div>

                      {item.sentences?.length > 0 ? (
                        <div className="interpretations-container">
                          {item.sentences.map((sentence, idx) => (
                            <div key={idx} className="sentence-interpretation">
                              {item.sentences.length > 1 && (
                                <div className="sentence-header">
                                  <strong>Sentence {idx + 1}:</strong> {sentence.sentence}
                                </div>
                              )}
                              {sentence.interpretations && (
                                <div className="interpretation-grid">
                                  <div className="interpretation-item">
                                    <div className="interpretation-icon">EN</div>
                                    <div>
                                      <strong>Translation</strong>
                                      <p>{sentence.interpretations.translation}</p>
                                    </div>
                                  </div>
                                  <div className="interpretation-item">
                                    <div className="interpretation-icon">Lit</div>
                                    <div>
                                      <strong>Literal</strong>
                                      <p>{sentence.interpretations.literal}</p>
                                    </div>
                                  </div>
                                  <div className="interpretation-item">
                                    <div className="interpretation-icon">Em</div>
                                    <div>
                                      <strong>Emotional</strong>
                                      <p>{sentence.interpretations.emotional}</p>
                                    </div>
                                  </div>
                                  <div className="interpretation-item">
                                    <div className="interpretation-icon">Ph</div>
                                    <div>
                                      <strong>Philosophical</strong>
                                      <p>{sentence.interpretations.philosophical}</p>
                                    </div>
                                  </div>
                                  <div className="interpretation-item">
                                    <div className="interpretation-icon">Cx</div>
                                    <div>
                                      <strong>Cultural</strong>
                                      <p>{sentence.interpretations.cultural}</p>
                                    </div>
                                  </div>
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      ) : (
                        <>
                          {item.translation && (
                            <div className="detail-section">
                              <strong>Translation</strong>
                              <p>{item.translation}</p>
                            </div>
                          )}
                          {item.explanation && (
                            <div className="detail-section">
                              <strong>Explanation</strong>
                              <p>{item.explanation}</p>
                            </div>
                          )}
                        </>
                      )}

                      <button
                        className="delete-item-btn"
                        onClick={(e) => { e.stopPropagation(); handleDelete(item._id); }}
                      >
                        Delete
                      </button>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default History;
