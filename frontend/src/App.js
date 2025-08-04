import { useState, useRef } from 'react';
import './App.css';

function App() {
  const [documentUrl, setDocumentUrl] = useState('');
  const [queryInputs, setQueryInputs] = useState([{ id: 1, value: '' }]);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const processingStartTime = useRef(null);

  const addQueryBox = () => {
    setQueryInputs([...queryInputs, { id: Date.now(), value: '' }]);
  };

  const removeQueryBox = (id) => {
    if (queryInputs.length > 1) {
      setQueryInputs(queryInputs.filter(input => input.id !== id));
    }
  };

  const handleQueryChange = (id, value) => {
    setQueryInputs(queryInputs.map(input => 
      input.id === id ? { ...input, value } : input
    ));
  };

  const processAllQueries = async (e) => {
    e.preventDefault();
    
    if (!documentUrl) {
      alert('Please enter a document URL');
      return;
    }

    processingStartTime.current = new Date();
    setLoading(true);
    setResults([]);
    
    // Simulate API processing for all queries
    setTimeout(() => {
      const processingEndTime = new Date();
      const processingTime = (processingEndTime - processingStartTime.current) / 1000;
      
      const processedResults = queryInputs.map(input => ({
        query: input.value,
        eligible: input.value.length > 5 && documentUrl.startsWith('http'),
        documentUrl,
        processingTime: processingTime.toFixed(2) + 's'
      }));
      
      setResults(processedResults);
      setLoading(false);
    }, 1500);
  };

  return (
    <div className={`app ${loading ? 'loading' : ''}`}>
      <h1>Insurance Query Checker</h1>
      
      <form className="query-form">
        {/* Document URL Input */}
        <div className="form-group">
          <label htmlFor="documentUrl">Document URL:</label>
          <input
            type="url"
            id="documentUrl"
            value={documentUrl}
            onChange={(e) => setDocumentUrl(e.target.value)}
            placeholder="https://example.com/document.pdf"
            required
          />
        </div>

        {/* Dynamic Query Inputs */}
        <div className="query-inputs">
          {queryInputs.map((input, index) => (
            <div key={input.id} className="query-group">
              <label htmlFor={`query-${input.id}`}>
                Query {index + 1}:
              </label>
              <div className="input-row">
                <textarea
                  id={`query-${input.id}`}
                  value={input.value}
                  onChange={(e) => handleQueryChange(input.id, e.target.value)}
                  placeholder="Enter your insurance query..."
                  rows={3}
                  required
                />
                {queryInputs.length > 1 && (
                  <button
                    type="button"
                    onClick={() => removeQueryBox(input.id)}
                    className="remove-btn"
                    aria-label="Remove query"
                  >
                    ×
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>

        {/* Action Buttons */}
        <div className="button-group">
          <button
            type="button"
            onClick={addQueryBox}
            className="secondary"
            disabled={loading}
          >
            + Add Another Query
          </button>
          
          <button 
            type="submit" 
            onClick={processAllQueries}
            disabled={loading || !documentUrl || queryInputs.some(input => !input.value)}
          >
            {loading ? (
              <>
                <span className="spinner"></span>
                Processing...
              </>
            ) : (
              'Check All Queries'
            )}
          </button>
        </div>
      </form>

      {/* Loading Overlay */}
      {loading && (
        <div className="loading-overlay">
          <div className="spinner large"></div>
          <p>Processing your queries...</p>
        </div>
      )}

      {/* Results Display */}
      {results.length > 0 && (
        <div className="results-container">
          <h2>Processing Results</h2>
          <div className="document-info">
            <strong>Document:</strong> <a href={documentUrl} target="_blank" rel="noopener noreferrer">{documentUrl}</a>
          </div>
          
          {results.map((result, index) => (
            <div 
              key={index} 
              className={`result ${result.eligible ? 'eligible' : 'not-eligible'}`}
            >
              <h3>Query {index + 1}:</h3>
              <p>{result.query || <em>No query entered</em>}</p>
              <div className="result-details">
                <p>Status: {result.eligible ? '✅ Eligible' : '❌ Not Eligible'}</p>
                <p className="processing-time">Processed in: {result.processingTime}</p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default App;