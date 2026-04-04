import { useState } from 'react'
import { ChatPanel } from './components/ChatPanel'
import { ResultsTab } from './components/ResultsTab'
import { LLAMA_URL, MISTRAL_URL } from './api'

const SUGGESTIONS = [
  { company: 'AAPL', text: "What are Apple's main supply chain risks?" },
  { company: 'AAPL', text: "How has Apple's R&D spend changed over 3 years?" },
  { company: 'MSFT', text: "How does Microsoft describe its cloud revenue growth?" },
  { company: 'MSFT', text: "What are Microsoft's key risk factors?" },
  { company: 'GOOGL', text: "How has Google's advertising revenue changed?" },
  { company: 'GOOGL', text: "What does Alphabet say about AI investment?" },
  { company: 'AMZN', text: "What cybersecurity risks does Amazon disclose?" },
  { company: 'AMZN', text: "How does AWS contribute to Amazon's operating income?" },
  { company: 'META', text: "What does Meta say about AI infrastructure investment?" },
  { company: 'META', text: "How has Meta's headcount changed after layoffs?" },
]

export default function App() {
  const [activeTab, setActiveTab]           = useState('chat')
  const [question, setQuestion]             = useState('')
  const [k, setK]                           = useState(5)
  const [mode, setMode]                     = useState('concise')
  const [triggerCount, setTriggerCount]     = useState(0)
  const [input, setInput]                   = useState('')
  const [llamaMetrics, setLlamaMetrics]     = useState(null)
  const [mistralMetrics, setMistralMetrics] = useState(null)
  const [llamaAnswer, setLlamaAnswer]       = useState('')
  const [mistralAnswer, setMistralAnswer]   = useState('')

  function submit() {
    const q = input.trim()
    if (!q) return
    setQuestion(q)
    setLlamaMetrics(null)
    setMistralMetrics(null)
    setLlamaAnswer('')
    setMistralAnswer('')
    setTriggerCount(c => c + 1)
  }

  function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  const llamaTtft   = llamaMetrics?.ttft_ms
  const mistralTtft = mistralMetrics?.ttft_ms
  const bothReady   = llamaTtft != null && mistralTtft != null
  const llamaBest   = bothReady && llamaTtft <= mistralTtft
  const mistralBest = bothReady && mistralTtft <= llamaTtft

  function handleExport() {
    const payload = {
      question,
      timestamp: new Date().toISOString(),
      k,
      mode,
      llama: {
        answer: llamaAnswer,
        metrics: llamaMetrics
          ? {
              ttft_ms:        llamaMetrics.ttft_ms,
              tpot_ms:        llamaMetrics.tpot_ms,
              tokens:         llamaMetrics.tokens,
              throughput_tps: llamaMetrics.throughput_tps,
            }
          : null,
        contexts: llamaMetrics?.contexts ?? [],
      },
      mistral: {
        answer: mistralAnswer,
        metrics: mistralMetrics
          ? {
              ttft_ms:        mistralMetrics.ttft_ms,
              tpot_ms:        mistralMetrics.tpot_ms,
              tokens:         mistralMetrics.tokens,
              throughput_tps: mistralMetrics.throughput_tps,
            }
          : null,
        contexts: mistralMetrics?.contexts ?? [],
      },
    }
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' })
    const url  = URL.createObjectURL(blob)
    const a    = document.createElement('a')
    a.href     = url
    a.download = `finsight_comparison_${Date.now()}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="app">
      <header className="header">
        <h1>FinSight <span className="header__sub">SEC 10-K RAG</span></h1>
      </header>

      <nav className="tab-bar">
        <button
          className={`tab-btn${activeTab === 'chat' ? ' tab-btn--active' : ''}`}
          onClick={() => setActiveTab('chat')}
        >
          Chat
        </button>
        <button
          className={`tab-btn${activeTab === 'results' ? ' tab-btn--active' : ''}`}
          onClick={() => setActiveTab('results')}
        >
          Results
        </button>
      </nav>

      {activeTab === 'chat' ? (
        <>
          <div className="panels">
            <ChatPanel
              modelLabel="LLaMA 3.1 8B"
              modelSub="Meta · 8B params"
              modalUrl={LLAMA_URL}
              question={question}
              k={k}
              mode={mode}
              triggerCount={triggerCount}
              isBest={llamaBest}
              onMetrics={setLlamaMetrics}
              onAnswer={setLlamaAnswer}
            />
            <ChatPanel
              modelLabel="Mistral 7B"
              modelSub="Mistral AI · 7B params"
              modalUrl={MISTRAL_URL}
              question={question}
              k={k}
              mode={mode}
              triggerCount={triggerCount}
              isBest={mistralBest}
              onMetrics={setMistralMetrics}
              onAnswer={setMistralAnswer}
            />
          </div>

          {bothReady && (
            <div className="winner-banner">
              <span>
                {mistralBest
                  ? `Mistral was ${(llamaTtft / mistralTtft).toFixed(1)}× faster on first token`
                  : `LLaMA was ${(mistralTtft / llamaTtft).toFixed(1)}× faster on first token`}
              </span>
              <button className="export-btn" onClick={handleExport}>
                Export JSON
              </button>
            </div>
          )}

          <div className="chat-controls">
            <div className="k-slider-group">
              <div className="k-slider-group__header">
                <label className="k-slider-group__label" htmlFor="k-slider">
                  Retrieved chunks (k)
                </label>
                <span className="k-slider-group__value">{k}</span>
              </div>
              <input
                id="k-slider"
                className="k-slider"
                type="range"
                min={2}
                max={10}
                step={1}
                value={k}
                onChange={e => setK(Number(e.target.value))}
              />
              <span className="k-slider-group__hint">Higher k = more context, slower retrieval.</span>
            </div>

            <div className="mode-toggle-group">
              <span className="mode-toggle-group__label">Answer mode</span>
              <div className="mode-toggle">
                <button
                  className={`mode-toggle__btn${mode === 'concise' ? ' mode-toggle__btn--active' : ''}`}
                  onClick={() => setMode('concise')}
                >
                  Concise
                </button>
                <button
                  className={`mode-toggle__btn${mode === 'detailed' ? ' mode-toggle__btn--active' : ''}`}
                  onClick={() => setMode('detailed')}
                >
                  Detailed
                </button>
              </div>
            </div>
          </div>

          <div className="input-area">
            <div className="input-row">
              <textarea
                className="input-row__textarea"
                rows={2}
                placeholder="Ask a question about SEC 10-K filings…"
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKey}
              />
              <button
                className="input-row__submit"
                onClick={submit}
                disabled={!input.trim()}
              >
                Send
              </button>
            </div>

            <div className="suggestions">
              {SUGGESTIONS.map((s, i) => (
                <button
                  key={i}
                  className="suggestion-chip"
                  onClick={() => setInput(s.text)}
                >
                  {s.text}
                </button>
              ))}
            </div>
          </div>
        </>
      ) : (
        <ResultsTab />
      )}
    </div>
  )
}
