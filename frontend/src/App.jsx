import { useState, useCallback } from 'react'
import { ChatPanel } from './components/ChatPanel'
import { LLAMA_URL, MISTRAL_URL } from './api'

export default function App() {
  const [question, setQuestion]         = useState('')
  const [k, setK]                       = useState(5)
  const [triggerCount, setTriggerCount] = useState(0)
  const [input, setInput]               = useState('')
  const [llamaMetrics, setLlamaMetrics] = useState(null)
  const [mistralMetrics, setMistralMetrics] = useState(null)

  function submit() {
    const q = input.trim()
    if (!q) return
    setQuestion(q)
    setLlamaMetrics(null)
    setMistralMetrics(null)
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

  return (
    <div className="app">
      <header className="header">
        <h1>FinSight <span className="header__sub">SEC 10-K RAG</span></h1>
        <label className="header__k">
          k:
          <select value={k} onChange={e => setK(Number(e.target.value))}>
            <option value={3}>3</option>
            <option value={5}>5</option>
            <option value={10}>10</option>
          </select>
        </label>
      </header>

      <div className="panels">
        <ChatPanel
          modelLabel="LLaMA 3.1 8B"
          modelSub="Meta · 8B params"
          modalUrl={LLAMA_URL}
          question={question}
          k={k}
          triggerCount={triggerCount}
          isBest={llamaBest}
          onMetrics={setLlamaMetrics}
        />
        <ChatPanel
          modelLabel="Mistral 7B"
          modelSub="Mistral AI · 7B params"
          modalUrl={MISTRAL_URL}
          question={question}
          k={k}
          triggerCount={triggerCount}
          isBest={mistralBest}
          onMetrics={setMistralMetrics}
        />
      </div>

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
    </div>
  )
}
