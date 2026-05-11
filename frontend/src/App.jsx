import { useMemo, useState } from 'react'
import {
  Activity,
  BarChart3,
  BrainCircuit,
  Cpu,
  Database,
  Download,
  FileSearch,
  Layers,
  LineChart,
  MessageSquareText,
  Network,
  Send,
  ShieldCheck,
  SlidersHorizontal,
  Timer,
} from 'lucide-react'
import { ChatPanel } from './components/ChatPanel'
import { ResultsTab } from './components/ResultsTab'
import { LLAMA_URL, MISTRAL_URL } from './api'
import benchmarkImage from './assets/finsight-benchmark-dashboard.png'
import heroImage from './assets/finsight-hero.png'
import heroVariantImage from './assets/finsight-hero-variant.png'
import infraImage from './assets/finsight-source-retrieval-infra.png'
import retrievalImage from './assets/finsight-source-retrieval.png'

const SUGGESTIONS = [
  { company: 'AAPL', text: "What are Apple's main supply chain risks?" },
  { company: 'MSFT', text: 'How does Microsoft describe its cloud revenue growth?' },
  { company: 'GOOGL', text: "How has Google's advertising revenue changed over 3 years?" },
  { company: 'AMZN', text: 'What cybersecurity risks does Amazon disclose?' },
  { company: 'META', text: 'What does Meta say about AI infrastructure investment?' },
]

const HERO_STATS = [
  { label: 'Filings indexed', value: '15', sub: 'AAPL, MSFT, GOOGL, AMZN, META', icon: Database },
  { label: 'Vector chunks', value: '4,782', sub: 'BGE-small + FAISS', icon: FileSearch },
  { label: 'Median TTFT', value: '198ms', sub: 'best live run', icon: Timer },
]

const SYSTEM_CARDS = [
  {
    title: 'Retrieval Map',
    label: 'Grounding layer',
    meta: 'BGE-small vectors, ranked 10-K context, visible citations',
    image: retrievalImage,
    icon: Network,
  },
  {
    title: 'Latency Board',
    label: 'SSE telemetry',
    meta: 'TTFT, TPOT, token count, and throughput per model lane',
    image: benchmarkImage,
    icon: LineChart,
  },
  {
    title: 'Modal Stack',
    label: 'Serving path',
    meta: 'FastAPI proxy, vLLM subprocess, A10G GPU, FAISS volume',
    image: infraImage,
    icon: Cpu,
  },
]

function MetricTile({ stat }) {
  const Icon = stat.icon
  return (
    <div className="metric-tile">
      <Icon size={17} aria-hidden="true" />
      <div>
        <span className="metric-tile__label">{stat.label}</span>
        <strong>{stat.value}</strong>
        <span>{stat.sub}</span>
      </div>
    </div>
  )
}

function RuntimePill({ icon: Icon, children }) {
  return (
    <span className="runtime-pill">
      <Icon size={15} aria-hidden="true" />
      {children}
    </span>
  )
}

function SystemCard({ card }) {
  const Icon = card.icon

  return (
    <article className="system-card">
      <img src={card.image} alt="" />
      <div className="system-card__shade" />
      <div className="system-card__body">
        <span className="system-card__label">
          <Icon size={15} aria-hidden="true" />
          {card.label}
        </span>
        <strong>{card.title}</strong>
        <small>{card.meta}</small>
      </div>
    </article>
  )
}

export default function App() {
  const [activeTab, setActiveTab] = useState('chat')
  const [question, setQuestion] = useState('')
  const [k, setK] = useState(5)
  const [maxTokens, setMaxTokens] = useState(400)
  const [mode, setMode] = useState('concise')
  const [triggerCount, setTriggerCount] = useState(0)
  const [input, setInput] = useState('')
  const [llamaMetrics, setLlamaMetrics] = useState(null)
  const [mistralMetrics, setMistralMetrics] = useState(null)
  const [llamaAnswer, setLlamaAnswer] = useState('')
  const [mistralAnswer, setMistralAnswer] = useState('')

  const comparison = useMemo(() => {
    const llamaTtft = llamaMetrics?.ttft_ms
    const mistralTtft = mistralMetrics?.ttft_ms
    const bothReady = llamaTtft != null && mistralTtft != null

    if (!bothReady) {
      return { bothReady: false, llamaBest: false, mistralBest: false, label: '' }
    }

    const tied = llamaTtft === mistralTtft
    const llamaBest = !tied && llamaTtft < mistralTtft
    const mistralBest = !tied && mistralTtft < llamaTtft

    let label
    if (tied) {
      label = `Both models matched first-token latency at ${llamaTtft}ms`
    } else {
      const faster = llamaBest ? 'LLaMA' : 'Mistral'
      const ratio = llamaBest
        ? (mistralTtft / Math.max(llamaTtft, 1)).toFixed(1)
        : (llamaTtft / Math.max(mistralTtft, 1)).toFixed(1)
      label = `${faster} led first-token latency by ${ratio}x`
    }

    return { bothReady, llamaBest, mistralBest, label }
  }, [llamaMetrics, mistralMetrics])

  function switchTab(tab) {
    setActiveTab(tab)
    requestAnimationFrame(() => window.scrollTo({ top: 0, behavior: 'smooth' }))
  }

  function runQuestion(nextQuestion = input) {
    const q = nextQuestion.trim()
    if (!q) return
    setQuestion(q)
    setInput(q)
    setLlamaMetrics(null)
    setMistralMetrics(null)
    setLlamaAnswer('')
    setMistralAnswer('')
    setTriggerCount(c => c + 1)
    setActiveTab('chat')
  }

  function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      runQuestion()
    }
  }

  function handleExport() {
    const payload = {
      question,
      timestamp: new Date().toISOString(),
      retrieval_k: k,
      max_tokens: maxTokens,
      mode,
      llama: {
        answer: llamaAnswer,
        metrics: llamaMetrics,
        contexts: llamaMetrics?.contexts ?? [],
      },
      mistral: {
        answer: mistralAnswer,
        metrics: mistralMetrics,
        contexts: mistralMetrics?.contexts ?? [],
      },
    }
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `finsight_comparison_${Date.now()}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  const canExport = Boolean(question && (llamaAnswer || mistralAnswer))

  return (
    <div className="app">
      <header className="topbar">
        <button className="brand-mark" onClick={() => switchTab('chat')} aria-label="Open FinSight chat">
          <BrainCircuit size={19} aria-hidden="true" />
          <span>FinSight</span>
        </button>

        <nav className="nav-tabs" aria-label="Primary views">
          <button
            className={`nav-tab${activeTab === 'chat' ? ' nav-tab--active' : ''}`}
            onClick={() => switchTab('chat')}
          >
            <MessageSquareText size={16} aria-hidden="true" />
            <span>Chat</span>
          </button>
          <button
            className={`nav-tab${activeTab === 'results' ? ' nav-tab--active' : ''}`}
            onClick={() => switchTab('results')}
          >
            <BarChart3 size={16} aria-hidden="true" />
            <span>Results</span>
          </button>
        </nav>

        <div className="endpoint-status">
          <Activity size={15} aria-hidden="true" />
          <span>SSE benchmark rig</span>
        </div>
      </header>

      <main className="app-main">
        <section className="hero-stage" aria-labelledby="hero-title">
          <img className="hero-stage__image" src={heroImage} alt="" />
          <div className="hero-stage__shade" />
          <div className="hero-stage__content">
            <div className="hero-copy">
              <span className="eyebrow">SEC 10-K RAG Command Center</span>
              <h1 id="hero-title">FinSight</h1>
              <p>Dual-model filing intelligence with live citations, first-token telemetry, and exportable evidence from the same prompt.</p>
              <div className="hero-actions" aria-label="Runtime capabilities">
                <RuntimePill icon={ShieldCheck}>Cited answers</RuntimePill>
                <RuntimePill icon={Layers}>Two model lanes</RuntimePill>
                <RuntimePill icon={Activity}>Live SSE</RuntimePill>
              </div>
            </div>

            <div className="hero-visual" aria-label="FinSight command preview">
              <img src={heroVariantImage} alt="" />
              <div className="hero-visual__scan" />
              <div className="hero-visual__hud hero-visual__hud--top">
                <span>Warm-cache p50</span>
                <strong>198ms</strong>
              </div>
              <div className="hero-visual__hud hero-visual__hud--bottom">
                <span>Indexed corpus</span>
                <strong>15 filings</strong>
              </div>
            </div>

            <div className="hero-metrics" aria-label="Project metrics">
              {HERO_STATS.map(stat => (
                <MetricTile key={stat.label} stat={stat} />
              ))}
            </div>
          </div>
        </section>

        {activeTab === 'chat' ? (
          <section className="workspace" aria-label="FinSight comparison workspace">
            <div className="system-rail" aria-label="FinSight system views">
              {SYSTEM_CARDS.map(card => (
                <SystemCard key={card.title} card={card} />
              ))}
            </div>

            <div className="query-console">
              <div className="query-console__header">
                <div>
                  <span className="section-kicker">Mission query</span>
                  <strong>Ask once. Watch both models stream.</strong>
                </div>
                <span className="console-live">
                  <Activity size={14} aria-hidden="true" />
                  Modal endpoints armed
                </span>
              </div>

              <div className="query-console__main">
                <textarea
                  className="query-console__input"
                  rows={3}
                  placeholder="Ask about revenue, risk factors, capex, cloud growth, AI infrastructure, or cybersecurity..."
                  value={input}
                  onChange={e => setInput(e.target.value)}
                  onKeyDown={handleKey}
                />
                <button
                  className="send-button"
                  onClick={() => runQuestion()}
                  disabled={!input.trim()}
                  aria-label="Run comparison"
                >
                  <Send size={18} aria-hidden="true" />
                  <span>Run</span>
                </button>
              </div>

              <div className="query-console__footer">
                <div className="quick-prompts" aria-label="Suggested questions">
                  {SUGGESTIONS.map(s => (
                    <button
                      key={`${s.company}-${s.text}`}
                      className="prompt-chip"
                      onClick={() => runQuestion(s.text)}
                    >
                      <span>{s.company}</span>
                      {s.text}
                    </button>
                  ))}
                </div>

                <div className="control-strip" aria-label="Query controls">
                  <div className="range-control">
                    <div className="control-label">
                      <SlidersHorizontal size={15} aria-hidden="true" />
                      <span>k</span>
                      <strong>{k}</strong>
                    </div>
                    <input
                      className="range-input"
                      type="range"
                      min={2}
                      max={10}
                      step={1}
                      value={k}
                      onChange={e => setK(Number(e.target.value))}
                      aria-label="Retrieved chunks"
                    />
                  </div>

                  <label className="number-control">
                    <span>Max tokens</span>
                    <input
                      type="number"
                      min={80}
                      max={800}
                      step={20}
                      value={maxTokens}
                      onChange={e => setMaxTokens(Math.min(800, Math.max(80, Number(e.target.value) || 80)))}
                    />
                  </label>

                  <div className="mode-toggle" aria-label="Answer mode">
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

                  <button
                    className="ghost-button"
                    onClick={handleExport}
                    disabled={!canExport}
                    aria-label="Export comparison JSON"
                  >
                    <Download size={16} aria-hidden="true" />
                    <span>Export</span>
                  </button>
                </div>
              </div>
            </div>

            {comparison.bothReady && (
              <div className="winner-banner">
                <Activity size={16} aria-hidden="true" />
                <span>{comparison.label}</span>
              </div>
            )}

            <div className="panels">
              <ChatPanel
                modelKey="llama"
                modelLabel="LLaMA 3.1 8B"
                modelSub="Meta | 8B params"
                modalUrl={LLAMA_URL}
                question={question}
                k={k}
                maxTokens={maxTokens}
                mode={mode}
                triggerCount={triggerCount}
                isBest={comparison.llamaBest}
                onMetrics={setLlamaMetrics}
                onAnswer={setLlamaAnswer}
              />
              <ChatPanel
                modelKey="mistral"
                modelLabel="Mistral 7B"
                modelSub="Mistral AI | 7B params"
                modalUrl={MISTRAL_URL}
                question={question}
                k={k}
                maxTokens={maxTokens}
                mode={mode}
                triggerCount={triggerCount}
                isBest={comparison.mistralBest}
                onMetrics={setMistralMetrics}
                onAnswer={setMistralAnswer}
              />
            </div>
          </section>
        ) : (
          <ResultsTab />
        )}
      </main>
    </div>
  )
}
