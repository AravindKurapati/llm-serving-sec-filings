import { Activity, BarChart3, CheckCircle2, Gauge, Layers, LineChart, Scale, Zap } from 'lucide-react'

const BENCHMARKS = [
  { label: 'TTFT p50', unit: 'ms', lowerBetter: true, llama: 198.3, mistral: 239.5 },
  { label: 'TTFT p95', unit: 'ms', lowerBetter: true, llama: 882.4, mistral: 1225.1 },
  { label: 'TPOT p50', unit: 'ms', lowerBetter: true, llama: 34.3, mistral: 31.6 },
  { label: 'Throughput avg', unit: 'tok/s', lowerBetter: false, llama: 27.6, mistral: 29.5 },
]

const QUALITY = [
  { label: 'Avg tokens', unit: '', lowerBetter: true, llama: 114.2, mistral: 87.2 },
  { label: 'Avg citations', unit: '', lowerBetter: false, llama: 5.4, mistral: 4.8 },
  { label: 'Repetition score', unit: '', lowerBetter: true, llama: 0.0107, mistral: 0.0029, precision: 4 },
]

const CONCURRENCY = [
  { level: 1, llama: '4.8s', mistral: '6.6s', note: 'single user' },
  { level: 4, llama: '17.8s', mistral: '16.6s', note: 'queued prefill' },
  { level: 8, llama: '20.8s', mistral: '28.0s', note: 'single GPU ceiling' },
]

function formatValue(value, metric) {
  const precision = metric.precision ?? (Number.isInteger(value) ? 0 : 1)
  const formatted = typeof value === 'number' ? value.toFixed(precision).replace(/\.0$/, '') : value
  return metric.unit ? `${formatted} ${metric.unit}` : formatted
}

function winnerFor(metric) {
  if (metric.llama === metric.mistral) return 'Tie'
  if (metric.lowerBetter) return metric.llama < metric.mistral ? 'LLaMA' : 'Mistral'
  return metric.llama > metric.mistral ? 'LLaMA' : 'Mistral'
}

function CompareRow({ metric }) {
  const winner = winnerFor(metric)
  const max = Math.max(metric.llama, metric.mistral)
  const llamaWidth = Math.max(6, Math.round((metric.llama / max) * 100))
  const mistralWidth = Math.max(6, Math.round((metric.mistral / max) * 100))

  return (
    <div className="compare-row">
      <div className="compare-row__head">
        <span>{metric.label}</span>
        <strong>{winner}</strong>
      </div>
      <div className="compare-bars">
        <div className={`compare-bar${winner === 'LLaMA' ? ' compare-bar--winner' : ''}`}>
          <span>LLaMA</span>
          <div><i style={{ width: `${llamaWidth}%` }} /></div>
          <strong>{formatValue(metric.llama, metric)}</strong>
        </div>
        <div className={`compare-bar compare-bar--mistral${winner === 'Mistral' ? ' compare-bar--winner' : ''}`}>
          <span>Mistral</span>
          <div><i style={{ width: `${mistralWidth}%` }} /></div>
          <strong>{formatValue(metric.mistral, metric)}</strong>
        </div>
      </div>
    </div>
  )
}

function Insight({ icon: Icon, label, value }) {
  return (
    <div className="insight">
      <Icon size={18} aria-hidden="true" />
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

export function ResultsTab() {
  return (
    <section className="results-tab" aria-label="Benchmark results">
      <div className="results-hero">
        <div>
          <span className="eyebrow">Live A10G Evidence</span>
          <h2>Infrastructure is close. Output behavior is the story.</h2>
        </div>
        <div className="results-hero__insights">
          <Insight icon={Activity} label="Best p50 TTFT" value="198ms" />
          <Insight icon={Zap} label="Best throughput" value="29.5 tok/s" />
          <Insight icon={CheckCircle2} label="Error rate under load" value="0%" />
        </div>
      </div>

      <div className="results-grid">
        <section className="result-panel result-panel--wide">
          <div className="section-title">
            <BarChart3 size={18} aria-hidden="true" />
            <h3>Latency And Throughput</h3>
          </div>
          <div className="compare-stack">
            {BENCHMARKS.map(metric => (
              <CompareRow key={metric.label} metric={metric} />
            ))}
          </div>
        </section>

        <section className="result-panel">
          <div className="section-title">
            <Scale size={18} aria-hidden="true" />
            <h3>Answer Quality</h3>
          </div>
          <div className="quality-stack">
            {QUALITY.map(metric => (
              <CompareRow key={metric.label} metric={metric} />
            ))}
          </div>
        </section>

        <section className="result-panel">
          <div className="section-title">
            <Gauge size={18} aria-hidden="true" />
            <h3>Load Behavior</h3>
          </div>
          <div className="load-table">
            <div className="load-table__head">
              <span>Users</span>
              <span>LLaMA</span>
              <span>Mistral</span>
            </div>
            {CONCURRENCY.map(row => (
              <div className="load-table__row" key={row.level}>
                <span>{row.level}x</span>
                <strong>{row.llama}</strong>
                <strong>{row.mistral}</strong>
                <em>{row.note}</em>
              </div>
            ))}
          </div>
        </section>

        <section className="result-panel result-panel--wide">
          <div className="section-title">
            <Layers size={18} aria-hidden="true" />
            <h3>Backend Upgrade Path</h3>
          </div>
          <div className="backend-list">
            <div>
              <LineChart size={17} aria-hidden="true" />
              <span>Expose queue depth and cold-start state in the metrics SSE event.</span>
            </div>
            <div>
              <LineChart size={17} aria-hidden="true" />
              <span>Add reranking after FAISS retrieval for cleaner citations at lower k.</span>
            </div>
            <div>
              <LineChart size={17} aria-hidden="true" />
              <span>Scale Modal GPU containers horizontally for concurrent users.</span>
            </div>
          </div>
        </section>
      </div>
    </section>
  )
}
