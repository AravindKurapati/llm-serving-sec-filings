// Hardcoded benchmark data from results/analysis.md
// Hardware: Modal A10G — 5 questions, 15 SEC 10-K filings
const INFRA = {
  ttft_p50:  { llama: 4616, mistral: 1015, max: 4616 },
  ttft_p95:  { llama: 4631, mistral: 2402, max: 4631 },
  tpot_p50:  { llama: '23.1ms', mistral: '23.5ms' },
  throughput: { llama: '28.9 t/s', mistral: '28.6 t/s' },
}

// RAGAS scores: pending — update values here when runs complete
const RAGAS = {
  llama:   { faithfulness: null, answer_relevancy: null, context_precision: null },
  mistral: { faithfulness: null, answer_relevancy: null, context_precision: null },
}

function BarRow({ model, value, max, isWinner }) {
  const pct = Math.round((value / max) * 100)
  return (
    <div className="chart-row">
      <span className={`chart-row__model${isWinner ? ' chart-row__model--winner' : ''}`}>
        {model}
      </span>
      <div className="chart-row__track">
        <div
          className={`chart-row__fill chart-row__fill--${isWinner ? 'mistral' : 'llama'}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className={`chart-row__value${isWinner ? ' chart-row__value--winner' : ''}`}>
        {value.toLocaleString()}ms
      </span>
    </div>
  )
}

function RagasCard({ model, modelId, scores }) {
  const metrics = [
    { key: 'faithfulness',       label: 'Faithfulness' },
    { key: 'answer_relevancy',   label: 'Answer Relevancy' },
    { key: 'context_precision',  label: 'Context Precision' },
  ]
  return (
    <div className="ragas-card">
      <div className="ragas-card__head">
        <div>
          <div className="ragas-card__model">{model}</div>
          <div className="ragas-card__sub">{modelId}</div>
        </div>
        <div className="pending-badge">
          <span className="pending-badge__dot" />
          Evaluation in progress
        </div>
      </div>
      <div className="ragas-metrics">
        {metrics.map(({ key, label }) => (
          <div key={key} className="ragas-metric-row">
            <span className="ragas-metric-row__name">{label}</span>
            <span className="ragas-metric-row__value">
              {scores[key] != null ? scores[key].toFixed(3) : '—'}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

export function ResultsTab() {
  return (
    <div className="results-tab">
      <div className="results-inner">

        {/* Key finding */}
        <div className="finding-callout">
          Mistral is 4.5× faster on prefill. Throughput and TPOT are
          identical — both models are memory-bandwidth bound on A10G.
        </div>

        {/* Infrastructure benchmark */}
        <div className="results-section">
          <h2 className="results-section__heading">Latency Benchmark — A10G GPU</h2>
          <div className="results-card">
            <div className="chart-area">

              <div className="chart-group">
                <div className="chart-group__label">TTFT · p50 (lower is faster)</div>
                <BarRow
                  model="LLaMA 3.1 8B"
                  value={INFRA.ttft_p50.llama}
                  max={INFRA.ttft_p50.max}
                  isWinner={false}
                />
                <BarRow
                  model="Mistral 7B"
                  value={INFRA.ttft_p50.mistral}
                  max={INFRA.ttft_p50.max}
                  isWinner={true}
                />
              </div>

              <div className="chart-group">
                <div className="chart-group__label">TTFT · p95 tail (lower is faster)</div>
                <BarRow
                  model="LLaMA 3.1 8B"
                  value={INFRA.ttft_p95.llama}
                  max={INFRA.ttft_p95.max}
                  isWinner={false}
                />
                <BarRow
                  model="Mistral 7B"
                  value={INFRA.ttft_p95.mistral}
                  max={INFRA.ttft_p95.max}
                  isWinner={true}
                />
              </div>

            </div>

            <div className="tie-table">
              <table>
                <thead>
                  <tr>
                    <th>Generation metrics</th>
                    <th>LLaMA 3.1 8B</th>
                    <th>Mistral 7B</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>
                      TPOT p50
                      <span className="tie-badge">Tied</span>
                    </td>
                    <td>{INFRA.tpot_p50.llama}</td>
                    <td>{INFRA.tpot_p50.mistral}</td>
                  </tr>
                  <tr>
                    <td>
                      Throughput avg
                      <span className="tie-badge">Tied</span>
                    </td>
                    <td>{INFRA.throughput.llama}</td>
                    <td>{INFRA.throughput.mistral}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* RAGAS quality scores */}
        <div className="results-section">
          <h2 className="results-section__heading">RAGAS Evaluation</h2>
          <div className="ragas-grid">
            <RagasCard
              model="LLaMA 3.1 8B"
              modelId="meta-llama/Meta-Llama-3.1-8B-Instruct"
              scores={RAGAS.llama}
            />
            <RagasCard
              model="Mistral 7B"
              modelId="mistralai/Mistral-7B-Instruct-v0.3"
              scores={RAGAS.mistral}
            />
          </div>
        </div>

      </div>
    </div>
  )
}
