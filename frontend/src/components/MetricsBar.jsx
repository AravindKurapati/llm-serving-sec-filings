export function MetricsBar({ metrics, isBest, ttftLive }) {
  const ttftDisplay = metrics?.ttft_ms ?? ttftLive

  if (ttftDisplay == null) return null

  return (
    <div className={`metrics-bar${isBest ? ' metrics-bar--best' : ''}`}>
      <div className="metrics-ttft">
        <span className="metrics-ttft__label">TTFT</span>
        <div className="metrics-ttft__row">
          <span className="metrics-ttft__value">{ttftDisplay}</span>
          <span className="metrics-ttft__unit">ms</span>
        </div>
      </div>

      {metrics && (
        <div className="metrics-secondary">
          <div className="metric-item">
            <span className="metric-item__label">TPOT</span>
            <span className="metric-item__value">{metrics.tpot_ms}ms</span>
          </div>
          <div className="metrics-divider" />
          <div className="metric-item">
            <span className="metric-item__label">Tokens</span>
            <span className="metric-item__value">{metrics.tokens}</span>
          </div>
          <div className="metrics-divider" />
          <div className="metric-item">
            <span className="metric-item__label">Throughput</span>
            <span className="metric-item__value">{metrics.throughput_tps} t/s</span>
          </div>
        </div>
      )}
    </div>
  )
}
