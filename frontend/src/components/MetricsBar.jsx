import { Gauge, Hash, Zap } from 'lucide-react'

function Metric({ icon: Icon, label, value }) {
  return (
    <div className="metric-item">
      <Icon size={15} aria-hidden="true" />
      <span className="metric-item__label">{label}</span>
      <strong className="metric-item__value">{value}</strong>
    </div>
  )
}

export function MetricsBar({ metrics, isBest, ttftLive }) {
  const ttftDisplay = metrics?.ttft_ms ?? ttftLive

  if (ttftDisplay == null) return null

  return (
    <div className={`metrics-bar${isBest ? ' metrics-bar--best' : ''}`}>
      <div className="metrics-ttft">
        <span className="metrics-ttft__label">{metrics ? 'TTFT' : 'Live TTFT'}</span>
        <div className="metrics-ttft__row">
          <strong className="metrics-ttft__value">{ttftDisplay}</strong>
          <span className="metrics-ttft__unit">ms</span>
        </div>
      </div>

      {metrics && (
        <div className="metrics-secondary">
          <Metric icon={Gauge} label="TPOT" value={`${metrics.tpot_ms}ms`} />
          <Metric icon={Hash} label="Tokens" value={metrics.tokens} />
          <Metric icon={Zap} label="Throughput" value={`${metrics.throughput_tps} t/s`} />
        </div>
      )}
    </div>
  )
}
