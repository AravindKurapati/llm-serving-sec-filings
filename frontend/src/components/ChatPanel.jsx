import { useEffect } from 'react'
import { useStream } from '../hooks/useStream'
import { MessageBubble } from './MessageBubble'
import { MetricsBar } from './MetricsBar'
import { SourceDrawer } from './SourceDrawer'

export function ChatPanel({ modelLabel, modelSub, modalUrl, question, k, triggerCount, isBest, onMetrics }) {
  const { stream, isStreaming, answer, metrics, error, reset } = useStream(modalUrl)

  useEffect(() => {
    if (triggerCount > 0 && question) {
      stream(question, k, 400)
    }
  }, [triggerCount]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (metrics) onMetrics?.(metrics)
  }, [metrics]) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="panel">
      <div className="panel__header">
        <div className="panel__model-wrap">
          <span className="panel__model">{modelLabel}</span>
          {modelSub && <span className="panel__model-sub">{modelSub}</span>}
        </div>
        {(answer || error) && (
          <button className="panel__reset" onClick={reset}>Clear</button>
        )}
      </div>

      <div className="panel__messages">
        {question && triggerCount > 0 && (
          <MessageBubble role="user" content={question} isStreaming={false} />
        )}
        {answer && (
          <MessageBubble role="assistant" content={answer} isStreaming={isStreaming} />
        )}
        {isStreaming && !answer && (
          <div className="panel__thinking">Thinking…</div>
        )}
        {error && (
          <div className="panel__error">Error: {error}</div>
        )}
      </div>

      <MetricsBar metrics={metrics} isBest={isBest} />
      <SourceDrawer contexts={metrics?.contexts} />
    </div>
  )
}
