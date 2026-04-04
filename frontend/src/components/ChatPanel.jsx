import { useEffect, useRef, useState } from 'react'
import { useRichStream } from '../hooks/useRichStream'
import { MessageBubble } from './MessageBubble'
import { MetricsBar } from './MetricsBar'
import { SourceDrawer } from './SourceDrawer'

export function ChatPanel({
  modelLabel, modelSub, modalUrl,
  question, k, mode, triggerCount,
  isBest, onMetrics, onAnswer,
}) {
  const { stream, isStreaming, answer, metrics, error, reset } = useRichStream(modalUrl)

  // TTFT race timer — useRef + requestAnimationFrame, never setInterval
  const rafRef        = useRef(null)
  const startRef      = useRef(null)
  const firstTokenRef = useRef(false)
  const [ttftLive, setTtftLive] = useState(null)

  // Start timer + kick stream when a new question is submitted
  useEffect(() => {
    if (triggerCount > 0 && question) {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      firstTokenRef.current = false
      startRef.current = performance.now()
      setTtftLive(0)

      function tick() {
        setTtftLive(Math.round(performance.now() - startRef.current))
        rafRef.current = requestAnimationFrame(tick)
      }
      rafRef.current = requestAnimationFrame(tick)

      stream(question, k, 400, mode)
    }
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
    }
  }, [triggerCount]) // eslint-disable-line react-hooks/exhaustive-deps

  // Freeze timer on first token
  useEffect(() => {
    if (answer && !firstTokenRef.current && rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current)
      rafRef.current = null
      firstTokenRef.current = true
      setTtftLive(Math.round(performance.now() - startRef.current))
    }
  }, [answer])

  // Notify parent of final metrics
  useEffect(() => {
    if (metrics) onMetrics?.(metrics)
  }, [metrics]) // eslint-disable-line react-hooks/exhaustive-deps

  // Notify parent of final answer (when streaming ends)
  useEffect(() => {
    if (!isStreaming && answer) onAnswer?.(answer)
  }, [isStreaming]) // eslint-disable-line react-hooks/exhaustive-deps

  function handleReset() {
    reset()
    if (rafRef.current) cancelAnimationFrame(rafRef.current)
    rafRef.current = null
    startRef.current = null
    firstTokenRef.current = false
    setTtftLive(null)
  }

  return (
    <div className="panel">
      <div className="panel__header">
        <div className="panel__model-wrap">
          <span className="panel__model">{modelLabel}</span>
          {modelSub && <span className="panel__model-sub">{modelSub}</span>}
        </div>
        {(answer || error) && (
          <button className="panel__reset" onClick={handleReset}>Clear</button>
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

      <MetricsBar metrics={metrics} isBest={isBest} ttftLive={ttftLive} />
      <SourceDrawer contexts={metrics?.contexts} />
    </div>
  )
}
