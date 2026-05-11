import { useEffect, useRef, useState } from 'react'
import { Bot, CheckCircle2, FileSearch, RotateCcw, SignalHigh } from 'lucide-react'
import { useRichStream } from '../hooks/useRichStream'
import { MessageBubble } from './MessageBubble'
import { MetricsBar } from './MetricsBar'
import { SourceDrawer } from './SourceDrawer'

const COLD_START_WARNING_MS = 15_000
const HARD_ABORT_MS = 240_000

export function ChatPanel({
  modelKey,
  modelLabel,
  modelSub,
  modalUrl,
  question,
  k,
  maxTokens,
  mode,
  triggerCount,
  isBest,
  onMetrics,
  onAnswer,
}) {
  const { stream, isStreaming, answer, metrics, error, reset } = useRichStream(modalUrl)
  const rafRef = useRef(null)
  const startRef = useRef(null)
  const firstTokenRef = useRef(false)
  const coldTimerRef = useRef(null)
  const abortTimerRef = useRef(null)
  const [ttftLive, setTtftLive] = useState(null)
  const [coldStartMsg, setColdStartMsg] = useState(null)

  useEffect(() => {
    if (triggerCount > 0 && question) {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      clearTimeout(coldTimerRef.current)
      clearTimeout(abortTimerRef.current)
      firstTokenRef.current = false
      startRef.current = performance.now()
      setTtftLive(0)
      setColdStartMsg('Connecting to Modal stream…')

      coldTimerRef.current = setTimeout(() => {
        setColdStartMsg('Cold starting Modal GPU — this may take 2–3 min…')
      }, COLD_START_WARNING_MS)

      abortTimerRef.current = setTimeout(() => {
        reset()
        setColdStartMsg(null)
      }, HARD_ABORT_MS)

      function tick() {
        setTtftLive(Math.round(performance.now() - startRef.current))
        rafRef.current = requestAnimationFrame(tick)
      }

      rafRef.current = requestAnimationFrame(tick)
      stream(question, k, maxTokens, mode)
    }

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
    }
  }, [triggerCount]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (answer && !firstTokenRef.current && rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current)
      rafRef.current = null
      firstTokenRef.current = true
      setTtftLive(Math.round(performance.now() - startRef.current))
      clearTimeout(coldTimerRef.current)
      clearTimeout(abortTimerRef.current)
      setColdStartMsg(null)
    }
  }, [answer])

  useEffect(() => {
    if (metrics) onMetrics?.(metrics)
  }, [metrics]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!isStreaming && answer) onAnswer?.(answer)
  }, [isStreaming]) // eslint-disable-line react-hooks/exhaustive-deps

  function handleReset() {
    reset()
    if (rafRef.current) cancelAnimationFrame(rafRef.current)
    clearTimeout(coldTimerRef.current)
    clearTimeout(abortTimerRef.current)
    rafRef.current = null
    startRef.current = null
    firstTokenRef.current = false
    setTtftLive(null)
    setColdStartMsg(null)
  }

  const status = error ? 'Issue' : isStreaming ? 'Streaming' : answer ? 'Complete' : 'Ready'

  return (
    <article className={`panel panel--${modelKey}${isBest ? ' panel--best' : ''}`}>
      <div className="panel__header">
        <div className="panel__identity">
          <div className="panel__badge">
            <Bot size={18} aria-hidden="true" />
          </div>
          <div className="panel__model-wrap">
            <h2 className="panel__model">{modelLabel}</h2>
            {modelSub && <span className="panel__model-sub">{modelSub}</span>}
          </div>
        </div>

        <div className={`panel__status panel__status--${status.toLowerCase()}`}>
          {status === 'Complete' ? <CheckCircle2 size={15} aria-hidden="true" /> : <SignalHigh size={15} aria-hidden="true" />}
          <span>{status}</span>
        </div>

        {(answer || error) && (
          <button className="icon-button" onClick={handleReset} aria-label={`Clear ${modelLabel}`}>
            <RotateCcw size={16} aria-hidden="true" />
          </button>
        )}
      </div>

      <div className="panel__messages">
        {question && triggerCount > 0 ? (
          <>
            <MessageBubble role="user" content={question} isStreaming={false} />
            {answer && (
              <MessageBubble role="assistant" content={answer} isStreaming={isStreaming} />
            )}
            {isStreaming && !answer && (
              <div className="panel__thinking">
                <span />
                {coldStartMsg ?? 'Reading filings and preparing the first token'}
              </div>
            )}
            {error && (
              <div className="panel__error">Error: {error}</div>
            )}
          </>
        ) : (
          <div className="panel__empty">
            <FileSearch className="panel__empty-icon" size={30} aria-hidden="true" />
            <strong>Ready for filing context</strong>
            <span>Results will stream here with citations and latency telemetry.</span>
          </div>
        )}
      </div>

      <MetricsBar metrics={metrics} isBest={isBest} ttftLive={ttftLive} />
      <SourceDrawer contexts={metrics?.contexts} />
    </article>
  )
}
