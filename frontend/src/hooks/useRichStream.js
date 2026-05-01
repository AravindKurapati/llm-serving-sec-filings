import { useCallback, useEffect, useRef, useState } from 'react'

const COLD_START_NOTICE_MS = 15000
const REQUEST_TIMEOUT_MS = 240000

export function useRichStream(modalUrl) {
  const [answer, setAnswer] = useState('')
  const [metrics, setMetrics] = useState(null)
  const [isStreaming, setStreaming] = useState(false)
  const [error, setError] = useState(null)
  const [statusMessage, setStatusMessage] = useState(null)
  const abortRef = useRef(null)
  const coldStartTimerRef = useRef(null)
  const timeoutRef = useRef(null)
  const decoder = useRef(new TextDecoder())

  const clearTimers = useCallback(() => {
    if (coldStartTimerRef.current) clearTimeout(coldStartTimerRef.current)
    if (timeoutRef.current) clearTimeout(timeoutRef.current)
    coldStartTimerRef.current = null
    timeoutRef.current = null
  }, [])

  const reset = useCallback(() => {
    if (abortRef.current) abortRef.current.abort()
    clearTimers()
    abortRef.current = null
    setAnswer('')
    setMetrics(null)
    setError(null)
    setStatusMessage(null)
    setStreaming(false)
  }, [clearTimers])

  useEffect(() => {
    return () => {
      if (abortRef.current) abortRef.current.abort()
      clearTimers()
    }
  }, [clearTimers])

  const stream = useCallback(async (question, k = 5, maxTokens = 400, mode = 'concise') => {
    reset()
    if (!modalUrl) {
      setError('Missing stream endpoint URL')
      return
    }

    const controller = new AbortController()
    abortRef.current = controller
    setStreaming(true)
    setStatusMessage('Connecting to Modal stream...')

    coldStartTimerRef.current = setTimeout(() => {
      setStatusMessage('Cold starting Modal GPU. First request can take 2-3 minutes.')
    }, COLD_START_NOTICE_MS)

    timeoutRef.current = setTimeout(() => {
      setError('Timed out waiting for Modal GPU startup. Please try again in a minute.')
      controller.abort()
    }, REQUEST_TIMEOUT_MS)

    function processLine(line) {
      if (!line.startsWith('data: ')) return false
      const data = line.slice(6).trim()
      if (data === '[DONE]') return true

      try {
        const parsed = JSON.parse(data)
        if (parsed.type === 'error') {
          setError(parsed.message || 'Streaming backend error')
        } else if (parsed.type === 'metrics') {
          setMetrics(parsed)
        } else if (parsed.type === 'status') {
          setStatusMessage(parsed.message || 'Backend is preparing the stream...')
        } else if (parsed.choices?.[0]?.delta?.content) {
          clearTimers()
          setStatusMessage(null)
          setAnswer(prev => prev + parsed.choices[0].delta.content)
        }
      } catch {
        // Ignore partial or malformed SSE lines.
      }
      return false
    }

    try {
      const res = await fetch(modalUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, k, max_tokens: maxTokens, mode }),
        signal: controller.signal,
      })

      if (!res.ok) {
        const details = await res.text().catch(() => '')
        let message = details || `HTTP ${res.status}`
        try {
          const parsed = JSON.parse(details)
          message = parsed.detail || message
        } catch {
          // Keep the raw response body when it is not JSON.
        }
        if (res.status === 503) {
          message = `Demo paused: ${message}`
        }
        throw new Error(message)
      }
      if (!res.body) throw new Error('Streaming response body is unavailable')

      const reader = res.body.getReader()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.current.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''

        for (const line of lines) {
          if (processLine(line)) {
            setStreaming(false)
            return
          }
        }
      }

      if (buffer) processLine(buffer)
    } catch (err) {
      if (err.name !== 'AbortError') setError(err.message)
    } finally {
      clearTimers()
      setStreaming(false)
      abortRef.current = null
    }
  }, [clearTimers, modalUrl, reset])

  return { stream, isStreaming, answer, metrics, error, statusMessage, reset }
}
