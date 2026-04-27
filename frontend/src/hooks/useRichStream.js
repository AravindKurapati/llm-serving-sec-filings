import { useCallback, useEffect, useRef, useState } from 'react'

export function useRichStream(modalUrl) {
  const [answer, setAnswer] = useState('')
  const [metrics, setMetrics] = useState(null)
  const [isStreaming, setStreaming] = useState(false)
  const [error, setError] = useState(null)
  const abortRef = useRef(null)
  const decoder = useRef(new TextDecoder())

  const reset = useCallback(() => {
    if (abortRef.current) abortRef.current.abort()
    abortRef.current = null
    setAnswer('')
    setMetrics(null)
    setError(null)
    setStreaming(false)
  }, [])

  useEffect(() => {
    return () => {
      if (abortRef.current) abortRef.current.abort()
    }
  }, [])

  const stream = useCallback(async (question, k = 5, maxTokens = 400, mode = 'concise') => {
    reset()
    if (!modalUrl) {
      setError('Missing stream endpoint URL')
      return
    }

    const controller = new AbortController()
    abortRef.current = controller
    setStreaming(true)

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
        } else if (parsed.choices?.[0]?.delta?.content) {
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
        throw new Error(details || `HTTP ${res.status}`)
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
      setStreaming(false)
      abortRef.current = null
    }
  }, [modalUrl, reset])

  return { stream, isStreaming, answer, metrics, error, reset }
}
