import { useState, useRef, useCallback } from 'react'

export function useStream(modalUrl) {
  const [answer, setAnswer]       = useState('')
  const [metrics, setMetrics]     = useState(null)
  const [isStreaming, setStreaming] = useState(false)
  const [error, setError]         = useState(null)
  const abortRef = useRef(null)
  const decoder  = useRef(new TextDecoder())

  const reset = useCallback(() => {
    if (abortRef.current) abortRef.current.abort()
    setAnswer('')
    setMetrics(null)
    setError(null)
    setStreaming(false)
  }, [])

  const stream = useCallback(async (question, k = 5, maxTokens = 400) => {
    reset()
    const controller = new AbortController()
    abortRef.current = controller

    setStreaming(true)

    try {
      const res = await fetch(modalUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, k, max_tokens: maxTokens }),
        signal: controller.signal,
      })

      if (!res.ok) throw new Error(`HTTP ${res.status}`)

      const reader = res.body.getReader()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.current.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() // keep incomplete last line

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const data = line.slice(6).trim()
          if (data === '[DONE]') {
            setStreaming(false)
            return
          }
          try {
            const parsed = JSON.parse(data)
            if (parsed.type === 'metrics') {
              setMetrics(parsed)
            } else if (parsed.choices?.[0]?.delta?.content) {
              setAnswer(prev => prev + parsed.choices[0].delta.content)
            }
          } catch {
            // ignore malformed lines
          }
        }
      }
    } catch (err) {
      if (err.name !== 'AbortError') {
        setError(err.message)
      }
    } finally {
      setStreaming(false)
    }
  }, [modalUrl, reset])

  return { stream, isStreaming, answer, metrics, error, reset }
}
