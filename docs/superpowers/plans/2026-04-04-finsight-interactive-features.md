# FinSight Interactive Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 5 interactive features to the FinSight React frontend — TTFT race timer, live k slider, answer mode toggle, question suggestion chips, and JSON export — implemented one at a time with visual verification between each.

**Architecture:** Features 1–4 live entirely in React (`App.jsx`, `ChatPanel.jsx`, `MetricsBar.jsx`, `index.css`, plus a new `useRichStream.js` hook). Feature 3 also touches the Python backend (`finsight.py`, `mock_stream_server.py`). Feature 5 adds state lifting for `answer` in `App.jsx` and a client-side JSON download. `useStream.js` and `api.js` are off-limits.

**Tech Stack:** React 18 (useState, useEffect, useRef, useCallback, requestAnimationFrame), Vite dev server, FastAPI (mock + real), Modal/vLLM backend

---

## Files Modified / Created

| File | Change |
|------|--------|
| `frontend/src/hooks/useRichStream.js` | **Create** — same as useStream but adds `mode` to POST body |
| `frontend/src/components/ChatPanel.jsx` | Modify — TTFT timer, mode prop, onAnswer callback, use useRichStream |
| `frontend/src/components/MetricsBar.jsx` | Modify — accept `ttftLive` prop, show counter before metrics arrive |
| `frontend/src/App.jsx` | Modify — remove header k-select, add k slider + mode toggle + suggestions + export |
| `frontend/src/index.css` | Modify — styles for slider, toggle, chips, export button |
| `scripts/mock_stream_server.py` | Modify — silently accept `mode` field |
| `v2_modal/finsight.py` | Modify — mode-aware `build_prompt` + `query_stream` + `stream_endpoint` |

---

## Task 1: Create useRichStream hook

Adds `mode` to the POST body without touching `useStream.js`. `ChatPanel` will swap to this hook in Task 2.

**Files:**
- Create: `frontend/src/hooks/useRichStream.js`

- [ ] **Step 1: Create the hook file**

```javascript
// frontend/src/hooks/useRichStream.js
import { useState, useRef, useCallback } from 'react'

export function useRichStream(modalUrl) {
  const [answer, setAnswer]        = useState('')
  const [metrics, setMetrics]      = useState(null)
  const [isStreaming, setStreaming] = useState(false)
  const [error, setError]          = useState(null)
  const abortRef = useRef(null)
  const decoder  = useRef(new TextDecoder())

  const reset = useCallback(() => {
    if (abortRef.current) abortRef.current.abort()
    setAnswer('')
    setMetrics(null)
    setError(null)
    setStreaming(false)
  }, [])

  const stream = useCallback(async (question, k = 5, maxTokens = 400, mode = 'concise') => {
    reset()
    const controller = new AbortController()
    abortRef.current = controller
    setStreaming(true)

    try {
      const res = await fetch(modalUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, k, max_tokens: maxTokens, mode }),
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
        buffer = lines.pop()

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const data = line.slice(6).trim()
          if (data === '[DONE]') { setStreaming(false); return }
          try {
            const parsed = JSON.parse(data)
            if (parsed.type === 'metrics') {
              setMetrics(parsed)
            } else if (parsed.choices?.[0]?.delta?.content) {
              setAnswer(prev => prev + parsed.choices[0].delta.content)
            }
          } catch { /* ignore malformed lines */ }
        }
      }
    } catch (err) {
      if (err.name !== 'AbortError') setError(err.message)
    } finally {
      setStreaming(false)
    }
  }, [modalUrl, reset])

  return { stream, isStreaming, answer, metrics, error, reset }
}
```

- [ ] **Step 2: Verify file exists**

```bash
ls frontend/src/hooks/
```
Expected output includes `useRichStream.js` and the original `useStream.js`.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/hooks/useRichStream.js
git commit -m "feat: add useRichStream hook with mode field in POST body"
```

---

## Task 2: TTFT race timer

Shows a live millisecond counter in each ChatPanel that starts when the user submits and freezes on first token. Uses `useRef` + `requestAnimationFrame` — no `setInterval`.

**Files:**
- Modify: `frontend/src/components/ChatPanel.jsx`
- Modify: `frontend/src/components/MetricsBar.jsx`

- [ ] **Step 1: Update MetricsBar to accept `ttftLive` prop**

Replace the entire contents of `frontend/src/components/MetricsBar.jsx`:

```jsx
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
```

- [ ] **Step 2: Update ChatPanel to use useRichStream and wire TTFT timer**

Replace the entire contents of `frontend/src/components/ChatPanel.jsx`:

```jsx
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
```

- [ ] **Step 3: Start mock server and dev server, verify timer**

In one terminal:
```bash
cd D:/Aru/NYU/llm-serving-sec-filings
uvicorn scripts.mock_stream_server:app --port 8001
```

In another:
```bash
cd D:/Aru/NYU/llm-serving-sec-filings/frontend
npm run dev
```

Open browser at http://localhost:5173 (or whatever Vite prints). Type any question and click Send.

Expected:
- The TTFT slot in MetricsBar appears immediately and counts up in ms (0, 16, 33…)
- The counter freezes the moment the first word appears in the chat bubble (≈310ms with the mock)
- After the mock finishes, the secondary metrics row (TPOT, Tokens, Throughput) appears below
- Clicking Clear resets the timer slot back to hidden

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/ChatPanel.jsx frontend/src/components/MetricsBar.jsx
git commit -m "feat: TTFT race timer with rAF counter in ChatPanel"
```

---

## Task 3: Live k slider

Removes the k dropdown from the header and replaces it with a proper slider in the Chat tab area.

**Files:**
- Modify: `frontend/src/App.jsx`
- Modify: `frontend/src/index.css`

- [ ] **Step 1: Remove the header k dropdown and add k slider + controls row in App.jsx**

In `frontend/src/App.jsx`, replace the entire file content:

```jsx
import { useState } from 'react'
import { ChatPanel } from './components/ChatPanel'
import { ResultsTab } from './components/ResultsTab'
import { LLAMA_URL, MISTRAL_URL } from './api'

const SUGGESTIONS = [
  { company: 'AAPL', text: "What are Apple's main supply chain risks?" },
  { company: 'AAPL', text: "How has Apple's R&D spend changed over 3 years?" },
  { company: 'MSFT', text: "How does Microsoft describe its cloud revenue growth?" },
  { company: 'MSFT', text: "What are Microsoft's key risk factors?" },
  { company: 'GOOGL', text: "How has Google's advertising revenue changed?" },
  { company: 'GOOGL', text: "What does Alphabet say about AI investment?" },
  { company: 'AMZN', text: "What cybersecurity risks does Amazon disclose?" },
  { company: 'AMZN', text: "How does AWS contribute to Amazon's operating income?" },
  { company: 'META', text: "What does Meta say about AI infrastructure investment?" },
  { company: 'META', text: "How has Meta's headcount changed after layoffs?" },
]

export default function App() {
  const [activeTab, setActiveTab]       = useState('chat')
  const [question, setQuestion]         = useState('')
  const [k, setK]                       = useState(5)
  const [mode, setMode]                 = useState('concise')
  const [triggerCount, setTriggerCount] = useState(0)
  const [input, setInput]               = useState('')
  const [llamaMetrics, setLlamaMetrics] = useState(null)
  const [mistralMetrics, setMistralMetrics] = useState(null)
  const [llamaAnswer, setLlamaAnswer]   = useState('')
  const [mistralAnswer, setMistralAnswer] = useState('')

  function submit() {
    const q = input.trim()
    if (!q) return
    setQuestion(q)
    setLlamaMetrics(null)
    setMistralMetrics(null)
    setLlamaAnswer('')
    setMistralAnswer('')
    setTriggerCount(c => c + 1)
  }

  function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  const llamaTtft   = llamaMetrics?.ttft_ms
  const mistralTtft = mistralMetrics?.ttft_ms
  const bothReady   = llamaTtft != null && mistralTtft != null
  const llamaBest   = bothReady && llamaTtft <= mistralTtft
  const mistralBest = bothReady && mistralTtft <= llamaTtft

  function handleExport() {
    const payload = {
      question,
      timestamp: new Date().toISOString(),
      k,
      mode,
      llama: {
        answer: llamaAnswer,
        metrics: llamaMetrics
          ? {
              ttft_ms:        llamaMetrics.ttft_ms,
              tpot_ms:        llamaMetrics.tpot_ms,
              tokens:         llamaMetrics.tokens,
              throughput_tps: llamaMetrics.throughput_tps,
            }
          : null,
        contexts: llamaMetrics?.contexts ?? [],
      },
      mistral: {
        answer: mistralAnswer,
        metrics: mistralMetrics
          ? {
              ttft_ms:        mistralMetrics.ttft_ms,
              tpot_ms:        mistralMetrics.tpot_ms,
              tokens:         mistralMetrics.tokens,
              throughput_tps: mistralMetrics.throughput_tps,
            }
          : null,
        contexts: mistralMetrics?.contexts ?? [],
      },
    }
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' })
    const url  = URL.createObjectURL(blob)
    const a    = document.createElement('a')
    a.href     = url
    a.download = `finsight_comparison_${Date.now()}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="app">
      <header className="header">
        <h1>FinSight <span className="header__sub">SEC 10-K RAG</span></h1>
      </header>

      <nav className="tab-bar">
        <button
          className={`tab-btn${activeTab === 'chat' ? ' tab-btn--active' : ''}`}
          onClick={() => setActiveTab('chat')}
        >
          Chat
        </button>
        <button
          className={`tab-btn${activeTab === 'results' ? ' tab-btn--active' : ''}`}
          onClick={() => setActiveTab('results')}
        >
          Results
        </button>
      </nav>

      {activeTab === 'chat' ? (
        <>
          <div className="panels">
            <ChatPanel
              modelLabel="LLaMA 3.1 8B"
              modelSub="Meta · 8B params"
              modalUrl={LLAMA_URL}
              question={question}
              k={k}
              mode={mode}
              triggerCount={triggerCount}
              isBest={llamaBest}
              onMetrics={setLlamaMetrics}
              onAnswer={setLlamaAnswer}
            />
            <ChatPanel
              modelLabel="Mistral 7B"
              modelSub="Mistral AI · 7B params"
              modalUrl={MISTRAL_URL}
              question={question}
              k={k}
              mode={mode}
              triggerCount={triggerCount}
              isBest={mistralBest}
              onMetrics={setMistralMetrics}
              onAnswer={setMistralAnswer}
            />
          </div>

          {bothReady && (
            <div className="winner-banner">
              <span>
                {mistralBest
                  ? `Mistral was ${(llamaTtft / mistralTtft).toFixed(1)}× faster on first token`
                  : `LLaMA was ${(mistralTtft / llamaTtft).toFixed(1)}× faster on first token`}
              </span>
              <button className="export-btn" onClick={handleExport}>
                Export JSON
              </button>
            </div>
          )}

          <div className="chat-controls">
            <div className="k-slider-group">
              <div className="k-slider-group__header">
                <label className="k-slider-group__label" htmlFor="k-slider">
                  Retrieved chunks (k)
                </label>
                <span className="k-slider-group__value">{k}</span>
              </div>
              <input
                id="k-slider"
                className="k-slider"
                type="range"
                min={2}
                max={10}
                step={1}
                value={k}
                onChange={e => setK(Number(e.target.value))}
              />
              <span className="k-slider-group__hint">Higher k = more context, slower retrieval.</span>
            </div>

            <div className="mode-toggle-group">
              <span className="mode-toggle-group__label">Answer mode</span>
              <div className="mode-toggle">
                <button
                  className={`mode-toggle__btn${mode === 'concise' ? ' mode-toggle__btn--active' : ''}`}
                  onClick={() => setMode('concise')}
                >
                  Concise
                </button>
                <button
                  className={`mode-toggle__btn${mode === 'detailed' ? ' mode-toggle__btn--active' : ''}`}
                  onClick={() => setMode('detailed')}
                >
                  Detailed
                </button>
              </div>
            </div>
          </div>

          <div className="input-area">
            <div className="input-row">
              <textarea
                className="input-row__textarea"
                rows={2}
                placeholder="Ask a question about SEC 10-K filings…"
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKey}
              />
              <button
                className="input-row__submit"
                onClick={submit}
                disabled={!input.trim()}
              >
                Send
              </button>
            </div>

            <div className="suggestions">
              {SUGGESTIONS.map((s, i) => (
                <button
                  key={i}
                  className="suggestion-chip"
                  onClick={() => setInput(s.text)}
                >
                  {s.text}
                </button>
              ))}
            </div>
          </div>
        </>
      ) : (
        <ResultsTab />
      )}
    </div>
  )
}
```

- [ ] **Step 2: Add CSS for k slider, mode toggle, chat controls, suggestions, export button**

Append the following to the END of `frontend/src/index.css` (do not remove existing styles):

```css
/* ── Chat controls (k slider + mode toggle) ────────────────── */

.chat-controls {
  display: flex;
  align-items: flex-start;
  gap: 2rem;
  padding: 0.75rem 1.5rem 0;
  background: var(--surface);
  border-top: 1px solid var(--border);
  flex-shrink: 0;
}

/* k slider */
.k-slider-group {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  min-width: 180px;
}

.k-slider-group__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.k-slider-group__label {
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--text-dim);
  letter-spacing: 0.01em;
}

.k-slider-group__value {
  font-size: 0.75rem;
  font-weight: 700;
  color: var(--accent);
  font-variant-numeric: tabular-nums;
  min-width: 1.25rem;
  text-align: right;
}

.k-slider-group__hint {
  font-size: 0.6875rem;
  color: var(--muted);
  line-height: 1.4;
}

.k-slider {
  -webkit-appearance: none;
  appearance: none;
  width: 100%;
  height: 4px;
  border-radius: 2px;
  background: var(--border);
  outline: none;
  cursor: pointer;
}

.k-slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 14px;
  height: 14px;
  border-radius: 50%;
  background: var(--accent);
  cursor: pointer;
  border: 2px solid var(--bg);
  box-shadow: 0 0 0 1px var(--accent);
  transition: box-shadow 0.15s;
}

.k-slider::-webkit-slider-thumb:hover {
  box-shadow: 0 0 0 3px var(--accent-glow);
}

.k-slider::-moz-range-thumb {
  width: 14px;
  height: 14px;
  border-radius: 50%;
  background: var(--accent);
  cursor: pointer;
  border: 2px solid var(--bg);
}

/* mode toggle */
.mode-toggle-group {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.mode-toggle-group__label {
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--text-dim);
  letter-spacing: 0.01em;
}

.mode-toggle {
  display: flex;
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  overflow: hidden;
}

.mode-toggle__btn {
  flex: 1;
  padding: 0.3125rem 0.875rem;
  background: transparent;
  border: none;
  font-size: 0.8125rem;
  font-weight: 500;
  font-family: inherit;
  color: var(--muted);
  cursor: pointer;
  transition: color 0.15s, background 0.15s;
  letter-spacing: 0.01em;
}

.mode-toggle__btn--active {
  background: var(--accent);
  color: #fff;
}

/* ── Input area wrapper ─────────────────────────────────────── */

.input-area {
  display: flex;
  flex-direction: column;
  background: var(--surface);
  border-top: 1px solid var(--border-soft);
  flex-shrink: 0;
}

/* ── Suggestion chips ───────────────────────────────────────── */

.suggestions {
  display: flex;
  gap: 0.5rem;
  padding: 0.5rem 1.5rem 0.75rem;
  overflow-x: auto;
  scrollbar-width: none;
}

.suggestions::-webkit-scrollbar { display: none; }

.suggestion-chip {
  flex-shrink: 0;
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: 2rem;
  padding: 0.25rem 0.75rem;
  font-size: 0.75rem;
  font-weight: 400;
  font-family: inherit;
  color: var(--muted);
  cursor: pointer;
  white-space: nowrap;
  transition: color 0.15s, border-color 0.15s, background 0.15s;
}

.suggestion-chip:hover {
  color: var(--text-dim);
  border-color: var(--muted);
  background: var(--surface);
}

/* ── Export button (inside winner banner) ───────────────────── */

.winner-banner {
  /* override: make it flex with space-between so export sits on the right */
  justify-content: space-between;
  padding: 0 1.5rem;
}

.export-btn {
  background: transparent;
  border: 1px solid rgba(34, 197, 94, 0.35);
  border-radius: var(--radius-sm);
  padding: 0.25rem 0.75rem;
  font-size: 0.75rem;
  font-weight: 500;
  font-family: inherit;
  color: var(--green);
  cursor: pointer;
  transition: background 0.15s, border-color 0.15s;
  letter-spacing: 0.01em;
}

.export-btn:hover {
  background: var(--green-dim);
  border-color: var(--green-ring);
}
```

- [ ] **Step 3: Remove the existing `.input-row` border-top rule conflict**

The `.input-row` CSS still has `border-top: 1px solid var(--border)`. Now `.input-area` owns that border. Open `frontend/src/index.css` and find the `.input-row` rule (around line 569) and remove `border-top: 1px solid var(--border);` from it so there isn't a double border.

The `.input-row` block should become:
```css
.input-row {
  display: flex;
  align-items: flex-end;
  gap: 0.75rem;
  padding: 0.875rem 1.5rem;
  background: var(--surface);
  flex-shrink: 0;
}
```

- [ ] **Step 4: Visual verification**

With mock server and dev server running, open the app:
- Header no longer shows the k dropdown
- Below the panels, a controls row shows a "Retrieved chunks (k)" slider (2–10) and "Answer mode" toggle (Concise/Detailed)
- Dragging the slider updates the number label next to it
- Clicking Concise/Detailed highlights the active choice
- Submit a question — check Network tab to confirm POST body contains `"k": <current value>` and `"mode": "concise"` or `"detailed"`

- [ ] **Step 5: Commit**

```bash
git add frontend/src/App.jsx frontend/src/index.css
git commit -m "feat: live k slider and answer mode toggle in chat controls"
```

---

## Task 4: Question suggestions

Suggestion chips appear below the input bar in the `.suggestions` row already added in Task 3 (the JSX and CSS were included). This task only verifies that they work correctly — no new code needed beyond what was added in Task 3.

**Files:**
- No new files (already added in Task 3)

- [ ] **Step 1: Verify suggestions render and populate the input**

With the dev server running:
- Scroll the suggestions row horizontally — 10 chips should be visible: 2 each for AAPL, MSFT, GOOGL, AMZN, META
- Click "What are Apple's main supply chain risks?" — the textarea should fill with that text, but NOT auto-submit
- The Send button becomes active; pressing Enter or clicking Send should submit
- After submitting, click a chip again — the textarea should overwrite with the chip text

- [ ] **Step 2: Commit (if any corrections were needed; otherwise skip)**

If no visual issues were found, no commit needed. If chips needed a CSS fix, commit that:
```bash
git add frontend/src/index.css
git commit -m "fix: suggestion chip alignment"
```

---

## Task 5: Backend mode support (mock + real)

Wires the `mode` field through the Python backend so the system prompt is selected at query time.

**Files:**
- Modify: `scripts/mock_stream_server.py`
- Modify: `v2_modal/finsight.py`

- [ ] **Step 1: Update mock server to accept (and ignore) mode**

In `scripts/mock_stream_server.py`, update the `stream_endpoint` function to extract `mode` (no behavior change):

```python
@app.post("/v1/stream")
async def stream_endpoint(item: dict):
    question   = item.get("question", "")
    k          = int(item.get("k", 5))
    max_tokens = int(item.get("max_tokens", 20))
    _mode      = item.get("mode", "concise")  # accepted, not used in mock

    return StreamingResponse(
        _token_stream(question, k, max_tokens),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
        },
    )
```

- [ ] **Step 2: Update finsight.py — build_prompt accepts mode**

In `v2_modal/finsight.py`, replace the `build_prompt` method (currently at lines 196–207) with:

```python
SYSTEM_PROMPTS = {
    "concise": (
        "You are a financial analyst. Answer using ONLY the context below. "
        "Be brief and direct. Cite sources as [1], [2]. 2-3 sentences maximum."
    ),
    "detailed": (
        "You are a financial analyst. Answer using ONLY the context below. "
        "Be thorough and structured. Cite sources as [1], [2]. "
        "Use bullet points for complex answers."
    ),
}

def build_prompt(self, question: str, contexts: list, mode: str = "concise") -> str:
    system = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["concise"])
    formatted = "\n\n".join(
        f"[{i+1}] (from {c['src']}):\n{c['text'][:600]}"
        for i, c in enumerate(contexts)
    )
    return (
        f"{system}\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{formatted}\n\n"
        "Answer:"
    )
```

Note: `SYSTEM_PROMPTS` is a module-level dict defined just before the `VLLMServer` class. Place it directly above `class VLLMServer:` (before the `@app.cls` decorator block is fine too — Python module scope).

- [ ] **Step 3: Update query_stream to accept and pass mode**

In `v2_modal/finsight.py`, update the `query_stream` method signature and the `build_prompt` call (currently at lines 210–296):

Change the method signature from:
```python
def query_stream(self, question: str, k: int = 5, max_tokens: int = 400):
```
to:
```python
def query_stream(self, question: str, k: int = 5, max_tokens: int = 400, mode: str = "concise"):
```

Change the `build_prompt` call from:
```python
prompt   = self.build_prompt(question, contexts)
```
to:
```python
prompt   = self.build_prompt(question, contexts, mode)
```

- [ ] **Step 4: Update stream_endpoint in _make_streaming_app to read and forward mode**

In `v2_modal/finsight.py`, update the `stream_endpoint` function inside `_make_streaming_app` (currently at lines 316–330):

```python
@web_app.post("/v1/stream")
async def stream_endpoint(item: dict):
    question   = item.get("question", "")
    k          = int(item.get("k", 5))
    max_tokens = int(item.get("max_tokens", 400))
    mode       = item.get("mode", "concise")

    server = VLLMServer(model_name=model_name)
    return StreamingResponse(
        server.query_stream.remote_gen(question, k, max_tokens, mode),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control":     "no-cache",
        },
    )
```

- [ ] **Step 5: Verify mock server still works**

Restart the mock server:
```bash
uvicorn scripts.mock_stream_server:app --port 8001
```

Switch mode toggle to "Detailed", submit a question. Expected: mock still returns the same fake tokens (mode ignored in mock). Check no server errors in the uvicorn console.

- [ ] **Step 6: Commit**

```bash
git add scripts/mock_stream_server.py v2_modal/finsight.py
git commit -m "feat: mode-aware system prompt in finsight.py; mock server accepts mode field"
```

---

## Task 6: Export comparison — verify end-to-end

Feature 5's logic was already included in the `App.jsx` rewrite in Task 3 (`handleExport`, `llamaAnswer`/`mistralAnswer` state, `onAnswer` callbacks, export button in the winner banner). This task verifies the full export flow.

**Files:**
- No new files

- [ ] **Step 1: End-to-end export test**

With mock server + dev server running:
1. Submit any question and wait for both panels to finish streaming
2. The winner banner should appear with an "Export JSON" button on the right
3. Click "Export JSON"
4. A file named `finsight_comparison_<timestamp>.json` should download
5. Open the file and verify it matches this shape (values will reflect mock data):

```json
{
  "question": "What are Apple's main supply chain risks?",
  "timestamp": "2026-04-04T...",
  "k": 5,
  "mode": "concise",
  "llama": {
    "answer": "Apple's primary supply chain risks include...",
    "metrics": {
      "ttft_ms": 312.4,
      "tpot_ms": 23.1,
      "tokens": 34,
      "throughput_tps": 28.6
    },
    "contexts": [{ "src": "AAPL_10K.txt", "text": "fake context chunk" }]
  },
  "mistral": {
    "answer": "Apple's primary supply chain risks include...",
    "metrics": { ... },
    "contexts": [...]
  }
}
```

- [ ] **Step 2: Commit if any fixes were required; otherwise final commit**

```bash
git add -p   # stage only intentional changes
git commit -m "feat: export comparison JSON download"
```

---

## Final verification checklist

- [ ] TTFT timer counts up from 0 in ms, freezes on first token, then final value from metrics replaces it
- [ ] k slider range 2–10 works; POST body reflects chosen k
- [ ] Mode toggle Concise/Detailed works; POST body contains `"mode"` field
- [ ] 10 suggestion chips visible, horizontally scrollable, clicking populates input (no auto-submit)
- [ ] Export button appears only after both panels finish; downloaded JSON has correct shape
- [ ] `useStream.js` and `api.js` are unmodified (confirm with `git diff`)
- [ ] Mock server handles all new fields without errors
