import { ChevronRight, FileText } from 'lucide-react'

export function SourceDrawer({ contexts }) {
  if (!contexts?.length) return null

  return (
    <details className="source-drawer">
      <summary>
        <ChevronRight className="source-drawer__chevron" size={16} aria-hidden="true" />
        <FileText size={15} aria-hidden="true" />
        <span>{contexts.length} retrieved source{contexts.length !== 1 ? 's' : ''}</span>
      </summary>
      <div className="source-cards">
        {contexts.map((c, i) => (
          <article key={`${c.src}-${i}`} className="source-card">
            <span className="source-card__num">{i + 1}</span>
            <div className="source-card__body">
              <strong className="source-card__src">{c.src}</strong>
              <p className="source-card__text">{c.text}</p>
            </div>
          </article>
        ))}
      </div>
    </details>
  )
}
