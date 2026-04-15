export function SourceDrawer({ contexts }) {
  if (!contexts?.length) return null

  return (
    <details className="source-drawer">
      <summary>
        {contexts.length} source{contexts.length !== 1 ? 's' : ''}
      </summary>
      <div className="source-cards">
        {contexts.map((c, i) => (
          <div key={i} className="source-card">
            <span className="source-card__num">{i + 1}</span>
            <div className="source-card__body">
              <span className="source-card__src">{c.src}</span>
              <p className="source-card__text">{c.text}</p>
            </div>
          </div>
        ))}
      </div>
    </details>
  )
}
