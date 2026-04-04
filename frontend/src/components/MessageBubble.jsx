export function MessageBubble({ role, content, isStreaming }) {
  return (
    <div className={`message message--${role}`}>
      <span className="message__role">{role === 'user' ? 'You' : 'FinSight'}</span>
      <p className="message__content">
        {content}
        {isStreaming && role === 'assistant' && <span className="cursor" />}
      </p>
    </div>
  )
}
