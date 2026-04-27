import { Bot, UserRound } from 'lucide-react'

export function MessageBubble({ role, content, isStreaming }) {
  const isUser = role === 'user'
  const Icon = isUser ? UserRound : Bot

  return (
    <div className={`message message--${role}`}>
      <span className="message__role">
        <Icon size={14} aria-hidden="true" />
        {isUser ? 'You' : 'FinSight'}
      </span>
      <p className="message__content">
        {content}
        {isStreaming && role === 'assistant' && <span className="cursor" />}
      </p>
    </div>
  )
}
