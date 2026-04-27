const STREAM_PATH = '/v1/stream'

function streamUrl(value) {
  if (!value) return ''
  const trimmed = value.trim().replace(/\/+$/, '')
  return trimmed.endsWith(STREAM_PATH) ? trimmed : `${trimmed}${STREAM_PATH}`
}

export const LLAMA_URL = streamUrl(import.meta.env.VITE_LLAMA_URL)
export const MISTRAL_URL = streamUrl(import.meta.env.VITE_MISTRAL_URL)
