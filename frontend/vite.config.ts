import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { existsSync, readFileSync } from 'node:fs'
import { fileURLToPath, URL } from 'node:url'

declare const process: { env: Record<string, string | undefined> }

const backend = process.env.SIGHTSEEN_BACKEND || 'http://localhost:8080'

// `npm run dev:phone` serves over HTTPS, because a phone only lets a page use its camera on a secure page (the
// landlord link's recorder). The certificate is self-signed for this PC's address (frontend/certs/, never committed):
// the phone shows a warning once, and the student-side pages keep working on plain `npm run dev`.
const phone = process.env.npm_lifecycle_event === 'dev:phone'
const certs = { key: 'certs/dev-key.pem', cert: 'certs/dev-cert.pem' }
const https = phone && existsSync(certs.key) && existsSync(certs.cert)
  ? { key: readFileSync(certs.key), cert: readFileSync(certs.cert) }
  : undefined

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  server: {
    https,
    proxy: {
      '/api': { target: backend, changeOrigin: true },
    },
  },
})
