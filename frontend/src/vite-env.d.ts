import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { existsSync, readFileSync } from 'node:fs'
import { fileURLToPath, URL } from 'node:url'

declare const process: { env: Record<string, string | undefined> }

const backend = process.env.SIGHTSEEN_BACKEND || 'http://localhost:8080'

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