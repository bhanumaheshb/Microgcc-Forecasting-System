import { defineConfig } from 'vite';

// Vite dev server (port 5173) proxies all API calls to the FastAPI backend
// running on port 8000. This keeps the frontend's relative fetch URLs working
// (e.g. `/states`, `/forecast/...`) without any code change.
export default defineConfig({
  server: {
    port: 5173,
    strictPort: true,
    open: true,
    proxy: {
      '/states': 'http://127.0.0.1:8000',
      '/health': 'http://127.0.0.1:8000',
      '/metrics': 'http://127.0.0.1:8000',
      '/forecast': 'http://127.0.0.1:8000',
      '/history': 'http://127.0.0.1:8000',
      '/docs': 'http://127.0.0.1:8000',
      '/openapi.json': 'http://127.0.0.1:8000',
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
});
