import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/upload': { target: 'http://localhost:8000', changeOrigin: true },
      '/query': { target: 'http://localhost:8000', changeOrigin: true },
      '/health': { target: 'http://localhost:8000', changeOrigin: true },
      '/status': { target: 'http://localhost:8000', changeOrigin: true },
      '/auth': { target: 'http://localhost:8000', changeOrigin: true },
      '/documents': { target: 'http://localhost:8000', changeOrigin: true },
      '/analytics': { target: 'http://localhost:8000', changeOrigin: true },
      '/index': { target: 'http://localhost:8000', changeOrigin: true },
      '/queries': { target: 'http://localhost:8000', changeOrigin: true },
      '/versions': { target: 'http://localhost:8000', changeOrigin: true },
      '/metrics': { target: 'http://localhost:8000', changeOrigin: true },
      '/resources': { target: 'http://localhost:8000', changeOrigin: true },
      '/settings': { target: 'http://localhost:8000', changeOrigin: true },
      '/files': { target: 'http://localhost:8000', changeOrigin: true },
      '/diagnosis': { target: 'http://localhost:8000', changeOrigin: true },
      '/experiments': { target: 'http://localhost:8000', changeOrigin: true },
      '/corpus': { target: 'http://localhost:8000', changeOrigin: true },
      '/embeddings': { target: 'http://localhost:8000', changeOrigin: true },
      '/linking': { target: 'http://localhost:8000', changeOrigin: true },
      '/clip': { target: 'http://localhost:8000', changeOrigin: true },
    },
  },
});
