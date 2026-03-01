import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const AUTH_ENABLED = import.meta.env.VITE_AUTH_ENABLED === 'true';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120_000, // 2 min — LLM can be slow on first load
  headers: { 'Content-Type': 'application/json' },
});

/* ─── Request Interceptor: attach JWT if auth enabled ──────────────────── */
api.interceptors.request.use((config) => {
  if (AUTH_ENABLED) {
    const token = localStorage.getItem('rag_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
  }
  return config;
});

/* ─── Response Interceptor: global error handling ──────────────────────── */
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (AUTH_ENABLED && error.response?.status === 401) {
      localStorage.removeItem('rag_token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  },
);

export { API_BASE_URL, AUTH_ENABLED };
export default api;
