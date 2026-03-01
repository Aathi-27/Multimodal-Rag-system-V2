import { useCallback, useState } from 'react';
import api, { AUTH_ENABLED } from '../api/axios';

/**
 * Lightweight auth hook.
 * When VITE_AUTH_ENABLED=false (default), isAuthenticated is always true.
 */
export function useAuth() {
  const [token, setToken] = useState<string | null>(() =>
    AUTH_ENABLED ? localStorage.getItem('rag_token') : '__bypass__',
  );

  const isAuthenticated = Boolean(token);

  const login = useCallback(async (email: string, password: string) => {
    if (!AUTH_ENABLED) return;
    const res = await api.post<{ token: string }>('/auth/login', { email, password });
    const jwt = res.data.token;
    localStorage.setItem('rag_token', jwt);
    setToken(jwt);
  }, []);

  const logout = useCallback(() => {
    localStorage.removeItem('rag_token');
    setToken(null);
  }, []);

  return { isAuthenticated, token, login, logout, authEnabled: AUTH_ENABLED };
}
