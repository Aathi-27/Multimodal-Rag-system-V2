import { useRoutes, Navigate, useLocation } from 'react-router-dom';
import { useAuth } from '@/shared/hooks/useAuth';
import { routes } from './routes';

/**
 * Root application component.
 *
 * When VITE_AUTH_ENABLED=true, unauthenticated users are redirected to /login.
 * When VITE_AUTH_ENABLED=false (default), the auth gate is skipped entirely.
 */
export default function App() {
  const { isAuthenticated, authEnabled } = useAuth();
  const location = useLocation();

  /* Auth gate: redirect to /login when required */
  if (authEnabled && !isAuthenticated && location.pathname !== '/login') {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  /* Render matched route tree */
  const element = useRoutes(routes);
  return <>{element}</>;
}
