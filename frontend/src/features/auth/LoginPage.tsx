import { FormEvent, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '@/shared/hooks/useAuth';
import ErrorMessage from '@/shared/components/ErrorMessage';
import Loader from '@/shared/components/Loader';

export default function LoginPage() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      await login(email, password);
      navigate('/', { replace: true });
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Login failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center px-4">
      <div className="w-full max-w-sm">
        {/* Brand */}
        <div className="text-center mb-8">
          <div className="mx-auto w-12 h-12 rounded-xl bg-blue-600 flex items-center justify-center text-white font-bold text-lg mb-3">
            R
          </div>
          <h1 className="text-xl font-semibold text-slate-100">Offline RAG</h1>
          <p className="text-sm text-slate-500 mt-1">Sign in to continue</p>
        </div>

        {error && <div className="mb-4"><ErrorMessage message={error} /></div>}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="email" className="block text-xs font-medium text-slate-400 mb-1.5">
              Email
            </label>
            <input
              id="email"
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-2.5
                         text-sm text-slate-100 placeholder-slate-500
                         focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 outline-none
                         transition-all duration-200"
              placeholder="you@company.com"
            />
          </div>

          <div>
            <label htmlFor="password" className="block text-xs font-medium text-slate-400 mb-1.5">
              Password
            </label>
            <input
              id="password"
              type="password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-2.5
                         text-sm text-slate-100 placeholder-slate-500
                         focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 outline-none
                         transition-all duration-200"
              placeholder="••••••••"
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full flex items-center justify-center gap-2 rounded-lg bg-blue-600
                       px-4 py-2.5 text-sm font-medium text-white
                       hover:bg-blue-500 disabled:opacity-50 transition-all duration-150
                       active:scale-[0.98]"
          >
            {loading ? <Loader size="sm" /> : 'Sign in'}
          </button>
        </form>
      </div>
    </div>
  );
}
