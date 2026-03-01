import { lazy, Suspense } from 'react';
import { Navigate, type RouteObject } from 'react-router-dom';
import Layout from '@/shared/components/Layout';
import Loader from '@/shared/components/Loader';

/* ── Lazy-loaded feature pages ─────────────────────────────── */
const ChatPage = lazy(() => import('@/features/chat/ChatPage'));
const UploadPage = lazy(() => import('@/features/upload/UploadPage'));
const KnowledgeBasePage = lazy(() => import('@/features/knowledge/KnowledgeBasePage'));
const SystemStatusPage = lazy(() => import('@/features/dashboard/SystemStatusPage'));
const QueryHistoryPage = lazy(() => import('@/features/history/QueryHistoryPage'));
const LoginPage = lazy(() => import('@/features/auth/LoginPage'));
const FailureDiagnosisPage = lazy(() => import('@/features/diagnosis/FailureDiagnosisPage'));
const ExperimentLabPage = lazy(() => import('@/features/experiments/ExperimentLabPage'));

/* ── Suspense wrapper ──────────────────────────────────────── */
function SuspenseWrap({ children }: { children: React.ReactNode }) {
  return (
    <Suspense
      fallback={
        <div className="flex h-screen items-center justify-center bg-slate-950">
          <Loader size="lg" label="Loading…" />
        </div>
      }
    >
      {children}
    </Suspense>
  );
}

/* ── Route definitions ─────────────────────────────────────── */
export const routes: RouteObject[] = [
  /* Public: login (only relevant when auth enabled) */
  {
    path: '/login',
    element: (
      <SuspenseWrap>
        <LoginPage />
      </SuspenseWrap>
    ),
  },

  /* Protected (or open) routes inside the Layout shell */
  {
    element: <Layout />,
    children: [
      {
        index: true,
        element: (
          <SuspenseWrap>
            <ChatPage />
          </SuspenseWrap>
        ),
      },
      {
        path: 'upload',
        element: (
          <SuspenseWrap>
            <UploadPage />
          </SuspenseWrap>
        ),
      },
      {
        path: 'knowledge',
        element: (
          <SuspenseWrap>
            <KnowledgeBasePage />
          </SuspenseWrap>
        ),
      },
      {
        path: 'history',
        element: (
          <SuspenseWrap>
            <QueryHistoryPage />
          </SuspenseWrap>
        ),
      },
      {
        path: 'status',
        element: (
          <SuspenseWrap>
            <SystemStatusPage />
          </SuspenseWrap>
        ),
      },
      {
        path: 'diagnosis',
        element: (
          <SuspenseWrap>
            <FailureDiagnosisPage />
          </SuspenseWrap>
        ),
      },
      {
        path: 'experiments',
        element: (
          <SuspenseWrap>
            <ExperimentLabPage />
          </SuspenseWrap>
        ),
      },
    ],
  },

  /* Catch-all → redirect to chat */
  {
    path: '*',
    element: <Navigate to="/" replace />,
  },
];
