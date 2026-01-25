import { BrowserRouter as Router, Routes, Route, Navigate, useLocation } from 'react-router-dom'
import { AuthProvider } from './context/AuthContext'
import { ThemeProvider } from './context/ThemeContext'
import { AnimatePresence } from 'framer-motion'
import { Suspense, lazy, memo } from 'react'
import Layout from './components/layout/Layout'
import PageTransition from './components/common/PageTransition'
import LoadingSpinner from './components/common/LoadingSpinner'
import { PerformanceErrorBoundary, usePerformanceMonitoring } from './utils/performance'

// Lazy load pages for better performance
const Landing = lazy(() => import('./pages/Landing'))
const Login = lazy(() => import('./pages/Login'))
const Register = lazy(() => import('./pages/Register'))
const Profile = lazy(() => import('./pages/Profile'))
const Dashboard = lazy(() => import('./pages/Dashboard'))
const Results = lazy(() => import('./pages/Results'))
const History = lazy(() => import('./pages/History'))
const About = lazy(() => import('./pages/About'))
const MedicationStep = lazy(() => import('./pages/MedicationStep'))
const DosageStep = lazy(() => import('./pages/DosageStep'))
const AnalysisStep = lazy(() => import('./pages/AnalysisStep'))
const MyMedPage = lazy(() => import('./pages/MyMedPage'))

// Optimized loading fallback
const PageLoadingFallback = memo(() => (
  <div className="min-h-screen flex items-center justify-center">
    <LoadingSpinner size="lg" />
  </div>
))

// Wrapper component to handle page transitions with performance monitoring
const AnimatedRoutes = memo(() => {
  const location = useLocation();
  const { measureRender } = usePerformanceMonitoring('AnimatedRoutes');
  
  return (
    <AnimatePresence mode="wait" initial={false}>
      <Routes location={location} key={location.pathname}>
        <Route 
          path="/" 
          element={
            <Suspense fallback={<PageLoadingFallback />}>
              <PageTransition variant="fade">
                <Landing />
              </PageTransition>
            </Suspense>
          } 
        />
        <Route 
          path="/login" 
          element={
            <Suspense fallback={<PageLoadingFallback />}>
              <PageTransition variant="slideUp">
                <Login />
              </PageTransition>
            </Suspense>
          } 
        />
        <Route 
          path="/register" 
          element={
            <Suspense fallback={<PageLoadingFallback />}>
              <PageTransition variant="slideUp">
                <Register />
              </PageTransition>
            </Suspense>
          } 
        />
        <Route 
          path="/results" 
          element={
            <Suspense fallback={<PageLoadingFallback />}>
              <PageTransition variant="slide">
                <Results />
              </PageTransition>
            </Suspense>
          } 
        />
        <Route 
          path="/history" 
          element={
            <Suspense fallback={<PageLoadingFallback />}>
              <PageTransition variant="slide">
                <History />
              </PageTransition>
            </Suspense>
          } 
        />
        <Route
          path="/profile"
          element={
            <Layout>
              <Suspense fallback={<PageLoadingFallback />}>
                <PageTransition variant="fade">
                  <Profile />
                </PageTransition>
              </Suspense>
            </Layout>
          }
        />
        <Route
          path="/dashboard"
          element={
            <Layout>
              <Suspense fallback={<PageLoadingFallback />}>
                <PageTransition variant="slide">
                  <Dashboard />
                </PageTransition>
              </Suspense>
            </Layout>
          }
        />
        <Route
          path="/my-med"
          element={
            <Layout>
              <Suspense fallback={<PageLoadingFallback />}>
                <PageTransition variant="slide">
                  <MyMedPage />
                </PageTransition>
              </Suspense>
            </Layout>
          }
        />
        <Route
          path="/check/medication"
          element={
            <Layout>
              <Suspense fallback={<PageLoadingFallback />}>
                <PageTransition variant="slide">
                  <MedicationStep />
                </PageTransition>
              </Suspense>
            </Layout>
          }
        />
        <Route
          path="/check/dosage"
          element={
            <Layout>
              <Suspense fallback={<PageLoadingFallback />}>
                <PageTransition variant="slide">
                  <DosageStep />
                </PageTransition>
              </Suspense>
            </Layout>
          }
        />
        <Route
          path="/check/analysis"
          element={
            <Layout>
              <Suspense fallback={<PageLoadingFallback />}>
                <PageTransition variant="slide">
                  <AnalysisStep />
                </PageTransition>
              </Suspense>
            </Layout>
          }
        />
        <Route
          path="/about"
          element={
            <Layout>
              <Suspense fallback={<PageLoadingFallback />}>
                <PageTransition variant="slideUp">
                  <About />
                </PageTransition>
              </Suspense>
            </Layout>
          }
        />
      </Routes>
    </AnimatePresence>
  );
});

AnimatedRoutes.displayName = 'AnimatedRoutes';

function App() {
  const { measureRender } = usePerformanceMonitoring('App');

  return (
    <PerformanceErrorBoundary>
      <ThemeProvider>
        <AuthProvider>
          <Router>
            <AnimatedRoutes />
          </Router>
        </AuthProvider>
      </ThemeProvider>
    </PerformanceErrorBoundary>
  )
}

export default memo(App)
