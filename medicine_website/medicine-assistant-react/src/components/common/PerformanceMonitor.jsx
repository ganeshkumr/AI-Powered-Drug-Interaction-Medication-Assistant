import React, { useEffect, useState, useRef } from 'react';
import { performanceMonitor, animationUtils } from '../../utils/performance';

/**
 * Performance monitoring component for the medical application
 * Tracks loading times, animation performance, and user experience metrics
 */
const PerformanceMonitor = ({ 
  enabled = process.env.NODE_ENV === 'development',
  showMetrics = false,
  onMetricsUpdate = null 
}) => {
  const [metrics, setMetrics] = useState({
    loadTime: 0,
    renderTime: 0,
    animationFrameRate: 0,
    memoryUsage: 0,
    bundleSize: 0,
  });
  
  const metricsRef = useRef(metrics);
  const frameCountRef = useRef(0);
  const lastTimeRef = useRef(performance.now());

  useEffect(() => {
    if (!enabled) return;

    // Monitor initial load performance
    const measureLoadTime = () => {
      const loadTime = performance.timing?.loadEventEnd - performance.timing?.navigationStart;
      if (loadTime > 0) {
        setMetrics(prev => ({ ...prev, loadTime }));
        performanceMonitor.mark('app-load-complete');
      }
    };

    // Monitor render performance
    const measureRenderTime = () => {
      performanceMonitor.mark('render-start');
      
      requestAnimationFrame(() => {
        performanceMonitor.mark('render-end');
        const renderTime = performanceMonitor.measure('render-time', 'render-start', 'render-end');
        if (renderTime) {
          setMetrics(prev => ({ ...prev, renderTime: Math.round(renderTime) }));
        }
      });
    };

    // Monitor animation frame rate
    const measureFrameRate = () => {
      const now = performance.now();
      frameCountRef.current++;
      
      if (now - lastTimeRef.current >= 1000) {
        const fps = Math.round((frameCountRef.current * 1000) / (now - lastTimeRef.current));
        setMetrics(prev => ({ ...prev, animationFrameRate: fps }));
        frameCountRef.current = 0;
        lastTimeRef.current = now;
      }
      
      requestAnimationFrame(measureFrameRate);
    };

    // Monitor memory usage (if available)
    const measureMemoryUsage = () => {
      if (performance.memory) {
        const memoryUsage = Math.round(performance.memory.usedJSHeapSize / 1024 / 1024);
        setMetrics(prev => ({ ...prev, memoryUsage }));
      }
    };

    // Monitor bundle size
    const measureBundleSize = () => {
      if (navigator.connection && navigator.connection.downlink) {
        // Estimate bundle size based on load time and connection speed
        const estimatedSize = Math.round((metrics.loadTime / 1000) * navigator.connection.downlink * 1024);
        setMetrics(prev => ({ ...prev, bundleSize: estimatedSize }));
      }
    };

    // Initialize measurements
    measureLoadTime();
    measureRenderTime();
    measureFrameRate();
    
    // Set up periodic measurements
    const memoryInterval = setInterval(measureMemoryUsage, 5000);
    const bundleInterval = setInterval(measureBundleSize, 10000);

    // Performance observer for navigation timing
    if ('PerformanceObserver' in window) {
      const observer = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        entries.forEach((entry) => {
          if (entry.entryType === 'navigation') {
            const loadTime = entry.loadEventEnd - entry.fetchStart;
            setMetrics(prev => ({ ...prev, loadTime: Math.round(loadTime) }));
          }
        });
      });
      
      observer.observe({ entryTypes: ['navigation'] });
      
      return () => {
        observer.disconnect();
        clearInterval(memoryInterval);
        clearInterval(bundleInterval);
      };
    }

    return () => {
      clearInterval(memoryInterval);
      clearInterval(bundleInterval);
    };
  }, [enabled]);

  // Update metrics reference and call callback
  useEffect(() => {
    metricsRef.current = metrics;
    if (onMetricsUpdate) {
      onMetricsUpdate(metrics);
    }
  }, [metrics, onMetricsUpdate]);

  // Performance validation
  const validatePerformance = () => {
    const issues = [];
    
    if (metrics.loadTime > 3000) {
      issues.push('Load time exceeds 3 seconds');
    }
    
    if (metrics.renderTime > 16) {
      issues.push('Render time exceeds 16ms (60fps threshold)');
    }
    
    if (metrics.animationFrameRate < 55) {
      issues.push('Animation frame rate below 55fps');
    }
    
    if (metrics.memoryUsage > 100) {
      issues.push('Memory usage exceeds 100MB');
    }
    
    return issues;
  };

  // Performance recommendations
  const getRecommendations = () => {
    const recommendations = [];
    const issues = validatePerformance();
    
    if (issues.includes('Load time exceeds 3 seconds')) {
      recommendations.push('Consider code splitting and lazy loading');
      recommendations.push('Optimize bundle size with tree shaking');
      recommendations.push('Enable compression (gzip/brotli)');
    }
    
    if (issues.includes('Render time exceeds 16ms (60fps threshold)')) {
      recommendations.push('Use React.memo for expensive components');
      recommendations.push('Optimize re-renders with useCallback and useMemo');
      recommendations.push('Consider virtualization for large lists');
    }
    
    if (issues.includes('Animation frame rate below 55fps')) {
      recommendations.push('Use CSS transforms instead of changing layout properties');
      recommendations.push('Enable hardware acceleration with transform3d');
      recommendations.push('Reduce animation complexity');
    }
    
    if (issues.includes('Memory usage exceeds 100MB')) {
      recommendations.push('Check for memory leaks in event listeners');
      recommendations.push('Clean up timers and intervals');
      recommendations.push('Optimize image sizes and formats');
    }
    
    return recommendations;
  };

  if (!enabled || !showMetrics) {
    return null;
  }

  const issues = validatePerformance();
  const recommendations = getRecommendations();

  return (
    <div className="fixed bottom-4 right-4 bg-white border border-gray-300 rounded-lg shadow-lg p-4 max-w-sm z-50">
      <h3 className="text-sm font-semibold text-gray-800 mb-2">Performance Metrics</h3>
      
      <div className="space-y-1 text-xs">
        <div className="flex justify-between">
          <span>Load Time:</span>
          <span className={metrics.loadTime > 3000 ? 'text-red-600' : 'text-green-600'}>
            {metrics.loadTime}ms
          </span>
        </div>
        
        <div className="flex justify-between">
          <span>Render Time:</span>
          <span className={metrics.renderTime > 16 ? 'text-red-600' : 'text-green-600'}>
            {metrics.renderTime}ms
          </span>
        </div>
        
        <div className="flex justify-between">
          <span>Frame Rate:</span>
          <span className={metrics.animationFrameRate < 55 ? 'text-red-600' : 'text-green-600'}>
            {metrics.animationFrameRate}fps
          </span>
        </div>
        
        <div className="flex justify-between">
          <span>Memory:</span>
          <span className={metrics.memoryUsage > 100 ? 'text-red-600' : 'text-green-600'}>
            {metrics.memoryUsage}MB
          </span>
        </div>
        
        {metrics.bundleSize > 0 && (
          <div className="flex justify-between">
            <span>Bundle Size:</span>
            <span className={metrics.bundleSize > 1000 ? 'text-red-600' : 'text-green-600'}>
              {metrics.bundleSize}KB
            </span>
          </div>
        )}
      </div>

      {issues.length > 0 && (
        <div className="mt-3 pt-2 border-t border-gray-200">
          <h4 className="text-xs font-semibold text-red-600 mb-1">Issues:</h4>
          <ul className="text-xs text-red-600 space-y-1">
            {issues.map((issue, index) => (
              <li key={index}>• {issue}</li>
            ))}
          </ul>
        </div>
      )}

      {recommendations.length > 0 && (
        <div className="mt-2">
          <h4 className="text-xs font-semibold text-blue-600 mb-1">Recommendations:</h4>
          <ul className="text-xs text-blue-600 space-y-1">
            {recommendations.slice(0, 3).map((rec, index) => (
              <li key={index}>• {rec}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

// Hook for using performance metrics in components
export const usePerformanceMetrics = () => {
  const [metrics, setMetrics] = useState({
    loadTime: 0,
    renderTime: 0,
    animationFrameRate: 0,
    memoryUsage: 0,
  });

  useEffect(() => {
    const updateMetrics = (newMetrics) => {
      setMetrics(newMetrics);
    };

    // This would be called by PerformanceMonitor
    window.performanceMetricsCallback = updateMetrics;

    return () => {
      delete window.performanceMetricsCallback;
    };
  }, []);

  return metrics;
};

// Performance validation utility
export const validateMedicalAppPerformance = () => {
  const checks = {
    loadTime: false,
    renderTime: false,
    accessibility: false,
    animations: false,
    memoryUsage: false,
  };

  // Check load time
  const loadTime = performance.timing?.loadEventEnd - performance.timing?.navigationStart;
  checks.loadTime = loadTime < 3000;

  // Check render time
  performanceMonitor.mark('validation-start');
  requestAnimationFrame(() => {
    performanceMonitor.mark('validation-end');
    const renderTime = performanceMonitor.measure('validation', 'validation-start', 'validation-end');
    checks.renderTime = renderTime < 16;
  });

  // Check accessibility
  checks.accessibility = document.querySelectorAll('[aria-label], [aria-labelledby], [role]').length > 0;

  // Check animations
  checks.animations = !animationUtils.prefersReducedMotion() || 
    document.querySelectorAll('[style*="animation: none"]').length > 0;

  // Check memory usage
  if (performance.memory) {
    const memoryUsage = performance.memory.usedJSHeapSize / 1024 / 1024;
    checks.memoryUsage = memoryUsage < 100;
  }

  return checks;
};

export default PerformanceMonitor;