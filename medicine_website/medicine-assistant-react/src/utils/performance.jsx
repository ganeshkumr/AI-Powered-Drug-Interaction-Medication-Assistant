import { lazy, Suspense, memo, useMemo, useCallback, useState, useEffect, useRef, Component } from 'react';
import LoadingSpinner from '../components/common/LoadingSpinner';

/**
 * Performance optimization utilities for the medical application
 */

// Lazy loading wrapper with error boundary
export const createLazyComponent = (importFn, fallback = <LoadingSpinner />) => {
  const LazyComponent = lazy(importFn);
  
  return memo((props) => (
    <Suspense fallback={fallback}>
      <LazyComponent {...props} />
    </Suspense>
  ));
};

// Optimized memo wrapper for components
export const createOptimizedComponent = (Component, propsAreEqual) => {
  return memo(Component, propsAreEqual);
};

// Performance-optimized callback hook
export const useOptimizedCallback = (callback, deps) => {
  return useCallback(callback, deps);
};

// Performance-optimized memo hook
export const useOptimizedMemo = (factory, deps) => {
  return useMemo(factory, deps);
};

// Debounce hook for performance optimization
export const useDebounce = (value, delay) => {
  const [debouncedValue, setDebouncedValue] = useState(value);

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => {
      clearTimeout(handler);
    };
  }, [value, delay]);

  return debouncedValue;
};

// Throttle hook for performance optimization
export const useThrottle = (value, limit) => {
  const [throttledValue, setThrottledValue] = useState(value);
  const lastRan = useRef(Date.now());

  useEffect(() => {
    const handler = setTimeout(() => {
      if (Date.now() - lastRan.current >= limit) {
        setThrottledValue(value);
        lastRan.current = Date.now();
      }
    }, limit - (Date.now() - lastRan.current));

    return () => {
      clearTimeout(handler);
    };
  }, [value, limit]);

  return throttledValue;
};

// Intersection Observer hook for lazy loading
export const useIntersectionObserver = (options = {}) => {
  const [isIntersecting, setIsIntersecting] = useState(false);
  const [element, setElement] = useState(null);

  useEffect(() => {
    if (!element) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        setIsIntersecting(entry.isIntersecting);
      },
      {
        threshold: 0.1,
        rootMargin: '50px',
        ...options,
      }
    );

    observer.observe(element);

    return () => {
      observer.disconnect();
    };
  }, [element, options]);

  return [setElement, isIntersecting];
};

// Performance monitoring utilities
export const performanceMonitor = {
  // Mark performance timing
  mark: (name) => {
    if (typeof performance !== 'undefined' && performance.mark) {
      performance.mark(name);
    }
  },

  // Measure performance between marks
  measure: (name, startMark, endMark) => {
    if (typeof performance !== 'undefined' && performance.measure) {
      try {
        performance.measure(name, startMark, endMark);
        const measure = performance.getEntriesByName(name)[0];
        return measure.duration;
      } catch (error) {
        console.warn('Performance measurement failed:', error);
        return null;
      }
    }
    return null;
  },

  // Clear performance marks and measures
  clear: (name) => {
    if (typeof performance !== 'undefined') {
      if (performance.clearMarks) performance.clearMarks(name);
      if (performance.clearMeasures) performance.clearMeasures(name);
    }
  },
};

// Animation performance utilities
export const animationUtils = {
  // Check if user prefers reduced motion
  prefersReducedMotion: () => {
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  },

  // Get optimized animation duration based on user preferences
  getAnimationDuration: (defaultDuration = 300) => {
    return animationUtils.prefersReducedMotion() ? 0 : defaultDuration;
  },

  // Create performance-optimized animation config
  createAnimationConfig: (config = {}) => {
    const prefersReduced = animationUtils.prefersReducedMotion();
    
    return {
      duration: prefersReduced ? 0 : (config.duration || 300),
      ease: prefersReduced ? 'linear' : (config.ease || 'easeOut'),
      delay: prefersReduced ? 0 : (config.delay || 0),
      ...config,
    };
  },

  // Optimize transform animations for better performance
  optimizeTransform: (element) => {
    if (element) {
      element.style.willChange = 'transform';
      element.style.backfaceVisibility = 'hidden';
      element.style.perspective = '1000px';
    }
  },

  // Clean up transform optimizations
  cleanupTransform: (element) => {
    if (element) {
      element.style.willChange = 'auto';
      element.style.backfaceVisibility = '';
      element.style.perspective = '';
    }
  },
};

// Bundle size optimization utilities
export const bundleUtils = {
  // Dynamically import modules only when needed
  dynamicImport: async (modulePath) => {
    try {
      const module = await import(/* @vite-ignore */ modulePath);
      return module.default || module;
    } catch (error) {
      console.error('Dynamic import failed:', error);
      return null;
    }
  },

  // Preload critical resources
  preloadResource: (href, as = 'script') => {
    const link = document.createElement('link');
    link.rel = 'preload';
    link.href = href;
    link.as = as;
    document.head.appendChild(link);
  },

  // Prefetch non-critical resources
  prefetchResource: (href) => {
    const link = document.createElement('link');
    link.rel = 'prefetch';
    link.href = href;
    document.head.appendChild(link);
  },
};

// Memory optimization utilities
export const memoryUtils = {
  // Clean up event listeners
  cleanupEventListeners: (element, events) => {
    if (element && events) {
      Object.entries(events).forEach(([event, handler]) => {
        element.removeEventListener(event, handler);
      });
    }
  },

  // Optimize large lists with virtualization
  shouldVirtualize: (itemCount, threshold = 100) => {
    return itemCount > threshold;
  },

  // Clean up timers and intervals
  cleanupTimers: (timers) => {
    timers.forEach(timer => {
      if (timer) {
        clearTimeout(timer);
        clearInterval(timer);
      }
    });
  },
};

// Error boundary for performance monitoring
export class PerformanceErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    // Log performance-related errors
    console.error('Performance Error:', error, errorInfo);
    
    // Report to monitoring service if available
    if (typeof window !== 'undefined' && window.reportError) {
      window.reportError(error);
    }
  }

  render() {
    if (this.state.hasError) {
      return (
        this.props.fallback || (
          <div className="p-4 text-center">
            <p className="text-danger-600">Something went wrong. Please refresh the page.</p>
          </div>
        )
      );
    }

    return this.props.children;
  }
}

// Performance monitoring hook
export const usePerformanceMonitoring = (componentName) => {
  useEffect(() => {
    performanceMonitor.mark(`${componentName}-mount-start`);
    
    return () => {
      performanceMonitor.mark(`${componentName}-unmount`);
      performanceMonitor.measure(
        `${componentName}-lifecycle`,
        `${componentName}-mount-start`,
        `${componentName}-unmount`
      );
    };
  }, [componentName]);

  const measureRender = useCallback((renderName) => {
    performanceMonitor.mark(`${componentName}-${renderName}-start`);
    
    return () => {
      performanceMonitor.mark(`${componentName}-${renderName}-end`);
      performanceMonitor.measure(
        `${componentName}-${renderName}`,
        `${componentName}-${renderName}-start`,
        `${componentName}-${renderName}-end`
      );
    };
  }, [componentName]);

  return { measureRender };
};

// Animation variants for framer-motion
export const animationVariants = {
  slideUp: {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
    exit: { opacity: 0, y: -20 }
  },
  slideDown: {
    initial: { opacity: 0, y: -20 },
    animate: { opacity: 1, y: 0 },
    exit: { opacity: 0, y: 20 }
  },
  slideLeft: {
    initial: { opacity: 0, x: 20 },
    animate: { opacity: 1, x: 0 },
    exit: { opacity: 0, x: -20 }
  },
  slideRight: {
    initial: { opacity: 0, x: -20 },
    animate: { opacity: 1, x: 0 },
    exit: { opacity: 0, x: 20 }
  },
  fadeIn: {
    initial: { opacity: 0 },
    animate: { opacity: 1 },
    exit: { opacity: 0 }
  },
  scaleIn: {
    initial: { opacity: 0, scale: 0.9 },
    animate: { opacity: 1, scale: 1 },
    exit: { opacity: 0, scale: 0.9 }
  }
};

// Animation configuration presets
export const animationConfigs = {
  fast: {
    duration: 0.2,
    ease: "easeOut"
  },
  medium: {
    duration: 0.3,
    ease: "easeInOut"
  },
  slow: {
    duration: 0.5,
    ease: "easeInOut"
  },
  spring: {
    type: "spring",
    stiffness: 300,
    damping: 30
  },
  bounce: {
    type: "spring",
    stiffness: 400,
    damping: 10
  }
};

export default {
  createLazyComponent,
  createOptimizedComponent,
  useOptimizedCallback,
  useOptimizedMemo,
  useDebounce,
  useThrottle,
  useIntersectionObserver,
  performanceMonitor,
  animationUtils,
  bundleUtils,
  memoryUtils,
  PerformanceErrorBoundary,
  usePerformanceMonitoring,
  animationVariants,
  animationConfigs,
};
