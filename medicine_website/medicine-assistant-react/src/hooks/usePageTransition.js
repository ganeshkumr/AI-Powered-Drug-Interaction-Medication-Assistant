import { useEffect, useRef } from 'react';
import { useLocation } from 'react-router-dom';

/**
 * Custom hook for managing page transitions and animations
 * Ensures animations don't interfere with business logic
 * 
 * Requirements: 6.1, 6.2, 6.3, 6.5
 */
export const usePageTransition = (options = {}) => {
  const {
    duration = 300,
    onTransitionStart,
    onTransitionEnd,
    preserveScrollPosition = false
  } = options;

  const location = useLocation();
  const previousLocation = useRef(location);
  const isTransitioning = useRef(false);

  useEffect(() => {
    // Only trigger transition if location actually changed
    if (previousLocation.current.pathname !== location.pathname) {
      isTransitioning.current = true;
      
      // Call transition start callback
      if (onTransitionStart) {
        onTransitionStart(previousLocation.current, location);
      }

      // Preserve scroll position if requested
      if (!preserveScrollPosition) {
        window.scrollTo(0, 0);
      }

      // Set transition end timer
      const timer = setTimeout(() => {
        isTransitioning.current = false;
        if (onTransitionEnd) {
          onTransitionEnd(previousLocation.current, location);
        }
      }, duration);

      // Update previous location
      previousLocation.current = location;

      return () => clearTimeout(timer);
    }
  }, [location, duration, onTransitionStart, onTransitionEnd, preserveScrollPosition]);

  return {
    isTransitioning: isTransitioning.current,
    currentLocation: location,
    previousLocation: previousLocation.current
  };
};

/**
 * Custom hook for step-based animations
 * Optimized for the 3-step medication check flow
 */
export const useStepTransition = (currentStep, totalSteps = 3) => {
  const progressPercentage = ((currentStep - 1) / (totalSteps - 1)) * 100;
  
  const getStepAnimation = (stepNumber) => {
    if (stepNumber < currentStep) {
      return 'completed';
    } else if (stepNumber === currentStep) {
      return 'active';
    } else {
      return 'inactive';
    }
  };

  const getProgressAnimation = () => ({
    width: `${progressPercentage}%`,
    transition: 'width 0.6s cubic-bezier(0.4, 0, 0.2, 1)'
  });

  return {
    progressPercentage,
    getStepAnimation,
    getProgressAnimation,
    isFirstStep: currentStep === 1,
    isLastStep: currentStep === totalSteps,
    canGoBack: currentStep > 1,
    canGoForward: currentStep < totalSteps
  };
};

/**
 * Custom hook for loading animations
 * Provides consistent loading states across the application
 */
export const useLoadingAnimation = (isLoading, options = {}) => {
  const {
    minDuration = 500, // Minimum loading duration for better UX
    onLoadingStart,
    onLoadingEnd
  } = options;

  const loadingStartTime = useRef(null);
  const loadingTimer = useRef(null);

  useEffect(() => {
    if (isLoading && !loadingStartTime.current) {
      loadingStartTime.current = Date.now();
      if (onLoadingStart) {
        onLoadingStart();
      }
    } else if (!isLoading && loadingStartTime.current) {
      const elapsedTime = Date.now() - loadingStartTime.current;
      const remainingTime = Math.max(0, minDuration - elapsedTime);

      if (remainingTime > 0) {
        loadingTimer.current = setTimeout(() => {
          if (onLoadingEnd) {
            onLoadingEnd();
          }
          loadingStartTime.current = null;
        }, remainingTime);
      } else {
        if (onLoadingEnd) {
          onLoadingEnd();
        }
        loadingStartTime.current = null;
      }
    }

    return () => {
      if (loadingTimer.current) {
        clearTimeout(loadingTimer.current);
      }
    };
  }, [isLoading, minDuration, onLoadingStart, onLoadingEnd]);

  return {
    isLoading,
    hasMinimumDuration: loadingStartTime.current !== null
  };
};

/**
 * Custom hook for managing animation performance
 * Reduces animations on low-performance devices
 */
export const useAnimationPerformance = () => {
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const isLowPerformance = navigator.hardwareConcurrency && navigator.hardwareConcurrency < 4;
  
  const shouldReduceAnimations = prefersReducedMotion || isLowPerformance;

  const getAnimationConfig = (baseConfig) => {
    if (shouldReduceAnimations) {
      return {
        ...baseConfig,
        duration: 0.1,
        ease: 'linear'
      };
    }
    return baseConfig;
  };

  return {
    shouldReduceAnimations,
    prefersReducedMotion,
    isLowPerformance,
    getAnimationConfig
  };
};