import React, { memo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useLocation } from 'react-router-dom';
import { animationUtils } from '../../utils/performance';

/**
 * PageTransition Component
 * 
 * Provides smooth page transitions between different sections of the application.
 * Implements medical-grade animations that don't interfere with business logic.
 * Optimized for 60fps performance with hardware acceleration.
 * 
 * Requirements: 6.1, 6.2, 6.3
 */
const PageTransition = memo(({ 
  children, 
  className = '',
  variant = 'slide',
  duration = 0.3,
  'data-testid': testId = 'page-transition'
}) => {
  const location = useLocation();

  // Get optimized animation duration based on user preferences
  const optimizedDuration = animationUtils.getAnimationDuration(duration * 1000) / 1000;

  // Animation variants optimized for performance
  const variants = {
    slide: {
      initial: { opacity: 0, x: 20, y: 0, z: 0 },
      animate: { opacity: 1, x: 0, y: 0, z: 0 },
      exit: { opacity: 0, x: -20, y: 0, z: 0 }
    },
    fade: {
      initial: { opacity: 0 },
      animate: { opacity: 1 },
      exit: { opacity: 0 }
    },
    slideUp: {
      initial: { opacity: 0, x: 0, y: 20, z: 0 },
      animate: { opacity: 1, x: 0, y: 0, z: 0 },
      exit: { opacity: 0, x: 0, y: -20, z: 0 }
    },
    scale: {
      initial: { opacity: 0, scale: 0.95, z: 0 },
      animate: { opacity: 1, scale: 1, z: 0 },
      exit: { opacity: 0, scale: 1.05, z: 0 }
    }
  };

  // Performance-optimized transition configuration
  const transition = animationUtils.createAnimationConfig({
    duration: optimizedDuration,
    ease: [0.4, 0, 0.2, 1], // Custom easing for smooth feel
    opacity: { duration: optimizedDuration * 0.8 }, // Slightly faster opacity
    scale: { duration: optimizedDuration * 1.2 }, // Slightly slower scale for smoothness
  });

  // Skip animations if user prefers reduced motion
  if (animationUtils.prefersReducedMotion()) {
    return (
      <div
        className={`w-full ${className}`}
        data-testid={testId}
      >
        {children}
      </div>
    );
  }

  return (
    <AnimatePresence mode="wait" initial={false}>
      <motion.div
        key={location.pathname}
        initial="initial"
        animate="animate"
        exit="exit"
        variants={variants[variant]}
        transition={transition}
        className={`w-full ${className}`}
        data-testid={testId}
        // Optimize for GPU acceleration and performance
        style={{
          willChange: 'transform, opacity',
          backfaceVisibility: 'hidden',
          perspective: 1000,
          transform: 'translateZ(0)', // Force hardware acceleration
        }}
        // Performance optimization callbacks
        onAnimationStart={() => {
          // Mark animation start for performance monitoring
          if (typeof performance !== 'undefined' && performance.mark) {
            performance.mark(`page-transition-${variant}-start`);
          }
        }}
        onAnimationComplete={() => {
          // Mark animation complete and clean up
          if (typeof performance !== 'undefined' && performance.mark) {
            performance.mark(`page-transition-${variant}-end`);
            performance.measure(
              `page-transition-${variant}`,
              `page-transition-${variant}-start`,
              `page-transition-${variant}-end`
            );
          }
        }}
      >
        {children}
      </motion.div>
    </AnimatePresence>
  );
});

PageTransition.displayName = 'PageTransition';

export default PageTransition;