import React from 'react';
import { motion } from 'framer-motion';
import { Brain, Shield, Pill, Activity } from 'lucide-react';

/**
 * AnalysisLoader Component
 * 
 * Specialized loading animation for AI analysis with medical-appropriate styling.
 * Provides visual feedback during medication safety analysis processing.
 * 
 * Requirements: 6.2, 6.3
 */
const AnalysisLoader = ({ 
  message = 'Analyzing Your Medications',
  submessage = 'Please wait while we check for interactions and safety concerns...',
  medicationCount = 0,
  className = '',
  'data-testid': testId = 'analysis-loader'
}) => {
  // Animation variants for the main loader
  const containerVariants = {
    initial: { opacity: 0, scale: 0.9 },
    animate: { 
      opacity: 1, 
      scale: 1,
      transition: {
        duration: 0.3,
        ease: 'easeOut',
        staggerChildren: 0.1
      }
    }
  };

  // Animation for floating icons
  const iconVariants = {
    initial: { opacity: 0, y: 20 },
    animate: { 
      opacity: 1, 
      y: 0,
      transition: {
        duration: 0.4,
        ease: 'easeOut'
      }
    }
  };

  // Pulsing animation for the main spinner
  const spinnerVariants = {
    animate: {
      rotate: 360,
      transition: {
        duration: 2,
        ease: 'linear',
        repeat: Infinity
      }
    }
  };

  // Floating animation for background icons
  const floatingVariants = {
    animate: {
      y: [-10, 10, -10],
      transition: {
        duration: 3,
        ease: 'easeInOut',
        repeat: Infinity
      }
    }
  };

  // Progress bar animation
  const progressVariants = {
    initial: { width: '0%' },
    animate: {
      width: ['0%', '30%', '60%', '90%', '100%'],
      transition: {
        duration: 4,
        ease: 'easeInOut',
        repeat: Infinity,
        repeatDelay: 1
      }
    }
  };

  return (
    <div 
      className={`min-h-screen bg-gradient-to-br from-neutral-50 to-primary-50 flex items-center justify-center p-4 ${className}`}
      data-testid={testId}
      role="status"
      aria-live="polite"
      aria-label={`${message}. ${submessage}`}
    >
      <motion.div
        variants={containerVariants}
        initial="initial"
        animate="animate"
        className="text-center max-w-md mx-auto relative"
      >
        {/* Background floating icons */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none" aria-hidden="true">
          <motion.div
            variants={floatingVariants}
            animate="animate"
            className="absolute top-4 left-4 text-primary-200"
          >
            <Pill className="w-6 h-6" />
          </motion.div>
          <motion.div
            variants={floatingVariants}
            animate="animate"
            style={{ animationDelay: '1s' }}
            className="absolute top-8 right-8 text-secondary-200"
          >
            <Shield className="w-5 h-5" />
          </motion.div>
          <motion.div
            variants={floatingVariants}
            animate="animate"
            style={{ animationDelay: '2s' }}
            className="absolute bottom-12 left-8 text-primary-200"
          >
            <Activity className="w-4 h-4" />
          </motion.div>
        </div>

        {/* Main spinner with brain icon */}
        <motion.div
          variants={iconVariants}
          className="mb-6 relative"
        >
          <motion.div
            variants={spinnerVariants}
            animate="animate"
            className="w-20 h-20 mx-auto relative"
          >
            {/* Outer ring */}
            <div className="absolute inset-0 border-4 border-primary-200 rounded-full"></div>
            {/* Spinning ring */}
            <div className="absolute inset-0 border-4 border-transparent border-t-primary-500 rounded-full"></div>
          </motion.div>
          
          {/* Brain icon in center */}
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.3, duration: 0.3 }}
            className="absolute inset-0 flex items-center justify-center"
          >
            <Brain className="w-8 h-8 text-primary-600" />
          </motion.div>
        </motion.div>

        {/* Loading text */}
        <motion.div variants={iconVariants} className="mb-6">
          <h2 className="text-2xl sm:text-3xl font-bold text-neutral-900 mb-3">
            {message}
          </h2>
          <p className="text-neutral-600 mb-4 leading-relaxed">
            {submessage}
          </p>
          
          {medicationCount > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              className="bg-primary-50 border border-primary-200 rounded-lg p-4"
            >
              <p className="text-sm text-primary-700 flex items-center justify-center space-x-2">
                <Pill className="w-4 h-4" />
                <span>
                  Analyzing {medicationCount} medication{medicationCount > 1 ? 's' : ''} for potential interactions
                </span>
              </p>
            </motion.div>
          )}
        </motion.div>

        {/* Progress bar */}
        <motion.div
          variants={iconVariants}
          className="w-full bg-neutral-200 rounded-full h-2 mb-4 overflow-hidden"
        >
          <motion.div
            variants={progressVariants}
            initial="initial"
            animate="animate"
            className="h-full bg-gradient-to-r from-primary-500 to-secondary-500 rounded-full"
          />
        </motion.div>

        {/* Processing steps indicator */}
        <motion.div
          variants={iconVariants}
          className="text-xs text-neutral-500 space-y-1"
        >
          <motion.div
            animate={{ opacity: [0.5, 1, 0.5] }}
            transition={{ duration: 2, repeat: Infinity }}
            className="flex items-center justify-center space-x-2"
          >
            <div className="w-2 h-2 bg-primary-400 rounded-full"></div>
            <span>Checking drug interactions</span>
          </motion.div>
          <motion.div
            animate={{ opacity: [0.5, 1, 0.5] }}
            transition={{ duration: 2, repeat: Infinity, delay: 0.5 }}
            className="flex items-center justify-center space-x-2"
          >
            <div className="w-2 h-2 bg-secondary-400 rounded-full"></div>
            <span>Analyzing safety profiles</span>
          </motion.div>
          <motion.div
            animate={{ opacity: [0.5, 1, 0.5] }}
            transition={{ duration: 2, repeat: Infinity, delay: 1 }}
            className="flex items-center justify-center space-x-2"
          >
            <div className="w-2 h-2 bg-primary-400 rounded-full"></div>
            <span>Generating recommendations</span>
          </motion.div>
        </motion.div>

        {/* Accessibility announcement */}
        <div className="sr-only" aria-live="polite" aria-atomic="true">
          AI analysis in progress. Please wait while we process your medication information.
        </div>
      </motion.div>
    </div>
  );
};

export default AnalysisLoader;