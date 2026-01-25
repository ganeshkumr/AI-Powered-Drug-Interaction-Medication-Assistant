import React from 'react';
import { motion } from 'framer-motion';
import { Check, Pill, Clipboard, Shield } from 'lucide-react';
import { handleKeyboardNavigation, announceToScreenReader } from '../../utils/accessibility';

/**
 * StepIndicator Component
 * 
 * Visual progress indicator for the three-step medication checking process.
 * Shows current step, completed steps, and allows navigation between steps.
 * Enhanced with mobile-first responsive design.
 * 
 * Requirements: 2.1, 2.2, 2.4, 8.1, 8.3, 8.4, 8.5
 */
const StepIndicator = ({ 
  currentStep, 
  completedSteps = [], 
  onStepClick,
  className = '',
  'data-testid': testId = 'step-indicator'
}) => {
  const steps = [
    {
      number: 1,
      title: 'Medication',
      description: 'Select your medications',
      shortDescription: 'Select meds',
      icon: Pill,
    },
    {
      number: 2,
      title: 'Dosage',
      description: 'Enter dosage details',
      shortDescription: 'Add dosage',
      icon: Clipboard,
    },
    {
      number: 3,
      title: 'Analysis',
      description: 'Safety analysis results',
      shortDescription: 'View results',
      icon: Shield,
    },
  ];

  const getStepStatus = (stepNumber) => {
    if (completedSteps.includes(stepNumber)) return 'completed';
    if (stepNumber === currentStep) return 'active';
    return 'inactive';
  };

  const getStepStyles = (status) => {
    switch (status) {
      case 'completed':
        return {
          circle: 'bg-success-500 border-success-500 text-white shadow-sm',
          text: 'text-success-600',
          line: 'bg-success-500',
        };
      case 'active':
        return {
          circle: 'bg-primary-500 border-primary-500 text-white shadow-md',
          text: 'text-primary-600',
          line: 'bg-neutral-200',
        };
      default:
        return {
          circle: 'bg-white border-neutral-300 text-neutral-400 shadow-sm',
          text: 'text-neutral-400',
          line: 'bg-neutral-200',
        };
    }
  };

  const handleStepClick = (stepNumber) => {
    // Only allow clicking on completed steps or current step
    if (onStepClick && (completedSteps.includes(stepNumber) || stepNumber === currentStep)) {
      onStepClick(stepNumber);
      announceToScreenReader(`Navigating to step ${stepNumber}: ${steps[stepNumber - 1]?.title}`);
    }
  };

  const handleStepKeyDown = (event, stepNumber) => {
    handleKeyboardNavigation(event, {
      onEnter: () => handleStepClick(stepNumber),
      onSpace: (e) => {
        e.preventDefault();
        handleStepClick(stepNumber);
      }
    });
  };

  return (
    <div 
      className={`w-full container-responsive py-4 sm:py-6 ${className}`}
      data-testid={testId}
      role="progressbar"
      aria-valuenow={currentStep}
      aria-valuemin={1}
      aria-valuemax={3}
      aria-label={`Step ${currentStep} of 3: ${steps[currentStep - 1]?.title}`}
    >
      {/* Mobile Step Indicator (Compact) */}
      <div className="block sm:hidden">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-3">
            <div className={`
              w-8 h-8 rounded-full border-2 flex items-center justify-center
              ${getStepStyles(getStepStatus(currentStep)).circle}
            `}>
              {getStepStatus(currentStep) === 'completed' ? (
                <Check className="w-4 h-4" aria-hidden="true" />
              ) : (
                React.createElement(steps[currentStep - 1].icon, { className: "w-4 h-4", "aria-hidden": "true" })
              )}
            </div>
            <div>
              <h3 className="text-sm font-semibold text-neutral-800">
                Step {currentStep}: {steps[currentStep - 1]?.title}
              </h3>
              <p className="text-xs text-neutral-500">
                {steps[currentStep - 1]?.shortDescription}
              </p>
            </div>
          </div>
          <div className="text-xs font-medium text-neutral-500 bg-neutral-100 px-2 py-1 rounded-full">
            {currentStep}/3
          </div>
        </div>
        
        {/* Mobile Progress Bar */}
        <div className="w-full bg-neutral-200 rounded-full h-2 overflow-hidden">
          <motion.div
            className="bg-gradient-to-r from-primary-500 to-secondary-500 h-2 rounded-full relative"
            initial={{ width: '0%' }}
            animate={{ 
              width: `${((currentStep - 1) / 2) * 100}%`
            }}
            transition={{ 
              duration: 0.6, 
              ease: [0.4, 0, 0.2, 1]
            }}
          >
            {/* Animated shimmer effect */}
            <motion.div
              className="absolute inset-0 bg-gradient-to-r from-transparent via-white to-transparent opacity-30"
              animate={{
                x: ['-100%', '100%']
              }}
              transition={{
                duration: 2,
                repeat: Infinity,
                ease: 'linear'
              }}
              style={{
                background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent)'
              }}
            />
          </motion.div>
        </div>
      </div>

      {/* Desktop Step Indicator (Full) */}
      <div className="hidden sm:block max-w-2xl mx-auto">
        <div className="relative">
          {/* Progress Line Background */}
          <div className="absolute top-5 left-0 right-0 h-0.5 bg-neutral-200" aria-hidden="true" />
          
          {/* Progress Line Fill */}
          <motion.div
            className="absolute top-5 left-0 h-0.5 bg-gradient-to-r from-primary-500 to-secondary-500"
            initial={{ width: '0%' }}
            animate={{ 
              width: currentStep === 1 ? '0%' : currentStep === 2 ? '50%' : '100%' 
            }}
            transition={{ 
              duration: 0.6, 
              ease: [0.4, 0, 0.2, 1],
              type: 'tween'
            }}
            aria-hidden="true"
          />
          
          {/* Animated progress glow effect */}
          <motion.div
            className="absolute top-5 left-0 h-0.5 bg-gradient-to-r from-primary-400 to-secondary-400 opacity-50 blur-sm"
            initial={{ width: '0%' }}
            animate={{ 
              width: currentStep === 1 ? '0%' : currentStep === 2 ? '50%' : '100%' 
            }}
            transition={{ 
              duration: 0.6, 
              ease: [0.4, 0, 0.2, 1],
              delay: 0.1
            }}
            aria-hidden="true"
          />

          {/* Steps Container */}
          <div className="relative flex justify-between">
            {steps.map((step, index) => {
              const status = getStepStatus(step.number);
              const styles = getStepStyles(status);
              const isClickable = completedSteps.includes(step.number) || step.number === currentStep;

              return (
                <motion.div
                  key={step.number}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1, duration: 0.3 }}
                  className="flex flex-col items-center space-y-2 relative"
                >
                  {/* Step Circle */}
                  <motion.button
                    whileHover={isClickable ? { scale: 1.05 } : {}}
                    whileTap={isClickable ? { scale: 0.95 } : {}}
                    onClick={() => handleStepClick(step.number)}
                    onKeyDown={(e) => handleStepKeyDown(e, step.number)}
                    disabled={!isClickable}
                    className={`
                      w-10 h-10 md:w-12 md:h-12 rounded-full border-2 flex items-center justify-center
                      transition-all duration-300 relative z-10 min-h-touch min-w-touch
                      ${styles.circle}
                      ${isClickable ? 'cursor-pointer hover:shadow-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2' : 'cursor-default'}
                      ${!isClickable ? 'opacity-60' : ''}
                    `}
                    aria-label={`Step ${step.number}: ${step.title}. ${status === 'completed' ? 'Completed' : status === 'active' ? 'Current step' : 'Not yet available'}`}
                    aria-current={status === 'active' ? 'step' : undefined}
                    aria-describedby={`step-${step.number}-description`}
                    tabIndex={isClickable ? 0 : -1}
                  >
                    {/* Animated background glow for active step */}
                    {status === 'active' && (
                      <motion.div
                        className="absolute inset-0 rounded-full bg-primary-500 opacity-20"
                        animate={{
                          scale: [1, 1.2, 1],
                          opacity: [0.2, 0.4, 0.2]
                        }}
                        transition={{
                          duration: 2,
                          repeat: Infinity,
                          ease: 'easeInOut'
                        }}
                      />
                    )}
                    
                    {status === 'completed' ? (
                      <motion.div
                        initial={{ scale: 0, rotate: -180 }}
                        animate={{ scale: 1, rotate: 0 }}
                        transition={{ 
                          delay: 0.2,
                          type: 'spring',
                          stiffness: 300,
                          damping: 20
                        }}
                      >
                        <Check className="w-5 h-5 md:w-6 md:h-6" aria-hidden="true" />
                      </motion.div>
                    ) : (
                      <motion.div
                        initial={{ scale: 0.8, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        transition={{ delay: index * 0.1 }}
                      >
                        <step.icon className="w-5 h-5 md:w-6 md:h-6" aria-hidden="true" />
                      </motion.div>
                    )}
                  </motion.button>

                  {/* Step Content */}
                  <div className="text-center min-w-0 max-w-20 sm:max-w-24 md:max-w-none" id={`step-${step.number}-description`}>
                    <h3 className={`text-sm md:text-base font-semibold ${styles.text} transition-colors duration-200`}>
                      {step.title}
                    </h3>
                    <p className={`text-xs md:text-sm ${styles.text} opacity-75 hidden md:block transition-colors duration-200`}>
                      {step.description}
                    </p>
                    <p className={`text-xs ${styles.text} opacity-75 block md:hidden transition-colors duration-200`}>
                      {step.shortDescription}
                    </p>
                  </div>

                  {/* Step Number Badge (Tablet) */}
                  <div className="sm:hidden md:block absolute -top-1 -right-1 w-5 h-5 bg-white border border-neutral-200 rounded-full flex items-center justify-center">
                    <span className="text-xs font-medium text-neutral-600">
                      {step.number}
                    </span>
                  </div>
                </motion.div>
              );
            })}
          </div>
        </div>

        {/* Current Step Description (Tablet) */}
        <motion.div
          key={currentStep}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="md:hidden mt-4 text-center"
        >
          <p className="text-sm text-neutral-600">
            Step {currentStep}: {steps[currentStep - 1]?.description}
          </p>
        </motion.div>
      </div>

      {/* Step Navigation Hints (Mobile) */}
      <div className="block sm:hidden mt-4">
        <div className="flex justify-between items-center text-xs text-neutral-500">
          <div className="flex items-center space-x-1">
            {completedSteps.length > 0 && (
              <>
                <Check className="w-3 h-3 text-success-500" />
                <span>{completedSteps.length} completed</span>
              </>
            )}
          </div>
          <div>
            {currentStep < 3 && (
              <span>{3 - currentStep} remaining</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default StepIndicator;