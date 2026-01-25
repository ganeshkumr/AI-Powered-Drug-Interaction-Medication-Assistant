import { motion } from 'framer-motion';
import { ChevronLeft, ChevronRight, Check } from 'lucide-react';
import Button from '../common/Button';
import { animationConfigs, animationVariants } from '../../utils/performance';
import { manageFocusTransition, announceToScreenReader } from '../../utils/accessibility';

/**
 * StepNavigation Component
 * 
 * Provides conditional navigation enabling based on step completion.
 * Implements Next/Back button functionality with validation and smooth transitions.
 * 
 * Requirements: 2.3, 2.5, 2.6, 3.5, 4.4, 7.5
 */
const StepNavigation = ({
  currentStep,
  canProceed = false,
  isLoading = false,
  onNext,
  onBack,
  onComplete,
  nextLabel,
  backLabel = 'Back',
  completeLabel = 'Complete',
  className = '',
  'data-testid': testId = 'step-navigation'
}) => {
  // Determine if we can go back (not on first step)
  const canGoBack = currentStep > 1;
  
  // Determine if this is the final step
  const isFinalStep = currentStep === 3;
  
  // Get appropriate next button label
  const getNextLabel = () => {
    if (nextLabel) return nextLabel;
    if (isFinalStep) return completeLabel;
    if (currentStep === 1) return 'Next: Dosage';
    if (currentStep === 2) return 'Check Safety';
    return 'Next';
  };

  // Handle next button click
  const handleNext = () => {
    if (!canProceed || isLoading) return;
    
    if (isFinalStep && onComplete) {
      announceToScreenReader('Completing medication safety check');
      onComplete();
    } else if (onNext) {
      announceToScreenReader(`Proceeding to step ${currentStep + 1}`);
      manageFocusTransition(currentStep + 1, 'forward');
      onNext();
    }
  };

  // Handle back button click
  const handleBack = () => {
    if (!canGoBack || isLoading || !onBack) return;
    announceToScreenReader(`Returning to step ${currentStep - 1}`);
    manageFocusTransition(currentStep - 1, 'backward');
    onBack();
  };

  return (
    <motion.div
      {...animationVariants.slideUp}
      transition={animationConfigs.fast}
      className={`
        flex items-center justify-between 
        w-full max-w-2xl mx-auto 
        px-4 py-6 mt-8
        ${className}
      `}
      data-testid={testId}
      role="navigation"
      aria-label="Step navigation controls"
    >
      {/* Back Button */}
      <div className="flex-shrink-0">
        {canGoBack ? (
          <Button
            variant="ghost"
            size="md"
            onClick={handleBack}
            disabled={isLoading}
            icon={<ChevronLeft className="w-5 h-5" />}
            className="text-neutral-600 hover:text-neutral-800"
            data-testid="step-navigation-back"
            aria-label={`Go back to step ${currentStep - 1}`}
          >
            {backLabel}
          </Button>
        ) : (
          // Invisible placeholder to maintain layout
          <div className="w-20" aria-hidden="true" />
        )}
      </div>

      {/* Step Progress Indicator (Optional) */}
      <div className="hidden sm:flex items-center space-x-2" aria-hidden="true">
        <span className="text-sm text-neutral-500">
          Step {currentStep} of 3
        </span>
      </div>

      {/* Next/Complete Button */}
      <div className="flex-shrink-0">
        <Button
          variant={isFinalStep ? "primary" : "primary"}
          size="md"
          onClick={handleNext}
          disabled={!canProceed || isLoading}
          loading={isLoading}
          icon={
            isFinalStep ? (
              <Check className="w-5 h-5" />
            ) : (
              <ChevronRight className="w-5 h-5" />
            )
          }
          className={`
            min-w-[120px] sm:min-w-[140px]
            ${!canProceed && !isLoading ? 'opacity-50 cursor-not-allowed' : ''}
          `}
          data-testid="step-navigation-next"
          aria-label={
            isFinalStep 
              ? 'Complete medication safety check'
              : `Proceed to step ${currentStep + 1}`
          }
        >
          {getNextLabel()}
        </Button>
      </div>
    </motion.div>
  );
};

export default StepNavigation;