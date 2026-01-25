import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import { describe, expect, vi, beforeEach, afterEach } from 'vitest'
import { fc, test } from '@fast-check/vitest'
import StepNavigation from './StepNavigation'

// Feature: frontend-ui-redesign, Property 3: Conditional Navigation Enabling

/**
 * Property-Based Test for Conditional Navigation Enabling
 * 
 * Validates: Requirements 2.3, 2.5, 2.6, 3.5, 4.4
 * 
 * Property 3: Conditional Navigation Enabling
 * For any step in the medication flow, navigation to the next step should only be 
 * enabled when all required inputs for the current step are completed and validated.
 */

// Test data generators
const stepArbitrary = fc.integer({ min: 1, max: 3 })
const canProceedArbitrary = fc.boolean()
const isLoadingArbitrary = fc.boolean()

// Generate step completion states that represent different validation scenarios
const stepCompletionArbitrary = fc.record({
  currentStep: stepArbitrary,
  canProceed: canProceedArbitrary,
  isLoading: isLoadingArbitrary,
  hasRequiredData: fc.boolean(),
  hasValidationErrors: fc.boolean()
})

// Generate navigation interaction scenarios
const navigationInteractionArbitrary = fc.record({
  stepState: stepCompletionArbitrary,
  userAction: fc.oneof(
    fc.constant('click_next'),
    fc.constant('click_back'),
    fc.constant('keyboard_enter'),
    fc.constant('keyboard_space')
  )
})

describe('Conditional Navigation Enabling Property Tests', () => {
  
  beforeEach(() => {
    vi.clearAllMocks()
  })
  
  afterEach(() => {
    cleanup()
  })
  
  test.prop([stepCompletionArbitrary])(
    'Next button enablement reflects step completion state',
    (stepState) => {
      const mockOnNext = vi.fn()
      const mockOnBack = vi.fn()
      const mockOnComplete = vi.fn()
      const testId = `step-navigation-${Math.random().toString(36).substr(2, 9)}`
      
      const { container } = render(
        <StepNavigation
          currentStep={stepState.currentStep}
          canProceed={stepState.canProceed}
          isLoading={stepState.isLoading}
          onNext={mockOnNext}
          onBack={mockOnBack}
          onComplete={mockOnComplete}
          data-testid={testId}
        />
      )
      
      const nextButton = container.querySelector('[data-testid="step-navigation-next"]')
      
      // Property: Next button should only be enabled when canProceed is true AND not loading
      const shouldBeEnabled = stepState.canProceed && !stepState.isLoading
      
      if (shouldBeEnabled) {
        expect(nextButton).not.toBeDisabled()
      } else {
        expect(nextButton).toBeDisabled()
      }
      
      // Verify button accessibility attributes
      expect(nextButton).toHaveAttribute('aria-label')
      expect(nextButton).toBeInTheDocument()
      
      cleanup()
    }
  )

  test.prop([stepArbitrary, canProceedArbitrary])(
    'Back button availability depends on current step position',
    (currentStep, canProceed) => {
      const mockOnNext = vi.fn()
      const mockOnBack = vi.fn()
      const testId = `step-navigation-${Math.random().toString(36).substr(2, 9)}`
      
      const { container } = render(
        <StepNavigation
          currentStep={currentStep}
          canProceed={canProceed}
          onNext={mockOnNext}
          onBack={mockOnBack}
          data-testid={testId}
        />
      )
      
      // Property: Back button should only be available when not on first step
      const shouldShowBackButton = currentStep > 1
      const backButton = container.querySelector('[data-testid="step-navigation-back"]')
      
      if (shouldShowBackButton) {
        expect(backButton).toBeInTheDocument()
        expect(backButton).not.toBeDisabled()
      } else {
        expect(backButton).not.toBeInTheDocument()
      }
      
      cleanup()
    }
  )

  test.prop([navigationInteractionArbitrary])(
    'Navigation interactions respect enablement state',
    (interaction) => {
      const mockOnNext = vi.fn()
      const mockOnBack = vi.fn()
      const mockOnComplete = vi.fn()
      
      const { stepState, userAction } = interaction
      const testId = `step-navigation-${Math.random().toString(36).substr(2, 9)}`
      
      const { container } = render(
        <StepNavigation
          currentStep={stepState.currentStep}
          canProceed={stepState.canProceed}
          isLoading={stepState.isLoading}
          onNext={mockOnNext}
          onBack={mockOnBack}
          onComplete={mockOnComplete}
          data-testid={testId}
        />
      )
      
      const nextButton = container.querySelector('[data-testid="step-navigation-next"]')
      const backButton = container.querySelector('[data-testid="step-navigation-back"]')
      
      // Property: Callbacks should only be invoked when buttons are enabled
      const canClickNext = stepState.canProceed && !stepState.isLoading
      const canClickBack = stepState.currentStep > 1 && !stepState.isLoading
      
      if (userAction === 'click_next' && nextButton) {
        fireEvent.click(nextButton)
        
        if (canClickNext) {
          if (stepState.currentStep === 3) {
            expect(mockOnComplete).toHaveBeenCalledTimes(1)
          } else {
            expect(mockOnNext).toHaveBeenCalledTimes(1)
          }
        } else {
          expect(mockOnNext).not.toHaveBeenCalled()
          expect(mockOnComplete).not.toHaveBeenCalled()
        }
      }
      
      if (userAction === 'click_back' && backButton) {
        fireEvent.click(backButton)
        
        if (canClickBack) {
          expect(mockOnBack).toHaveBeenCalledTimes(1)
        } else {
          expect(mockOnBack).not.toHaveBeenCalled()
        }
      }
      
      cleanup()
    }
  )

  test.prop([stepArbitrary])(
    'Step-specific button labels and behavior are consistent',
    (currentStep) => {
      const mockOnNext = vi.fn()
      const mockOnComplete = vi.fn()
      const testId = `step-navigation-${Math.random().toString(36).substr(2, 9)}`
      
      const { container } = render(
        <StepNavigation
          currentStep={currentStep}
          canProceed={true}
          onNext={mockOnNext}
          onComplete={mockOnComplete}
          data-testid={testId}
        />
      )
      
      const nextButton = container.querySelector('[data-testid="step-navigation-next"]')
      
      // Property: Button labels should reflect current step context
      if (currentStep === 1) {
        expect(nextButton).toHaveTextContent('Next: Dosage')
      } else if (currentStep === 2) {
        expect(nextButton).toHaveTextContent('Check Safety')
      } else if (currentStep === 3) {
        expect(nextButton).toHaveTextContent('Complete')
      }
      
      // Property: Aria labels should provide context about navigation
      const ariaLabel = nextButton.getAttribute('aria-label')
      expect(ariaLabel).toBeTruthy()
      
      if (currentStep === 3) {
        expect(ariaLabel).toContain('Complete')
      } else {
        expect(ariaLabel).toContain(`step ${currentStep + 1}`)
      }
      
      cleanup()
    }
  )

  test.prop([stepCompletionArbitrary])(
    'Loading state prevents all navigation interactions',
    (stepState) => {
      const mockOnNext = vi.fn()
      const mockOnBack = vi.fn()
      const mockOnComplete = vi.fn()
      const testId = `step-navigation-${Math.random().toString(36).substr(2, 9)}`
      
      const { container } = render(
        <StepNavigation
          currentStep={stepState.currentStep}
          canProceed={stepState.canProceed}
          isLoading={true} // Force loading state
          onNext={mockOnNext}
          onBack={mockOnBack}
          onComplete={mockOnComplete}
          data-testid={testId}
        />
      )
      
      const nextButton = container.querySelector('[data-testid="step-navigation-next"]')
      const backButton = container.querySelector('[data-testid="step-navigation-back"]')
      
      // Property: Loading state should disable all navigation
      expect(nextButton).toBeDisabled()
      
      if (backButton) {
        expect(backButton).toBeDisabled()
      }
      
      // Property: Clicking disabled buttons should not trigger callbacks
      if (nextButton) {
        fireEvent.click(nextButton)
        expect(mockOnNext).not.toHaveBeenCalled()
        expect(mockOnComplete).not.toHaveBeenCalled()
      }
      
      if (backButton) {
        fireEvent.click(backButton)
        expect(mockOnBack).not.toHaveBeenCalled()
      }
      
      cleanup()
    }
  )

  test.prop([fc.record({
    currentStep: stepArbitrary,
    canProceed: fc.boolean(),
    customNextLabel: fc.option(fc.string({ minLength: 1, maxLength: 20 })),
    customBackLabel: fc.option(fc.string({ minLength: 1, maxLength: 20 })),
    customCompleteLabel: fc.option(fc.string({ minLength: 1, maxLength: 20 }))
  })])(
    'Custom labels preserve navigation behavior',
    (config) => {
      const mockOnNext = vi.fn()
      const mockOnBack = vi.fn()
      const mockOnComplete = vi.fn()
      const testId = `step-navigation-${Math.random().toString(36).substr(2, 9)}`
      
      const { container } = render(
        <StepNavigation
          currentStep={config.currentStep}
          canProceed={config.canProceed}
          nextLabel={config.customNextLabel}
          backLabel={config.customBackLabel}
          completeLabel={config.customCompleteLabel}
          onNext={mockOnNext}
          onBack={mockOnBack}
          onComplete={mockOnComplete}
          data-testid={testId}
        />
      )
      
      const nextButton = container.querySelector('[data-testid="step-navigation-next"]')
      
      // Property: Custom labels should not affect enablement logic
      const shouldBeEnabled = config.canProceed
      
      if (shouldBeEnabled) {
        expect(nextButton).not.toBeDisabled()
      } else {
        expect(nextButton).toBeDisabled()
      }
      
      // Property: Custom labels should be displayed when provided
      // The component prioritizes nextLabel over completeLabel for step 3
      if (config.customNextLabel && config.customNextLabel.trim()) {
        expect(nextButton.textContent.trim()).toBe(config.customNextLabel.trim())
      } else if (config.customCompleteLabel && config.customCompleteLabel.trim() && config.currentStep === 3 && !config.customNextLabel) {
        expect(nextButton.textContent.trim()).toBe(config.customCompleteLabel.trim())
      }
      
      if (config.currentStep > 1) {
        const backButton = container.querySelector('[data-testid="step-navigation-back"]')
        if (backButton && config.customBackLabel && config.customBackLabel.trim()) {
          expect(backButton.textContent.trim()).toBe(config.customBackLabel.trim())
        }
      }
      
      cleanup()
    }
  )

  test.prop([stepArbitrary, canProceedArbitrary])(
    'Navigation component maintains accessibility standards',
    (currentStep, canProceed) => {
      const testId = `step-navigation-${Math.random().toString(36).substr(2, 9)}`
      
      const { container } = render(
        <StepNavigation
          currentStep={currentStep}
          canProceed={canProceed}
          onNext={vi.fn()}
          onBack={vi.fn()}
          onComplete={vi.fn()}
          data-testid={testId}
        />
      )
      
      const navigationContainer = container.querySelector(`[data-testid="${testId}"]`)
      const nextButton = container.querySelector('[data-testid="step-navigation-next"]')
      
      // Property: All interactive elements should have proper accessibility attributes
      expect(navigationContainer).toBeInTheDocument()
      expect(nextButton).toHaveAttribute('aria-label')
      
      // Property: Disabled state should be properly communicated
      if (!canProceed) {
        expect(nextButton).toBeDisabled()
        expect(nextButton).toHaveAttribute('disabled')
      }
      
      // Property: Back button accessibility when present
      if (currentStep > 1) {
        const backButton = container.querySelector('[data-testid="step-navigation-back"]')
        if (backButton) {
          expect(backButton).toHaveAttribute('aria-label')
          expect(backButton.getAttribute('aria-label')).toContain(`step ${currentStep - 1}`)
        }
      }
      
      cleanup()
    }
  )
})