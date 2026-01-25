import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import StepNavigation from './StepNavigation';

describe('StepNavigation', () => {
  const defaultProps = {
    currentStep: 1,
    canProceed: true,
    onNext: vi.fn(),
    onBack: vi.fn(),
    onComplete: vi.fn(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders correctly for step 1', () => {
    render(<StepNavigation {...defaultProps} />);
    
    expect(screen.getByTestId('step-navigation')).toBeInTheDocument();
    expect(screen.getByTestId('step-navigation-next')).toBeInTheDocument();
    expect(screen.queryByTestId('step-navigation-back')).not.toBeInTheDocument();
  });

  it('renders back button for steps 2 and 3', () => {
    render(<StepNavigation {...defaultProps} currentStep={2} />);
    
    expect(screen.getByTestId('step-navigation-back')).toBeInTheDocument();
    expect(screen.getByTestId('step-navigation-next')).toBeInTheDocument();
  });

  it('disables next button when canProceed is false', () => {
    render(<StepNavigation {...defaultProps} canProceed={false} />);
    
    const nextButton = screen.getByTestId('step-navigation-next');
    expect(nextButton).toBeDisabled();
  });

  it('enables next button when canProceed is true', () => {
    render(<StepNavigation {...defaultProps} canProceed={true} />);
    
    const nextButton = screen.getByTestId('step-navigation-next');
    expect(nextButton).not.toBeDisabled();
  });

  it('calls onNext when next button is clicked', () => {
    const onNext = vi.fn();
    render(<StepNavigation {...defaultProps} onNext={onNext} />);
    
    fireEvent.click(screen.getByTestId('step-navigation-next'));
    expect(onNext).toHaveBeenCalledTimes(1);
  });

  it('calls onBack when back button is clicked', () => {
    const onBack = vi.fn();
    render(<StepNavigation {...defaultProps} currentStep={2} onBack={onBack} />);
    
    fireEvent.click(screen.getByTestId('step-navigation-back'));
    expect(onBack).toHaveBeenCalledTimes(1);
  });

  it('shows complete button on final step', () => {
    render(<StepNavigation {...defaultProps} currentStep={3} />);
    
    const nextButton = screen.getByTestId('step-navigation-next');
    expect(nextButton).toHaveTextContent('Complete');
  });

  it('calls onComplete when complete button is clicked on final step', () => {
    const onComplete = vi.fn();
    render(<StepNavigation {...defaultProps} currentStep={3} onComplete={onComplete} />);
    
    fireEvent.click(screen.getByTestId('step-navigation-next'));
    expect(onComplete).toHaveBeenCalledTimes(1);
  });

  it('shows loading state correctly', () => {
    render(<StepNavigation {...defaultProps} isLoading={true} />);
    
    const nextButton = screen.getByTestId('step-navigation-next');
    expect(nextButton).toBeDisabled();
    expect(screen.getByText('Loading...')).toBeInTheDocument();
  });

  it('uses custom labels when provided', () => {
    render(
      <StepNavigation 
        {...defaultProps} 
        currentStep={2}
        nextLabel="Custom Next"
        backLabel="Custom Back"
      />
    );
    
    expect(screen.getByText('Custom Next')).toBeInTheDocument();
    expect(screen.getByText('Custom Back')).toBeInTheDocument();
  });

  it('shows appropriate step labels for each step', () => {
    // Step 1
    const { rerender } = render(<StepNavigation {...defaultProps} currentStep={1} />);
    expect(screen.getByText('Next: Dosage')).toBeInTheDocument();

    // Step 2
    rerender(<StepNavigation {...defaultProps} currentStep={2} />);
    expect(screen.getByText('Check Safety')).toBeInTheDocument();

    // Step 3
    rerender(<StepNavigation {...defaultProps} currentStep={3} />);
    expect(screen.getByText('Complete')).toBeInTheDocument();
  });
});