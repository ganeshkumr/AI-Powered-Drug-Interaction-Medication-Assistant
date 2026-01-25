import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import WarningModal from './WarningModal'

// Mock framer-motion
vi.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }) => <div {...props}>{children}</div>,
  },
  AnimatePresence: ({ children }) => children,
}))

// Mock Modal component
vi.mock('./Modal', () => ({
  default: ({ isOpen, onClose, children, ...props }) => 
    isOpen ? (
      <div data-testid="modal" {...props}>
        <button onClick={onClose} data-testid="close-button">Close</button>
        {children}
      </div>
    ) : null
}))

// Mock other components
vi.mock('./Button', () => ({
  default: ({ children, onClick, ...props }) => (
    <button onClick={onClick} {...props}>{children}</button>
  )
}))

vi.mock('../risk/RiskBadge', () => ({
  default: ({ riskLevel }) => <div data-testid="risk-badge">{riskLevel}</div>
}))

// Mock accessibility utils
vi.mock('../../utils/accessibility', () => ({
  announceToScreenReader: vi.fn(),
}))

describe('WarningModal', () => {
  const mockRiskData = {
    overallRisk: 85,
    overallVerdict: 'HIGH RISK',
    results: [
      {
        medication: 'Aspirin',
        ai_response: 'High risk of bleeding when combined with other medications.',
        gnn_risk: 85,
        verdict: 'UNSAFE'
      }
    ]
  }

  const defaultProps = {
    isOpen: true,
    onClose: vi.fn(),
    onProceed: vi.fn(),
    riskData: mockRiskData,
    medications: ['Aspirin', 'Warfarin'],
    title: 'Medication Safety Warning'
  }

  it('renders warning modal when open', () => {
    render(<WarningModal {...defaultProps} />)
    
    expect(screen.getByTestId('modal')).toBeInTheDocument()
    expect(screen.getByText('Medication Safety Warning')).toBeInTheDocument()
    expect(screen.getByText('Our AI analysis has identified potential safety concerns')).toBeInTheDocument()
  })

  it('does not render when closed', () => {
    render(<WarningModal {...defaultProps} isOpen={false} />)
    
    expect(screen.queryByTestId('modal')).not.toBeInTheDocument()
  })

  it('displays risk assessment information', () => {
    render(<WarningModal {...defaultProps} />)
    
    expect(screen.getByText('Risk Assessment')).toBeInTheDocument()
    expect(screen.getByText('85%')).toBeInTheDocument()
    expect(screen.getByText('HIGH RISK')).toBeInTheDocument()
  })

  it('displays AI analysis results', () => {
    render(<WarningModal {...defaultProps} />)
    
    expect(screen.getByText('AI Safety Analysis')).toBeInTheDocument()
    // Use more specific selector to avoid ambiguity with multiple "Aspirin" elements
    expect(screen.getByRole('heading', { name: /💊 Aspirin/ })).toBeInTheDocument()
    expect(screen.getByText(/High risk of bleeding/)).toBeInTheDocument()
  })

  it('displays medications being analyzed', () => {
    render(<WarningModal {...defaultProps} />)
    
    expect(screen.getByText('Medications Analyzed:')).toBeInTheDocument()
    // Use getAllByText to handle multiple instances and check the count
    const aspirinElements = screen.getAllByText('Aspirin')
    const warfarinElements = screen.getAllByText('Warfarin')
    expect(aspirinElements.length).toBeGreaterThan(0)
    expect(warfarinElements.length).toBeGreaterThan(0)
  })

  it('calls onClose when close button is clicked', () => {
    const onClose = vi.fn()
    render(<WarningModal {...defaultProps} onClose={onClose} />)
    
    fireEvent.click(screen.getByTestId('close-button'))
    expect(onClose).toHaveBeenCalled()
  })

  it('calls onProceed when proceed button is clicked', () => {
    const onProceed = vi.fn()
    render(<WarningModal {...defaultProps} onProceed={onProceed} />)
    
    const proceedButton = screen.getByText('I Understand, Continue')
    fireEvent.click(proceedButton)
    expect(onProceed).toHaveBeenCalled()
  })

  it('shows custom proceed button text', () => {
    render(
      <WarningModal 
        {...defaultProps} 
        proceedButtonText="Custom Proceed Text" 
      />
    )
    
    expect(screen.getByText('Custom Proceed Text')).toBeInTheDocument()
  })

  it('hides proceed button when showProceedButton is false', () => {
    render(<WarningModal {...defaultProps} showProceedButton={false} />)
    
    expect(screen.queryByText('I Understand, Continue')).not.toBeInTheDocument()
  })

  it('displays important notice', () => {
    render(<WarningModal {...defaultProps} />)
    
    expect(screen.getByText('Important Notice:')).toBeInTheDocument()
    expect(screen.getByText(/This analysis is for informational purposes only/)).toBeInTheDocument()
  })

  it('handles missing risk data gracefully', () => {
    render(<WarningModal {...defaultProps} riskData={null} />)
    
    expect(screen.getByTestId('modal')).toBeInTheDocument()
    expect(screen.getByText('Medication Safety Warning')).toBeInTheDocument()
  })

  it('formats AI response with HTML', () => {
    const riskDataWithFormatting = {
      ...mockRiskData,
      results: [
        {
          medication: 'Test Drug',
          ai_response: '**Bold text** and *italic text*\nNew line text',
          gnn_risk: 50,
          verdict: 'CAUTION'
        }
      ]
    }

    render(<WarningModal {...defaultProps} riskData={riskDataWithFormatting} />)
    
    // The formatted HTML should be rendered
    expect(screen.getByText('Test Drug')).toBeInTheDocument()
  })
})