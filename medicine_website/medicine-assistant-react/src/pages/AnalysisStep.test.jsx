import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { BrowserRouter } from 'react-router-dom'
import { vi } from 'vitest'
import { fc, test } from '@fast-check/vitest'
import AnalysisStep from './AnalysisStep'
import { medicationAPI } from '../services/api'

// Mock lucide-react icons
vi.mock('lucide-react', () => ({
  ArrowLeft: () => <div data-testid="arrow-left-icon" />,
  Save: () => <div data-testid="save-icon" />,
  MessageCircle: () => <div data-testid="message-circle-icon" />,
  CheckCircle: () => <div data-testid="check-circle-icon" />,
  AlertCircle: () => <div data-testid="alert-circle-icon" />,
  Loader: () => <div data-testid="loader-icon" />
}))

// Mock the navigate function
const mockNavigate = vi.fn()
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom')
  return {
    ...actual,
    useNavigate: () => mockNavigate
  }
})

// Mock the API
vi.mock('../services/api', () => ({
  medicationAPI: {
    checkBeforeAdding: vi.fn(),
    addMedication: vi.fn()
  }
}))

// Mock components
vi.mock('../components/navigation/StepIndicator', () => ({
  default: ({ currentStep, completedSteps }) => (
    <div data-testid="step-indicator">
      Step {currentStep}, Completed: {completedSteps.join(',')}
    </div>
  )
}))

vi.mock('../components/risk/RiskBadge', () => ({
  default: ({ riskLevel, size }) => (
    <div data-testid="risk-badge" data-risk-level={riskLevel} data-size={size}>
      Risk: {riskLevel}
    </div>
  )
}))

vi.mock('../components/common/Button', () => ({
  default: ({ children, onClick, disabled, loading, variant, size, icon, className }) => (
    <button 
      onClick={onClick} 
      disabled={disabled || loading} 
      className={className}
      data-variant={variant}
      data-size={size}
      data-loading={loading}
    >
      {icon && <span data-testid="button-icon">{icon}</span>}
      {children}
    </button>
  )
}))

vi.mock('../components/common/Card', () => ({
  default: ({ children, className }) => (
    <div className={className} data-testid="card">{children}</div>
  )
}))

const renderWithRouter = (component) => {
  return render(
    <BrowserRouter>
      {component}
    </BrowserRouter>
  )
}

describe('AnalysisStep', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    sessionStorage.clear()
    
    // Set up default session storage data
    sessionStorage.setItem('selectedMedications', JSON.stringify(['Aspirin', 'Lisinopril']))
    sessionStorage.setItem('dosageData', JSON.stringify({
      'Aspirin': {
        dosage_amount: '100',
        dosage_unit: 'mg',
        frequency: 'Once daily',
        start_date: '2024-01-01'
      },
      'Lisinopril': {
        dosage_amount: '10',
        dosage_unit: 'mg',
        frequency: 'Once daily',
        start_date: '2024-01-01'
      }
    }))
  })

  describe('Component Initialization', () => {
    test('redirects to medication step when no data in sessionStorage', () => {
      sessionStorage.clear()
      renderWithRouter(<AnalysisStep />)
      
      expect(mockNavigate).toHaveBeenCalledWith('/check/medication')
    })

    test('shows loading state initially', () => {
      medicationAPI.checkBeforeAdding.mockImplementation(() => new Promise(() => {})) // Never resolves
      
      renderWithRouter(<AnalysisStep />)
      
      expect(screen.getByText('Analyzing Your Medications')).toBeInTheDocument()
      expect(screen.getByText('Please wait while we check for interactions and safety concerns...')).toBeInTheDocument()
      expect(screen.getByTestId('loader-icon')).toBeInTheDocument()
    })

    test('displays step indicator with correct step and completion status', async () => {
      medicationAPI.checkBeforeAdding.mockResolvedValue({
        data: { verdict: 'SAFE', ai_response: 'Safe to use', gnn_risk: 20 }
      })
      
      renderWithRouter(<AnalysisStep />)
      
      await waitFor(() => {
        expect(screen.getByTestId('step-indicator')).toBeInTheDocument()
        expect(screen.getByText('Step 3, Completed: 1,2')).toBeInTheDocument()
      })
    })
  })

  describe('Safety Analysis Results Display', () => {
    test('displays safe analysis results correctly', async () => {
      medicationAPI.checkBeforeAdding.mockResolvedValue({
        data: { verdict: 'SAFE', ai_response: 'This medication is safe to use.', gnn_risk: 20 }
      })
      
      renderWithRouter(<AnalysisStep />)
      
      await waitFor(() => {
        expect(screen.getByText('Safety Analysis Results')).toBeInTheDocument()
        expect(screen.getByText('SAFE')).toBeInTheDocument()
        expect(screen.getByTestId('risk-badge')).toHaveAttribute('data-risk-level', 'safe')
        expect(screen.getByTestId('risk-badge')).toHaveAttribute('data-size', 'large')
      })
    })

    test('displays caution analysis results correctly', async () => {
      medicationAPI.checkBeforeAdding.mockResolvedValue({
        data: { verdict: 'CAUTION', ai_response: 'Use with caution.', gnn_risk: 50 }
      })
      
      renderWithRouter(<AnalysisStep />)
      
      await waitFor(() => {
        expect(screen.getByText('CAUTION')).toBeInTheDocument()
        expect(screen.getByTestId('risk-badge')).toHaveAttribute('data-risk-level', 'caution')
      })
    })

    test('displays high risk analysis results correctly', async () => {
      medicationAPI.checkBeforeAdding.mockResolvedValue({
        data: { verdict: 'UNSAFE', ai_response: 'High risk detected.', gnn_risk: 80 }
      })
      
      renderWithRouter(<AnalysisStep />)
      
      await waitFor(() => {
        expect(screen.getByText('HIGH RISK')).toBeInTheDocument()
        expect(screen.getByTestId('risk-badge')).toHaveAttribute('data-risk-level', 'high-risk')
      })
    })

    test('displays error state when analysis fails', async () => {
      medicationAPI.checkBeforeAdding.mockRejectedValue(new Error('API Error'))
      
      renderWithRouter(<AnalysisStep />)
      
      await waitFor(() => {
        expect(screen.getByText('ERROR')).toBeInTheDocument()
        // The error is displayed in individual medication results, not in a separate "Analysis Error" section
        expect(screen.getByText('Failed to analyze Aspirin. Please try again.')).toBeInTheDocument()
        expect(screen.getByText('Failed to analyze Lisinopril. Please try again.')).toBeInTheDocument()
      })
    })
  })

  describe('AI Explanation Display', () => {
    test('displays AI explanation box with proper formatting', async () => {
      medicationAPI.checkBeforeAdding.mockResolvedValue({
        data: { 
          verdict: 'SAFE', 
          ai_response: 'This medication is safe.\nNo interactions detected.', 
          gnn_risk: 20 
        }
      })
      
      renderWithRouter(<AnalysisStep />)
      
      await waitFor(() => {
        expect(screen.getByText('AI Safety Analysis')).toBeInTheDocument()
        expect(screen.getByText('Personalized for your medications')).toBeInTheDocument()
        
        // Check that AI response is displayed - the text gets split by HTML breaks
        expect(screen.getByText('This medication is safe.')).toBeInTheDocument()
        expect(screen.getByText('No interactions detected.')).toBeInTheDocument()
      })
    })

    test('displays combined analysis summary for multiple medications', async () => {
      medicationAPI.checkBeforeAdding
        .mockResolvedValueOnce({
          data: { verdict: 'SAFE', ai_response: 'Aspirin is safe.', gnn_risk: 20 }
        })
        .mockResolvedValueOnce({
          data: { verdict: 'SAFE', ai_response: 'Lisinopril is safe.', gnn_risk: 15 }
        })
      
      renderWithRouter(<AnalysisStep />)
      
      await waitFor(() => {
        expect(screen.getByText('Combined Analysis Summary')).toBeInTheDocument()
        expect(screen.getByText(/We've analyzed all 2 medications together/)).toBeInTheDocument()
      })
    })

    test('displays individual medication analysis results', async () => {
      medicationAPI.checkBeforeAdding
        .mockResolvedValueOnce({
          data: { verdict: 'SAFE', ai_response: 'Aspirin is safe to use.', gnn_risk: 20 }
        })
        .mockResolvedValueOnce({
          data: { verdict: 'SAFE', ai_response: 'Lisinopril is safe to use.', gnn_risk: 15 }
        })
      
      renderWithRouter(<AnalysisStep />)
      
      await waitFor(() => {
        // Check that both medications appear in the analysis section
        const aspirinElements = screen.getAllByText('Aspirin')
        const lisinoprilElements = screen.getAllByText('Lisinopril')
        expect(aspirinElements.length).toBeGreaterThan(0)
        expect(lisinoprilElements.length).toBeGreaterThan(0)
        
        // Check risk scores are displayed
        expect(screen.getByText('Risk Score: 20%')).toBeInTheDocument()
        expect(screen.getByText('Risk Score: 15%')).toBeInTheDocument()
        
        // Check AI responses are displayed
        expect(screen.getByText('Aspirin is safe to use.')).toBeInTheDocument()
        expect(screen.getByText('Lisinopril is safe to use.')).toBeInTheDocument()
      })
    })
  })

  describe('Medication Summary Display', () => {
    test('displays medication summary with dosage information', async () => {
      medicationAPI.checkBeforeAdding.mockResolvedValue({
        data: { verdict: 'SAFE', ai_response: 'Safe to use', gnn_risk: 20 }
      })
      
      renderWithRouter(<AnalysisStep />)
      
      await waitFor(() => {
        expect(screen.getByText('Medication Summary')).toBeInTheDocument()
        expect(screen.getByText('100 mg - Once daily')).toBeInTheDocument()
        expect(screen.getByText('10 mg - Once daily')).toBeInTheDocument()
        
        // Check that start dates are displayed (there are two, so use getAllByText)
        const startDates = screen.getAllByText('Start: 2024-01-01')
        expect(startDates).toHaveLength(2) // One for each medication
      })
    })
  })

  describe('Save Medication Functionality', () => {
    test('displays save button when analysis is safe', async () => {
      medicationAPI.checkBeforeAdding.mockResolvedValue({
        data: { verdict: 'SAFE', ai_response: 'Safe to use', gnn_risk: 20 }
      })
      
      renderWithRouter(<AnalysisStep />)
      
      await waitFor(() => {
        const saveButton = screen.getByText('Save to My Medications')
        expect(saveButton).toBeInTheDocument()
        expect(saveButton).not.toBeDisabled()
        expect(saveButton).toHaveAttribute('data-variant', 'primary')
        expect(saveButton).toHaveAttribute('data-size', 'lg')
      })
    })

    test('hides save button when analysis shows high risk', async () => {
      medicationAPI.checkBeforeAdding.mockResolvedValue({
        data: { verdict: 'UNSAFE', ai_response: 'High risk detected', gnn_risk: 80 }
      })
      
      renderWithRouter(<AnalysisStep />)
      
      await waitFor(() => {
        expect(screen.queryByText('Save to My Medications')).not.toBeInTheDocument()
      })
    })

    test('saves medications and navigates to dashboard when save button clicked', async () => {
      medicationAPI.checkBeforeAdding.mockResolvedValue({
        data: { verdict: 'SAFE', ai_response: 'Safe to use', gnn_risk: 20 }
      })
      medicationAPI.addMedication.mockResolvedValue({})
      
      renderWithRouter(<AnalysisStep />)
      
      await waitFor(() => {
        const saveButton = screen.getByText('Save to My Medications')
        fireEvent.click(saveButton)
      })
      
      await waitFor(() => {
        expect(medicationAPI.addMedication).toHaveBeenCalledTimes(2)
        expect(medicationAPI.addMedication).toHaveBeenCalledWith({
          drug_name: 'Aspirin',
          dosage_amount: '100',
          dosage_unit: 'mg',
          frequency: 'Once daily',
          start_date: '2024-01-01',
          end_date: undefined
        })
        expect(mockNavigate).toHaveBeenCalledWith('/dashboard')
      })
    })

    test('shows loading state during save operation', async () => {
      medicationAPI.checkBeforeAdding.mockResolvedValue({
        data: { verdict: 'SAFE', ai_response: 'Safe to use', gnn_risk: 20 }
      })
      medicationAPI.addMedication.mockImplementation(() => new Promise(() => {})) // Never resolves
      
      renderWithRouter(<AnalysisStep />)
      
      await waitFor(() => {
        const saveButton = screen.getByText('Save to My Medications')
        fireEvent.click(saveButton)
      })
      
      await waitFor(() => {
        const saveButton = screen.getByText('Save to My Medications')
        expect(saveButton).toHaveAttribute('data-loading', 'true')
        expect(saveButton).toBeDisabled()
      })
    })
  })

  describe('Chatbot Integration', () => {
    test('displays Ask AI Assistant button', async () => {
      medicationAPI.checkBeforeAdding.mockResolvedValue({
        data: { verdict: 'SAFE', ai_response: 'Safe to use', gnn_risk: 20 }
      })
      
      renderWithRouter(<AnalysisStep />)
      
      await waitFor(() => {
        const chatbotButton = screen.getByText('Ask AI Assistant')
        expect(chatbotButton).toBeInTheDocument()
        expect(chatbotButton).toHaveAttribute('data-variant', 'secondary')
        expect(chatbotButton).toHaveAttribute('data-size', 'lg')
        
        // Check that the chatbot button has an icon (there are multiple button icons, so check specifically for message circle)
        expect(screen.getByTestId('message-circle-icon')).toBeInTheDocument()
      })
    })

    test('chatbot button click logs interaction', async () => {
      const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {})
      
      medicationAPI.checkBeforeAdding.mockResolvedValue({
        data: { verdict: 'SAFE', ai_response: 'Safe to use', gnn_risk: 20 }
      })
      
      renderWithRouter(<AnalysisStep />)
      
      await waitFor(() => {
        const chatbotButton = screen.getByText('Ask AI Assistant')
        fireEvent.click(chatbotButton)
        
        expect(consoleSpy).toHaveBeenCalledWith('Opening chatbot for follow-up questions')
      })
      
      consoleSpy.mockRestore()
    })
  })

  describe('Navigation', () => {
    test('displays back button that navigates to dosage step', async () => {
      medicationAPI.checkBeforeAdding.mockResolvedValue({
        data: { verdict: 'SAFE', ai_response: 'Safe to use', gnn_risk: 20 }
      })
      
      renderWithRouter(<AnalysisStep />)
      
      await waitFor(() => {
        const backButton = screen.getByText('Back: Dosage')
        expect(backButton).toBeInTheDocument()
        
        fireEvent.click(backButton)
        expect(mockNavigate).toHaveBeenCalledWith('/check/dosage')
      })
    })

    test('displays start new check button', async () => {
      medicationAPI.checkBeforeAdding.mockResolvedValue({
        data: { verdict: 'SAFE', ai_response: 'Safe to use', gnn_risk: 20 }
      })
      
      renderWithRouter(<AnalysisStep />)
      
      await waitFor(() => {
        const newCheckButton = screen.getByText('Start New Check')
        expect(newCheckButton).toBeInTheDocument()
        
        fireEvent.click(newCheckButton)
        expect(mockNavigate).toHaveBeenCalledWith('/check/medication')
      })
    })
  })

  describe('Error Handling', () => {
    test('handles API errors gracefully', async () => {
      medicationAPI.checkBeforeAdding.mockRejectedValue(new Error('Network error'))
      
      renderWithRouter(<AnalysisStep />)
      
      await waitFor(() => {
        expect(screen.getByText('ERROR')).toBeInTheDocument()
        // The error is displayed in individual medication results
        expect(screen.getByText('Failed to analyze Aspirin. Please try again.')).toBeInTheDocument()
        expect(screen.getByText('Failed to analyze Lisinopril. Please try again.')).toBeInTheDocument()
      })
    })

    test('handles save medication errors', async () => {
      const alertSpy = vi.spyOn(window, 'alert').mockImplementation(() => {})
      
      medicationAPI.checkBeforeAdding.mockResolvedValue({
        data: { verdict: 'SAFE', ai_response: 'Safe to use', gnn_risk: 20 }
      })
      medicationAPI.addMedication.mockRejectedValue(new Error('Save failed'))
      
      renderWithRouter(<AnalysisStep />)
      
      await waitFor(() => {
        const saveButton = screen.getByText('Save to My Medications')
        fireEvent.click(saveButton)
      })
      
      await waitFor(() => {
        expect(alertSpy).toHaveBeenCalledWith('Failed to save medications. Please try again.')
      })
      
      alertSpy.mockRestore()
    })
  })

  describe('Risk Level Calculation', () => {
    test('calculates overall risk correctly for multiple medications', async () => {
      medicationAPI.checkBeforeAdding
        .mockResolvedValueOnce({
          data: { verdict: 'SAFE', ai_response: 'Safe', gnn_risk: 30 }
        })
        .mockResolvedValueOnce({
          data: { verdict: 'CAUTION', ai_response: 'Caution', gnn_risk: 60 }
        })
      
      renderWithRouter(<AnalysisStep />)
      
      await waitFor(() => {
        // Should use the highest risk score (60%) in the overall assessment
        const riskScores = screen.getAllByText('Risk Score: 60%')
        expect(riskScores.length).toBeGreaterThan(0) // Should appear at least once
        expect(screen.getByTestId('risk-badge')).toHaveAttribute('data-risk-level', 'caution')
      })
    })

    test('determines high risk verdict when any medication is unsafe', async () => {
      medicationAPI.checkBeforeAdding
        .mockResolvedValueOnce({
          data: { verdict: 'SAFE', ai_response: 'Safe', gnn_risk: 20 }
        })
        .mockResolvedValueOnce({
          data: { verdict: 'UNSAFE', ai_response: 'Unsafe', gnn_risk: 80 } // Use higher risk score to match 'high-risk' level
        })
      
      renderWithRouter(<AnalysisStep />)
      
      await waitFor(() => {
        expect(screen.getByText('HIGH RISK')).toBeInTheDocument()
        expect(screen.getByTestId('risk-badge')).toHaveAttribute('data-risk-level', 'high-risk')
      })
    })
  })

  // Feature: frontend-ui-redesign, Property 6: Analysis Step Display Elements
  test.prop([
    fc.integer({ min: 0, max: 100 }), // risk score
    fc.constantFrom('SAFE', 'CAUTION', 'UNSAFE', 'ERROR'), // verdict
    fc.string({ minLength: 10, maxLength: 200 }) // ai response
  ], { timeout: 15000 })('Property 6: Analysis Step Display Elements - validates Requirements 5.1, 5.2, 5.3, 5.4', async (riskScore, verdict, aiResponse) => {
    medicationAPI.checkBeforeAdding.mockResolvedValue({
      data: { verdict, ai_response: aiResponse, gnn_risk: riskScore }
    })
    
    renderWithRouter(<AnalysisStep />)
    
    await waitFor(() => {
      // Property: For any analysis results, the interface should display prominent risk badge,
      // AI explanation box, medication summary, and appropriate action buttons
      
      // Requirement 5.1: Prominent risk badge display
      expect(screen.getByTestId('risk-badge')).toBeInTheDocument()
      expect(screen.getByTestId('risk-badge')).toHaveAttribute('data-size', 'large')
      
      // Requirement 5.2: AI explanation box with readable formatting
      expect(screen.getByText('AI Safety Analysis')).toBeInTheDocument()
      expect(screen.getByText('Personalized for your medications')).toBeInTheDocument()
      
      // Requirement 5.3: Save Medication button (when appropriate)
      if (verdict === 'SAFE' || (verdict === 'CAUTION' && riskScore <= 70)) {
        expect(screen.getByText('Save to My Medications')).toBeInTheDocument()
      }
      
      // Requirement 5.4: Chatbot icon integration
      expect(screen.getByText('Ask AI Assistant')).toBeInTheDocument()
      
      // Additional interface elements that should always be present
      expect(screen.getByText('Safety Analysis Results')).toBeInTheDocument()
      expect(screen.getByText('Medication Summary')).toBeInTheDocument()
      expect(screen.getByText('Back: Dosage')).toBeInTheDocument()
      expect(screen.getByText('Start New Check')).toBeInTheDocument()
    }, { timeout: 10000 }) // Increase timeout for property-based test
  })
})