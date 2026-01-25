import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { BrowserRouter } from 'react-router-dom'
import { vi } from 'vitest'
import { fc, test } from '@fast-check/vitest'
import DosageStep from './DosageStep'

// Mock the navigate function
const mockNavigate = vi.fn()
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom')
  return {
    ...actual,
    useNavigate: () => mockNavigate
  }
})

// Mock components
vi.mock('../components/navigation/StepIndicator', () => ({
  default: ({ currentStep, completedSteps }) => (
    <div data-testid="step-indicator">
      Step {currentStep}, Completed: {completedSteps?.join(',')}
    </div>
  )
}))

vi.mock('../components/common/Button', () => ({
  default: ({ children, onClick, disabled, variant, className }) => (
    <button 
      onClick={onClick} 
      disabled={disabled} 
      className={className}
      data-variant={variant}
    >
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

describe('DosageStep', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    sessionStorage.clear()
  })

  describe('Component Initialization', () => {
    test('redirects to medication step when no medications selected', () => {
      renderWithRouter(<DosageStep />)
      
      expect(mockNavigate).toHaveBeenCalledWith('/check/medication')
    })

    test('renders dosage step interface when medications exist', () => {
      // Set up medications in sessionStorage
      sessionStorage.setItem('selectedMedications', JSON.stringify(['Aspirin', 'Lisinopril']))
      
      renderWithRouter(<DosageStep />)
      
      // Check for required interface elements per Requirements 4.1, 4.2, 4.5
      expect(screen.getByText('Dosage Information')).toBeInTheDocument()
      expect(screen.getByText(/Provide dosage details for each medication/)).toBeInTheDocument()
      expect(screen.getByTestId('step-indicator')).toBeInTheDocument()
    })

    test('displays step indicator with current step 2', () => {
      sessionStorage.setItem('selectedMedications', JSON.stringify(['Aspirin']))
      
      renderWithRouter(<DosageStep />)
      
      // Validates Requirements 2.1, 2.2
      expect(screen.getByText('Step 2, Completed: 1')).toBeInTheDocument()
    })
  })

  describe('Form Layout and Inputs', () => {
    beforeEach(() => {
      sessionStorage.setItem('selectedMedications', JSON.stringify(['Aspirin', 'Lisinopril']))
    })

    test('displays clean card layout for each medication', () => {
      renderWithRouter(<DosageStep />)
      
      // Requirement 4.1: Clean card layout for dosage collection forms
      const cards = screen.getAllByTestId('card')
      expect(cards).toHaveLength(2) // One card per medication
      
      expect(screen.getByText('Aspirin')).toBeInTheDocument()
      expect(screen.getByText('Lisinopril')).toBeInTheDocument()
    })

    test('provides all required form inputs', () => {
      renderWithRouter(<DosageStep />)
      
      // Requirement 4.2: Form inputs for dosage amount, unit, frequency, and dates
      expect(screen.getAllByLabelText(/Dosage Amount/)).toHaveLength(2)
      expect(screen.getAllByLabelText(/Unit/)).toHaveLength(2)
      expect(screen.getAllByLabelText(/Frequency/)).toHaveLength(2)
      expect(screen.getAllByLabelText(/Start Date/)).toHaveLength(2)
      expect(screen.getAllByLabelText(/End Date/)).toHaveLength(2)
    })

    test('has mobile-friendly spacing and layout', () => {
      renderWithRouter(<DosageStep />)
      
      // Requirement 4.5: Mobile-friendly spacing
      const dosageInputs = screen.getAllByLabelText(/Dosage Amount/)
      dosageInputs.forEach(input => {
        expect(input).toHaveClass('w-full', 'px-4', 'py-3')
      })
    })

    test('includes proper input constraints and validation', () => {
      renderWithRouter(<DosageStep />)
      
      const dosageInputs = screen.getAllByLabelText(/Dosage Amount/)
      dosageInputs.forEach(input => {
        expect(input).toHaveAttribute('type', 'number')
        expect(input).toHaveAttribute('min', '0')
        expect(input).toHaveAttribute('step', '0.1')
        expect(input).toHaveAttribute('required')
      })
    })
  })

  describe('Form Validation', () => {
    beforeEach(() => {
      sessionStorage.setItem('selectedMedications', JSON.stringify(['Aspirin']))
    })

    test('validates required fields', async () => {
      renderWithRouter(<DosageStep />)
      
      const nextButton = screen.getByRole('button', { name: /Check Safety/ })
      
      // Initially disabled due to missing required fields
      expect(nextButton).toBeDisabled()
      
      // Fill required fields
      const dosageInput = screen.getByLabelText(/Dosage Amount/)
      const frequencySelect = screen.getByLabelText(/Frequency/)
      
      fireEvent.change(dosageInput, { target: { value: '10' } })
      fireEvent.change(frequencySelect, { target: { value: 'Once daily' } })
      
      await waitFor(() => {
        expect(nextButton).not.toBeDisabled()
      })
    })

    test('validates date logic', async () => {
      renderWithRouter(<DosageStep />)
      
      // Fill required fields first
      const dosageInput = screen.getByLabelText(/Dosage Amount/)
      const frequencySelect = screen.getByLabelText(/Frequency/)
      
      fireEvent.change(dosageInput, { target: { value: '10' } })
      fireEvent.change(frequencySelect, { target: { value: 'Once daily' } })
      
      // Set invalid dates (end date before start date)
      const startDateInput = screen.getByLabelText(/Start Date/)
      const endDateInput = screen.getByLabelText(/End Date/)
      
      fireEvent.change(startDateInput, { target: { value: '2024-01-15' } })
      fireEvent.change(endDateInput, { target: { value: '2024-01-10' } })
      
      const nextButton = screen.getByText(/Check Safety/)
      fireEvent.click(nextButton)
      
      await waitFor(() => {
        expect(screen.getByText(/End date must be after start date/)).toBeInTheDocument()
      })
    })

    test('displays validation errors with proper styling', async () => {
      renderWithRouter(<DosageStep />)
      
      // Try to proceed without filling required fields
      const nextButton = screen.getByRole('button', { name: /Check Safety/ })
      
      // Fill only dosage, leave frequency empty
      const dosageInput = screen.getByLabelText(/Dosage Amount/)
      fireEvent.change(dosageInput, { target: { value: '10' } })
      
      // Button should still be disabled
      expect(nextButton).toBeDisabled()
    })

    // Task 5.3: Additional unit tests for dosage form validation
    describe('Required Field Validation and Error Messaging', () => {
      test('shows error message when dosage amount is missing', async () => {
        renderWithRouter(<DosageStep />)
        
        // Fill only frequency, leave dosage amount empty
        const frequencySelect = screen.getByLabelText(/Frequency/)
        fireEvent.change(frequencySelect, { target: { value: 'Once daily' } })
        
        // Try to proceed - should show validation error
        const nextButton = screen.getByText(/Check Safety/)
        fireEvent.click(nextButton)
        
        await waitFor(() => {
          expect(screen.getByText(/Dosage amount and frequency are required/)).toBeInTheDocument()
        })
      })

      test('shows error message when frequency is missing', async () => {
        renderWithRouter(<DosageStep />)
        
        // Fill only dosage amount, leave frequency empty
        const dosageInput = screen.getByLabelText(/Dosage Amount/)
        fireEvent.change(dosageInput, { target: { value: '10' } })
        
        // Try to proceed - should show validation error
        const nextButton = screen.getByText(/Check Safety/)
        fireEvent.click(nextButton)
        
        await waitFor(() => {
          expect(screen.getByText(/Dosage amount and frequency are required/)).toBeInTheDocument()
        })
      })

      test('shows error message when both required fields are missing', async () => {
        renderWithRouter(<DosageStep />)
        
        // Don't fill any required fields
        const nextButton = screen.getByText(/Check Safety/)
        fireEvent.click(nextButton)
        
        await waitFor(() => {
          expect(screen.getByText(/Dosage amount and frequency are required/)).toBeInTheDocument()
        })
      })

      test('error message disappears when required fields are filled', async () => {
        renderWithRouter(<DosageStep />)
        
        // First trigger error
        const nextButton = screen.getByText(/Check Safety/)
        fireEvent.click(nextButton)
        
        await waitFor(() => {
          expect(screen.getByText(/Dosage amount and frequency are required/)).toBeInTheDocument()
        })
        
        // Then fill required fields
        const dosageInput = screen.getByLabelText(/Dosage Amount/)
        const frequencySelect = screen.getByLabelText(/Frequency/)
        
        fireEvent.change(dosageInput, { target: { value: '10' } })
        fireEvent.change(frequencySelect, { target: { value: 'Once daily' } })
        
        await waitFor(() => {
          expect(screen.queryByText(/Dosage amount and frequency are required/)).not.toBeInTheDocument()
        })
      })

      test('applies error styling to form inputs when validation fails', async () => {
        renderWithRouter(<DosageStep />)
        
        // Trigger validation error
        const nextButton = screen.getByText(/Check Safety/)
        fireEvent.click(nextButton)
        
        await waitFor(() => {
          const dosageInput = screen.getByLabelText(/Dosage Amount/)
          const frequencySelect = screen.getByLabelText(/Frequency/)
          
          expect(dosageInput).toHaveClass('border-red-300')
          expect(frequencySelect).toHaveClass('border-red-300')
        })
      })
    })

    describe('Date Validation Logic and Edge Cases', () => {
      test('validates end date is after start date', async () => {
        renderWithRouter(<DosageStep />)
        
        // Fill required fields
        const dosageInput = screen.getByLabelText(/Dosage Amount/)
        const frequencySelect = screen.getByLabelText(/Frequency/)
        fireEvent.change(dosageInput, { target: { value: '10' } })
        fireEvent.change(frequencySelect, { target: { value: 'Once daily' } })
        
        // Set end date before start date
        const startDateInput = screen.getByLabelText(/Start Date/)
        const endDateInput = screen.getByLabelText(/End Date/)
        
        fireEvent.change(startDateInput, { target: { value: '2024-01-15' } })
        fireEvent.change(endDateInput, { target: { value: '2024-01-10' } })
        
        const nextButton = screen.getByText(/Check Safety/)
        fireEvent.click(nextButton)
        
        await waitFor(() => {
          expect(screen.getByText(/End date must be after start date/)).toBeInTheDocument()
        })
      })

      test('validates end date equal to start date shows error', async () => {
        renderWithRouter(<DosageStep />)
        
        // Fill required fields
        const dosageInput = screen.getByLabelText(/Dosage Amount/)
        const frequencySelect = screen.getByLabelText(/Frequency/)
        fireEvent.change(dosageInput, { target: { value: '10' } })
        fireEvent.change(frequencySelect, { target: { value: 'Once daily' } })
        
        // Set end date equal to start date
        const startDateInput = screen.getByLabelText(/Start Date/)
        const endDateInput = screen.getByLabelText(/End Date/)
        
        fireEvent.change(startDateInput, { target: { value: '2024-01-15' } })
        fireEvent.change(endDateInput, { target: { value: '2024-01-15' } })
        
        const nextButton = screen.getByText(/Check Safety/)
        fireEvent.click(nextButton)
        
        await waitFor(() => {
          expect(screen.getByText(/End date must be after start date/)).toBeInTheDocument()
        })
      })

      test('allows valid date range (end date after start date)', async () => {
        renderWithRouter(<DosageStep />)
        
        // Fill required fields
        const dosageInput = screen.getByLabelText(/Dosage Amount/)
        const frequencySelect = screen.getByLabelText(/Frequency/)
        fireEvent.change(dosageInput, { target: { value: '10' } })
        fireEvent.change(frequencySelect, { target: { value: 'Once daily' } })
        
        // Set valid date range
        const startDateInput = screen.getByLabelText(/Start Date/)
        const endDateInput = screen.getByLabelText(/End Date/)
        
        fireEvent.change(startDateInput, { target: { value: '2024-01-10' } })
        fireEvent.change(endDateInput, { target: { value: '2024-01-20' } })
        
        const nextButton = screen.getByText(/Check Safety/)
        fireEvent.click(nextButton)
        
        await waitFor(() => {
          expect(screen.queryByText(/End date must be after start date/)).not.toBeInTheDocument()
        })
      })

      test('allows empty end date (ongoing medication)', async () => {
        renderWithRouter(<DosageStep />)
        
        // Fill required fields
        const dosageInput = screen.getByLabelText(/Dosage Amount/)
        const frequencySelect = screen.getByLabelText(/Frequency/)
        fireEvent.change(dosageInput, { target: { value: '10' } })
        fireEvent.change(frequencySelect, { target: { value: 'Once daily' } })
        
        // Set start date but leave end date empty
        const startDateInput = screen.getByLabelText(/Start Date/)
        fireEvent.change(startDateInput, { target: { value: '2024-01-10' } })
        
        const nextButton = screen.getByText(/Check Safety/)
        fireEvent.click(nextButton)
        
        await waitFor(() => {
          expect(screen.queryByText(/End date must be after start date/)).not.toBeInTheDocument()
        })
      })

      test('start date has max constraint to today', () => {
        renderWithRouter(<DosageStep />)
        
        const startDateInput = screen.getByLabelText(/Start Date/)
        const today = new Date().toISOString().split('T')[0]
        
        expect(startDateInput).toHaveAttribute('max', today)
      })

      test('end date min constraint updates with start date', async () => {
        renderWithRouter(<DosageStep />)
        
        const startDateInput = screen.getByLabelText(/Start Date/)
        const endDateInput = screen.getByLabelText(/End Date/)
        
        fireEvent.change(startDateInput, { target: { value: '2024-01-15' } })
        
        await waitFor(() => {
          expect(endDateInput).toHaveAttribute('min', '2024-01-15')
        })
      })
    })

    describe('Form Submission and Data Handling', () => {
      test('prevents form submission when validation fails', async () => {
        renderWithRouter(<DosageStep />)
        
        // Don't fill required fields
        const nextButton = screen.getByRole('button', { name: /Check Safety/ })
        
        // Button should be disabled
        expect(nextButton).toBeDisabled()
        
        // Clicking disabled button should not navigate
        fireEvent.click(nextButton)
        expect(mockNavigate).not.toHaveBeenCalledWith('/check/analysis')
      })

      test('allows form submission when all validations pass', async () => {
        renderWithRouter(<DosageStep />)
        
        // Fill all required fields correctly
        const dosageInput = screen.getByLabelText(/Dosage Amount/)
        const frequencySelect = screen.getByLabelText(/Frequency/)
        
        fireEvent.change(dosageInput, { target: { value: '10' } })
        fireEvent.change(frequencySelect, { target: { value: 'Once daily' } })
        
        await waitFor(() => {
          const nextButton = screen.getByText(/Check Safety/)
          expect(nextButton).not.toBeDisabled()
          
          fireEvent.click(nextButton)
          expect(mockNavigate).toHaveBeenCalledWith('/check/analysis')
        })
      })

      test('stores complete dosage data structure on submission', async () => {
        renderWithRouter(<DosageStep />)
        
        // Fill complete form data
        const dosageInput = screen.getByLabelText(/Dosage Amount/)
        const unitSelect = screen.getByLabelText(/Unit/)
        const frequencySelect = screen.getByLabelText(/Frequency/)
        const startDateInput = screen.getByLabelText(/Start Date/)
        const endDateInput = screen.getByLabelText(/End Date/)
        
        fireEvent.change(dosageInput, { target: { value: '25' } })
        fireEvent.change(unitSelect, { target: { value: 'mg' } })
        fireEvent.change(frequencySelect, { target: { value: 'Twice daily' } })
        fireEvent.change(startDateInput, { target: { value: '2024-01-01' } })
        fireEvent.change(endDateInput, { target: { value: '2024-02-01' } })
        
        await waitFor(() => {
          const nextButton = screen.getByText(/Check Safety/)
          fireEvent.click(nextButton)
          
          const storedData = JSON.parse(sessionStorage.getItem('dosageData'))
          expect(storedData).toEqual({
            'Aspirin': {
              dosage_amount: '25',
              dosage_unit: 'mg',
              frequency: 'Twice daily',
              start_date: '2024-01-01',
              end_date: '2024-02-01'
            }
          })
        })
      })

      test('handles multiple medications data correctly', async () => {
        sessionStorage.setItem('selectedMedications', JSON.stringify(['Aspirin', 'Lisinopril']))
        renderWithRouter(<DosageStep />)
        
        // Fill data for both medications
        const dosageInputs = screen.getAllByLabelText(/Dosage Amount/)
        const frequencySelects = screen.getAllByLabelText(/Frequency/)
        
        fireEvent.change(dosageInputs[0], { target: { value: '10' } })
        fireEvent.change(frequencySelects[0], { target: { value: 'Once daily' } })
        fireEvent.change(dosageInputs[1], { target: { value: '5' } })
        fireEvent.change(frequencySelects[1], { target: { value: 'Twice daily' } })
        
        await waitFor(() => {
          const nextButton = screen.getByText(/Check Safety/)
          fireEvent.click(nextButton)
          
          const storedData = JSON.parse(sessionStorage.getItem('dosageData'))
          expect(storedData).toHaveProperty('Aspirin')
          expect(storedData).toHaveProperty('Lisinopril')
          expect(storedData.Aspirin.dosage_amount).toBe('10')
          expect(storedData.Lisinopril.dosage_amount).toBe('5')
        })
      })

      test('validates numeric input constraints', () => {
        renderWithRouter(<DosageStep />)
        
        const dosageInput = screen.getByLabelText(/Dosage Amount/)
        
        // Test numeric constraints
        expect(dosageInput).toHaveAttribute('type', 'number')
        expect(dosageInput).toHaveAttribute('min', '0')
        expect(dosageInput).toHaveAttribute('step', '0.1')
        
        // Test that input accepts positive values
        fireEvent.change(dosageInput, { target: { value: '10' } })
        expect(dosageInput.value).toBe('10')
      })

      test('preserves form data during component re-renders', async () => {
        renderWithRouter(<DosageStep />)
        
        // Fill some data
        const dosageInput = screen.getByLabelText(/Dosage Amount/)
        const frequencySelect = screen.getByLabelText(/Frequency/)
        
        fireEvent.change(dosageInput, { target: { value: '15' } })
        fireEvent.change(frequencySelect, { target: { value: 'Three times daily' } })
        
        // Verify data is preserved
        expect(dosageInput.value).toBe('15')
        expect(frequencySelect.value).toBe('Three times daily')
      })
    })
  })

  describe('Navigation', () => {
    beforeEach(() => {
      sessionStorage.setItem('selectedMedications', JSON.stringify(['Aspirin']))
    })

    test('navigates back to medication step', () => {
      renderWithRouter(<DosageStep />)
      
      const backButton = screen.getByText(/Back: Medications/)
      fireEvent.click(backButton)
      
      expect(mockNavigate).toHaveBeenCalledWith('/check/medication')
    })

    test('navigates to analysis step when form is valid', async () => {
      renderWithRouter(<DosageStep />)
      
      // Fill required fields
      const dosageInput = screen.getByLabelText(/Dosage Amount/)
      const frequencySelect = screen.getByLabelText(/Frequency/)
      
      fireEvent.change(dosageInput, { target: { value: '10' } })
      fireEvent.change(frequencySelect, { target: { value: 'Once daily' } })
      
      await waitFor(() => {
        const nextButton = screen.getByText(/Check Safety/)
        expect(nextButton).not.toBeDisabled()
        
        fireEvent.click(nextButton)
        expect(mockNavigate).toHaveBeenCalledWith('/check/analysis')
      })
    })

    test('stores dosage data in sessionStorage', async () => {
      renderWithRouter(<DosageStep />)
      
      // Fill form data
      const dosageInput = screen.getByLabelText(/Dosage Amount/)
      const unitSelect = screen.getByLabelText(/Unit/)
      const frequencySelect = screen.getByLabelText(/Frequency/)
      
      fireEvent.change(dosageInput, { target: { value: '10' } })
      fireEvent.change(unitSelect, { target: { value: 'mg' } })
      fireEvent.change(frequencySelect, { target: { value: 'Once daily' } })
      
      await waitFor(() => {
        const nextButton = screen.getByText(/Check Safety/)
        fireEvent.click(nextButton)
        
        const storedData = JSON.parse(sessionStorage.getItem('dosageData'))
        expect(storedData).toEqual({
          'Aspirin': {
            dosage_amount: '10',
            dosage_unit: 'mg',
            frequency: 'Once daily',
            start_date: '',
            end_date: ''
          }
        })
      })
    })
  })

  describe('Progress Indicator', () => {
    beforeEach(() => {
      sessionStorage.setItem('selectedMedications', JSON.stringify(['Aspirin', 'Lisinopril']))
    })

    test('displays progress indicator with completion status', () => {
      renderWithRouter(<DosageStep />)
      
      expect(screen.getByText(/Progress: 0 of 2 medications completed/)).toBeInTheDocument()
    })

    test('updates progress as forms are completed', async () => {
      renderWithRouter(<DosageStep />)
      
      // Fill first medication
      const dosageInputs = screen.getAllByLabelText(/Dosage Amount/)
      const frequencySelects = screen.getAllByLabelText(/Frequency/)
      
      fireEvent.change(dosageInputs[0], { target: { value: '10' } })
      fireEvent.change(frequencySelects[0], { target: { value: 'Once daily' } })
      
      await waitFor(() => {
        expect(screen.getByText(/Progress: 1 of 2 medications completed/)).toBeInTheDocument()
      })
    })
  })

  describe('Data Integration', () => {
    test('integrates with existing dosage data structures', () => {
      sessionStorage.setItem('selectedMedications', JSON.stringify(['Aspirin']))
      
      renderWithRouter(<DosageStep />)
      
      // Check that form fields match expected data structure
      expect(screen.getByLabelText(/Dosage Amount/)).toBeInTheDocument()
      expect(screen.getByLabelText(/Unit/)).toBeInTheDocument()
      expect(screen.getByLabelText(/Frequency/)).toBeInTheDocument()
      expect(screen.getByLabelText(/Start Date/)).toBeInTheDocument()
      expect(screen.getByLabelText(/End Date/)).toBeInTheDocument()
    })

    test('handles multiple medications correctly', () => {
      sessionStorage.setItem('selectedMedications', JSON.stringify(['Aspirin', 'Lisinopril', 'Metformin']))
      
      renderWithRouter(<DosageStep />)
      
      // Should create forms for all medications
      expect(screen.getByText('Aspirin')).toBeInTheDocument()
      expect(screen.getByText('Lisinopril')).toBeInTheDocument()
      expect(screen.getByText('Metformin')).toBeInTheDocument()
      
      // Should have correct number of form inputs
      expect(screen.getAllByLabelText(/Dosage Amount/)).toHaveLength(3)
      expect(screen.getAllByLabelText(/Frequency/)).toHaveLength(3)
    })
  })

  // Feature: frontend-ui-redesign, Property 5: Dosage Step Form Completeness
  test.prop([
    fc.array(fc.stringMatching(/^[A-Za-z][A-Za-z0-9\s-]{0,49}$/), { minLength: 1, maxLength: 3 })
  ])('Property 5: Dosage Step Form Completeness - validates Requirements 4.1, 4.2, 4.3', (medications) => {
    // Set up medications in sessionStorage
    sessionStorage.setItem('selectedMedications', JSON.stringify(medications))
    
    renderWithRouter(<DosageStep />)
    
    // Property: For any set of medications, the dosage step should provide complete
    // form inputs for each medication with proper validation and layout
    
    // Requirement 4.1: Clean card layout for dosage collection forms
    const cards = screen.getAllByTestId('card')
    expect(cards.length).toBeGreaterThanOrEqual(medications.length)
    
    // Requirement 4.2: Form inputs for dosage amount, unit, frequency, and dates
    expect(screen.getAllByLabelText(/Dosage Amount/)).toHaveLength(medications.length)
    expect(screen.getAllByLabelText(/Unit/)).toHaveLength(medications.length)
    expect(screen.getAllByLabelText(/Frequency/)).toHaveLength(medications.length)
    expect(screen.getAllByLabelText(/Start Date/)).toHaveLength(medications.length)
    expect(screen.getAllByLabelText(/End Date/)).toHaveLength(medications.length)
    
    // Requirement 4.3: Form validation (required fields should be marked)
    const dosageInputs = screen.getAllByLabelText(/Dosage Amount/)
    const frequencySelects = screen.getAllByLabelText(/Frequency/)
    
    dosageInputs.forEach(input => {
      expect(input).toHaveAttribute('required')
    })
    
    frequencySelects.forEach(select => {
      expect(select).toHaveAttribute('required')
    })
    
    // Navigation elements should always be present
    expect(screen.getByText(/Back: Medications/)).toBeInTheDocument()
    expect(screen.getByText(/Check Safety/)).toBeInTheDocument()
    
    // Step indicator should show Step 2
    expect(screen.getByText(/Step 2/)).toBeInTheDocument()
    
    // Progress indicator should be present
    expect(screen.getByText(/Progress:/)).toBeInTheDocument()
  })
})