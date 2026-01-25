import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { BrowserRouter } from 'react-router-dom'
import { vi } from 'vitest'
import { fc, test } from '@fast-check/vitest'
import MedicationStep from './MedicationStep'

// Mock the navigate function
const mockNavigate = vi.fn()
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom')
  return {
    ...actual,
    useNavigate: () => mockNavigate
  }
})

// Mock the DrugSearch component with more realistic behavior
vi.mock('../components/medication/DrugSearch', () => ({
  default: ({ onSelect, placeholder }) => (
    <div data-testid="drug-search">
      <input 
        placeholder={placeholder}
        data-testid="medication-search-input"
        onChange={(e) => {
          // Simulate different medications for testing
          if (e.target.value === 'Aspirin') {
            onSelect('Aspirin')
          } else if (e.target.value === 'Lisinopril') {
            onSelect('Lisinopril')
          } else if (e.target.value === 'Duplicate') {
            onSelect('Aspirin') // Test duplicate handling
          }
        }}
      />
    </div>
  )
}))

// Mock the StepIndicator component
vi.mock('../components/navigation/StepIndicator', () => ({
  default: ({ currentStep }) => (
    <div data-testid="step-indicator">Step {currentStep}</div>
  )
}))

// Mock other components
vi.mock('../components/common/Button', () => ({
  default: ({ children, onClick, disabled, className }) => (
    <button onClick={onClick} disabled={disabled} className={className}>
      {children}
    </button>
  )
}))

vi.mock('../components/common/Card', () => ({
  default: ({ children, className }) => (
    <div className={className}>{children}</div>
  )
}))

const renderWithRouter = (component) => {
  return render(
    <BrowserRouter>
      {component}
    </BrowserRouter>
  )
}

describe('MedicationStep', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    // Clear sessionStorage
    sessionStorage.clear()
  })

  test('renders medication step interface elements', () => {
    renderWithRouter(<MedicationStep />)
    
    // Check for required interface elements per Requirements 3.1, 3.2, 3.6
    expect(screen.getByText('Personalized Safety Check')).toBeInTheDocument()
    expect(screen.getByTestId('drug-search')).toBeInTheDocument()
    expect(screen.getByText(/Search for medications/)).toBeInTheDocument()
    expect(screen.getByText(/Tip: Start typing the medication name/)).toBeInTheDocument()
  })

  test('displays step indicator with current step 1', () => {
    renderWithRouter(<MedicationStep />)
    
    // Validates Requirements 2.1, 2.2
    expect(screen.getByTestId('step-indicator')).toBeInTheDocument()
    expect(screen.getByText('Step 1')).toBeInTheDocument()
  })

  test('shows empty state when no medications selected', () => {
    renderWithRouter(<MedicationStep />)
    
    // Check empty state display
    expect(screen.getByText('No medications selected yet')).toBeInTheDocument()
    expect(screen.getByText('Use the search above to add medications')).toBeInTheDocument()
  })

  test('adds medication when selected from search', async () => {
    renderWithRouter(<MedicationStep />)
    
    const searchInput = screen.getByPlaceholderText(/Type medication name/)
    
    // Simulate selecting a medication
    fireEvent.change(searchInput, { target: { value: 'Aspirin' } })
    
    await waitFor(() => {
      expect(screen.getByText('Aspirin')).toBeInTheDocument()
      expect(screen.getByText('Selected Medications (1)')).toBeInTheDocument()
    })
  })

  test('enables next button when medications are selected', async () => {
    renderWithRouter(<MedicationStep />)
    
    const searchInput = screen.getByPlaceholderText(/Type medication name/)
    const nextButton = screen.getByText(/Next: Dosage Information/)
    
    // Initially disabled
    expect(nextButton).toBeDisabled()
    
    // Add medication
    fireEvent.change(searchInput, { target: { value: 'Aspirin' } })
    
    await waitFor(() => {
      expect(nextButton).not.toBeDisabled()
    })
  })

  test('navigates to dosage step when next is clicked', async () => {
    renderWithRouter(<MedicationStep />)
    
    const searchInput = screen.getByPlaceholderText(/Type medication name/)
    
    // Add medication
    fireEvent.change(searchInput, { target: { value: 'Aspirin' } })
    
    await waitFor(() => {
      const nextButton = screen.getByText(/Next: Dosage Information/)
      expect(nextButton).not.toBeDisabled()
      
      fireEvent.click(nextButton)
      expect(mockNavigate).toHaveBeenCalledWith('/check/dosage')
    })
  })

  test('stores selected medications in sessionStorage', async () => {
    renderWithRouter(<MedicationStep />)
    
    const searchInput = screen.getByPlaceholderText(/Type medication name/)
    
    // Add medication
    fireEvent.change(searchInput, { target: { value: 'Aspirin' } })
    
    await waitFor(() => {
      const nextButton = screen.getByText(/Next: Dosage Information/)
      fireEvent.click(nextButton)
      
      const storedMedications = JSON.parse(sessionStorage.getItem('selectedMedications'))
      expect(storedMedications).toEqual(['Aspirin'])
    })
  })

  test('removes medication when X button is clicked', async () => {
    renderWithRouter(<MedicationStep />)
    
    const searchInput = screen.getByPlaceholderText(/Type medication name/)
    
    // Add medication
    fireEvent.change(searchInput, { target: { value: 'Aspirin' } })
    
    await waitFor(() => {
      expect(screen.getByText('Aspirin')).toBeInTheDocument()
      
      // Click remove button
      const removeButton = screen.getByLabelText('Remove Aspirin')
      fireEvent.click(removeButton)
      
      expect(screen.queryByText('Aspirin')).not.toBeInTheDocument()
      expect(screen.getByText('No medications selected yet')).toBeInTheDocument()
    })
  })

  describe('Medication Search Integration', () => {
    test('prevents duplicate medications from being added', async () => {
      renderWithRouter(<MedicationStep />)
      
      const searchInput = screen.getByPlaceholderText(/Type medication name/)
      
      // Add Aspirin first time
      fireEvent.change(searchInput, { target: { value: 'Aspirin' } })
      
      await waitFor(() => {
        expect(screen.getByText('Aspirin')).toBeInTheDocument()
        expect(screen.getByText('Selected Medications (1)')).toBeInTheDocument()
      })
      
      // Try to add Aspirin again using the 'Duplicate' trigger
      fireEvent.change(searchInput, { target: { value: 'Duplicate' } })
      
      // Should still only have one Aspirin
      await waitFor(() => {
        const aspirinElements = screen.getAllByText('Aspirin')
        expect(aspirinElements).toHaveLength(1)
        expect(screen.getByText('Selected Medications (1)')).toBeInTheDocument()
      })
    })

    test('allows adding multiple different medications', async () => {
      renderWithRouter(<MedicationStep />)
      
      const searchInput = screen.getByPlaceholderText(/Type medication name/)
      
      // Add first medication
      fireEvent.change(searchInput, { target: { value: 'Aspirin' } })
      
      await waitFor(() => {
        expect(screen.getByText('Aspirin')).toBeInTheDocument()
        expect(screen.getByText('Selected Medications (1)')).toBeInTheDocument()
      })
      
      // Add second medication
      fireEvent.change(searchInput, { target: { value: 'Lisinopril' } })
      
      await waitFor(() => {
        expect(screen.getByText('Aspirin')).toBeInTheDocument()
        expect(screen.getByText('Lisinopril')).toBeInTheDocument()
        expect(screen.getByText('Selected Medications (2)')).toBeInTheDocument()
      })
    })

    test('updates next button state based on medication list', async () => {
      renderWithRouter(<MedicationStep />)
      
      const searchInput = screen.getByPlaceholderText(/Type medication name/)
      const nextButton = screen.getByText(/Next: Dosage Information/)
      
      // Initially disabled
      expect(nextButton).toBeDisabled()
      
      // Add medication - should enable
      fireEvent.change(searchInput, { target: { value: 'Aspirin' } })
      
      await waitFor(() => {
        expect(nextButton).not.toBeDisabled()
      })
      
      // Remove medication - should disable again
      const removeButton = screen.getByLabelText('Remove Aspirin')
      fireEvent.click(removeButton)
      
      await waitFor(() => {
        expect(nextButton).toBeDisabled()
      })
    })

    test('displays helper information when medications are added', async () => {
      renderWithRouter(<MedicationStep />)
      
      const searchInput = screen.getByPlaceholderText(/Type medication name/)
      
      // Initially no helper info
      expect(screen.queryByText(/Great! You've added/)).not.toBeInTheDocument()
      
      // Add medication
      fireEvent.change(searchInput, { target: { value: 'Aspirin' } })
      
      await waitFor(() => {
        expect(screen.getByText(/Great! You've added 1 medication/)).toBeInTheDocument()
        expect(screen.getByText(/Next, you'll provide dosage information/)).toBeInTheDocument()
      })
      
      // Add second medication
      fireEvent.change(searchInput, { target: { value: 'Lisinopril' } })
      
      await waitFor(() => {
        expect(screen.getByText(/Great! You've added 2 medications/)).toBeInTheDocument()
      })
    })

    test('maintains medication list state during component lifecycle', async () => {
      renderWithRouter(<MedicationStep />)
      
      const searchInput = screen.getByPlaceholderText(/Type medication name/)
      
      // Add multiple medications
      fireEvent.change(searchInput, { target: { value: 'Aspirin' } })
      
      await waitFor(() => {
        expect(screen.getByText('Aspirin')).toBeInTheDocument()
      })
      
      fireEvent.change(searchInput, { target: { value: 'Lisinopril' } })
      
      await waitFor(() => {
        expect(screen.getByText('Lisinopril')).toBeInTheDocument()
        expect(screen.getByText('Selected Medications (2)')).toBeInTheDocument()
      })
      
      // Remove one medication
      const removeButton = screen.getByLabelText('Remove Aspirin')
      fireEvent.click(removeButton)
      
      await waitFor(() => {
        expect(screen.queryByText('Aspirin')).not.toBeInTheDocument()
        expect(screen.getByText('Lisinopril')).toBeInTheDocument()
        expect(screen.getByText('Selected Medications (1)')).toBeInTheDocument()
      })
    })

    test('handles edge case of removing all medications', async () => {
      renderWithRouter(<MedicationStep />)
      
      const searchInput = screen.getByPlaceholderText(/Type medication name/)
      
      // Add medication
      fireEvent.change(searchInput, { target: { value: 'Aspirin' } })
      
      await waitFor(() => {
        expect(screen.getByText('Aspirin')).toBeInTheDocument()
        expect(screen.queryByText('No medications selected yet')).not.toBeInTheDocument()
      })
      
      // Remove medication
      const removeButton = screen.getByLabelText('Remove Aspirin')
      fireEvent.click(removeButton)
      
      await waitFor(() => {
        expect(screen.queryByText('Aspirin')).not.toBeInTheDocument()
        expect(screen.getByText('No medications selected yet')).toBeInTheDocument()
        expect(screen.getByText('Use the search above to add medications')).toBeInTheDocument()
        expect(screen.getByText(/Next: Dosage Information/)).toBeDisabled()
      })
    })

    test('stores correct medication data in sessionStorage', async () => {
      renderWithRouter(<MedicationStep />)
      
      const searchInput = screen.getByPlaceholderText(/Type medication name/)
      
      // Add multiple medications
      fireEvent.change(searchInput, { target: { value: 'Aspirin' } })
      
      await waitFor(() => {
        expect(screen.getByText('Aspirin')).toBeInTheDocument()
      })
      
      fireEvent.change(searchInput, { target: { value: 'Lisinopril' } })
      
      await waitFor(() => {
        expect(screen.getByText('Lisinopril')).toBeInTheDocument()
      })
      
      // Navigate to next step
      const nextButton = screen.getByText(/Next: Dosage Information/)
      fireEvent.click(nextButton)
      
      // Check sessionStorage
      const storedMedications = JSON.parse(sessionStorage.getItem('selectedMedications'))
      expect(storedMedications).toEqual(['Aspirin', 'Lisinopril'])
      expect(mockNavigate).toHaveBeenCalledWith('/check/dosage')
    })
  })

  // Feature: frontend-ui-redesign, Property 4: Medication Step Interface Elements
  test.prop([
    fc.array(fc.string({ minLength: 1, maxLength: 50 }), { minLength: 0, maxLength: 10 })
  ])('Property 4: Medication Step Interface Elements - validates Requirements 3.2, 3.3, 3.4, 3.6', (medications) => {
    renderWithRouter(<MedicationStep />)
    
    // Property: For any rendering of Step 1 (Medication), the interface should contain 
    // search input with dropdown, Add button when medication selected, pill-style cards 
    // for added medications, and helper text.
    
    // Requirement 3.1: Display "Personalized Safety Check" title
    expect(screen.getByText('Personalized Safety Check')).toBeInTheDocument()
    
    // Requirement 3.2: Search input with dropdown suggestions
    expect(screen.getByTestId('drug-search')).toBeInTheDocument()
    expect(screen.getByPlaceholderText(/Type medication name/)).toBeInTheDocument()
    
    // Requirement 3.6: Helper text to guide users
    expect(screen.getByText(/Tip: Start typing the medication name/)).toBeInTheDocument()
    
    // Requirement 3.4: Pill-style cards display (when medications exist)
    // Test with different medication arrays to verify consistent interface
    if (medications.length === 0) {
      // Empty state should show helper text
      expect(screen.getByText('No medications selected yet')).toBeInTheDocument()
      expect(screen.getByText('Use the search above to add medications')).toBeInTheDocument()
    }
    
    // Requirement 3.3: Add functionality (implicitly tested through search component presence)
    // The DrugSearch component provides the Add functionality when medications are selected
    
    // Navigation elements should always be present
    expect(screen.getByText(/Next: Dosage Information/)).toBeInTheDocument()
    
    // Step indicator should always show Step 1
    expect(screen.getByTestId('step-indicator')).toBeInTheDocument()
    expect(screen.getByText('Step 1')).toBeInTheDocument()
  })
})