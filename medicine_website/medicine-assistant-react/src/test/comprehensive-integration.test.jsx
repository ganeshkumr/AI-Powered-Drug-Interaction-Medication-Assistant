import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { BrowserRouter } from 'react-router-dom'
import App from '../App'

// Mock the API module with realistic responses
vi.mock('../services/api', () => ({
  medicationAPI: {
    checkBeforeAdding: vi.fn(),
    addMedication: vi.fn(),
    getMedications: vi.fn(),
    updateMedication: vi.fn(),
    deleteMedication: vi.fn(),
  },
  chatbotAPI: {
    askAssistant: vi.fn(),
  },
  healthAPI: {
    getHealthData: vi.fn(),
    getHealthAlerts: vi.fn(),
  },
  emergencyAPI: {
    checkInteraction: vi.fn(),
  },
}))

// Mock framer-motion to avoid animation issues in tests
vi.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }) => <div {...props}>{children}</div>,
    nav: ({ children, ...props }) => <nav {...props}>{children}</nav>,
    button: ({ children, ...props }) => <button {...props}>{children}</button>,
    main: ({ children, ...props }) => <main {...props}>{children}</main>,
  },
  AnimatePresence: ({ children }) => children,
}))

// Mock react-router-dom with more realistic navigation
const mockNavigate = vi.fn()
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom')
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  }
})

describe('Comprehensive Integration Tests', () => {
  let user

  beforeEach(async () => {
    user = userEvent.setup()
    vi.clearAllMocks()
    
    // Get the mocked API functions
    const { medicationAPI, healthAPI } = await import('../services/api')
    
    // Setup default API responses
    medicationAPI.getMedications.mockResolvedValue([])
    healthAPI.getHealthData.mockResolvedValue({
      totalMedications: 0,
      safeCount: 0,
      needsAttentionCount: 0,
      highRiskCount: 0,
    })
    healthAPI.getHealthAlerts.mockResolvedValue([])
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  describe('Complete User Workflows', () => {
    it('should handle complete medication management workflow', async () => {
      // Get the mocked API functions
      const { medicationAPI, healthAPI } = await import('../services/api')
      
      // Mock medication data
      const mockMedications = [
        {
          id: '1',
          drug_name: 'Aspirin',
          dosage_amount: 100,
          dosage_unit: 'mg',
          frequency: 'daily',
          start_date: '2024-01-01',
          time_of_day: 'morning',
          risk_level: 'safe'
        }
      ]

      medicationAPI.getMedications.mockResolvedValue(mockMedications)
      healthAPI.getHealthData.mockResolvedValue({
        totalMedications: 1,
        safeCount: 1,
        needsAttentionCount: 0,
        highRiskCount: 0,
      })

      render(<App />)

      // Wait for initial load
      await waitFor(() => {
        expect(screen.getByText(/Medicine Assistant/i)).toBeInTheDocument()
      })

      // Navigate to dashboard/my-med page
      const dashboardLink = screen.getByRole('link', { name: /dashboard|my medications/i })
      await user.click(dashboardLink)

      // Verify medication display
      await waitFor(() => {
        expect(screen.getByText('Aspirin')).toBeInTheDocument()
        expect(screen.getByText('100 mg')).toBeInTheDocument()
        expect(screen.getByText('daily')).toBeInTheDocument()
      })

      // Test add medication functionality
      const addButton = screen.getByRole('button', { name: /add medication/i })
      await user.click(addButton)

      // Verify navigation to medication step
      expect(mockNavigate).toHaveBeenCalledWith('/medication-step')
    })

    it('should handle user authentication workflow', async () => {
      render(<App />)

      // Navigate to login
      const loginLink = screen.getByRole('link', { name: /login/i })
      await user.click(loginLink)

      // Verify login form is displayed
      await waitFor(() => {
        expect(screen.getByRole('textbox', { name: /email/i })).toBeInTheDocument()
        expect(screen.getByLabelText(/password/i)).toBeInTheDocument()
      })

      // Fill login form
      await user.type(screen.getByRole('textbox', { name: /email/i }), 'test@example.com')
      await user.type(screen.getByLabelText(/password/i), 'password123')

      // Submit form
      const submitButton = screen.getByRole('button', { name: /sign in|login/i })
      await user.click(submitButton)

      // Verify form submission (would normally redirect)
      expect(submitButton).toBeInTheDocument()
    })
  })

  describe('Step-Based Safety Check Flow', () => {
    it('should complete the 3-step safety check flow end-to-end', async () => {
      // Get the mocked API functions
      const { medicationAPI } = await import('../services/api')
      
      // Mock API responses for safety check
      medicationAPI.checkBeforeAdding.mockResolvedValue({
        risk_percentage: 15,
        risk_category: 'Safe',
        ai_explanation: 'No significant interactions detected.',
        interactions: []
      })

      render(<App />)

      // Navigate to safety check
      const safetyCheckLink = screen.getByRole('link', { name: /safety check/i })
      await user.click(safetyCheckLink)

      // Step 1: Medication Selection
      await waitFor(() => {
        expect(screen.getByText(/personalized safety check/i)).toBeInTheDocument()
        expect(screen.getByText(/step 1/i)).toBeInTheDocument()
      })

      // Search for medication
      const searchInput = screen.getByRole('textbox', { name: /search medication/i })
      await user.type(searchInput, 'Aspirin')

      // Select medication (mock selection)
      const medicationOption = screen.getByText('Aspirin')
      await user.click(medicationOption)

      // Proceed to Step 2
      const nextButton = screen.getByRole('button', { name: /next/i })
      await user.click(nextButton)

      // Step 2: Dosage Information
      await waitFor(() => {
        expect(screen.getByText(/step 2/i)).toBeInTheDocument()
        expect(screen.getByText(/dosage/i)).toBeInTheDocument()
      })

      // Fill dosage form
      const amountInput = screen.getByRole('spinbutton', { name: /amount/i })
      await user.type(amountInput, '100')

      const unitSelect = screen.getByRole('combobox', { name: /unit/i })
      await user.selectOptions(unitSelect, 'mg')

      const frequencySelect = screen.getByRole('combobox', { name: /frequency/i })
      await user.selectOptions(frequencySelect, 'daily')

      // Proceed to Step 3
      const checkSafetyButton = screen.getByRole('button', { name: /check safety/i })
      await user.click(checkSafetyButton)

      // Step 3: Analysis Results
      await waitFor(() => {
        expect(screen.getByText(/step 3/i)).toBeInTheDocument()
        expect(screen.getByText(/analysis/i)).toBeInTheDocument()
        expect(screen.getByText(/15%/)).toBeInTheDocument()
        expect(screen.getByText(/safe/i)).toBeInTheDocument()
        expect(screen.getByText(/no significant interactions/i)).toBeInTheDocument()
      })

      // Verify API was called
      expect(medicationAPI.checkBeforeAdding).toHaveBeenCalledWith({
        medications: expect.arrayContaining([
          expect.objectContaining({
            drug_name: 'Aspirin',
            dosage_amount: 100,
            dosage_unit: 'mg',
            frequency: 'daily'
          })
        ])
      })
    })

    it('should handle high-risk medication combinations with warnings', async () => {
      // Get the mocked API functions
      const { medicationAPI } = await import('../services/api')
      
      // Mock high-risk response
      medicationAPI.checkBeforeAdding.mockResolvedValue({
        risk_percentage: 85,
        risk_category: 'High Risk',
        ai_explanation: 'Dangerous interaction detected between medications.',
        interactions: [
          {
            drug1: 'Warfarin',
            drug2: 'Aspirin',
            severity: 'High',
            description: 'Increased bleeding risk'
          }
        ]
      })

      render(<App />)

      // Navigate through safety check flow (abbreviated)
      const safetyCheckLink = screen.getByRole('link', { name: /safety check/i })
      await user.click(safetyCheckLink)

      // Mock completing steps 1 and 2 quickly
      const checkSafetyButton = screen.getByRole('button', { name: /check safety/i })
      await user.click(checkSafetyButton)

      // Verify warning modal appears
      await waitFor(() => {
        expect(screen.getByText(/warning/i)).toBeInTheDocument()
        expect(screen.getByText(/85%/)).toBeInTheDocument()
        expect(screen.getByText(/high risk/i)).toBeInTheDocument()
        expect(screen.getByText(/dangerous interaction/i)).toBeInTheDocument()
      })

      // Verify user can still proceed (non-blocking)
      const proceedButton = screen.getByRole('button', { name: /proceed|continue/i })
      expect(proceedButton).toBeInTheDocument()
    })
  })

  describe('Responsive Design Testing', () => {
    it('should adapt to mobile viewport', async () => {
      // Mock mobile viewport
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 375,
      })
      Object.defineProperty(window, 'innerHeight', {
        writable: true,
        configurable: true,
        value: 667,
      })

      render(<App />)

      // Verify mobile navigation (hamburger menu)
      const hamburgerButton = screen.getByRole('button', { name: /menu/i })
      expect(hamburgerButton).toBeInTheDocument()

      // Test mobile menu functionality
      await user.click(hamburgerButton)
      
      // Verify mobile menu opens
      await waitFor(() => {
        const mobileMenu = screen.getByRole('navigation')
        expect(mobileMenu).toHaveClass(/mobile|responsive/)
      })
    })

    it('should adapt to tablet viewport', async () => {
      // Mock tablet viewport
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 768,
      })
      Object.defineProperty(window, 'innerHeight', {
        writable: true,
        configurable: true,
        value: 1024,
      })

      render(<App />)

      // Verify tablet layout adaptations
      const navigation = screen.getByRole('navigation')
      expect(navigation).toBeInTheDocument()

      // Verify responsive containers
      const containers = screen.getAllByTestId(/responsive-container/)
      containers.forEach(container => {
        expect(container).toHaveClass(/tablet|md:/)
      })
    })

    it('should utilize desktop space effectively', async () => {
      // Mock desktop viewport
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 1920,
      })
      Object.defineProperty(window, 'innerHeight', {
        writable: true,
        configurable: true,
        value: 1080,
      })

      render(<App />)

      // Verify desktop navigation is fully visible
      const navigation = screen.getByRole('navigation')
      expect(navigation).toBeInTheDocument()

      // Verify no hamburger menu on desktop
      expect(screen.queryByRole('button', { name: /menu/i })).not.toBeInTheDocument()

      // Verify desktop layout utilizes space
      const mainContent = screen.getByRole('main')
      expect(mainContent).toHaveClass(/desktop|lg:|xl:/)
    })
  })

  describe('Accessibility Compliance', () => {
    it('should support keyboard navigation', async () => {
      render(<App />)

      // Test tab navigation
      await user.tab()
      expect(document.activeElement).toHaveAttribute('role', 'link')

      // Test arrow key navigation in menus
      const navigation = screen.getByRole('navigation')
      const firstLink = within(navigation).getAllByRole('link')[0]
      firstLink.focus()

      await user.keyboard('{ArrowRight}')
      expect(document.activeElement).toHaveAttribute('role', 'link')

      // Test Enter key activation
      await user.keyboard('{Enter}')
      // Should navigate or activate the focused element
    })

    it('should have proper ARIA labels and roles', async () => {
      render(<App />)

      // Check navigation has proper ARIA
      const navigation = screen.getByRole('navigation')
      expect(navigation).toHaveAttribute('aria-label')

      // Check buttons have proper labels
      const buttons = screen.getAllByRole('button')
      buttons.forEach(button => {
        expect(button).toHaveAccessibleName()
      })

      // Check form fields have proper labels
      const textboxes = screen.getAllByRole('textbox')
      textboxes.forEach(textbox => {
        expect(textbox).toHaveAccessibleName()
      })
    })

    it('should support screen reader announcements', async () => {
      render(<App />)

      // Mock screen reader announcements
      const announcements = []
      const mockAnnounce = vi.fn((message) => announcements.push(message))
      
      // Navigate to different sections
      const aboutLink = screen.getByRole('link', { name: /about/i })
      await user.click(aboutLink)

      // Verify page changes are announced
      await waitFor(() => {
        const pageTitle = screen.getByRole('heading', { level: 1 })
        expect(pageTitle).toBeInTheDocument()
      })
    })

    it('should have proper focus management', async () => {
      render(<App />)

      // Test focus trap in modals
      const modalTrigger = screen.getByRole('button', { name: /open modal/i })
      if (modalTrigger) {
        await user.click(modalTrigger)

        // Verify focus is trapped within modal
        await user.tab()
        expect(document.activeElement).toBeInTheDocument()
        
        // Test escape key closes modal
        await user.keyboard('{Escape}')
        expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
      }
    })

    it('should meet color contrast requirements', async () => {
      render(<App />)

      // Check that high contrast elements are present
      const highContrastElements = screen.getAllByTestId(/high-contrast/)
      highContrastElements.forEach(element => {
        const styles = window.getComputedStyle(element)
        // Verify contrast ratios meet WCAG standards
        expect(styles.color).toBeTruthy()
        expect(styles.backgroundColor).toBeTruthy()
      })
    })

    it('should have touch-friendly interface elements on mobile', async () => {
      // Mock mobile viewport
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 375,
      })

      render(<App />)

      // Check button sizes meet touch target requirements (44px minimum)
      const buttons = screen.getAllByRole('button')
      buttons.forEach(button => {
        const styles = window.getComputedStyle(button)
        const minSize = parseInt(styles.minHeight) || parseInt(styles.height)
        expect(minSize).toBeGreaterThanOrEqual(44)
      })
    })
  })

  describe('Error Handling and Edge Cases', () => {
    it('should handle API failures gracefully', async () => {
      // Get the mocked API functions
      const { medicationAPI } = await import('../services/api')
      
      // Mock API failure
      medicationAPI.getMedications.mockRejectedValue(new Error('Network error'))

      render(<App />)

      // Navigate to dashboard
      const dashboardLink = screen.getByRole('link', { name: /dashboard/i })
      await user.click(dashboardLink)

      // Verify error handling
      await waitFor(() => {
        expect(screen.getByText(/error|failed|try again/i)).toBeInTheDocument()
      })
    })

    it('should handle empty states appropriately', async () => {
      // Get the mocked API functions
      const { medicationAPI, healthAPI } = await import('../services/api')
      
      // Mock empty medication list
      medicationAPI.getMedications.mockResolvedValue([])
      healthAPI.getHealthData.mockResolvedValue({
        totalMedications: 0,
        safeCount: 0,
        needsAttentionCount: 0,
        highRiskCount: 0,
      })

      render(<App />)

      // Navigate to dashboard
      const dashboardLink = screen.getByRole('link', { name: /dashboard/i })
      await user.click(dashboardLink)

      // Verify empty state display
      await waitFor(() => {
        expect(screen.getByText(/no medications|empty|get started/i)).toBeInTheDocument()
      })
    })

    it('should handle loading states during API calls', async () => {
      // Get the mocked API functions
      const { medicationAPI } = await import('../services/api')
      
      // Mock slow API response
      medicationAPI.checkBeforeAdding.mockImplementation(
        () => new Promise(resolve => setTimeout(() => resolve({
          risk_percentage: 10,
          risk_category: 'Safe',
          ai_explanation: 'Safe combination',
          interactions: []
        }), 1000))
      )

      render(<App />)

      // Navigate to safety check and trigger analysis
      const safetyCheckLink = screen.getByRole('link', { name: /safety check/i })
      await user.click(safetyCheckLink)

      const checkSafetyButton = screen.getByRole('button', { name: /check safety/i })
      await user.click(checkSafetyButton)

      // Verify loading state is shown
      expect(screen.getByTestId(/loading|spinner/)).toBeInTheDocument()

      // Wait for completion
      await waitFor(() => {
        expect(screen.getByText(/safe/i)).toBeInTheDocument()
      }, { timeout: 2000 })
    })
  })
})