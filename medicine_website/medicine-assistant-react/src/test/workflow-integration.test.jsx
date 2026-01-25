import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { BrowserRouter } from 'react-router-dom'
import App from '../App'

// Mock API responses for specific workflows
const mockAPI = {
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
}

vi.mock('../services/api', () => mockAPI)

// Mock framer-motion
vi.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }) => <div {...props}>{children}</div>,
    nav: ({ children, ...props }) => <nav {...props}>{children}</nav>,
    button: ({ children, ...props }) => <button {...props}>{children}</button>,
    main: ({ children, ...props }) => <main {...props}>{children}</main>,
  },
  AnimatePresence: ({ children }) => children,
}))

describe('Workflow Integration Tests', () => {
  let user

  beforeEach(() => {
    user = userEvent.setup()
    vi.clearAllMocks()
    
    // Default API responses
    mockAPI.medicationAPI.getMedications.mockResolvedValue([])
    mockAPI.healthAPI.getHealthData.mockResolvedValue({
      totalMedications: 0,
      safeCount: 0,
      needsAttentionCount: 0,
      highRiskCount: 0,
    })
    mockAPI.healthAPI.getHealthAlerts.mockResolvedValue([])
  })

  describe('Medication Management Workflow', () => {
    it('should handle complete CRUD operations for medications', async () => {
      const mockMedication = {
        id: '1',
        drug_name: 'Lisinopril',
        dosage_amount: 10,
        dosage_unit: 'mg',
        frequency: 'daily',
        start_date: '2024-01-01',
        time_of_day: 'morning',
        risk_level: 'safe'
      }

      // Mock API responses for CRUD operations
      mockAPI.medicationAPI.getMedications.mockResolvedValue([mockMedication])
      mockAPI.medicationAPI.addMedication.mockResolvedValue(mockMedication)
      mockAPI.medicationAPI.updateMedication.mockResolvedValue({ ...mockMedication, dosage_amount: 20 })
      mockAPI.medicationAPI.deleteMedication.mockResolvedValue({ success: true })

      render(<App />)

      // Navigate to My Medications page
      const myMedLink = screen.getByRole('link', { name: /my medications|dashboard/i })
      await user.click(myMedLink)

      // Verify medication is displayed
      await waitFor(() => {
        expect(screen.getByText('Lisinopril')).toBeInTheDocument()
        expect(screen.getByText('10 mg')).toBeInTheDocument()
      })

      // Test Edit functionality
      const editButton = screen.getByRole('button', { name: /edit/i })
      await user.click(editButton)

      // Modify dosage
      const dosageInput = screen.getByDisplayValue('10')
      await user.clear(dosageInput)
      await user.type(dosageInput, '20')

      // Save changes
      const saveButton = screen.getByRole('button', { name: /save/i })
      await user.click(saveButton)

      // Verify API was called
      expect(mockAPI.medicationAPI.updateMedication).toHaveBeenCalledWith('1', expect.objectContaining({
        dosage_amount: 20
      }))

      // Test Delete functionality
      const deleteButton = screen.getByRole('button', { name: /delete/i })
      await user.click(deleteButton)

      // Confirm deletion
      const confirmButton = screen.getByRole('button', { name: /confirm|yes/i })
      await user.click(confirmButton)

      // Verify API was called
      expect(mockAPI.medicationAPI.deleteMedication).toHaveBeenCalledWith('1')
    })

    it('should handle medication search and selection workflow', async () => {
      const searchResults = [
        { id: '1', name: 'Aspirin', generic_name: 'acetylsalicylic acid' },
        { id: '2', name: 'Ibuprofen', generic_name: 'ibuprofen' },
        { id: '3', name: 'Acetaminophen', generic_name: 'paracetamol' }
      ]

      mockAPI.medicationAPI.searchMedications = vi.fn().mockResolvedValue(searchResults)

      render(<App />)

      // Navigate to medication search
      const addMedButton = screen.getByRole('button', { name: /add medication/i })
      await user.click(addMedButton)

      // Search for medication
      const searchInput = screen.getByRole('textbox', { name: /search/i })
      await user.type(searchInput, 'aspirin')

      // Verify search results
      await waitFor(() => {
        expect(screen.getByText('Aspirin')).toBeInTheDocument()
        expect(screen.getByText('acetylsalicylic acid')).toBeInTheDocument()
      })

      // Select medication
      const aspirinOption = screen.getByText('Aspirin')
      await user.click(aspirinOption)

      // Verify selection
      expect(screen.getByDisplayValue('Aspirin')).toBeInTheDocument()
    })
  })

  describe('Safety Check Workflow Variations', () => {
    it('should handle multiple medication interactions', async () => {
      const multiDrugResponse = {
        risk_percentage: 65,
        risk_category: 'Warning',
        ai_explanation: 'Multiple interactions detected. Monitor closely.',
        interactions: [
          {
            drug1: 'Warfarin',
            drug2: 'Aspirin',
            severity: 'Moderate',
            description: 'Increased bleeding risk'
          },
          {
            drug1: 'Warfarin',
            drug2: 'Ibuprofen',
            severity: 'High',
            description: 'Significantly increased bleeding risk'
          }
        ]
      }

      mockAPI.medicationAPI.checkBeforeAdding.mockResolvedValue(multiDrugResponse)

      render(<App />)

      // Navigate to safety check
      const safetyCheckLink = screen.getByRole('link', { name: /safety check/i })
      await user.click(safetyCheckLink)

      // Add multiple medications (simplified)
      const medications = ['Warfarin', 'Aspirin', 'Ibuprofen']
      for (const med of medications) {
        const searchInput = screen.getByRole('textbox', { name: /search/i })
        await user.type(searchInput, med)
        const option = screen.getByText(med)
        await user.click(option)
      }

      // Proceed through steps
      const nextButton = screen.getByRole('button', { name: /next/i })
      await user.click(nextButton)

      const checkSafetyButton = screen.getByRole('button', { name: /check safety/i })
      await user.click(checkSafetyButton)

      // Verify multiple interactions are displayed
      await waitFor(() => {
        expect(screen.getByText(/65%/)).toBeInTheDocument()
        expect(screen.getByText(/warning/i)).toBeInTheDocument()
        expect(screen.getByText(/multiple interactions/i)).toBeInTheDocument()
        expect(screen.getByText(/warfarin.*aspirin/i)).toBeInTheDocument()
        expect(screen.getByText(/warfarin.*ibuprofen/i)).toBeInTheDocument()
      })
    })

    it('should handle emergency interaction checks', async () => {
      const emergencyResponse = {
        risk_percentage: 95,
        risk_category: 'Emergency',
        ai_explanation: 'CRITICAL: Seek immediate medical attention.',
        interactions: [
          {
            drug1: 'MAO Inhibitor',
            drug2: 'SSRI',
            severity: 'Critical',
            description: 'Serotonin syndrome risk - potentially fatal'
          }
        ],
        emergency: true
      }

      mockAPI.emergencyAPI.checkInteraction.mockResolvedValue(emergencyResponse)

      render(<App />)

      // Trigger emergency check
      const emergencyButton = screen.getByRole('button', { name: /emergency check/i })
      await user.click(emergencyButton)

      // Verify emergency warning
      await waitFor(() => {
        expect(screen.getByText(/emergency|critical/i)).toBeInTheDocument()
        expect(screen.getByText(/95%/)).toBeInTheDocument()
        expect(screen.getByText(/seek immediate medical attention/i)).toBeInTheDocument()
        expect(screen.getByText(/serotonin syndrome/i)).toBeInTheDocument()
      })

      // Verify emergency contact information is displayed
      expect(screen.getByText(/call 911|emergency services/i)).toBeInTheDocument()
    })
  })

  describe('Chatbot Integration Workflow', () => {
    it('should handle chatbot interactions within the application', async () => {
      const chatbotResponse = {
        message: 'I can help you understand your medication interactions. What would you like to know?',
        suggestions: [
          'Tell me about my current medications',
          'What are the side effects?',
          'How should I take this medication?'
        ]
      }

      mockAPI.chatbotAPI.askAssistant.mockResolvedValue(chatbotResponse)

      render(<App />)

      // Open chatbot
      const chatbotButton = screen.getByRole('button', { name: /chat|assistant/i })
      await user.click(chatbotButton)

      // Verify chatbot window opens
      await waitFor(() => {
        expect(screen.getByText(/assistant|chat/i)).toBeInTheDocument()
      })

      // Send message to chatbot
      const chatInput = screen.getByRole('textbox', { name: /message|chat/i })
      await user.type(chatInput, 'What medications am I taking?')

      const sendButton = screen.getByRole('button', { name: /send/i })
      await user.click(sendButton)

      // Verify response
      await waitFor(() => {
        expect(screen.getByText(/help you understand/i)).toBeInTheDocument()
        expect(screen.getByText(/current medications/i)).toBeInTheDocument()
      })

      // Test suggestion interaction
      const suggestion = screen.getByText(/current medications/i)
      await user.click(suggestion)

      // Verify suggestion was processed
      expect(mockAPI.chatbotAPI.askAssistant).toHaveBeenCalledWith(
        expect.objectContaining({
          message: expect.stringContaining('current medications')
        })
      )
    })
  })

  describe('Health Monitoring Workflow', () => {
    it('should display health alerts and monitoring data', async () => {
      const healthData = {
        totalMedications: 3,
        safeCount: 2,
        needsAttentionCount: 1,
        highRiskCount: 0,
        lastCheck: '2024-01-15T10:30:00Z'
      }

      const healthAlerts = [
        {
          id: '1',
          type: 'reminder',
          message: 'Time to take your morning medications',
          priority: 'medium',
          timestamp: '2024-01-15T08:00:00Z'
        },
        {
          id: '2',
          type: 'warning',
          message: 'Medication interaction detected - consult your doctor',
          priority: 'high',
          timestamp: '2024-01-15T09:15:00Z'
        }
      ]

      mockAPI.healthAPI.getHealthData.mockResolvedValue(healthData)
      mockAPI.healthAPI.getHealthAlerts.mockResolvedValue(healthAlerts)

      render(<App />)

      // Navigate to dashboard
      const dashboardLink = screen.getByRole('link', { name: /dashboard/i })
      await user.click(dashboardLink)

      // Verify health summary cards
      await waitFor(() => {
        expect(screen.getByText('3')).toBeInTheDocument() // Total medications
        expect(screen.getByText('2')).toBeInTheDocument() // Safe count
        expect(screen.getByText('1')).toBeInTheDocument() // Needs attention
      })

      // Verify health alerts
      expect(screen.getByText(/time to take your morning medications/i)).toBeInTheDocument()
      expect(screen.getByText(/medication interaction detected/i)).toBeInTheDocument()

      // Test alert interaction
      const highPriorityAlert = screen.getByText(/medication interaction detected/i)
      await user.click(highPriorityAlert)

      // Verify alert details are shown
      expect(screen.getByText(/consult your doctor/i)).toBeInTheDocument()
    })
  })

  describe('Cross-Browser Compatibility', () => {
    it('should handle different user agent strings', async () => {
      // Mock different browsers
      const browsers = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15'
      ]

      for (const userAgent of browsers) {
        Object.defineProperty(navigator, 'userAgent', {
          value: userAgent,
          configurable: true
        })

        render(<App />)

        // Verify basic functionality works across browsers
        expect(screen.getByText(/medicine assistant/i)).toBeInTheDocument()
        
        // Test navigation
        const aboutLink = screen.getByRole('link', { name: /about/i })
        await user.click(aboutLink)

        await waitFor(() => {
          expect(screen.getByText(/why this matters/i)).toBeInTheDocument()
        })
      }
    })
  })

  describe('Performance and Loading', () => {
    it('should handle slow network conditions gracefully', async () => {
      // Mock slow API responses
      mockAPI.medicationAPI.getMedications.mockImplementation(
        () => new Promise(resolve => setTimeout(() => resolve([]), 3000))
      )

      render(<App />)

      // Navigate to dashboard
      const dashboardLink = screen.getByRole('link', { name: /dashboard/i })
      await user.click(dashboardLink)

      // Verify loading state is shown
      expect(screen.getByTestId(/loading|spinner/)).toBeInTheDocument()

      // Verify timeout handling (if implemented)
      await waitFor(() => {
        expect(screen.getByText(/loading|please wait/i)).toBeInTheDocument()
      }, { timeout: 1000 })
    })

    it('should handle offline scenarios', async () => {
      // Mock network error
      mockAPI.medicationAPI.getMedications.mockRejectedValue(new Error('Network Error'))

      render(<App />)

      // Navigate to dashboard
      const dashboardLink = screen.getByRole('link', { name: /dashboard/i })
      await user.click(dashboardLink)

      // Verify offline message
      await waitFor(() => {
        expect(screen.getByText(/offline|network error|connection/i)).toBeInTheDocument()
      })

      // Verify retry functionality
      const retryButton = screen.getByRole('button', { name: /retry|try again/i })
      expect(retryButton).toBeInTheDocument()
    })
  })
})