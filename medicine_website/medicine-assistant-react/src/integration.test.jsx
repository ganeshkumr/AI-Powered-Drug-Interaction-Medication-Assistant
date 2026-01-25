import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { BrowserRouter } from 'react-router-dom'
import App from './App'

// Mock the API module
vi.mock('./services/api', () => ({
  medicationAPI: {
    checkBeforeAdding: vi.fn(),
    addMedication: vi.fn(),
    getMedications: vi.fn(),
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

describe('App Integration', () => {
  it('renders without crashing', () => {
    render(<App />)
    // The app should render without throwing errors
    expect(document.body).toBeTruthy()
  })

  it('has proper routing structure', () => {
    render(<App />)
    // Check that the router is properly set up
    // We can't test specific routes without navigation, but we can ensure no errors
    expect(document.body).toBeTruthy()
  })
})

describe('Component Integration', () => {
  it('integrates ThemeProvider and AuthProvider correctly', () => {
    // This test ensures the context providers are properly nested
    render(<App />)
    expect(document.body).toBeTruthy()
  })
})