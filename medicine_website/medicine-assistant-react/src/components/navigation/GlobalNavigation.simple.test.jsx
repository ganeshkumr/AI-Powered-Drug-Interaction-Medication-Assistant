import { render, screen } from '@testing-library/react'
import { describe, expect, vi, test } from 'vitest'
import GlobalNavigation from './GlobalNavigation'

describe('GlobalNavigation Simple Test', () => {
  test('renders basic navigation elements', () => {
    const mockOnChatbotToggle = vi.fn()
    
    render(
      <GlobalNavigation
        currentPage="/"
        user={null}
        onChatbotToggle={mockOnChatbotToggle}
      />
    )

    // Check if basic elements are present
    expect(screen.getByText('AI-HealthMate')).toBeInTheDocument()
    expect(screen.getByText('Safety Check')).toBeInTheDocument()
    expect(screen.getByText('My Med')).toBeInTheDocument()
    expect(screen.getByText('About')).toBeInTheDocument()
  })
})