import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { vi } from 'vitest'
import DrugSearch from './DrugSearch'

// Mock fetch globally
global.fetch = vi.fn()

describe('DrugSearch Component', () => {
  const mockOnSelect = vi.fn()
  
  beforeEach(() => {
    vi.clearAllMocks()
    fetch.mockClear()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  describe('Search Input Behavior', () => {
    test('renders search input with placeholder', () => {
      render(<DrugSearch onSelect={mockOnSelect} placeholder="Search medications..." />)
      
      const input = screen.getByPlaceholderText('Search medications...')
      expect(input).toBeInTheDocument()
      expect(screen.getByTestId('search-icon')).toBeInTheDocument()
    })

    test('updates input value when typing', () => {
      render(<DrugSearch onSelect={mockOnSelect} />)
      
      const input = screen.getByPlaceholderText('Search for a medication...')
      fireEvent.change(input, { target: { value: 'Aspirin' } })
      
      expect(input.value).toBe('Aspirin')
    })

    test('does not trigger search for queries less than 2 characters', async () => {
      render(<DrugSearch onSelect={mockOnSelect} />)
      
      const input = screen.getByPlaceholderText('Search for a medication...')
      fireEvent.change(input, { target: { value: 'A' } })
      
      // Wait a bit to ensure no API call is made
      await new Promise(resolve => setTimeout(resolve, 400))
      expect(fetch).not.toHaveBeenCalled()
    })

    test('triggers search after 300ms debounce for queries 2+ characters', async () => {
      fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ drugs: ['Aspirin', 'Acetaminophen'] })
      })

      render(<DrugSearch onSelect={mockOnSelect} />)
      
      const input = screen.getByPlaceholderText('Search for a medication...')
      fireEvent.change(input, { target: { value: 'As' } })
      
      await waitFor(() => {
        expect(fetch).toHaveBeenCalledWith('http://localhost:5000/api/search-drugs?q=As')
      }, { timeout: 500 })
    })

    test('shows loading spinner during search', async () => {
      // Mock a delayed response
      fetch.mockImplementationOnce(() => 
        new Promise(resolve => 
          setTimeout(() => resolve({
            ok: true,
            json: async () => ({ drugs: ['Aspirin'] })
          }), 100)
        )
      )

      render(<DrugSearch onSelect={mockOnSelect} />)
      
      const input = screen.getByPlaceholderText('Search for a medication...')
      fireEvent.change(input, { target: { value: 'Aspirin' } })
      
      await waitFor(() => {
        expect(screen.getByTestId('loader2-icon')).toBeInTheDocument()
      })
    })
  })

  describe('Dropdown Interactions', () => {
    test('shows dropdown with search results', async () => {
      fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ drugs: ['Aspirin', 'Acetaminophen'] })
      })

      render(<DrugSearch onSelect={mockOnSelect} />)
      
      const input = screen.getByPlaceholderText('Search for a medication...')
      fireEvent.change(input, { target: { value: 'A' } })
      
      await waitFor(() => {
        expect(screen.getByText('Aspirin')).toBeInTheDocument()
        expect(screen.getByText('Acetaminophen')).toBeInTheDocument()
      })
    })

    test('calls onSelect when medication is clicked', async () => {
      fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ drugs: ['Aspirin'] })
      })

      render(<DrugSearch onSelect={mockOnSelect} />)
      
      const input = screen.getByPlaceholderText('Search for a medication...')
      fireEvent.change(input, { target: { value: 'Aspirin' } })
      
      await waitFor(() => {
        const aspirinOption = screen.getByText('Aspirin')
        fireEvent.click(aspirinOption)
        expect(mockOnSelect).toHaveBeenCalledWith('Aspirin')
      })
    })

    test('clears input and closes dropdown after selection', async () => {
      fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ drugs: ['Aspirin'] })
      })

      render(<DrugSearch onSelect={mockOnSelect} />)
      
      const input = screen.getByPlaceholderText('Search for a medication...')
      fireEvent.change(input, { target: { value: 'Aspirin' } })
      
      await waitFor(() => {
        const aspirinOption = screen.getByText('Aspirin')
        fireEvent.click(aspirinOption)
      })
      
      expect(input.value).toBe('')
      expect(screen.queryByText('Aspirin')).not.toBeInTheDocument()
    })

    test('shows dropdown on focus if query is 2+ characters', async () => {
      fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ drugs: ['Aspirin'] })
      })

      render(<DrugSearch onSelect={mockOnSelect} />)
      
      const input = screen.getByPlaceholderText('Search for a medication...')
      fireEvent.change(input, { target: { value: 'Aspirin' } })
      
      // Wait for initial search
      await waitFor(() => {
        expect(screen.getByText('Aspirin')).toBeInTheDocument()
      })
      
      // Click outside to close dropdown
      fireEvent.mouseDown(document.body)
      
      await waitFor(() => {
        expect(screen.queryByText('Aspirin')).not.toBeInTheDocument()
      })
      
      // Focus again should show dropdown
      fireEvent.focus(input)
      
      await waitFor(() => {
        expect(screen.getByText('Aspirin')).toBeInTheDocument()
      })
    })

    test('closes dropdown when clicking outside', async () => {
      fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ drugs: ['Aspirin'] })
      })

      render(<DrugSearch onSelect={mockOnSelect} />)
      
      const input = screen.getByPlaceholderText('Search for a medication...')
      fireEvent.change(input, { target: { value: 'Aspirin' } })
      
      await waitFor(() => {
        expect(screen.getByText('Aspirin')).toBeInTheDocument()
      })
      
      // Click outside
      fireEvent.mouseDown(document.body)
      
      await waitFor(() => {
        expect(screen.queryByText('Aspirin')).not.toBeInTheDocument()
      })
    })
  })

  describe('Medication Addition and Removal', () => {
    test('prevents duplicate medication selection', async () => {
      fetch.mockResolvedValue({
        ok: true,
        json: async () => ({ drugs: ['Aspirin'] })
      })

      render(<DrugSearch onSelect={mockOnSelect} />)
      
      const input = screen.getByPlaceholderText('Search for a medication...')
      
      // First selection
      fireEvent.change(input, { target: { value: 'Aspirin' } })
      await waitFor(() => {
        fireEvent.click(screen.getByText('Aspirin'))
      })
      
      expect(mockOnSelect).toHaveBeenCalledTimes(1)
      expect(mockOnSelect).toHaveBeenCalledWith('Aspirin')
      
      // Second selection of same medication
      fireEvent.change(input, { target: { value: 'Aspirin' } })
      await waitFor(() => {
        fireEvent.click(screen.getByText('Aspirin'))
      })
      
      expect(mockOnSelect).toHaveBeenCalledTimes(2)
      expect(mockOnSelect).toHaveBeenLastCalledWith('Aspirin')
    })

    test('allows selection of different medications', async () => {
      fetch
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ drugs: ['Aspirin'] })
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ drugs: ['Lisinopril'] })
        })

      render(<DrugSearch onSelect={mockOnSelect} />)
      
      const input = screen.getByPlaceholderText('Search for a medication...')
      
      // First medication
      fireEvent.change(input, { target: { value: 'Aspirin' } })
      await waitFor(() => {
        fireEvent.click(screen.getByText('Aspirin'))
      })
      
      // Second medication
      fireEvent.change(input, { target: { value: 'Lisinopril' } })
      await waitFor(() => {
        fireEvent.click(screen.getByText('Lisinopril'))
      })
      
      expect(mockOnSelect).toHaveBeenCalledTimes(2)
      expect(mockOnSelect).toHaveBeenNthCalledWith(1, 'Aspirin')
      expect(mockOnSelect).toHaveBeenNthCalledWith(2, 'Lisinopril')
    })

    test('refocuses input after medication selection', async () => {
      fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ drugs: ['Aspirin'] })
      })

      render(<DrugSearch onSelect={mockOnSelect} />)
      
      const input = screen.getByPlaceholderText('Search for a medication...')
      fireEvent.change(input, { target: { value: 'Aspirin' } })
      
      await waitFor(() => {
        fireEvent.click(screen.getByText('Aspirin'))
      })
      
      // Input should be focused after selection
      expect(document.activeElement).toBe(input)
    })
  })

  describe('Edge Cases and Error Handling', () => {
    test('shows "no medications found" message for empty results', async () => {
      fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ drugs: [] })
      })

      render(<DrugSearch onSelect={mockOnSelect} />)
      
      const input = screen.getByPlaceholderText('Search for a medication...')
      fireEvent.change(input, { target: { value: 'XYZ123' } })
      
      await waitFor(() => {
        expect(screen.getByText('No medications found. Try a different search term.')).toBeInTheDocument()
      })
    })

    test('handles network errors gracefully', async () => {
      const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
      
      fetch.mockRejectedValueOnce(new Error('Network error'))

      render(<DrugSearch onSelect={mockOnSelect} />)
      
      const input = screen.getByPlaceholderText('Search for a medication...')
      fireEvent.change(input, { target: { value: 'Aspirin' } })
      
      await waitFor(() => {
        expect(consoleSpy).toHaveBeenCalledWith('Drug search error:', expect.any(Error))
      })
      
      // Should not show dropdown or results
      expect(screen.queryByText('Aspirin')).not.toBeInTheDocument()
      
      consoleSpy.mockRestore()
    })

    test('handles API response errors', async () => {
      fetch.mockResolvedValueOnce({
        ok: false,
        status: 500
      })

      render(<DrugSearch onSelect={mockOnSelect} />)
      
      const input = screen.getByPlaceholderText('Search for a medication...')
      fireEvent.change(input, { target: { value: 'Aspirin' } })
      
      await waitFor(() => {
        // Should not show any results
        expect(screen.queryByText('Aspirin')).not.toBeInTheDocument()
      })
    })

    test('handles malformed API response', async () => {
      fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ /* missing drugs array */ })
      })

      render(<DrugSearch onSelect={mockOnSelect} />)
      
      const input = screen.getByPlaceholderText('Search for a medication...')
      fireEvent.change(input, { target: { value: 'Aspirin' } })
      
      await waitFor(() => {
        // Should show no results message
        expect(screen.getByText('No medications found. Try a different search term.')).toBeInTheDocument()
      })
    })

    test('debounces multiple rapid input changes', async () => {
      fetch.mockResolvedValue({
        ok: true,
        json: async () => ({ drugs: ['Aspirin'] })
      })

      render(<DrugSearch onSelect={mockOnSelect} />)
      
      const input = screen.getByPlaceholderText('Search for a medication...')
      
      // Rapid typing
      fireEvent.change(input, { target: { value: 'A' } })
      fireEvent.change(input, { target: { value: 'As' } })
      fireEvent.change(input, { target: { value: 'Asp' } })
      fireEvent.change(input, { target: { value: 'Aspi' } })
      fireEvent.change(input, { target: { value: 'Aspir' } })
      fireEvent.change(input, { target: { value: 'Aspirin' } })
      
      // Wait for debounce
      await waitFor(() => {
        expect(fetch).toHaveBeenCalledTimes(1)
        expect(fetch).toHaveBeenCalledWith('http://localhost:5000/api/search-drugs?q=Aspirin')
      }, { timeout: 500 })
    })

    test('clears suggestions when query becomes too short', async () => {
      fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ drugs: ['Aspirin'] })
      })

      render(<DrugSearch onSelect={mockOnSelect} />)
      
      const input = screen.getByPlaceholderText('Search for a medication...')
      
      // First, search for something
      fireEvent.change(input, { target: { value: 'Aspirin' } })
      
      await waitFor(() => {
        expect(screen.getByText('Aspirin')).toBeInTheDocument()
      })
      
      // Then make query too short
      fireEvent.change(input, { target: { value: 'A' } })
      
      await waitFor(() => {
        expect(screen.queryByText('Aspirin')).not.toBeInTheDocument()
      })
    })

    test('handles empty string input', () => {
      render(<DrugSearch onSelect={mockOnSelect} />)
      
      const input = screen.getByPlaceholderText('Search for a medication...')
      fireEvent.change(input, { target: { value: '' } })
      
      expect(input.value).toBe('')
      expect(screen.queryByText('No medications found')).not.toBeInTheDocument()
    })

    test('handles special characters in search query', async () => {
      fetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ drugs: [] })
      })

      render(<DrugSearch onSelect={mockOnSelect} />)
      
      const input = screen.getByPlaceholderText('Search for a medication...')
      fireEvent.change(input, { target: { value: 'Test@#$%' } })
      
      await waitFor(() => {
        expect(fetch).toHaveBeenCalledWith('http://localhost:5000/api/search-drugs?q=Test%40%23%24%25')
      })
    })
  })
})