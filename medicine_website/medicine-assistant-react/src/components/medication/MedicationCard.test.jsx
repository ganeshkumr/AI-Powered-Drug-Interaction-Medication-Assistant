import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import MedicationCard from './MedicationCard'

describe('MedicationCard', () => {
  const mockMedication = {
    name: 'Aspirin',
    dosage: '100mg',
    frequency: 'Once daily',
    timeOfDay: 'morning',
    riskLevel: 'safe',
    nextIntakeTime: '2024-01-15T08:00:00Z',
    startDate: '2024-01-01',
    endDate: '2024-12-31'
  }

  describe('Selection Variant', () => {
    it('renders medication name and dosage in selection variant', () => {
      render(
        <MedicationCard 
          medication={mockMedication} 
          variant="selection" 
          data-testid="medication-card"
        />
      )
      
      expect(screen.getByText('Aspirin')).toBeInTheDocument()
      expect(screen.getByText('100mg')).toBeInTheDocument()
      expect(screen.getByRole('listitem')).toBeInTheDocument()
    })

    it('renders risk badge when risk level is provided', () => {
      render(
        <MedicationCard 
          medication={mockMedication} 
          variant="selection" 
          data-testid="medication-card"
        />
      )
      
      expect(screen.getByRole('status')).toBeInTheDocument()
    })

    it('calls onRemove when remove button is clicked in selection variant', () => {
      const onRemove = vi.fn()
      render(
        <MedicationCard 
          medication={mockMedication} 
          variant="selection" 
          onRemove={onRemove}
        />
      )
      
      const removeButton = screen.getByLabelText('Remove Aspirin from medication list')
      fireEvent.click(removeButton)
      
      expect(onRemove).toHaveBeenCalledTimes(1)
    })

    it('does not render remove button when onRemove is not provided', () => {
      render(
        <MedicationCard 
          medication={mockMedication} 
          variant="selection" 
        />
      )
      
      expect(screen.queryByLabelText('Remove Aspirin from medication list')).not.toBeInTheDocument()
    })

    it('renders without dosage when not provided', () => {
      const medicationWithoutDosage = { name: 'Aspirin' }
      render(
        <MedicationCard 
          medication={medicationWithoutDosage} 
          variant="selection" 
        />
      )
      
      expect(screen.getByText('Aspirin')).toBeInTheDocument()
      expect(screen.queryByText('100mg')).not.toBeInTheDocument()
    })
  })

  describe('Dashboard Variant', () => {
    it('renders all medication information in dashboard variant', () => {
      render(
        <MedicationCard 
          medication={mockMedication} 
          variant="dashboard" 
          data-testid="medication-card"
        />
      )
      
      expect(screen.getByText('Aspirin')).toBeInTheDocument()
      expect(screen.getByText('Dosage:')).toBeInTheDocument()
      expect(screen.getByText('100mg')).toBeInTheDocument()
      expect(screen.getByText('Once daily')).toBeInTheDocument()
      expect(screen.getByText('Morning')).toBeInTheDocument()
      expect(screen.getByRole('status')).toBeInTheDocument() // Risk badge
      expect(screen.getByRole('article')).toBeInTheDocument()
    })

    it('renders next intake time when provided', () => {
      render(
        <MedicationCard 
          medication={mockMedication} 
          variant="dashboard" 
        />
      )
      
      expect(screen.getByText('Next:')).toBeInTheDocument()
    })

    it('renders start and end dates when provided', () => {
      render(
        <MedicationCard 
          medication={mockMedication} 
          variant="dashboard" 
        />
      )
      
      expect(screen.getByText(/Started:/)).toBeInTheDocument()
      expect(screen.getByText(/Ends:/)).toBeInTheDocument()
    })

    it('calls onEdit when edit button is clicked in dashboard variant', () => {
      const onEdit = vi.fn()
      render(
        <MedicationCard 
          medication={mockMedication} 
          variant="dashboard" 
          onEdit={onEdit}
        />
      )
      
      const editButton = screen.getByLabelText('Edit Aspirin medication details')
      fireEvent.click(editButton)
      
      expect(onEdit).toHaveBeenCalledTimes(1)
    })

    it('calls onRemove when remove button is clicked in dashboard variant', () => {
      const onRemove = vi.fn()
      render(
        <MedicationCard 
          medication={mockMedication} 
          variant="dashboard" 
          onRemove={onRemove}
        />
      )
      
      const removeButton = screen.getByLabelText('Remove Aspirin from medication list')
      fireEvent.click(removeButton)
      
      expect(onRemove).toHaveBeenCalledTimes(1)
    })

    it('formats time of day correctly', () => {
      const afternoonMedication = { ...mockMedication, timeOfDay: 'afternoon' }
      render(
        <MedicationCard 
          medication={afternoonMedication} 
          variant="dashboard" 
        />
      )
      
      expect(screen.getByText('Afternoon')).toBeInTheDocument()
    })

    it('renders without optional fields when not provided', () => {
      const minimalMedication = { name: 'Aspirin' }
      render(
        <MedicationCard 
          medication={minimalMedication} 
          variant="dashboard" 
        />
      )
      
      expect(screen.getByText('Aspirin')).toBeInTheDocument()
      expect(screen.queryByText('Dosage:')).not.toBeInTheDocument()
      expect(screen.queryByText('Next:')).not.toBeInTheDocument()
      expect(screen.queryByText(/Started:/)).not.toBeInTheDocument()
    })

    it('supports backward compatibility with snake_case properties', () => {
      const snakeCaseMedication = {
        name: 'Aspirin',
        dosage: '100mg',
        risk_level: 'warning',
        next_intake_time: '2024-01-15T08:00:00Z',
        start_date: '2024-01-01'
      }
      
      render(
        <MedicationCard 
          medication={snakeCaseMedication} 
          variant="dashboard" 
        />
      )
      
      expect(screen.getByText('Aspirin')).toBeInTheDocument()
      expect(screen.getByRole('status')).toBeInTheDocument() // Risk badge
      expect(screen.getByText('Next:')).toBeInTheDocument()
      expect(screen.getByText(/Started:/)).toBeInTheDocument()
    })
  })

  describe('Analysis Variant', () => {
    it('renders medication information in compact analysis variant', () => {
      render(
        <MedicationCard 
          medication={mockMedication} 
          variant="analysis" 
          data-testid="medication-card"
        />
      )
      
      expect(screen.getByText('Aspirin')).toBeInTheDocument()
      expect(screen.getByText('100mg')).toBeInTheDocument()
      expect(screen.getByText('Once daily')).toBeInTheDocument()
      expect(screen.getByText('Morning')).toBeInTheDocument()
      expect(screen.getByRole('status')).toBeInTheDocument() // Risk badge
      expect(screen.getByRole('listitem')).toBeInTheDocument()
    })

    it('renders without optional fields in analysis variant', () => {
      const minimalMedication = { name: 'Aspirin' }
      render(
        <MedicationCard 
          medication={minimalMedication} 
          variant="analysis" 
        />
      )
      
      expect(screen.getByText('Aspirin')).toBeInTheDocument()
      expect(screen.queryByText('100mg')).not.toBeInTheDocument()
      expect(screen.queryByRole('status')).not.toBeInTheDocument() // No risk badge
    })
  })

  describe('Edge Cases', () => {
    it('handles empty medication name gracefully', () => {
      const emptyMedication = { name: '' }
      const { container } = render(
        <MedicationCard 
          medication={emptyMedication} 
          variant="selection" 
        />
      )
      
      // Should return null for empty medication name
      expect(container.firstChild).toBeNull()
    })

    it('applies custom className when provided', () => {
      render(
        <MedicationCard 
          medication={mockMedication} 
          variant="selection" 
          className="custom-class"
          data-testid="medication-card"
        />
      )
      
      const card = screen.getByTestId('medication-card')
      expect(card).toHaveClass('custom-class')
    })

    it('returns null for invalid variant', () => {
      const { container } = render(
        <MedicationCard 
          medication={mockMedication} 
          variant="invalid" 
        />
      )
      
      expect(container.firstChild).toBeNull()
    })

    it('handles long medication names with truncation', () => {
      const longNameMedication = { 
        name: 'Very Long Medication Name That Should Be Truncated' 
      }
      render(
        <MedicationCard 
          medication={longNameMedication} 
          variant="selection" 
        />
      )
      
      const nameElement = screen.getByText('Very Long Medication Name That Should Be Truncated')
      expect(nameElement).toHaveClass('truncate')
    })

    it('handles invalid date formats gracefully', () => {
      const invalidDateMedication = {
        ...mockMedication,
        nextIntakeTime: 'invalid-date',
        startDate: 'invalid-date'
      }
      
      render(
        <MedicationCard 
          medication={invalidDateMedication} 
          variant="dashboard" 
        />
      )
      
      expect(screen.getByText('Aspirin')).toBeInTheDocument()
      // Should not crash and should handle invalid dates gracefully
    })
  })

  describe('Accessibility', () => {
    it('provides proper aria-labels for action buttons', () => {
      const onRemove = vi.fn()
      const onEdit = vi.fn()
      
      render(
        <MedicationCard 
          medication={mockMedication} 
          variant="dashboard" 
          onRemove={onRemove}
          onEdit={onEdit}
        />
      )
      
      expect(screen.getByLabelText('Edit Aspirin medication details')).toBeInTheDocument()
      expect(screen.getByLabelText('Remove Aspirin from medication list')).toBeInTheDocument()
    })

    it('supports data-testid for testing', () => {
      render(
        <MedicationCard 
          medication={mockMedication} 
          variant="selection" 
          data-testid="test-card"
        />
      )
      
      expect(screen.getByTestId('test-card')).toBeInTheDocument()
    })
  })
})