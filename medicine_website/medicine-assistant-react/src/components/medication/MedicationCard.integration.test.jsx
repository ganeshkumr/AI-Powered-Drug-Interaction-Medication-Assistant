import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MedicationCard } from './index'

describe('MedicationCard Integration', () => {
  it('can be imported and rendered from index', () => {
    const medication = {
      name: 'Aspirin',
      dosage: '100mg',
      frequency: 'Once daily',
      timeOfDay: 'morning'
    }

    render(
      <MedicationCard 
        medication={medication} 
        variant="selection" 
        data-testid="integration-test"
      />
    )

    expect(screen.getByTestId('integration-test')).toBeInTheDocument()
    expect(screen.getByText('Aspirin')).toBeInTheDocument()
  })

  it('works with all three variants', () => {
    const medication = {
      name: 'Test Medication',
      dosage: '50mg',
      frequency: 'Twice daily',
      timeOfDay: 'morning'
    }

    const variants = ['selection', 'dashboard', 'analysis']
    
    variants.forEach((variant, index) => {
      const { unmount } = render(
        <MedicationCard 
          medication={medication} 
          variant={variant} 
          data-testid={`variant-${variant}`}
        />
      )
      
      expect(screen.getByTestId(`variant-${variant}`)).toBeInTheDocument()
      expect(screen.getByText('Test Medication')).toBeInTheDocument()
      
      unmount()
    })
  })
})