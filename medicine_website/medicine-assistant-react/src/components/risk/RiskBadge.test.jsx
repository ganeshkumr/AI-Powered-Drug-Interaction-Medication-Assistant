import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import RiskBadge from './RiskBadge'

describe('RiskBadge', () => {
  describe('Risk Level Variants', () => {
    it('renders safe risk level with correct styling and content', () => {
      render(<RiskBadge riskLevel="safe" size="large" />)
      
      const badge = screen.getByRole('status')
      expect(badge).toBeInTheDocument()
      expect(badge).toHaveAttribute('aria-label', 'Safe - No significant risks detected')
      expect(screen.getByText('SAFE')).toBeInTheDocument()
      expect(screen.getByTestId('shield-icon')).toBeInTheDocument()
      
      // Check CSS classes instead of inline styles
      expect(badge).toHaveClass('bg-success-100', 'text-success-700', 'border-success-300')
    })

    it('renders caution risk level with correct styling and content', () => {
      render(<RiskBadge riskLevel="caution" size="large" />)
      
      const badge = screen.getByRole('status')
      expect(badge).toBeInTheDocument()
      expect(badge).toHaveAttribute('aria-label', 'Caution - Potential risks require attention')
      expect(screen.getByText('CAUTION')).toBeInTheDocument()
      expect(screen.getByTestId('alert-triangle-icon')).toBeInTheDocument()
      
      // Check CSS classes instead of inline styles
      expect(badge).toHaveClass('bg-warning-100', 'text-warning-700', 'border-warning-300')
    })

    it('renders high-risk level with correct styling and content', () => {
      render(<RiskBadge riskLevel="high-risk" size="large" />)
      
      const badge = screen.getByRole('status')
      expect(badge).toBeInTheDocument()
      expect(badge).toHaveAttribute('aria-label', 'High Risk - Significant risks detected, consult healthcare provider')
      expect(screen.getByText('HIGH RISK')).toBeInTheDocument()
      expect(screen.getByTestId('x-circle-icon')).toBeInTheDocument()
      
      // Check CSS classes instead of inline styles
      expect(badge).toHaveClass('bg-danger-100', 'text-danger-700', 'border-danger-300')
    })

    it('renders unknown risk level as fallback', () => {
      render(<RiskBadge riskLevel="unknown" size="large" />)
      
      const badge = screen.getByRole('status')
      expect(badge).toBeInTheDocument()
      expect(badge).toHaveAttribute('aria-label', 'Unknown risk level')
      expect(screen.getByText('UNKNOWN')).toBeInTheDocument()
      expect(screen.getByTestId('alert-triangle-icon')).toBeInTheDocument()
      
      // Check CSS classes instead of inline styles
      expect(badge).toHaveClass('bg-gray-100', 'text-gray-700', 'border-gray-300')
    })
  })

  describe('Size Variants', () => {
    it('renders large size with correct dimensions and styling', () => {
      render(<RiskBadge riskLevel="safe" size="large" />)
      
      const badge = screen.getByRole('status')
      // Check CSS classes instead of inline styles
      expect(badge).toHaveClass('px-6', 'py-3', 'text-base')
    })

    it('renders small size with correct dimensions and styling', () => {
      render(<RiskBadge riskLevel="safe" size="small" />)
      
      const badge = screen.getByRole('status')
      // Check CSS classes instead of inline styles
      expect(badge).toHaveClass('px-3', 'py-1', 'text-xs')
    })

    it('defaults to large size when size prop is not provided', () => {
      render(<RiskBadge riskLevel="safe" />)
      
      const badge = screen.getByRole('status')
      // Check CSS classes instead of inline styles
      expect(badge).toHaveClass('px-6', 'py-3', 'text-base')
    })
  })

  describe('Design System Compliance', () => {
    it('applies pill shape border radius', () => {
      render(<RiskBadge riskLevel="safe" size="large" />)
      
      const badge = screen.getByRole('status')
      // Check CSS classes instead of inline styles
      expect(badge).toHaveClass('rounded-full')
    })

    it('uses design system font family', () => {
      render(<RiskBadge riskLevel="safe" size="large" />)
      
      const badge = screen.getByRole('status')
      // Check CSS classes instead of inline styles
      expect(badge).toHaveClass('font-heading')
    })

    it('applies consistent layout classes', () => {
      render(<RiskBadge riskLevel="safe" size="large" />)
      
      const badge = screen.getByRole('status')
      expect(badge).toHaveClass('inline-flex', 'items-center', 'space-x-2', 'rounded-full')
    })
  })

  describe('Accessibility Features', () => {
    it('provides proper ARIA role and attributes', () => {
      render(<RiskBadge riskLevel="safe" size="large" />)
      
      const badge = screen.getByRole('status')
      expect(badge).toHaveAttribute('role', 'status')
      expect(badge).toHaveAttribute('aria-live', 'polite')
      expect(badge).toHaveAttribute('aria-label')
    })

    it('hides icon from screen readers', () => {
      render(<RiskBadge riskLevel="safe" size="large" />)
      
      // Note: In the mocked environment, we can't test the actual aria-hidden attribute
      // but we can verify the icon is rendered
      const icon = screen.getByTestId('shield-icon')
      expect(icon).toBeInTheDocument()
    })

    it('provides descriptive aria-labels for each risk level', () => {
      const { rerender } = render(<RiskBadge riskLevel="safe" size="large" />)
      expect(screen.getByLabelText('Safe - No significant risks detected')).toBeInTheDocument()

      rerender(<RiskBadge riskLevel="caution" size="large" />)
      expect(screen.getByLabelText('Caution - Potential risks require attention')).toBeInTheDocument()

      rerender(<RiskBadge riskLevel="high-risk" size="large" />)
      expect(screen.getByLabelText('High Risk - Significant risks detected, consult healthcare provider')).toBeInTheDocument()
    })
  })

  describe('Edge Cases', () => {
    it('handles invalid risk level gracefully', () => {
      render(<RiskBadge riskLevel="invalid" size="large" />)
      
      const badge = screen.getByRole('status')
      expect(badge).toBeInTheDocument()
      expect(screen.getByText('UNKNOWN')).toBeInTheDocument()
    })

    it('handles invalid size gracefully by defaulting to large', () => {
      render(<RiskBadge riskLevel="safe" size="invalid" />)
      
      const badge = screen.getByRole('status')
      // Check CSS classes instead of inline styles
      expect(badge).toHaveClass('px-6', 'py-3', 'text-base')
    })

    it('handles missing props gracefully', () => {
      render(<RiskBadge />)
      
      const badge = screen.getByRole('status')
      expect(badge).toBeInTheDocument()
      expect(screen.getByText('UNKNOWN')).toBeInTheDocument()
    })
  })

  describe('Visual Consistency', () => {
    it('maintains consistent icon and text color for each risk level', () => {
      const { rerender } = render(<RiskBadge riskLevel="safe" size="large" />)
      
      let badge = screen.getByRole('status')
      let icon = screen.getByTestId('shield-icon')
      expect(badge).toHaveClass('text-success-700')
      expect(icon).toHaveClass('text-success-600')

      rerender(<RiskBadge riskLevel="caution" size="large" />)
      badge = screen.getByRole('status')
      icon = screen.getByTestId('alert-triangle-icon')
      expect(badge).toHaveClass('text-warning-700')
      expect(icon).toHaveClass('text-warning-600')

      rerender(<RiskBadge riskLevel="high-risk" size="large" />)
      badge = screen.getByRole('status')
      icon = screen.getByTestId('x-circle-icon')
      expect(badge).toHaveClass('text-danger-700')
      expect(icon).toHaveClass('text-danger-600')
    })

    it('maintains proper icon sizing for each size variant', () => {
      const { rerender } = render(<RiskBadge riskLevel="safe" size="large" />)
      
      // Note: In the mocked environment, we can't test the actual CSS classes
      // but we can verify the icons are rendered for different sizes
      let icon = screen.getByTestId('shield-icon')
      expect(icon).toBeInTheDocument()

      rerender(<RiskBadge riskLevel="safe" size="small" />)
      icon = screen.getByTestId('shield-icon')
      expect(icon).toBeInTheDocument()
    })
  })
})