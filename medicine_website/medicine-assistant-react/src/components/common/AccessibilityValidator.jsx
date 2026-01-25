import { useEffect, useState } from 'react'
import { validateAccessibility, validateColorContrast } from '../../utils/accessibility'

/**
 * Accessibility Validator Component
 * 
 * Validates accessibility compliance across the application.
 * This component runs accessibility checks and reports issues.
 * Should only be used in development mode.
 */
const AccessibilityValidator = ({ enabled = process.env.NODE_ENV === 'development' }) => {
  const [issues, setIssues] = useState([])
  const [isValidating, setIsValidating] = useState(false)

  useEffect(() => {
    if (!enabled) return

    const validatePage = () => {
      setIsValidating(true)
      
      // Run validation after a short delay to ensure DOM is ready
      setTimeout(() => {
        const pageIssues = validateAccessibility(document.body)
        setIssues(pageIssues)
        setIsValidating(false)
        
        // Log issues to console for developers
        if (pageIssues.length > 0) {
          console.group('🔍 Accessibility Issues Found')
          pageIssues.forEach((issue, index) => {
            const logMethod = issue.severity === 'error' ? console.error : console.warn
            logMethod(`${index + 1}. ${issue.message}`, issue.element)
          })
          console.groupEnd()
        } else {
          console.log('✅ No accessibility issues found')
        }
      }, 1000)
    }

    // Validate on mount and when DOM changes
    validatePage()

    // Set up mutation observer to re-validate when DOM changes
    const observer = new MutationObserver(() => {
      validatePage()
    })

    observer.observe(document.body, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ['class', 'id', 'aria-label', 'aria-labelledby', 'aria-describedby']
    })

    return () => {
      observer.disconnect()
    }
  }, [enabled])

  // Test color contrast for key design system colors
  useEffect(() => {
    if (!enabled) return

    const testColorContrast = () => {
      const colorTests = [
        { name: 'Primary on White', fg: '#0EA5E9', bg: '#FFFFFF' },
        { name: 'Primary 600 on White', fg: '#0284C7', bg: '#FFFFFF' },
        { name: 'Neutral 600 on White', fg: '#475569', bg: '#FFFFFF' },
        { name: 'Neutral 700 on White', fg: '#334155', bg: '#FFFFFF' },
        { name: 'Success 700 on Success 50', fg: '#047857', bg: '#F0FDF4' },
        { name: 'Warning 700 on Warning 50', fg: '#B45309', bg: '#FFFBEB' },
        { name: 'Danger 700 on Danger 50', fg: '#B91C1C', bg: '#FEF2F2' },
        { name: 'White on Primary', fg: '#FFFFFF', bg: '#0EA5E9' },
      ]

      console.group('🎨 Color Contrast Test Results')
      colorTests.forEach(test => {
        const result = validateColorContrast(test.fg, test.bg)
        const status = result.passesAA ? '✅' : '❌'
        console.log(`${status} ${test.name}: ${result.ratio}:1 (AA: ${result.passesAA}, AAA: ${result.passesAAA})`)
      })
      console.groupEnd()
    }

    testColorContrast()
  }, [enabled])

  if (!enabled) return null

  return (
    <div className="fixed bottom-4 right-4 z-50 max-w-sm">
      {isValidating && (
        <div className="bg-blue-100 border border-blue-300 rounded-lg p-3 mb-2 text-sm">
          🔍 Validating accessibility...
        </div>
      )}
      
      {issues.length > 0 && (
        <div className="bg-yellow-100 border border-yellow-300 rounded-lg p-3 text-sm">
          <div className="font-semibold text-yellow-800 mb-2">
            ⚠️ {issues.length} Accessibility Issue{issues.length > 1 ? 's' : ''} Found
          </div>
          <div className="text-yellow-700">
            Check console for details
          </div>
        </div>
      )}
      
      {!isValidating && issues.length === 0 && (
        <div className="bg-green-100 border border-green-300 rounded-lg p-3 text-sm">
          <div className="font-semibold text-green-800">
            ✅ No Accessibility Issues
          </div>
        </div>
      )}
    </div>
  )
}

export default AccessibilityValidator