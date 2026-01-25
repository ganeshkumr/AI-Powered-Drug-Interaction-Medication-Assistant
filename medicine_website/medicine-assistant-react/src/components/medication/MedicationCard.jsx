import React from 'react'
import { motion } from 'framer-motion'
import { Pill, X, Edit3, Clock, Calendar, AlertCircle } from 'lucide-react'
import { handleKeyboardNavigation, generateId } from '../../utils/accessibility'
import RiskBadge from '../risk/RiskBadge'

/**
 * MedicationCard Component
 * 
 * A versatile card component for displaying medication information across different contexts.
 * Supports three variants: selection, dashboard, and analysis views.
 * Enhanced with risk badges, improved UX for edit/delete actions, and responsive design.
 * 
 * Requirements: 3.3, 3.4, 6.3, 7.1, 7.2
 */
const MedicationCard = ({ 
  medication, 
  variant = 'selection', 
  onRemove, 
  onEdit,
  className = '',
  'data-testid': testId
}) => {
  // Handle edge cases and invalid data
  if (!medication || typeof medication !== 'object') {
    return null
  }

  const { 
    name, 
    dosage, 
    frequency, 
    timeOfDay, 
    riskLevel, 
    risk_level, 
    nextIntakeTime,
    next_intake_time,
    startDate,
    start_date,
    endDate,
    end_date
  } = medication
  
  // Validate and sanitize medication name
  const medicationName = (name || '').toString().trim()
  if (!medicationName) {
    return null
  }
  
  // Support both camelCase and snake_case for backward compatibility
  const risk = riskLevel || risk_level
  const nextIntake = nextIntakeTime || next_intake_time
  const start = startDate || start_date
  const end = endDate || end_date
  
  const cardId = generateId('medication-card')
  const removeButtonId = generateId('remove-button')
  const editButtonId = generateId('edit-button')

  // Base card styles using design system variables with enhanced responsive design
  const baseCardStyles = `
    bg-white rounded-lg shadow-md border border-gray-200 
    transition-all duration-200 ease-in-out
    hover:shadow-lg hover:border-primary-300
    focus-within:ring-2 focus-within:ring-primary-500 focus-within:ring-offset-2
    touch-manipulation
  `

  // Variant-specific styles with improved responsive design
  const variantStyles = {
    selection: `
      ${baseCardStyles}
      p-3 sm:p-4 inline-flex items-center space-x-2 sm:space-x-3
      hover:bg-primary-50 cursor-pointer
      min-w-0 max-w-full
    `,
    dashboard: `
      ${baseCardStyles}
      p-4 sm:p-6 w-full
      hover:bg-neutral-25
    `,
    analysis: `
      ${baseCardStyles}
      p-3 sm:p-4 w-full bg-neutral-50 border-neutral-300
      hover:bg-neutral-75
    `
  }

  // Time of day display formatting
  const formatTimeOfDay = (time) => {
    if (!time) return ''
    return time.charAt(0).toUpperCase() + time.slice(1)
  }

  // Format next intake time for display
  const formatNextIntakeTime = (time) => {
    if (!time) return ''
    try {
      const date = new Date(time)
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    } catch {
      return time
    }
  }

  // Get risk badge size based on variant
  const getRiskBadgeSize = () => {
    return variant === 'selection' ? 'small' : 'small'
  }

  const handleRemoveKeyDown = (event) => {
    handleKeyboardNavigation(event, {
      onEnter: (e) => {
        e.stopPropagation();
        onRemove();
      },
      onSpace: (e) => {
        e.preventDefault();
        e.stopPropagation();
        onRemove();
      }
    });
  };

  const handleEditKeyDown = (event) => {
    handleKeyboardNavigation(event, {
      onEnter: onEdit,
      onSpace: (e) => {
        e.preventDefault();
        onEdit();
      }
    });
  };

  // Render selection variant (pill-style for Step 1) with risk badge
  if (variant === 'selection') {
    return (
      <motion.div
        initial={{ scale: 0, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0, opacity: 0 }}
        whileHover={{ scale: 1.02 }}
        className={`${variantStyles.selection} ${className}`}
        data-testid={testId}
        id={cardId}
        role="listitem"
        aria-label={`Medication: ${medicationName}${dosage ? `, ${dosage}` : ''}${risk ? `, Risk: ${risk}` : ''}`}
      >
        <Pill className="w-4 h-4 sm:w-5 sm:h-5 text-primary-600 flex-shrink-0" aria-hidden="true" />
        <div className="flex flex-col min-w-0 flex-1">
          <span className="text-sm font-medium text-neutral-900 truncate">
            {medicationName}
          </span>
          <div className="flex items-center space-x-2 mt-1">
            {dosage && (
              <span className="text-xs text-neutral-600 truncate">
                {dosage}
              </span>
            )}
            {risk && (
              <RiskBadge riskLevel={risk} size="small" />
            )}
          </div>
        </div>
        {onRemove && (
          <button
            id={removeButtonId}
            onClick={(e) => {
              e.stopPropagation()
              onRemove()
            }}
            onKeyDown={handleRemoveKeyDown}
            className="ml-2 p-1 hover:bg-primary-200 rounded-full transition-colors flex-shrink-0 min-h-touch min-w-touch touch-manipulation focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-1"
            aria-label={`Remove ${medicationName} from medication list`}
            aria-describedby={cardId}
          >
            <X className="w-3 h-3 sm:w-4 sm:h-4 text-primary-700" aria-hidden="true" />
          </button>
        )}
      </motion.div>
    )
  }

  // Render dashboard variant (organized by time periods) with complete medication information
  if (variant === 'dashboard') {
    return (
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className={`${variantStyles.dashboard} ${className}`}
        data-testid={testId}
        id={cardId}
        role="article"
        aria-labelledby={`${cardId}-title`}
      >
        <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between space-y-4 sm:space-y-0">
          <div className="flex items-start space-x-3 flex-1 min-w-0">
            <div className="p-2 bg-primary-100 rounded-lg flex-shrink-0" aria-hidden="true">
              <Pill className="w-5 h-5 text-primary-600" />
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between sm:space-x-4">
                <div className="flex-1 min-w-0">
                  <h3 id={`${cardId}-title`} className="text-base sm:text-lg font-semibold text-neutral-900 truncate">
                    {medicationName}
                  </h3>
                  {risk && (
                    <div className="mt-2">
                      <RiskBadge riskLevel={risk} size={getRiskBadgeSize()} />
                    </div>
                  )}
                </div>
              </div>
              
              <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-3">
                {dosage && (
                  <div className="flex items-center text-sm text-neutral-600">
                    <span className="font-medium text-neutral-700 mr-2">Dosage:</span>
                    <span>{dosage}</span>
                  </div>
                )}
                {frequency && (
                  <div className="flex items-center text-sm text-neutral-600">
                    <Clock className="w-4 h-4 mr-2 text-neutral-500" aria-hidden="true" />
                    <span>{frequency}</span>
                  </div>
                )}
                {timeOfDay && (
                  <div className="flex items-center text-sm text-neutral-600">
                    <Calendar className="w-4 h-4 mr-2 text-neutral-500" aria-hidden="true" />
                    <span>{formatTimeOfDay(timeOfDay)}</span>
                  </div>
                )}
                {nextIntake && (
                  <div className="flex items-center text-sm text-neutral-600">
                    <AlertCircle className="w-4 h-4 mr-2 text-neutral-500" aria-hidden="true" />
                    <span className="font-medium text-neutral-700 mr-1">Next:</span>
                    <span>{formatNextIntakeTime(nextIntake)}</span>
                  </div>
                )}
              </div>
              
              {(start || end) && (
                <div className="mt-3 pt-3 border-t border-neutral-200">
                  <div className="flex flex-col sm:flex-row sm:space-x-4 space-y-1 sm:space-y-0 text-xs text-neutral-500">
                    {start && (
                      <span>Started: {new Date(start).toLocaleDateString()}</span>
                    )}
                    {end && (
                      <span>Ends: {new Date(end).toLocaleDateString()}</span>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
          
          <div className="flex items-center justify-end space-x-2 sm:ml-4 flex-shrink-0">
            {onEdit && (
              <button
                id={editButtonId}
                onClick={onEdit}
                onKeyDown={handleEditKeyDown}
                className="p-2 hover:bg-neutral-100 rounded-lg transition-colors min-h-touch min-w-touch touch-manipulation focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-1 group"
                aria-label={`Edit ${medicationName} medication details`}
                aria-describedby={cardId}
              >
                <Edit3 className="w-4 h-4 text-neutral-600 group-hover:text-primary-600 transition-colors" aria-hidden="true" />
              </button>
            )}
            {onRemove && (
              <button
                id={removeButtonId}
                onClick={onRemove}
                onKeyDown={handleRemoveKeyDown}
                className="p-2 hover:bg-red-100 rounded-lg transition-colors min-h-touch min-w-touch touch-manipulation focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-1 group"
                aria-label={`Remove ${medicationName} from medication list`}
                aria-describedby={cardId}
              >
                <X className="w-4 h-4 text-red-600 group-hover:text-red-700 transition-colors" aria-hidden="true" />
              </button>
            )}
          </div>
        </div>
      </motion.div>
    )
  }

  // Render analysis variant (for Step 3 results) with risk information
  if (variant === 'analysis') {
    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className={`${variantStyles.analysis} ${className}`}
        data-testid={testId}
        id={cardId}
        role="listitem"
        aria-labelledby={`${cardId}-title`}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3 flex-1 min-w-0">
            <div className="p-2 bg-secondary-100 rounded-lg flex-shrink-0" aria-hidden="true">
              <Pill className="w-4 h-4 sm:w-5 sm:h-5 text-secondary-600" />
            </div>
            <div className="flex-1 min-w-0">
              <h4 id={`${cardId}-title`} className="text-sm font-semibold text-neutral-900 truncate">
                {medicationName}
              </h4>
              <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-neutral-600">
                {dosage && <span className="bg-neutral-200 px-2 py-1 rounded">{dosage}</span>}
                {frequency && <span className="bg-neutral-200 px-2 py-1 rounded">{frequency}</span>}
                {timeOfDay && <span className="bg-neutral-200 px-2 py-1 rounded">{formatTimeOfDay(timeOfDay)}</span>}
              </div>
            </div>
          </div>
          {risk && (
            <div className="ml-3 flex-shrink-0">
              <RiskBadge riskLevel={risk} size="small" />
            </div>
          )}
        </div>
      </motion.div>
    )
  }

  return null
}

export default MedicationCard