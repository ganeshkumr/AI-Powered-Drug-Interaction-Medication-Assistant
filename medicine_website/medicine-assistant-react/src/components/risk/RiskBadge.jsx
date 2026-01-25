import { motion } from 'framer-motion'
import { Shield, AlertTriangle, XCircle } from 'lucide-react'

const RiskBadge = ({ riskLevel, verdict, size = 'large' }) => {
  // Use riskLevel if provided, otherwise fall back to verdict for backward compatibility
  const level = riskLevel || verdict
  
  // Determine badge configuration based on risk level
  const getBadgeConfig = () => {
    if (!level) {
      return {
        text: 'UNKNOWN',
        icon: AlertTriangle,
        bgColor: 'bg-gray-100',
        textColor: 'text-gray-700',
        borderColor: 'border-gray-300',
        iconColor: 'text-gray-600',
        ariaLabel: 'Unknown risk level'
      }
    }
    
    const levelLower = level.toLowerCase()
    
    if (levelLower.includes('safe')) {
      return {
        text: 'SAFE',
        icon: Shield,
        bgColor: 'bg-success-100',
        textColor: 'text-success-700',
        borderColor: 'border-success-300',
        iconColor: 'text-success-600',
        ariaLabel: 'Safe - No significant risks detected'
      }
    } else if (levelLower.includes('caution') || levelLower.includes('warning')) {
      return {
        text: 'CAUTION',
        icon: AlertTriangle,
        bgColor: 'bg-warning-100',
        textColor: 'text-warning-700',
        borderColor: 'border-warning-300',
        iconColor: 'text-warning-600',
        ariaLabel: 'Caution - Potential risks require attention'
      }
    } else if (levelLower.includes('high') || levelLower.includes('dangerous') || levelLower.includes('unsafe')) {
      return {
        text: 'HIGH RISK',
        icon: XCircle,
        bgColor: 'bg-danger-100',
        textColor: 'text-danger-700',
        borderColor: 'border-danger-300',
        iconColor: 'text-danger-600',
        ariaLabel: 'High Risk - Significant risks detected, consult healthcare provider'
      }
    } else {
      return {
        text: 'UNKNOWN',
        icon: AlertTriangle,
        bgColor: 'bg-gray-100',
        textColor: 'text-gray-700',
        borderColor: 'border-gray-300',
        iconColor: 'text-gray-600',
        ariaLabel: 'Unknown risk level'
      }
    }
  }

  const config = getBadgeConfig()
  const Icon = config.icon

  // Size configurations
  const sizes = {
    small: { padding: 'px-3 py-1', text: 'text-xs', icon: 'w-3 h-3' },
    large: { padding: 'px-6 py-3', text: 'text-base', icon: 'w-5 h-5' }
  }

  const sizeConfig = sizes[size] || sizes.large

  return (
    <motion.div
      role="status"
      aria-label={config.ariaLabel}
      aria-live="polite"
      initial={{ scale: 0, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ type: 'spring', delay: 0.3 }}
      whileHover={{ scale: 1.05 }}
      className={`inline-flex items-center space-x-2 ${sizeConfig.padding} ${config.bgColor} ${config.textColor} border-2 ${config.borderColor} rounded-full font-bold font-heading ${sizeConfig.text} shadow-soft`}
    >
      <Icon className={`${sizeConfig.icon} ${config.iconColor}`} aria-hidden="true" />
      <span>{config.text}</span>
    </motion.div>
  )
}

export default RiskBadge
