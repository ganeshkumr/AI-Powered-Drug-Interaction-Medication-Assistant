import { motion } from 'framer-motion'
import { Shield, AlertTriangle, XCircle } from 'lucide-react'

const RiskBadge = ({ verdict, size = 'md' }) => {
  // Determine badge configuration based on verdict
  const getBadgeConfig = () => {
    const verdictLower = verdict.toLowerCase()
    
    if (verdictLower.includes('safe')) {
      return {
        text: 'SAFE',
        icon: Shield,
        bgColor: 'bg-success-100',
        textColor: 'text-success-700',
        borderColor: 'border-success-300',
        iconColor: 'text-success-600'
      }
    } else if (verdictLower.includes('caution')) {
      return {
        text: 'CAUTION',
        icon: AlertTriangle,
        bgColor: 'bg-warning-100',
        textColor: 'text-warning-700',
        borderColor: 'border-warning-300',
        iconColor: 'text-warning-600'
      }
    } else {
      return {
        text: 'DANGEROUS',
        icon: XCircle,
        bgColor: 'bg-danger-100',
        textColor: 'text-danger-700',
        borderColor: 'border-danger-300',
        iconColor: 'text-danger-600'
      }
    }
  }

  const config = getBadgeConfig()
  const Icon = config.icon

  // Size configurations
  const sizes = {
    sm: { padding: 'px-3 py-1', text: 'text-xs', icon: 'w-3 h-3' },
    md: { padding: 'px-4 py-2', text: 'text-sm', icon: 'w-4 h-4' },
    lg: { padding: 'px-6 py-3', text: 'text-base', icon: 'w-5 h-5' },
    xl: { padding: 'px-8 py-4', text: 'text-lg', icon: 'w-6 h-6' }
  }

  const sizeConfig = sizes[size]

  return (
    <motion.div
      initial={{ scale: 0, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ type: 'spring', delay: 0.3 }}
      whileHover={{ scale: 1.05 }}
      className={`inline-flex items-center space-x-2 ${sizeConfig.padding} ${config.bgColor} ${config.textColor} border-2 ${config.borderColor} rounded-full font-bold font-heading ${sizeConfig.text} shadow-soft`}
    >
      <Icon className={`${sizeConfig.icon} ${config.iconColor}`} />
      <span>{config.text}</span>
    </motion.div>
  )
}

export default RiskBadge
