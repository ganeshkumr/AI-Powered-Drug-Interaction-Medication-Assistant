import { motion } from 'framer-motion'
import { useEffect, useState } from 'react'

const RiskGauge = ({ risk, size = 'lg' }) => {
  const [animatedRisk, setAnimatedRisk] = useState(0)

  useEffect(() => {
    // Animate the risk value
    const timer = setTimeout(() => {
      setAnimatedRisk(risk)
    }, 100)
    return () => clearTimeout(timer)
  }, [risk])

  // Determine color based on risk level
  const getColor = () => {
    if (risk < 30) return { stroke: '#16A34A', bg: '#DCFCE7', text: '#15803D' } // Green
    if (risk < 70) return { stroke: '#F59E0B', bg: '#FEF3C7', text: '#D97706' } // Yellow
    return { stroke: '#EF4444', bg: '#FEE2E2', text: '#DC2626' } // Red
  }

  const colors = getColor()

  // Size configurations
  const sizes = {
    sm: { radius: 40, strokeWidth: 6, fontSize: 'text-xl' },
    md: { radius: 60, strokeWidth: 8, fontSize: 'text-3xl' },
    lg: { radius: 80, strokeWidth: 10, fontSize: 'text-4xl' },
    xl: { radius: 100, strokeWidth: 12, fontSize: 'text-5xl' }
  }

  const config = sizes[size]
  const circumference = 2 * Math.PI * config.radius
  const offset = circumference - (animatedRisk / 100) * circumference

  return (
    <div className="flex items-center justify-center">
      <motion.div
        initial={{ scale: 0, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ type: 'spring', duration: 0.8 }}
        className="relative"
        style={{ width: config.radius * 2 + 40, height: config.radius * 2 + 40 }}
      >
        {/* Background circle */}
        <svg
          className="transform -rotate-90"
          width={config.radius * 2 + 40}
          height={config.radius * 2 + 40}
        >
          {/* Background track */}
          <circle
            cx={(config.radius * 2 + 40) / 2}
            cy={(config.radius * 2 + 40) / 2}
            r={config.radius}
            stroke="#E5E7EB"
            strokeWidth={config.strokeWidth}
            fill="none"
          />
          
          {/* Animated progress circle */}
          <motion.circle
            cx={(config.radius * 2 + 40) / 2}
            cy={(config.radius * 2 + 40) / 2}
            r={config.radius}
            stroke={colors.stroke}
            strokeWidth={config.strokeWidth}
            fill="none"
            strokeLinecap="round"
            strokeDasharray={circumference}
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset: offset }}
            transition={{ duration: 1.5, ease: 'easeOut' }}
          />
        </svg>

        {/* Center content */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.5, type: 'spring' }}
            className="text-center"
          >
            <div className={`${config.fontSize} font-bold font-heading`} style={{ color: colors.text }}>
              {Math.round(animatedRisk)}%
            </div>
            <div className="text-xs text-gray-500 font-medium mt-1">
              Risk Score
            </div>
          </motion.div>
        </div>

        {/* Pulse animation for high risk */}
        {risk >= 70 && (
          <motion.div
            className="absolute inset-0 rounded-full"
            style={{ backgroundColor: colors.stroke }}
            animate={{
              scale: [1, 1.1, 1],
              opacity: [0.3, 0, 0.3]
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              ease: 'easeInOut'
            }}
          />
        )}
      </motion.div>
    </div>
  )
}

export default RiskGauge
