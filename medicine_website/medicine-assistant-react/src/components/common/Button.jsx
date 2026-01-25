import { motion } from 'framer-motion'
import { Loader } from 'lucide-react'

/**
 * Medical-Grade Button Component
 * 
 * Implements medical-appropriate styling with:
 * - High contrast colors for accessibility
 * - Consistent spacing and typography
 * - Touch-friendly minimum sizes
 * - Professional hover and focus states
 * - WCAG compliant focus indicators
 * - Proper ARIA attributes
 */
const Button = ({ 
  children, 
  variant = 'primary', 
  size = 'md', 
  icon, 
  loading = false, 
  disabled = false,
  className = '',
  'aria-label': ariaLabel,
  'aria-describedby': ariaDescribedby,
  'aria-expanded': ariaExpanded,
  'aria-pressed': ariaPressed,
  type = 'button',
  ...props 
}) => {
  const baseStyles = `
    inline-flex items-center justify-center font-medium rounded-lg 
    transition-all duration-base focus:outline-none focus:ring-2 focus:ring-offset-2 
    disabled:opacity-60 disabled:cursor-not-allowed disabled:transform-none
    medical-focus-ring touch-target relative
  `
  
  const variants = {
    primary: `
      bg-primary-500 text-white border-2 border-primary-500
      hover:bg-primary-600 hover:border-primary-600 hover:shadow-md hover:-translate-y-0.5
      focus:ring-primary-500 active:translate-y-0 active:shadow-sm
      shadow-sm medical-button-primary
    `,
    secondary: `
      bg-white text-primary-600 border-2 border-primary-500
      hover:bg-primary-50 hover:border-primary-600 hover:shadow-md hover:-translate-y-0.5
      focus:ring-primary-500 active:translate-y-0 active:shadow-sm
      shadow-sm medical-button-secondary
    `,
    success: `
      bg-success-500 text-white border-2 border-success-500
      hover:bg-success-600 hover:border-success-600 hover:shadow-md hover:-translate-y-0.5
      focus:ring-success-500 active:translate-y-0 active:shadow-sm
      shadow-sm medical-button-primary
    `,
    warning: `
      bg-warning-500 text-white border-2 border-warning-500
      hover:bg-warning-600 hover:border-warning-600 hover:shadow-md hover:-translate-y-0.5
      focus:ring-warning-500 active:translate-y-0 active:shadow-sm
      shadow-sm medical-button-primary
    `,
    danger: `
      bg-danger-500 text-white border-2 border-danger-500
      hover:bg-danger-600 hover:border-danger-600 hover:shadow-md hover:-translate-y-0.5
      focus:ring-danger-500 active:translate-y-0 active:shadow-sm
      shadow-sm medical-button-primary
    `,
    ghost: `
      bg-transparent text-primary-600 border-2 border-transparent
      hover:bg-primary-50 hover:border-primary-200 hover:shadow-sm
      focus:ring-primary-500 focus:border-primary-300
      medical-hover-scale
    `,
    outline: `
      bg-transparent text-neutral-700 border-2 border-neutral-300
      hover:bg-neutral-50 hover:border-neutral-400 hover:shadow-sm
      focus:ring-neutral-500 focus:border-neutral-400
      medical-hover-scale
    `,
  }
  
  const sizes = {
    sm: 'px-3 py-2 text-sm min-h-[36px] gap-1.5',
    md: 'px-4 py-2.5 text-base min-h-[44px] gap-2',
    lg: 'px-6 py-3 text-lg min-h-[52px] gap-2.5',
    xl: 'px-8 py-4 text-xl min-h-[60px] gap-3',
  }

  // Enhanced accessibility props
  const accessibilityProps = {
    type,
    'aria-label': ariaLabel,
    'aria-describedby': ariaDescribedby,
    'aria-expanded': ariaExpanded,
    'aria-pressed': ariaPressed,
    'aria-busy': loading,
    'aria-disabled': disabled || loading,
    role: props.role || 'button',
  };

  // Remove undefined props
  Object.keys(accessibilityProps).forEach(key => {
    if (accessibilityProps[key] === undefined) {
      delete accessibilityProps[key];
    }
  });
  
  return (
    <motion.button
      whileHover={{ 
        scale: disabled || loading ? 1 : 1.02,
        y: disabled || loading ? 0 : -2,
        transition: { duration: 0.15, ease: [0, 0, 0.2, 1] }
      }}
      whileTap={{ 
        scale: disabled || loading ? 1 : 0.98,
        y: disabled || loading ? 0 : 0,
        transition: { duration: 0.1, ease: [0, 0, 0.2, 1] }
      }}
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.2, ease: [0, 0, 0.2, 1] }}
      className={`${baseStyles} ${variants[variant]} ${sizes[size]} ${className}`}
      disabled={disabled || loading}
      {...accessibilityProps}
      {...props}
    >
      {loading ? (
        <>
          <Loader className="w-4 h-4 animate-spin" aria-hidden="true" />
          <span>Loading...</span>
          <span className="sr-only">Please wait, processing your request</span>
        </>
      ) : (
        <>
          {icon && (
            <span className="flex-shrink-0" aria-hidden="true">
              {icon}
            </span>
          )}
          <span className="font-medium">{children}</span>
        </>
      )}
    </motion.button>
  )
}

export default Button
