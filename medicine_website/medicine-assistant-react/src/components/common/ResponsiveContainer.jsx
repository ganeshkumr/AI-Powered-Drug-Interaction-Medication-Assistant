import React from 'react'
import { motion } from 'framer-motion'

/**
 * ResponsiveContainer Component
 * 
 * A flexible container component that provides consistent responsive behavior
 * across different screen sizes with mobile-first design principles.
 * 
 * Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
 */
const ResponsiveContainer = ({ 
  children, 
  className = '', 
  size = 'default',
  padding = 'responsive',
  animate = false,
  as = 'div',
  ...props 
}) => {
  // Container size variants
  const sizeClasses = {
    'full': 'w-full',
    'default': 'w-full max-w-7xl mx-auto',
    'narrow': 'w-full max-w-4xl mx-auto',
    'wide': 'w-full max-w-screen-2xl mx-auto',
    'mobile-full': 'w-full max-w-screen-mobile mobile:max-w-full',
  }

  // Padding variants with mobile-first approach
  const paddingClasses = {
    'none': '',
    'responsive': 'px-mobile-x sm:px-5 md:px-desktop-x lg:px-8 xl:px-10',
    'mobile': 'px-mobile-x py-mobile-y',
    'desktop': 'px-desktop-x py-desktop-y',
    'safe': 'px-mobile-x py-mobile-y safe-area-inset-top safe-area-inset-bottom',
    'compact': 'px-4 py-3 sm:px-6 sm:py-4',
    'comfortable': 'px-mobile-x py-6 sm:px-6 sm:py-8 lg:px-8 lg:py-10',
  }

  const containerClasses = `
    ${sizeClasses[size] || sizeClasses.default}
    ${paddingClasses[padding] || paddingClasses.responsive}
    ${className}
  `.trim()

  const Component = as

  if (animate) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, ease: 'easeOut' }}
        className={containerClasses}
        {...props}
      >
        {children}
      </motion.div>
    )
  }

  return (
    <Component className={containerClasses} {...props}>
      {children}
    </Component>
  )
}

/**
 * ResponsiveGrid Component
 * 
 * A responsive grid component that adapts column count based on screen size
 */
export const ResponsiveGrid = ({ 
  children, 
  columns = { mobile: 1, tablet: 2, desktop: 3 },
  gap = 'responsive',
  className = '',
  ...props 
}) => {
  const gapClasses = {
    'none': 'gap-0',
    'small': 'gap-3 sm:gap-4',
    'responsive': 'gap-4 sm:gap-5 md:gap-6',
    'large': 'gap-6 sm:gap-8',
  }

  const gridClasses = `
    grid
    grid-cols-${columns.mobile}
    sm:grid-cols-${columns.tablet || columns.mobile}
    lg:grid-cols-${columns.desktop || columns.tablet || columns.mobile}
    ${gapClasses[gap] || gapClasses.responsive}
    ${className}
  `.trim()

  return (
    <div className={gridClasses} {...props}>
      {children}
    </div>
  )
}

/**
 * ResponsiveFlex Component
 * 
 * A responsive flex component that changes direction based on screen size
 */
export const ResponsiveFlex = ({ 
  children, 
  direction = { mobile: 'col', desktop: 'row' },
  align = 'start',
  justify = 'start',
  gap = 'responsive',
  wrap = false,
  className = '',
  ...props 
}) => {
  const gapClasses = {
    'none': 'gap-0',
    'small': 'gap-3 sm:gap-4',
    'responsive': 'gap-4 sm:gap-6',
    'large': 'gap-6 sm:gap-8',
  }

  const directionClasses = {
    'col': 'flex-col',
    'row': 'flex-row',
  }

  const alignClasses = {
    'start': 'items-start',
    'center': 'items-center',
    'end': 'items-end',
    'stretch': 'items-stretch',
  }

  const justifyClasses = {
    'start': 'justify-start',
    'center': 'justify-center',
    'end': 'justify-end',
    'between': 'justify-between',
    'around': 'justify-around',
    'evenly': 'justify-evenly',
  }

  const flexClasses = `
    flex
    ${directionClasses[direction.mobile] || 'flex-col'}
    ${direction.desktop ? `lg:${directionClasses[direction.desktop]}` : ''}
    ${alignClasses[align]}
    ${justifyClasses[justify]}
    ${gapClasses[gap] || gapClasses.responsive}
    ${wrap ? 'flex-wrap' : ''}
    ${className}
  `.trim()

  return (
    <div className={flexClasses} {...props}>
      {children}
    </div>
  )
}

/**
 * ResponsiveText Component
 * 
 * A text component that scales appropriately across screen sizes
 */
export const ResponsiveText = ({ 
  children, 
  size = 'base',
  weight = 'normal',
  color = 'primary',
  align = { mobile: 'center', desktop: 'left' },
  className = '',
  as = 'p',
  ...props 
}) => {
  const sizeClasses = {
    'xs': 'text-xs sm:text-sm',
    'sm': 'text-sm sm:text-base',
    'base': 'text-base sm:text-lg',
    'lg': 'text-lg sm:text-xl',
    'xl': 'text-xl sm:text-2xl',
    '2xl': 'text-2xl sm:text-3xl',
    '3xl': 'text-3xl sm:text-4xl',
    '4xl': 'text-4xl sm:text-5xl',
  }

  const weightClasses = {
    'light': 'font-light',
    'normal': 'font-normal',
    'medium': 'font-medium',
    'semibold': 'font-semibold',
    'bold': 'font-bold',
  }

  const colorClasses = {
    'primary': 'text-neutral-900',
    'secondary': 'text-neutral-600',
    'tertiary': 'text-neutral-500',
    'inverse': 'text-white',
    'success': 'text-success-600',
    'warning': 'text-warning-600',
    'danger': 'text-danger-600',
  }

  const alignClasses = {
    'left': 'text-left',
    'center': 'text-center',
    'right': 'text-right',
  }

  const textClasses = `
    ${sizeClasses[size] || sizeClasses.base}
    ${weightClasses[weight]}
    ${colorClasses[color]}
    ${alignClasses[align.mobile] || 'text-center'}
    ${align.desktop ? `lg:${alignClasses[align.desktop]}` : ''}
    ${className}
  `.trim()

  const Component = as

  return (
    <Component className={textClasses} {...props}>
      {children}
    </Component>
  )
}

/**
 * ResponsiveButton Component
 * 
 * A button component optimized for touch interactions across devices
 */
export const ResponsiveButton = ({ 
  children, 
  size = 'default',
  variant = 'primary',
  fullWidth = { mobile: true, desktop: false },
  className = '',
  ...props 
}) => {
  const sizeClasses = {
    'small': 'min-h-touch px-4 py-2 text-sm',
    'default': 'min-h-touch px-6 py-3 text-base',
    'large': 'min-h-touch-comfortable px-8 py-4 text-lg',
  }

  const variantClasses = {
    'primary': 'bg-primary-500 hover:bg-primary-600 text-white',
    'secondary': 'bg-white hover:bg-neutral-50 text-primary-600 border border-primary-500',
    'outline': 'bg-transparent hover:bg-primary-50 text-primary-600 border border-primary-300',
  }

  const widthClasses = `
    ${fullWidth.mobile ? 'w-full' : 'w-auto'}
    ${fullWidth.desktop === false ? 'lg:w-auto' : ''}
    ${fullWidth.desktop === true ? 'lg:w-full' : ''}
  `

  const buttonClasses = `
    inline-flex items-center justify-center
    ${sizeClasses[size]}
    ${variantClasses[variant]}
    ${widthClasses}
    rounded-lg font-medium
    transition-all duration-200
    focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2
    disabled:opacity-50 disabled:cursor-not-allowed
    touch-manipulation
    ${className}
  `.trim()

  return (
    <button className={buttonClasses} {...props}>
      {children}
    </button>
  )
}

/**
 * ResponsiveCard Component
 * 
 * A card component that adapts its padding and styling based on screen size
 */
export const ResponsiveCard = ({ 
  children, 
  padding = 'responsive',
  shadow = 'default',
  hover = false,
  className = '',
  ...props 
}) => {
  const paddingClasses = {
    'none': 'p-0',
    'small': 'p-3 sm:p-4',
    'responsive': 'p-4 sm:p-6',
    'large': 'p-6 sm:p-8',
  }

  const shadowClasses = {
    'none': 'shadow-none',
    'small': 'shadow-sm',
    'default': 'shadow-card',
    'large': 'shadow-lg',
  }

  const cardClasses = `
    bg-white
    border border-neutral-200
    rounded-lg sm:rounded-xl
    ${paddingClasses[padding]}
    ${shadowClasses[shadow]}
    ${hover ? 'hover:shadow-card-hover transition-shadow duration-200' : ''}
    ${className}
  `.trim()

  return (
    <div className={cardClasses} {...props}>
      {children}
    </div>
  )
}

export default ResponsiveContainer