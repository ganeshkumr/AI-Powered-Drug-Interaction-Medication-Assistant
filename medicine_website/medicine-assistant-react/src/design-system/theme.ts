/**
 * Design System Theme Configuration
 * Medical-Grade Theme for the Drug Interaction & Safety Application
 * 
 * This theme implements medical-grade styling with:
 * - Soft blue/green color scheme with high contrast
 * - Clean typography appropriate for healthcare
 * - Consistent spacing and layout patterns
 * - Accessibility-compliant design tokens
 */

// Medical-Grade Color Palette
export const colors = {
  // Primary Medical Colors (Soft Blue)
  primary: {
    50: '#E6F7F5',
    100: '#CCF0EB', 
    200: '#99E0D7',
    300: '#66D1C3',
    400: '#33C1AF',
    500: '#0EA5E9', // Primary brand color
    600: '#0284C7',
    700: '#0369A1',
    800: '#075985',
    900: '#0C4A6E',
  },
  
  // Secondary Medical Colors (Soft Green/Teal)
  secondary: {
    50: '#F0FDFA',
    100: '#CCFBF1',
    200: '#99F6E4',
    300: '#5EEAD4',
    400: '#2DD4BF',
    500: '#14B8A6', // Secondary brand color
    600: '#0D9488',
    700: '#0F766E',
    800: '#115E59',
    900: '#134E4A',
  },

  // Success/Safe Colors (Medical Green)
  success: {
    50: '#F0FDF4',
    100: '#DCFCE7',
    200: '#BBF7D0',
    300: '#86EFAC',
    400: '#4ADE80',
    500: '#10B981', // Safe medication indicator
    600: '#059669',
    700: '#047857',
    800: '#065F46',
    900: '#064E3B',
  },

  // Warning/Caution Colors (Medical Amber)
  warning: {
    50: '#FFFBEB',
    100: '#FEF3C7',
    200: '#FDE68A',
    300: '#FCD34D',
    400: '#FBBF24',
    500: '#F59E0B', // Caution indicator
    600: '#D97706',
    700: '#B45309',
    800: '#92400E',
    900: '#78350F',
  },

  // Danger/High Risk Colors (Medical Red)
  danger: {
    50: '#FEF2F2',
    100: '#FEE2E2',
    200: '#FECACA',
    300: '#FCA5A5',
    400: '#F87171',
    500: '#EF4444', // High risk indicator
    600: '#DC2626',
    700: '#B91C1C',
    800: '#991B1B',
    900: '#7F1D1D',
  },

  // Neutral Colors (Medical Gray Scale)
  neutral: {
    50: '#F8FAFC',  // Lightest background
    100: '#F1F5F9', // Light background
    200: '#E2E8F0', // Border light
    300: '#CBD5E1', // Border medium
    400: '#94A3B8', // Border dark
    500: '#64748B', // Text tertiary
    600: '#475569', // Text secondary
    700: '#334155', // Text primary dark
    800: '#1E293B', // Text primary darker
    900: '#0F172A', // Text primary darkest
  },

  // Background Colors
  background: {
    primary: '#FFFFFF',   // Pure white for cards and forms
    secondary: '#F8FAFC', // Light gray for page backgrounds
    tertiary: '#F1F5F9',  // Slightly darker for sections
  },

  // Text Colors (High Contrast for Medical Readability)
  text: {
    primary: '#0F172A',   // Darkest for headings and important text
    secondary: '#475569', // Medium for body text
    tertiary: '#64748B',  // Light for captions and labels
    inverse: '#FFFFFF',   // White for dark backgrounds
  },

  // Border Colors
  border: {
    light: '#E2E8F0',   // Subtle borders
    medium: '#CBD5E1',  // Standard borders
    dark: '#94A3B8',    // Emphasized borders
  },
} as const;

// Medical-Grade Typography Scale
export const typography = {
  fontFamily: {
    sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'Helvetica Neue', 'Arial', 'sans-serif'],
    heading: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'Helvetica Neue', 'Arial', 'sans-serif'],
    mono: ['SF Mono', 'Monaco', 'Cascadia Code', 'Roboto Mono', 'Consolas', 'Courier New', 'monospace'],
  },
  
  fontSize: {
    xs: ['12px', { lineHeight: '16px' }],
    sm: ['14px', { lineHeight: '20px' }],
    base: ['16px', { lineHeight: '24px' }],
    lg: ['18px', { lineHeight: '28px' }],
    xl: ['20px', { lineHeight: '28px' }],
    '2xl': ['24px', { lineHeight: '32px' }],
    '3xl': ['30px', { lineHeight: '36px' }],
    '4xl': ['36px', { lineHeight: '40px' }],
    '5xl': ['48px', { lineHeight: '56px' }],
  },
  
  fontWeight: {
    light: '300',
    normal: '400',
    medium: '500',
    semibold: '600',
    bold: '700',
    extrabold: '800',
  },
} as const;

// Medical-Grade Spacing System (8px base unit for consistency)
export const spacing = {
  0: '0px',
  1: '4px',   // 0.5 * base
  2: '8px',   // 1 * base
  3: '12px',  // 1.5 * base
  4: '16px',  // 2 * base
  5: '20px',  // 2.5 * base
  6: '24px',  // 3 * base
  8: '32px',  // 4 * base
  10: '40px', // 5 * base
  12: '48px', // 6 * base
  16: '64px', // 8 * base
  20: '80px', // 10 * base
  24: '96px', // 12 * base
  32: '128px', // 16 * base
} as const;

// Responsive Breakpoints
export const breakpoints = {
  sm: '640px',   // Mobile landscape
  md: '768px',   // Tablet portrait
  lg: '1024px',  // Tablet landscape / Small desktop
  xl: '1280px',  // Desktop
  '2xl': '1536px', // Large desktop
} as const;

// Medical-Grade Border Radius
export const borderRadius = {
  none: '0px',
  sm: '4px',     // Small elements
  base: '8px',   // Standard elements
  md: '12px',    // Cards and containers
  lg: '16px',    // Large containers
  xl: '24px',    // Hero sections
  full: '9999px', // Pills and avatars
} as const;

// Medical-Grade Box Shadows (Subtle and Professional)
export const boxShadow = {
  xs: '0 1px 2px 0 rgba(0, 0, 0, 0.03)',
  sm: '0 1px 3px 0 rgba(0, 0, 0, 0.05)',
  base: '0 2px 8px 0 rgba(0, 0, 0, 0.06)',
  md: '0 4px 16px 0 rgba(0, 0, 0, 0.08)',
  lg: '0 8px 32px 0 rgba(0, 0, 0, 0.10)',
  xl: '0 16px 64px 0 rgba(0, 0, 0, 0.12)',
  '2xl': '0 24px 96px 0 rgba(0, 0, 0, 0.15)',
  
  // Medical-specific shadows
  card: '0 2px 8px 0 rgba(15, 23, 42, 0.06)',
  cardHover: '0 4px 16px 0 rgba(15, 23, 42, 0.08)',
  modal: '0 20px 80px 0 rgba(15, 23, 42, 0.15)',
  dropdown: '0 8px 32px 0 rgba(15, 23, 42, 0.12)',
} as const;

// Z-Index Scale for Layering
export const zIndex = {
  hide: -1,
  auto: 'auto',
  base: 0,
  docked: 10,
  dropdown: 1000,
  sticky: 1100,
  banner: 1200,
  overlay: 1300,
  modal: 1400,
  popover: 1500,
  skipLink: 1600,
  toast: 1700,
  tooltip: 1800,
} as const;

// Medical-Grade Animation System
export const animation = {
  duration: {
    fast: '150ms',    // Quick interactions
    base: '200ms',    // Standard transitions
    slow: '300ms',    // Smooth animations
    slower: '500ms',  // Emphasis animations
  },
  
  easing: {
    linear: 'linear',
    in: 'cubic-bezier(0.4, 0, 1, 1)',
    out: 'cubic-bezier(0, 0, 0.2, 1)',
    inOut: 'cubic-bezier(0.4, 0, 0.2, 1)',
  },
} as const;

// Medical-Grade Component Tokens
export const components = {
  navigation: {
    height: '64px',
    heightMobile: '56px',
    zIndex: zIndex.sticky,
  },
  
  stepIndicator: {
    height: '80px',
    heightMobile: '120px',
  },
  
  card: {
    padding: spacing[4],
    borderRadius: borderRadius.md,
    shadow: boxShadow.card,
    shadowHover: boxShadow.cardHover,
  },
  
  button: {
    height: '44px',
    borderRadius: borderRadius.base,
    fontWeight: typography.fontWeight.medium,
  },
  
  input: {
    height: '44px',
    borderRadius: borderRadius.base,
    borderWidth: '2px',
  },
  
  modal: {
    borderRadius: borderRadius.lg,
    shadow: boxShadow.modal,
    zIndex: zIndex.modal,
  },
} as const;

// Medical Risk Level Mappings
export const riskLevels = {
  safe: {
    color: colors.success[600],
    background: colors.success[50],
    border: colors.success[200],
    label: 'Safe',
  },
  warning: {
    color: colors.warning[700],
    background: colors.warning[50],
    border: colors.warning[200],
    label: 'Caution',
  },
  danger: {
    color: colors.danger[700],
    background: colors.danger[50],
    border: colors.danger[200],
    label: 'High Risk',
  },
} as const;

// Accessibility Tokens
export const accessibility = {
  focusRing: {
    width: '2px',
    color: colors.primary[500],
    offset: '2px',
  },
  
  touchTarget: {
    min: '44px',
    comfortable: '48px',
  },
  
  contrast: {
    aa: 4.5,    // WCAG AA standard
    aaa: 7,     // WCAG AAA standard
  },
} as const;