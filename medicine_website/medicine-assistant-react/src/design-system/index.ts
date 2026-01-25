/**
 * Design System Index
 * Centralized exports for the medical-focused design system
 */

// Theme configuration
export * from './theme';
export * from './types';

// CSS variables are imported in the main CSS file
export { default as cssVariables } from './css-variables.css';

// Re-export theme object as default
export { 
  colors, 
  typography, 
  spacing, 
  breakpoints, 
  borderRadius, 
  boxShadow, 
  zIndex, 
  animation 
} from './theme';