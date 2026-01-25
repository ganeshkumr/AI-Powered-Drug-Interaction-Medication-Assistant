/**
 * TypeScript interfaces for theme configuration
 */

// Color palette type definitions
export interface ColorScale {
  50: string;
  100: string;
  200: string;
  300: string;
  400: string;
  500: string;
  600: string;
  700: string;
  800: string;
  900: string;
}

export interface BackgroundColors {
  primary: string;
  secondary: string;
  tertiary: string;
}

export interface TextColors {
  primary: string;
  secondary: string;
  tertiary: string;
  inverse: string;
}

export interface BorderColors {
  light: string;
  medium: string;
  dark: string;
}

export interface Colors {
  primary: ColorScale;
  secondary: ColorScale;
  success: ColorScale;
  warning: ColorScale;
  danger: ColorScale;
  neutral: ColorScale;
  background: BackgroundColors;
  text: TextColors;
  border: BorderColors;
}

// Typography type definitions
export interface FontFamily {
  sans: string[];
  heading: string[];
}

export interface FontSize {
  xs: [string, { lineHeight: string }];
  sm: [string, { lineHeight: string }];
  base: [string, { lineHeight: string }];
  lg: [string, { lineHeight: string }];
  xl: [string, { lineHeight: string }];
  '2xl': [string, { lineHeight: string }];
  '3xl': [string, { lineHeight: string }];
  '4xl': [string, { lineHeight: string }];
}

export interface FontWeight {
  normal: string;
  medium: string;
  semibold: string;
  bold: string;
}

export interface Typography {
  fontFamily: FontFamily;
  fontSize: FontSize;
  fontWeight: FontWeight;
}

// Spacing and layout type definitions
export interface Spacing {
  0: string;
  1: string;
  2: string;
  3: string;
  4: string;
  5: string;
  6: string;
  8: string;
  10: string;
  12: string;
  16: string;
  20: string;
  24: string;
  32: string;
}

export interface Breakpoints {
  sm: string;
  md: string;
  lg: string;
  xl: string;
  '2xl': string;
}

export interface BorderRadius {
  none: string;
  sm: string;
  base: string;
  md: string;
  lg: string;
  xl: string;
  full: string;
}

export interface BoxShadow {
  sm: string;
  base: string;
  md: string;
  lg: string;
  xl: string;
}

export interface ZIndex {
  hide: number;
  auto: string;
  base: number;
  docked: number;
  dropdown: number;
  sticky: number;
  banner: number;
  overlay: number;
  modal: number;
  popover: number;
  skipLink: number;
  toast: number;
  tooltip: number;
}

export interface AnimationDuration {
  fast: string;
  base: string;
  slow: string;
  slower: string;
}

export interface AnimationEasing {
  linear: string;
  in: string;
  out: string;
  inOut: string;
}

export interface Animation {
  duration: AnimationDuration;
  easing: AnimationEasing;
}

// Complete theme interface
export interface Theme {
  colors: Colors;
  typography: Typography;
  spacing: Spacing;
  breakpoints: Breakpoints;
  borderRadius: BorderRadius;
  boxShadow: BoxShadow;
  zIndex: ZIndex;
  animation: Animation;
}

// Component variant types
export type ComponentSize = 'sm' | 'base' | 'lg' | 'xl';
export type ComponentVariant = 'primary' | 'secondary' | 'success' | 'warning' | 'danger';
export type RiskLevel = 'safe' | 'caution' | 'high-risk';

// UI state types
export interface StepState {
  currentStep: 1 | 2 | 3;
  completedSteps: number[];
  canProceed: boolean;
  validationErrors: string[];
}

export interface UIPreferences {
  theme: 'light';
  reducedMotion: boolean;
  fontSize: 'normal' | 'large';
}

export interface NavigationState {
  isMenuOpen: boolean;
  isChatbotOpen: boolean;
  currentRoute: string;
}

// Component prop interfaces
export interface BaseComponentProps {
  className?: string;
  children?: React.ReactNode;
  'data-testid'?: string;
}

export interface NavigationProps extends BaseComponentProps {
  currentPage: string;
  user?: any;
  onChatbotToggle: () => void;
}

export interface StepIndicatorProps extends BaseComponentProps {
  currentStep: 1 | 2 | 3;
  completedSteps: number[];
  onStepClick?: (step: number) => void;
}

export interface MedicationCardProps extends BaseComponentProps {
  medication: {
    name: string;
    dosage?: string;
    frequency?: string;
    timeOfDay?: 'morning' | 'afternoon' | 'night';
  };
  variant: 'selection' | 'dashboard' | 'analysis';
  onRemove?: () => void;
  onEdit?: () => void;
}

export interface RiskBadgeProps extends BaseComponentProps {
  riskLevel: RiskLevel;
  size: 'small' | 'large';
}