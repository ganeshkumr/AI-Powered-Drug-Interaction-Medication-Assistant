import '@testing-library/jest-dom'

// Mock framer-motion to avoid animation issues in tests
vi.mock('framer-motion', () => ({
  motion: {
    div: ({ children, whileHover, whileTap, initial, animate, exit, transition, ...props }) => <div {...props}>{children}</div>,
    button: ({ children, whileHover, whileTap, initial, animate, exit, transition, ...props }) => <button {...props}>{children}</button>,
    nav: ({ children, whileHover, whileTap, initial, animate, exit, transition, ...props }) => <nav {...props}>{children}</nav>,
  },
  AnimatePresence: ({ children }) => children,
}))

// Mock lucide-react icons
vi.mock('lucide-react', () => ({
  Pill: () => <div data-testid="pill-icon" />,
  X: () => <div data-testid="x-icon" />,
  Edit3: () => <div data-testid="edit-icon" />,
  Clock: () => <div data-testid="clock-icon" />,
  Calendar: () => <div data-testid="calendar-icon" />,
  Shield: () => <div data-testid="shield-icon" />,
  AlertTriangle: () => <div data-testid="alert-triangle-icon" />,
  XCircle: () => <div data-testid="x-circle-icon" />,
  ChevronLeft: () => <div data-testid="chevron-left-icon" />,
  ChevronRight: () => <div data-testid="chevron-right-icon" />,
  Check: () => <div data-testid="check-icon" />,
  Clipboard: () => <div data-testid="clipboard-icon" />,
  Loader: () => <div data-testid="loader-icon" />,
  Plus: () => <div data-testid="plus-icon" />,
  ArrowRight: () => <div data-testid="arrow-right-icon" />,
  ArrowLeft: () => <div data-testid="arrow-left-icon" />,
  Save: () => <div data-testid="save-icon" />,
  MessageCircle: () => <div data-testid="message-circle-icon" />,
  CheckCircle: () => <div data-testid="check-circle-icon" />,
  AlertCircle: () => <div data-testid="alert-circle-icon" />,
  Activity: () => <div data-testid="activity-icon" />,
  Search: () => <div data-testid="search-icon" />,
  Loader2: () => <div data-testid="loader2-icon" />,
  User: () => <div data-testid="user-icon" />,
  Menu: () => <div data-testid="menu-icon" />,
  LogOut: () => <div data-testid="logout-icon" />,
  Home: () => <div data-testid="home-icon" />,
  Info: () => <div data-testid="info-icon" />,
}))

// Mock react-router-dom
vi.mock('react-router-dom', () => ({
  Link: ({ children, to, className, ...props }) => (
    <a href={to} className={className} {...props}>
      {children}
    </a>
  ),
  useLocation: () => ({
    pathname: '/',
  }),
  useNavigate: () => vi.fn(),
}))

// Mock AuthContext
vi.mock('../context/AuthContext', () => ({
  useAuth: () => ({
    logout: vi.fn(),
  }),
}))

// Mock accessibility utils
vi.mock('../utils/accessibility', () => ({
  trapFocus: vi.fn(() => vi.fn()),
  handleKeyboardNavigation: vi.fn(),
  announceToScreenReader: vi.fn(),
  manageFocusTransition: vi.fn(),
  generateId: vi.fn((prefix = 'element') => `${prefix}-test-id`),
  validateColorContrast: vi.fn(() => true),
  validateTouchTargetSize: vi.fn(() => true),
  manageLoadingAnnouncements: vi.fn(),
  createAccessibleFormField: vi.fn((config) => ({
    fieldProps: {
      id: config.id || 'test-field',
      'aria-label': config.label,
      'aria-describedby': config.description ? `${config.id}-desc` : undefined,
      'aria-required': config.required || false,
      'aria-invalid': config.invalid || false,
    },
    labelProps: {
      htmlFor: config.id || 'test-field',
    },
  })),
}))