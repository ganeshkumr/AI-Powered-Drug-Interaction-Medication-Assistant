import { useState, useEffect, useRef } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Activity, 
  User, 
  MessageCircle, 
  Menu, 
  X, 
  LogOut,
  Pill
} from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import { trapFocus, handleAdvancedKeyboardNavigation, announceToScreenReader } from '../../utils/accessibility';

/**
 * GlobalNavigation Component
 * 
 * Provides consistent navigation across all pages with medical-appropriate styling.
 * Features sticky positioning, responsive design, and mobile hamburger menu.
 * 
 * Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 8.1, 8.4, 8.5
 */
const GlobalNavigation = ({ currentPage, user, onChatbotToggle }) => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [showUserDropdown, setShowUserDropdown] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();
  const { logout } = useAuth();
  
  // Refs for focus management
  const mobileMenuRef = useRef(null);
  const userDropdownRef = useRef(null);
  const menuButtonRef = useRef(null);
  const userButtonRef = useRef(null);
  
  // Focus trap cleanup functions
  const mobileMenuCleanup = useRef(null);
  const userDropdownCleanup = useRef(null);

  // Close mobile menu when route changes
  useEffect(() => {
    setIsMenuOpen(false);
    setShowUserDropdown(false);
    
    // Announce route changes to screen readers
    const currentRoute = navigationLinks.find(link => link.active);
    if (currentRoute) {
      announceToScreenReader(`Navigated to ${currentRoute.name} page`);
    }
  }, [location.pathname]);

  // Manage focus trapping for mobile menu
  useEffect(() => {
    if (isMenuOpen && mobileMenuRef.current) {
      mobileMenuCleanup.current = trapFocus(mobileMenuRef.current);
      announceToScreenReader('Mobile menu opened');
    } else if (mobileMenuCleanup.current) {
      mobileMenuCleanup.current();
      mobileMenuCleanup.current = null;
      if (menuButtonRef.current) {
        menuButtonRef.current.focus();
      }
      announceToScreenReader('Mobile menu closed');
    }
    
    return () => {
      if (mobileMenuCleanup.current) {
        mobileMenuCleanup.current();
      }
    };
  }, [isMenuOpen]);

  // Manage focus trapping for user dropdown
  useEffect(() => {
    if (showUserDropdown && userDropdownRef.current) {
      userDropdownCleanup.current = trapFocus(userDropdownRef.current);
      announceToScreenReader('User menu opened');
    } else if (userDropdownCleanup.current) {
      userDropdownCleanup.current();
      userDropdownCleanup.current = null;
      if (userButtonRef.current) {
        userButtonRef.current.focus();
      }
      announceToScreenReader('User menu closed');
    }
    
    return () => {
      if (userDropdownCleanup.current) {
        userDropdownCleanup.current();
      }
    };
  }, [showUserDropdown]);

  // Close dropdowns when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (!event.target.closest('.user-dropdown') && !event.target.closest('.user-button')) {
        setShowUserDropdown(false);
      }
      if (!event.target.closest('.mobile-nav') && !event.target.closest('.mobile-menu-button')) {
        setIsMenuOpen(false);
      }
    };

    document.addEventListener('click', handleClickOutside);
    return () => document.removeEventListener('click', handleClickOutside);
  }, []);

  // Prevent body scroll when mobile menu is open
  useEffect(() => {
    if (isMenuOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = 'unset';
    }
    
    return () => {
      document.body.style.overflow = 'unset';
    };
  }, [isMenuOpen]);

  const handleLogout = async () => {
    await logout();
    navigate('/login');
    setShowUserDropdown(false);
    announceToScreenReader('Logged out successfully');
  };

  const handleMenuToggle = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  const handleUserDropdownToggle = () => {
    setShowUserDropdown(!showUserDropdown);
  };

  const handleKeyboardNavigation = (event, action) => {
    handleAdvancedKeyboardNavigation(event, {
      onEnter: action,
      onSpace: (e) => {
        e.preventDefault();
        action();
      },
      onEscape: () => {
        if (isMenuOpen) setIsMenuOpen(false);
        if (showUserDropdown) setShowUserDropdown(false);
      }
    });
  };

  const navigationLinks = [
    { 
      name: 'Safety Check', 
      path: '/check/medication', 
      icon: Activity,
      active: location.pathname.includes('/check/medication') || 
             location.pathname.includes('/check/dosage') || 
             location.pathname.includes('/check/analysis') ||
             location.pathname.includes('/medication-step') || 
             location.pathname.includes('/dosage-step') || 
             location.pathname.includes('/analysis-step')
    },
    { 
      name: 'My Med', 
      path: '/my-med', 
      icon: Pill,
      active: location.pathname === '/my-med' || location.pathname === '/dashboard'
    },
    { 
      name: 'About', 
      path: '/about', 
      icon: null,
      active: location.pathname === '/about' 
    }
  ];

  return (
    <>
      <motion.nav
        initial={{ y: -64 }}
        animate={{ y: 0 }}
        transition={{ duration: 0.3, ease: 'easeOut' }}
        className="sticky top-0 z-sticky bg-white border-b border-neutral-200 shadow-card medical-section"
        role="navigation"
        aria-label="Main navigation"
      >
        <div className="medical-container">
          <div className="flex justify-between items-center h-navigation-mobile sm:h-navigation px-mobile-x sm:px-0">
            
            {/* Logo Section - Left */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.1 }}
              className="flex items-center space-x-2 sm:space-x-3 cursor-pointer min-w-0 touch-manipulation"
              onClick={() => navigate('/')}
            >
              <div className="w-8 h-8 sm:w-10 sm:h-10 medical-gradient-bg rounded-lg sm:rounded-xl flex items-center justify-center shadow-card flex-shrink-0">
                <Activity className="w-4 h-4 sm:w-6 sm:h-6 text-white" aria-hidden="true" />
              </div>
              <div className="hidden xs:block min-w-0">
                <h1 className="text-responsive-lg sm:text-responsive-xl font-bold medical-gradient-text truncate mb-0">
                  AI-HealthMate
                </h1>
                <p className="text-responsive-xs text-neutral-500 leading-none hidden sm:block mb-0">
                  Medicine Safety Assistant
                </p>
              </div>
            </motion.div>

            {/* Navigation Links - Center (Desktop) */}
            <div className="hidden lg:flex items-center space-x-6 xl:space-x-8">
              {navigationLinks.map((link, index) => (
                <motion.div
                  key={link.name}
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.2 + index * 0.1 }}
                  whileHover={{ y: -2 }}
                  whileTap={{ y: 0 }}
                >
                  <Link
                    to={link.path}
                    className={`
                      flex items-center space-x-2 px-3 py-2 rounded-lg text-sm font-medium
                      transition-all duration-200 hover:bg-primary-50 min-h-touch touch-manipulation
                      medical-nav-item relative
                      ${link.active 
                        ? 'text-primary-600 bg-primary-50 active' 
                        : 'text-neutral-600 hover:text-primary-600'
                      }
                    `}
                    aria-current={link.active ? 'page' : undefined}
                  >
                    {link.icon && (
                      <motion.div
                        whileHover={{ rotate: 5, scale: 1.1 }}
                        transition={{ duration: 0.2 }}
                      >
                        <link.icon className="w-4 h-4" aria-hidden="true" />
                      </motion.div>
                    )}
                    <span>{link.name}</span>
                  </Link>
                </motion.div>
              ))}
            </div>

            {/* User Actions - Right */}
            <div className="flex items-center space-x-2 sm:space-x-3 md:space-x-4">
              
              {/* Chatbot Button */}
              <motion.button
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.3 }}
                whileHover={{ 
                  scale: 1.05,
                  rotate: 5,
                  transition: { duration: 0.2 }
                }}
                whileTap={{ 
                  scale: 0.95,
                  rotate: -5,
                  transition: { duration: 0.1 }
                }}
                onClick={onChatbotToggle}
                onKeyDown={(e) => handleKeyboardNavigation(e, onChatbotToggle)}
                className="p-2 sm:p-2.5 rounded-lg bg-secondary-50 text-secondary-600 hover:bg-secondary-100 transition-all duration-200 min-h-touch min-w-touch touch-manipulation focus:outline-none focus:ring-2 focus:ring-secondary-500 focus:ring-offset-2 medical-hover-glow"
                aria-label="Open AI chatbot for assistance"
                title="Open AI chatbot"
              >
                <MessageCircle className="w-4 h-4 sm:w-5 sm:h-5" aria-hidden="true" />
              </motion.button>

              {/* User Profile Dropdown */}
              {user && (
                <div className="relative">
                  <motion.button
                    ref={userButtonRef}
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.4 }}
                    whileHover={{ 
                      scale: 1.05,
                      transition: { duration: 0.2 }
                    }}
                    whileTap={{ 
                      scale: 0.98,
                      transition: { duration: 0.1 }
                    }}
                    onClick={handleUserDropdownToggle}
                    onKeyDown={(e) => handleKeyboardNavigation(e, handleUserDropdownToggle)}
                    className="user-button flex items-center space-x-1 sm:space-x-2 p-1.5 sm:p-2 rounded-lg hover:bg-neutral-50 transition-all duration-200 min-h-touch touch-manipulation focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 medical-hover-lift"
                    aria-label={`User menu for ${user?.name || 'User'}`}
                    aria-expanded={showUserDropdown}
                    aria-haspopup="true"
                    id="user-menu-button"
                  >
                    <motion.div 
                      className="w-7 h-7 sm:w-8 sm:h-8 bg-gradient-to-br from-primary-500 to-secondary-500 rounded-full flex items-center justify-center text-white text-xs sm:text-sm font-semibold shadow-sm"
                      whileHover={{ 
                        rotate: 360,
                        transition: { duration: 0.5 }
                      }}
                    >
                      {user?.name?.[0]?.toUpperCase() || 'U'}
                    </motion.div>
                    <span className="text-sm font-medium text-neutral-700 hidden md:block max-w-20 lg:max-w-none truncate">
                      {user?.name || 'User'}
                    </span>
                  </motion.button>

                  {/* User Dropdown Menu */}
                  <AnimatePresence>
                    {showUserDropdown && (
                      <motion.div
                        ref={userDropdownRef}
                        initial={{ opacity: 0, y: -10, scale: 0.95 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: -10, scale: 0.95 }}
                        transition={{ duration: 0.2 }}
                        className="user-dropdown absolute right-0 mt-2 w-48 bg-white rounded-lg shadow-lg border border-neutral-200 overflow-hidden z-dropdown"
                        role="menu"
                        aria-orientation="vertical"
                        aria-labelledby="user-menu-button"
                        data-close-modal="true"
                      >
                        <Link
                          to="/profile"
                          className="flex items-center space-x-3 px-4 py-3 hover:bg-neutral-50 transition-colors duration-200 text-neutral-700 min-h-touch"
                          role="menuitem"
                          onClick={() => setShowUserDropdown(false)}
                        >
                          <User className="w-4 h-4 text-primary-500" aria-hidden="true" />
                          <span className="text-sm font-medium">Profile</span>
                        </Link>
                        <div className="border-t border-neutral-100" />
                        <button
                          onClick={handleLogout}
                          className="w-full flex items-center space-x-3 px-4 py-3 hover:bg-danger-50 transition-colors duration-200 text-danger-600 min-h-touch"
                          role="menuitem"
                        >
                          <LogOut className="w-4 h-4" aria-hidden="true" />
                          <span className="text-sm font-medium">Logout</span>
                        </button>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              )}

              {/* Mobile Menu Button */}
              <motion.button
                ref={menuButtonRef}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.5 }}
                whileTap={{ scale: 0.95 }}
                onClick={handleMenuToggle}
                onKeyDown={(e) => handleKeyboardNavigation(e, handleMenuToggle)}
                className="mobile-menu-button lg:hidden p-2 rounded-lg hover:bg-neutral-50 transition-colors duration-200 min-h-touch min-w-touch touch-manipulation focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2"
                aria-label={isMenuOpen ? 'Close mobile menu' : 'Open mobile menu'}
                aria-expanded={isMenuOpen}
                aria-controls="mobile-navigation"
              >
                {isMenuOpen ? (
                  <X className="w-5 h-5 sm:w-6 sm:h-6 text-neutral-600" aria-hidden="true" />
                ) : (
                  <Menu className="w-5 h-5 sm:w-6 sm:h-6 text-neutral-600" aria-hidden="true" />
                )}
              </motion.button>
            </div>
          </div>
        </div>
      </motion.nav>

      {/* Mobile Navigation Overlay */}
      <AnimatePresence>
        {isMenuOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="mobile-nav-overlay lg:hidden"
            onClick={() => setIsMenuOpen(false)}
          />
        )}
      </AnimatePresence>

      {/* Mobile Navigation Menu */}
      <AnimatePresence>
        {isMenuOpen && (
          <motion.div
            ref={mobileMenuRef}
            initial={{ opacity: 0, x: '100%' }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: '100%' }}
            transition={{ duration: 0.3, ease: 'easeInOut' }}
            className="mobile-nav lg:hidden fixed top-14 sm:top-16 right-0 bottom-0 w-64 sm:w-72 bg-white border-l border-neutral-200 shadow-xl z-mobile-nav overflow-y-auto scroll-smooth-mobile"
            id="mobile-navigation"
            role="navigation"
            aria-label="Mobile navigation menu"
            data-close-modal="true"
          >
            <div className="p-4 space-y-2">
              {/* Mobile Navigation Header */}
              <div className="pb-4 border-b border-neutral-100">
                <h2 className="text-lg font-semibold text-neutral-800">Navigation</h2>
              </div>

              {/* Navigation Links */}
              <div className="space-y-1">
                {navigationLinks.map((link) => (
                  <Link
                    key={link.name}
                    to={link.path}
                    className={`
                      flex items-center space-x-3 px-4 py-4 rounded-lg text-base font-medium
                      transition-colors duration-200 min-h-touch touch-manipulation
                      ${link.active 
                        ? 'text-primary-600 bg-primary-50' 
                        : 'text-neutral-600 hover:text-primary-600 hover:bg-neutral-50'
                      }
                    `}
                    aria-current={link.active ? 'page' : undefined}
                    onClick={() => setIsMenuOpen(false)}
                  >
                    {link.icon && (
                      <link.icon className="w-5 h-5" aria-hidden="true" />
                    )}
                    <span>{link.name}</span>
                  </Link>
                ))}
              </div>

              {/* Mobile User Section */}
              {user && (
                <div className="pt-4 border-t border-neutral-100 space-y-1">
                  <div className="px-4 py-2">
                    <p className="text-sm font-medium text-neutral-800">
                      {user?.name || 'User'}
                    </p>
                    <p className="text-xs text-neutral-500">
                      {user?.email || 'user@example.com'}
                    </p>
                  </div>
                  
                  <Link
                    to="/profile"
                    className="flex items-center space-x-3 px-4 py-4 rounded-lg text-base font-medium text-neutral-600 hover:text-primary-600 hover:bg-neutral-50 transition-colors duration-200 min-h-touch touch-manipulation"
                    onClick={() => setIsMenuOpen(false)}
                  >
                    <User className="w-5 h-5" aria-hidden="true" />
                    <span>Profile</span>
                  </Link>
                  
                  <button
                    onClick={() => {
                      handleLogout();
                      setIsMenuOpen(false);
                    }}
                    className="w-full flex items-center space-x-3 px-4 py-4 rounded-lg text-base font-medium text-danger-600 hover:bg-danger-50 transition-colors duration-200 min-h-touch touch-manipulation"
                  >
                    <LogOut className="w-5 h-5" aria-hidden="true" />
                    <span>Logout</span>
                  </button>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
};

export default GlobalNavigation;