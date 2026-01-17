/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#2EA79B',
          50: '#E6F7F5',
          100: '#CCF0EB',
          200: '#99E0D7',
          300: '#66D1C3',
          400: '#33C1AF',
          500: '#2EA79B',
          600: '#25867C',
          700: '#1C645D',
          800: '#13433E',
          900: '#0A211F',
        },
        accent: {
          DEFAULT: '#F4B400',
          50: '#FEF9E6',
          100: '#FDF3CC',
          200: '#FBE799',
          300: '#F9DB66',
          400: '#F7CF33',
          500: '#F4B400',
          600: '#C39000',
          700: '#926C00',
          800: '#614800',
          900: '#302400',
        },
        neutral: {
          bg: '#F8FAFB',
          text: '#0F172A',
        },
        status: {
          safe: '#16A34A',
          warning: '#F59E0B',
          danger: '#EF4444',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        heading: ['Poppins', 'Inter', 'system-ui', 'sans-serif'],
      },
      fontSize: {
        body: ['16px', '1.5'],
        'body-lg': ['18px', '1.6'],
      },
      borderRadius: {
        card: '12px',
        'card-lg': '16px',
      },
      boxShadow: {
        soft: '0 2px 8px rgba(0, 0, 0, 0.08)',
        'soft-lg': '0 4px 16px rgba(0, 0, 0, 0.1)',
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-in-out',
        'slide-in': 'slideIn 0.3s ease-out',
        'slide-up': 'slideUp 0.4s ease-out',
        'bounce-slow': 'bounce 2s infinite',
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'pill-drop': 'pillDrop 0.6s ease-out',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideIn: {
          '0%': { transform: 'translateX(100%)', opacity: '0' },
          '100%': { transform: 'translateX(0)', opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(20px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        pillDrop: {
          '0%': { transform: 'translateY(-20px) scale(0.8)', opacity: '0' },
          '50%': { transform: 'translateY(5px) scale(1.05)' },
          '100%': { transform: 'translateY(0) scale(1)', opacity: '1' },
        },
      },
    },
  },
  plugins: [],
}
