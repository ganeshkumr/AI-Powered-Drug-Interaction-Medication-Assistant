import { motion } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import { Activity, Shield, Zap, CheckCircle, Users, Clock, Pill } from 'lucide-react'
import { useState } from 'react'
import Button from '../components/common/Button'
import Modal from '../components/common/Modal'
import QuickCheckModal from '../components/landing/QuickCheckModal'

const Landing = () => {
  const navigate = useNavigate()
  const [showQuickCheck, setShowQuickCheck] = useState(false)

  const features = [
    {
      icon: <Zap className="w-6 h-6" />,
      title: 'AI-Powered Analysis',
      description: 'Advanced GNN model predicts drug interactions with high accuracy'
    },
    {
      icon: <Shield className="w-6 h-6" />,
      title: 'Dosage Safety',
      description: 'Validates dosages against medical databases for your safety'
    },
    {
      icon: <CheckCircle className="w-6 h-6" />,
      title: 'Personalized Checks',
      description: 'Tailored recommendations based on your health profile'
    }
  ]

  // Floating pill animations
  const floatingPills = [
    { delay: 0, duration: 20, x: '10%', y: '20%' },
    { delay: 2, duration: 25, x: '80%', y: '15%' },
    { delay: 4, duration: 22, x: '15%', y: '70%' },
    { delay: 1, duration: 23, x: '85%', y: '75%' },
    { delay: 3, duration: 21, x: '50%', y: '85%' },
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-50 via-white to-accent-50 relative overflow-hidden">
      {/* Animated Background Pattern */}
      <div className="absolute inset-0 opacity-30">
        <div className="absolute inset-0" style={{
          backgroundImage: `radial-gradient(circle at 20% 50%, rgba(46, 167, 155, 0.1) 0%, transparent 50%),
                           radial-gradient(circle at 80% 80%, rgba(244, 180, 0, 0.1) 0%, transparent 50%),
                           radial-gradient(circle at 40% 90%, rgba(46, 167, 155, 0.08) 0%, transparent 50%)`
        }} />
      </div>

      {/* Floating Pills Animation */}
      {floatingPills.map((pill, index) => (
        <motion.div
          key={index}
          className="absolute"
          style={{ left: pill.x, top: pill.y }}
          animate={{
            y: [0, -30, 0],
            rotate: [0, 360],
            opacity: [0.1, 0.3, 0.1]
          }}
          transition={{
            duration: pill.duration,
            repeat: Infinity,
            delay: pill.delay,
            ease: "easeInOut"
          }}
        >
          <Pill className="w-8 h-8 text-primary" />
        </motion.div>
      ))}

      {/* Animated Grid Pattern */}
      <div className="absolute inset-0 opacity-5">
        <div className="absolute inset-0" style={{
          backgroundImage: `linear-gradient(rgba(46, 167, 155, 0.3) 1px, transparent 1px),
                           linear-gradient(90deg, rgba(46, 167, 155, 0.3) 1px, transparent 1px)`,
          backgroundSize: '50px 50px'
        }} />
      </div>
      {/* Hero Section */}
      <div className="container mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 pt-20 pb-16 relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center max-w-4xl mx-auto"
        >
          {/* Logo with Pulse Animation */}
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.2, type: 'spring' }}
            className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-br from-primary to-primary-600 rounded-2xl shadow-soft-lg mb-8 relative"
          >
            <motion.div
              className="absolute inset-0 bg-primary rounded-2xl"
              animate={{
                scale: [1, 1.2, 1],
                opacity: [0.5, 0, 0.5]
              }}
              transition={{
                duration: 2,
                repeat: Infinity,
                ease: "easeInOut"
              }}
            />
            <Activity className="w-10 h-10 text-white relative z-10" />
          </motion.div>

          {/* Headline with Stagger Animation */}
          <motion.h1 
            className="text-5xl md:text-6xl font-heading font-bold text-neutral-text mb-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3, duration: 0.6 }}
          >
            Is that medicine{' '}
            <motion.span 
              className="gradient-text"
              animate={{
                backgroundPosition: ['0% 50%', '100% 50%', '0% 50%']
              }}
              transition={{
                duration: 5,
                repeat: Infinity,
                ease: "linear"
              }}
              style={{
                backgroundSize: '200% auto'
              }}
            >
              safe for you?
            </motion.span>
          </motion.h1>

          <motion.p 
            className="text-xl text-gray-600 mb-8 leading-relaxed"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4, duration: 0.6 }}
          >
            Quickly check drug interactions with AI-powered analysis.
            <br />
            <span className="text-primary font-medium">No signup required.</span>
          </motion.p>

          {/* CTA Buttons with Stagger Animation */}
          <motion.div 
            className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-4"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5, duration: 0.6 }}
          >
            <motion.div
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Button
                variant="primary"
                size="lg"
                onClick={() => setShowQuickCheck(true)}
                icon={<Activity className="w-5 h-5" />}
                className="w-full sm:w-auto px-8"
              >
                Check Interaction Now
              </Button>
            </motion.div>
            
            <motion.div
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Button
                variant="secondary"
                size="lg"
                onClick={() => navigate('/register')}
                className="w-full sm:w-auto px-8"
              >
                Create Account
              </Button>
            </motion.div>
          </motion.div>

          <motion.p 
            className="text-sm text-gray-500"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.6, duration: 0.6 }}
          >
            Already have an account?{' '}
            <button
              onClick={() => navigate('/login')}
              className="text-primary hover:text-primary-600 font-medium transition-colors"
            >
              Sign in
            </button>
          </motion.p>
        </motion.div>

        {/* Features Grid with Enhanced Animations */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7 }}
          className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-20"
        >
          {features.map((feature, index) => (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.8 + index * 0.1 }}
              whileHover={{ 
                y: -8,
                transition: { duration: 0.3 }
              }}
              className="bg-white/80 backdrop-blur-sm rounded-card-lg p-6 shadow-soft hover:shadow-soft-lg transition-all duration-300 border border-gray-100"
            >
              <motion.div 
                className="w-12 h-12 bg-primary-50 rounded-xl flex items-center justify-center mb-4 text-primary"
                whileHover={{ 
                  rotate: 360,
                  scale: 1.1
                }}
                transition={{ duration: 0.5 }}
              >
                {feature.icon}
              </motion.div>
              <h3 className="text-lg font-heading font-bold text-neutral-text mb-2">
                {feature.title}
              </h3>
              <p className="text-gray-600 text-sm leading-relaxed">
                {feature.description}
              </p>
            </motion.div>
          ))}
        </motion.div>

        {/* Trust Indicators with Fade-in Animation */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.1 }}
          className="mt-16 flex flex-wrap items-center justify-center gap-6"
        >
          {[
            { icon: Shield, text: 'Health data processed locally & encrypted', delay: 0 },
            { icon: Users, text: 'Trusted by healthcare professionals', delay: 0.1 },
            { icon: Clock, text: 'Results in seconds', delay: 0.2 }
          ].map((item, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 1.1 + item.delay }}
              whileHover={{ scale: 1.05 }}
              className="inline-flex items-center space-x-2 px-4 py-2 bg-white/80 backdrop-blur-sm rounded-full shadow-soft hover:shadow-soft-lg transition-all duration-300"
            >
              <item.icon className="w-4 h-4 text-primary" />
              <span className="text-sm text-gray-600">
                {item.text}
              </span>
            </motion.div>
          ))}
        </motion.div>
      </div>

      {/* Quick Check Modal */}
      <Modal
        isOpen={showQuickCheck}
        onClose={() => setShowQuickCheck(false)}
        title="Quick Interaction Check"
        size="lg"
      >
        <QuickCheckModal onClose={() => setShowQuickCheck(false)} />
      </Modal>
    </div>
  )
}

export default Landing
