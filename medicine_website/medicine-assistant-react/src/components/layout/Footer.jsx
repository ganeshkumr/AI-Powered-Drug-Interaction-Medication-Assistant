import { Heart, Shield, FileText } from 'lucide-react'
import { Link } from 'react-router-dom'

const Footer = () => {
  return (
    <footer className="bg-white dark:bg-slate-800 border-t border-gray-100 mt-auto">
      <div className="container mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* About */}
          <div>
            <h3 className="font-heading font-bold text-lg mb-4 text-neutral-text">
              About AI-HealthMate
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
              Your trusted AI-powered medication safety assistant. We help you check drug interactions and manage your medications safely.
            </p>
          </div>

          {/* Quick Links */}
          <div>
            <h3 className="font-heading font-bold text-lg mb-4 text-neutral-text">
              Quick Links
            </h3>
            <ul className="space-y-2">
              <li>
                <Link to="/dashboard" className="text-sm text-gray-600 hover:text-primary transition-colors">
                  Check Interaction
                </Link>
              </li>
              <li>
                <Link to="/dashboard" className="text-sm text-gray-600 hover:text-primary transition-colors">
                  My Medications
                </Link>
              </li>
              <li>
                <Link to="/profile" className="text-sm text-gray-600 hover:text-primary transition-colors">
                  Health Profile
                </Link>
              </li>
            </ul>
          </div>

          {/* Legal & Contact */}
          <div>
            <h3 className="font-heading font-bold text-lg mb-4 text-neutral-text">
              Legal & Privacy
            </h3>
            <ul className="space-y-2">
              <li>
                <a href="#" className="text-sm text-gray-600 hover:text-primary transition-colors flex items-center space-x-2">
                  <Shield className="w-4 h-4" />
                  <span>Privacy Policy</span>
                </a>
              </li>
              <li>
                <a href="#" className="text-sm text-gray-600 hover:text-primary transition-colors flex items-center space-x-2">
                  <FileText className="w-4 h-4" />
                  <span>Terms of Service</span>
                </a>
              </li>
              <li>
                <a href="#" className="text-sm text-gray-600 hover:text-primary transition-colors">
                  About Us
                </a>
              </li>
            </ul>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="mt-8 pt-8 border-t border-gray-100">
          <div className="flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0">
            <p className="text-sm text-gray-600 dark:text-gray-400 flex items-center space-x-2">
              <Shield className="w-4 h-4 text-primary" />
              <span>Health data processed locally & encrypted. Not shared.</span>
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400 flex items-center space-x-1">
              <span>Made with</span>
              <Heart className="w-4 h-4 text-status-danger fill-current" />
              <span>for your health</span>
            </p>
          </div>
        </div>
      </div>
    </footer>
  )
}

export default Footer
