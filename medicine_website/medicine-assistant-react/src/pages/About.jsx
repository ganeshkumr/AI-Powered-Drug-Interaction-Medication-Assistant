import { motion } from 'framer-motion'
import { 
  Shield, 
  Brain, 
  Users, 
  AlertTriangle, 
  Database, 
  Stethoscope,
  CheckCircle,
  Zap,
  Heart,
  Lock,
  FileText,
  ExternalLink
} from 'lucide-react'
import Card from '../components/common/Card'
import Button from '../components/common/Button'

const About = () => {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  }

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 }
  }

  return (
    <div className="max-w-4xl mx-auto">
      {/* Page Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-12"
      >
        <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-primary to-primary-600 rounded-2xl shadow-soft-lg mb-6">
          <Stethoscope className="w-8 h-8 text-white" />
        </div>
        <h1 className="text-4xl font-heading font-bold text-neutral-text mb-4">
          About AI-HealthMate
        </h1>
        <p className="text-xl text-gray-600 leading-relaxed">
          Your trusted companion for medication safety and drug interaction analysis
        </p>
      </motion.div>

      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="space-y-12"
      >
        {/* Why This Matters Section */}
        <motion.section variants={itemVariants}>
          <Card className="border-l-4 border-l-danger">
            <div className="flex items-start space-x-4">
              <div className="flex-shrink-0">
                <div className="w-12 h-12 bg-danger-50 rounded-xl flex items-center justify-center">
                  <Heart className="w-6 h-6 text-danger" />
                </div>
              </div>
              <div className="flex-1">
                <h2 className="text-2xl font-heading font-bold text-neutral-text mb-4">
                  Why This Matters
                </h2>
                <div className="prose prose-lg text-gray-600 space-y-4">
                  <p>
                    <strong>Medication safety is a critical healthcare concern.</strong> Every year, 
                    adverse drug interactions affect millions of people worldwide, leading to 
                    hospitalizations, complications, and even fatalities that could have been prevented.
                  </p>
                  <p>
                    Many patients take multiple medications prescribed by different healthcare providers, 
                    creating complex interaction patterns that can be difficult to track manually. 
                    Even healthcare professionals can miss potential interactions when managing 
                    complex medication regimens.
                  </p>
                  <div className="bg-danger-50 border border-danger-200 rounded-lg p-4 my-6">
                    <div className="flex items-center space-x-2 mb-2">
                      <AlertTriangle className="w-5 h-5 text-danger" />
                      <span className="font-semibold text-danger">Critical Statistics</span>
                    </div>
                    <ul className="text-sm text-danger-700 space-y-1">
                      <li>• Over 125,000 deaths annually in the US from medication errors</li>
                      <li>• 1 in 5 adults take medications that could interact dangerously</li>
                      <li>• 70% of adverse drug events are preventable with proper screening</li>
                    </ul>
                  </div>
                  <p>
                    Our mission is to bridge this safety gap by providing intelligent, 
                    accessible medication interaction analysis that empowers both patients 
                    and healthcare providers to make informed decisions.
                  </p>
                </div>
              </div>
            </div>
          </Card>
        </motion.section>

        {/* What This System Does Section */}
        <motion.section variants={itemVariants}>
          <Card className="border-l-4 border-l-primary">
            <div className="flex items-start space-x-4">
              <div className="flex-shrink-0">
                <div className="w-12 h-12 bg-primary-50 rounded-xl flex items-center justify-center">
                  <Brain className="w-6 h-6 text-primary" />
                </div>
              </div>
              <div className="flex-1">
                <h2 className="text-2xl font-heading font-bold text-neutral-text mb-4">
                  What This System Does
                </h2>
                <div className="prose prose-lg text-gray-600 space-y-4">
                  <p>
                    AI-HealthMate is a comprehensive medication safety platform that combines 
                    advanced artificial intelligence with extensive medical databases to 
                    provide real-time drug interaction analysis.
                  </p>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 my-6">
                    <div className="bg-primary-50 rounded-lg p-4">
                      <div className="flex items-center space-x-2 mb-2">
                        <Zap className="w-5 h-5 text-primary" />
                        <span className="font-semibold text-primary">AI-Powered Analysis</span>
                      </div>
                      <p className="text-sm text-primary-700">
                        Advanced Graph Neural Network (GNN) models analyze complex 
                        molecular interactions and predict potential risks with high accuracy.
                      </p>
                    </div>
                    
                    <div className="bg-secondary-50 rounded-lg p-4">
                      <div className="flex items-center space-x-2 mb-2">
                        <Shield className="w-5 h-5 text-secondary" />
                        <span className="font-semibold text-secondary">Safety Validation</span>
                      </div>
                      <p className="text-sm text-secondary-700">
                        Cross-references dosages against medical databases and 
                        validates safety parameters for your specific health profile.
                      </p>
                    </div>
                    
                    <div className="bg-accent-50 rounded-lg p-4">
                      <div className="flex items-center space-x-2 mb-2">
                        <CheckCircle className="w-5 h-5 text-accent" />
                        <span className="font-semibold text-accent">Personalized Checks</span>
                      </div>
                      <p className="text-sm text-accent-700">
                        Tailored recommendations based on your medication history, 
                        health conditions, and individual risk factors.
                      </p>
                    </div>
                    
                    <div className="bg-success-50 rounded-lg p-4">
                      <div className="flex items-center space-x-2 mb-2">
                        <Users className="w-5 h-5 text-success" />
                        <span className="font-semibold text-success">Healthcare Integration</span>
                      </div>
                      <p className="text-sm text-success-700">
                        Designed to complement your healthcare provider's expertise, 
                        not replace professional medical advice.
                      </p>
                    </div>
                  </div>

                  <p>
                    The system processes your medication information through multiple 
                    validation layers, providing clear risk assessments, detailed 
                    explanations, and actionable recommendations in seconds.
                  </p>
                </div>
              </div>
            </div>
          </Card>
        </motion.section>

        {/* How It's Different Section */}
        <motion.section variants={itemVariants}>
          <Card className="border-l-4 border-l-accent">
            <div className="flex items-start space-x-4">
              <div className="flex-shrink-0">
                <div className="w-12 h-12 bg-accent-50 rounded-xl flex items-center justify-center">
                  <Zap className="w-6 h-6 text-accent" />
                </div>
              </div>
              <div className="flex-1">
                <h2 className="text-2xl font-heading font-bold text-neutral-text mb-4">
                  How It's Different
                </h2>
                <div className="prose prose-lg text-gray-600 space-y-4">
                  <p>
                    Unlike traditional drug interaction checkers, AI-HealthMate offers 
                    a comprehensive, intelligent approach to medication safety:
                  </p>
                  
                  <div className="space-y-4">
                    <div className="flex items-start space-x-3">
                      <div className="w-8 h-8 bg-primary-100 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
                        <Brain className="w-4 h-4 text-primary" />
                      </div>
                      <div>
                        <h3 className="font-semibold text-neutral-text mb-1">Advanced AI Technology</h3>
                        <p className="text-gray-600">
                          Utilizes cutting-edge Graph Neural Networks trained on extensive 
                          pharmaceutical databases, going beyond simple rule-based checking 
                          to understand complex molecular interactions.
                        </p>
                      </div>
                    </div>
                    
                    <div className="flex items-start space-x-3">
                      <div className="w-8 h-8 bg-secondary-100 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
                        <Users className="w-4 h-4 text-secondary" />
                      </div>
                      <div>
                        <h3 className="font-semibold text-neutral-text mb-1">User-Centric Design</h3>
                        <p className="text-gray-600">
                          Built with both patients and healthcare providers in mind, 
                          featuring intuitive interfaces, clear explanations, and 
                          professional-grade reliability.
                        </p>
                      </div>
                    </div>
                    
                    <div className="flex items-start space-x-3">
                      <div className="w-8 h-8 bg-accent-100 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
                        <Lock className="w-4 h-4 text-accent" />
                      </div>
                      <div>
                        <h3 className="font-semibold text-neutral-text mb-1">Privacy-First Approach</h3>
                        <p className="text-gray-600">
                          Your health data is processed locally and encrypted, ensuring 
                          maximum privacy while providing comprehensive analysis. No data 
                          is shared with third parties without your explicit consent.
                        </p>
                      </div>
                    </div>
                    
                    <div className="flex items-start space-x-3">
                      <div className="w-8 h-8 bg-success-100 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
                        <CheckCircle className="w-4 h-4 text-success" />
                      </div>
                      <div>
                        <h3 className="font-semibold text-neutral-text mb-1">Continuous Learning</h3>
                        <p className="text-gray-600">
                          Our AI models are continuously updated with the latest medical 
                          research and drug interaction data, ensuring you always have 
                          access to the most current safety information.
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </Card>
        </motion.section>

        {/* Data Disclaimer Section */}
        <motion.section variants={itemVariants}>
          <Card className="border-l-4 border-l-warning">
            <div className="flex items-start space-x-4">
              <div className="flex-shrink-0">
                <div className="w-12 h-12 bg-warning-50 rounded-xl flex items-center justify-center">
                  <Database className="w-6 h-6 text-warning" />
                </div>
              </div>
              <div className="flex-1">
                <h2 className="text-2xl font-heading font-bold text-neutral-text mb-4">
                  Data Disclaimer
                </h2>
                <div className="prose prose-lg text-gray-600 space-y-4">
                  <div className="bg-warning-50 border border-warning-200 rounded-lg p-4">
                    <div className="flex items-center space-x-2 mb-2">
                      <FileText className="w-5 h-5 text-warning" />
                      <span className="font-semibold text-warning">Important Information</span>
                    </div>
                    <p className="text-sm text-warning-700">
                      Please read this section carefully to understand how your data is used 
                      and the limitations of our analysis.
                    </p>
                  </div>
                  
                  <div className="space-y-4">
                    <div>
                      <h3 className="font-semibold text-neutral-text mb-2">Data Sources</h3>
                      <p>
                        Our analysis is based on comprehensive pharmaceutical databases including 
                        FDA drug labels, clinical trial data, medical literature, and established 
                        drug interaction databases. While extensive, these sources may not capture 
                        every possible interaction or individual variation.
                      </p>
                    </div>
                    
                    <div>
                      <h3 className="font-semibold text-neutral-text mb-2">Data Usage</h3>
                      <ul className="list-disc list-inside space-y-1 text-gray-600">
                        <li>Your medication information is processed locally on secure servers</li>
                        <li>Personal health data is encrypted and never shared with third parties</li>
                        <li>Anonymized interaction patterns may be used to improve our AI models</li>
                        <li>You can request deletion of your data at any time</li>
                      </ul>
                    </div>
                    
                    <div>
                      <h3 className="font-semibold text-neutral-text mb-2">Limitations</h3>
                      <ul className="list-disc list-inside space-y-1 text-gray-600">
                        <li>Analysis is based on available scientific data and may not reflect the most recent discoveries</li>
                        <li>Individual responses to medications can vary significantly</li>
                        <li>The system cannot account for all personal health factors or genetic variations</li>
                        <li>Results should always be verified with healthcare professionals</li>
                      </ul>
                    </div>
                    
                    <div>
                      <h3 className="font-semibold text-neutral-text mb-2">Data Accuracy</h3>
                      <p>
                        While we strive for the highest accuracy, medical information is constantly 
                        evolving. Our AI models are regularly updated, but there may be delays 
                        between new medical discoveries and their integration into our system.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </Card>
        </motion.section>

        {/* Safety Notice Section */}
        <motion.section variants={itemVariants}>
          <Card className="border-l-4 border-l-danger bg-danger-50">
            <div className="flex items-start space-x-4">
              <div className="flex-shrink-0">
                <div className="w-12 h-12 bg-danger-100 rounded-xl flex items-center justify-center">
                  <AlertTriangle className="w-6 h-6 text-danger" />
                </div>
              </div>
              <div className="flex-1">
                <h2 className="text-2xl font-heading font-bold text-danger mb-4">
                  Safety Notice
                </h2>
                <div className="prose prose-lg text-danger-800 space-y-4">
                  <div className="bg-white border border-danger-200 rounded-lg p-4">
                    <div className="flex items-center space-x-2 mb-3">
                      <AlertTriangle className="w-5 h-5 text-danger" />
                      <span className="font-bold text-danger">MEDICAL DISCLAIMER</span>
                    </div>
                    <p className="text-sm text-danger-700 font-medium">
                      This system is designed to assist with medication safety analysis but 
                      IS NOT A SUBSTITUTE for professional medical advice, diagnosis, or treatment.
                    </p>
                  </div>
                  
                  <div className="space-y-4">
                    <div>
                      <h3 className="font-semibold text-danger mb-2">Always Consult Healthcare Professionals</h3>
                      <ul className="list-disc list-inside space-y-1 text-danger-700">
                        <li>Before starting, stopping, or changing any medication</li>
                        <li>If you experience any adverse reactions or side effects</li>
                        <li>For personalized medical advice based on your complete health history</li>
                        <li>When managing complex medical conditions or multiple medications</li>
                      </ul>
                    </div>
                    
                    <div>
                      <h3 className="font-semibold text-danger mb-2">Emergency Situations</h3>
                      <p className="text-danger-700">
                        <strong>If you are experiencing a medical emergency, call emergency services immediately.</strong> 
                        Do not rely on this system for emergency medical decisions. Seek immediate 
                        professional medical attention for any serious symptoms or reactions.
                      </p>
                    </div>
                    
                    <div>
                      <h3 className="font-semibold text-danger mb-2">System Limitations</h3>
                      <ul className="list-disc list-inside space-y-1 text-danger-700">
                        <li>Cannot replace clinical judgment and professional medical expertise</li>
                        <li>May not detect all possible interactions or individual sensitivities</li>
                        <li>Does not consider your complete medical history or current health status</li>
                        <li>Should not be used as the sole basis for medication decisions</li>
                      </ul>
                    </div>
                    
                    <div>
                      <h3 className="font-semibold text-danger mb-2">Legal Notice</h3>
                      <p className="text-danger-700">
                        By using this system, you acknowledge that you understand these limitations 
                        and agree that the developers and operators of AI-HealthMate are not liable 
                        for any medical decisions made based on the information provided. Always 
                        verify results with qualified healthcare professionals.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </Card>
        </motion.section>

        {/* Call to Action */}
        <motion.section variants={itemVariants} className="text-center">
          <Card className="bg-gradient-to-br from-primary-50 to-secondary-50">
            <div className="max-w-2xl mx-auto">
              <h2 className="text-2xl font-heading font-bold text-neutral-text mb-4">
                Ready to Check Your Medications?
              </h2>
              <p className="text-gray-600 mb-6">
                Start with our quick interaction check or create an account for 
                comprehensive medication management.
              </p>
              <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
                <Button
                  variant="primary"
                  size="lg"
                  icon={<Shield className="w-5 h-5" />}
                  onClick={() => window.location.href = '/'}
                >
                  Quick Safety Check
                </Button>
                <Button
                  variant="secondary"
                  size="lg"
                  icon={<Users className="w-5 h-5" />}
                  onClick={() => window.location.href = '/register'}
                >
                  Create Account
                </Button>
              </div>
            </div>
          </Card>
        </motion.section>
      </motion.div>
    </div>
  )
}

export default About