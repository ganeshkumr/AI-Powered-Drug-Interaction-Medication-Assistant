import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Heart, Activity, Flame } from 'lucide-react'
import { Line } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js'
import { healthAPI } from '../../services/api'

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
)

const HealthMonitoring = () => {
  const [healthData, setHealthData] = useState({
    heartRate: '--',
    steps: '--',
    calories: '--',
    trends: []
  })

  useEffect(() => {
    fetchHealthData()
    const interval = setInterval(fetchHealthData, 10000)
    return () => clearInterval(interval)
  }, [])

  const fetchHealthData = async () => {
    try {
      const response = await healthAPI.getHealthData()
      setHealthData({
        heartRate: response.data.current.heart_rate,
        steps: response.data.current.steps.toLocaleString(),
        calories: response.data.current.calories,
        trends: response.data.trends
      })
    } catch (error) {
      console.error('Failed to fetch health data:', error)
    }
  }

  const chartData = {
    labels: healthData.trends.map(d => new Date(d.date).toLocaleDateString('en-US', { weekday: 'short' })),
    datasets: [
      {
        label: 'Heart Rate',
        data: healthData.trends.map(d => d.heart_rate),
        borderColor: 'rgb(239, 68, 68)',
        backgroundColor: 'rgba(239, 68, 68, 0.1)',
        tension: 0.4,
        fill: true,
      },
      {
        label: 'Steps (K)',
        data: healthData.trends.map(d => d.steps / 1000),
        borderColor: 'rgb(34, 197, 94)',
        backgroundColor: 'rgba(34, 197, 94, 0.1)',
        tension: 0.4,
        fill: true,
        yAxisID: 'y1',
      }
    ]
  }

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    scales: {
      y: {
        type: 'linear',
        display: true,
        position: 'left',
        grid: {
          color: 'rgba(0,0,0,0.1)'
        }
      },
      y1: {
        type: 'linear',
        display: true,
        position: 'right',
        grid: {
          drawOnChartArea: false,
        }
      }
    },
    plugins: {
      legend: {
        position: 'top',
      },
      tooltip: {
        backgroundColor: 'rgba(0,0,0,0.8)',
      }
    }
  }

  const metrics = [
    {
      icon: Heart,
      label: 'Heart Rate',
      value: healthData.heartRate,
      unit: 'BPM',
      color: 'red',
      bgColor: 'bg-red-100 dark:bg-red-900/20'
    },
    {
      icon: Activity,
      label: 'Steps Today',
      value: healthData.steps,
      unit: 'steps',
      color: 'green',
      bgColor: 'bg-green-100 dark:bg-green-900/20'
    },
    {
      icon: Flame,
      label: 'Calories Burned',
      value: healthData.calories,
      unit: 'kcal',
      color: 'orange',
      bgColor: 'bg-orange-100 dark:bg-orange-900/20'
    }
  ]

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-slate-800 dark:to-slate-700 p-6 rounded-xl shadow-lg border border-blue-200 dark:border-slate-600"
    >
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold text-slate-800 dark:text-white">
          📱 Live Health Monitoring
        </h2>
        <div className="flex items-center space-x-2">
          <motion.div
            animate={{ scale: [1, 1.2, 1] }}
            transition={{ repeat: Infinity, duration: 2 }}
            className="w-2 h-2 bg-green-500 rounded-full"
          />
          <span className="text-sm text-slate-600 dark:text-slate-300">Connected</span>
        </div>
      </div>

      {/* Health Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        {metrics.map((metric, index) => (
          <motion.div
            key={metric.label}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: index * 0.1 }}
            whileHover={{ scale: 1.05 }}
            className="bg-white dark:bg-slate-800 p-4 rounded-lg border border-slate-200 dark:border-slate-600 shadow-sm"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-slate-600 dark:text-slate-400">{metric.label}</p>
                <p className={`text-3xl font-bold text-${metric.color}-500`}>
                  {metric.value}
                </p>
                <p className="text-xs text-slate-500 dark:text-slate-400">{metric.unit}</p>
              </div>
              <div className={`w-12 h-12 ${metric.bgColor} rounded-full flex items-center justify-center`}>
                <metric.icon className={`w-6 h-6 text-${metric.color}-500`} />
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Health Chart */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
        className="bg-white dark:bg-slate-800 p-4 rounded-lg border border-slate-200 dark:border-slate-600"
      >
        <h3 className="font-semibold text-slate-800 dark:text-white mb-4">
          📊 Health Trends (Last 7 Days)
        </h3>
        <div className="h-64">
          <Line data={chartData} options={chartOptions} />
        </div>
      </motion.div>
    </motion.div>
  )
}

export default HealthMonitoring