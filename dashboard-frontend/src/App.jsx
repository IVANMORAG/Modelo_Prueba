import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  LayoutDashboard, BarChart3, AlertCircle, Lightbulb, 
  Settings, Moon, Sun, Menu, X
} from 'lucide-react';
import Dashboard from './pages/Dashboard';

// React Query Client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 5 * 60 * 1000,
    },
  },
});

function App() {
  const [darkMode, setDarkMode] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [currentView, setCurrentView] = useState('dashboard');

  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
    document.documentElement.classList.toggle('dark');
  };

  const menuItems = [
    { id: 'dashboard', icon: LayoutDashboard, label: 'Dashboard', color: 'blue' },
    { id: 'forecast', icon: BarChart3, label: 'Pronóstico', color: 'purple' },
    { id: 'anomalies', icon: AlertCircle, label: 'Anomalías', color: 'red' },
    { id: 'recommendations', icon: Lightbulb, label: 'Recomendaciones', color: 'yellow' },
    { id: 'settings', icon: Settings, label: 'Configuración', color: 'gray' },
  ];

  return (
    <QueryClientProvider client={queryClient}>
      <div className={`min-h-screen ${darkMode ? 'dark' : ''}`}>
        <div className="flex h-screen bg-gray-50 dark:bg-gray-900">
          
          {/* Sidebar Desktop */}
          <aside className="hidden md:flex md:flex-col w-64 bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700">
            {/* Logo */}
            <div className="p-6 border-b border-gray-200 dark:border-gray-700">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                  <BarChart3 className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h1 className="text-lg font-bold text-gray-900 dark:text-white">
                    Forecast API
                  </h1>
                  <p className="text-xs text-gray-500">Dashboard v1.0</p>
                </div>
              </div>
            </div>

            {/* Menu Items */}
            <nav className="flex-1 p-4 space-y-2">
              {menuItems.map((item) => {
                const Icon = item.icon;
                const isActive = currentView === item.id;
                
                return (
                  <button
                    key={item.id}
                    onClick={() => setCurrentView(item.id)}
                    className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all ${
                      isActive
                        ? 'bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400'
                        : 'text-gray-600 dark:text-gray-400 hover:bg-gray-50 dark:hover:bg-gray-700/50'
                    }`}
                  >
                    <Icon className="w-5 h-5" />
                    <span className="font-medium">{item.label}</span>
                    {isActive && (
                      <motion.div
                        layoutId="activeTab"
                        className="ml-auto w-1.5 h-1.5 bg-blue-600 rounded-full"
                      />
                    )}
                  </button>
                );
              })}
            </nav>

            {/* Dark Mode Toggle */}
            <div className="p-4 border-t border-gray-200 dark:border-gray-700">
              <button
                onClick={toggleDarkMode}
                className="w-full flex items-center justify-between px-4 py-3 rounded-lg bg-gray-50 dark:bg-gray-700/50 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
              >
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  {darkMode ? 'Modo Claro' : 'Modo Oscuro'}
                </span>
                {darkMode ? (
                  <Sun className="w-5 h-5 text-yellow-500" />
                ) : (
                  <Moon className="w-5 h-5 text-gray-600" />
                )}
              </button>
            </div>
          </aside>

          {/* Sidebar Mobile */}
          <AnimatePresence>
            {sidebarOpen && (
              <>
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  onClick={() => setSidebarOpen(false)}
                  className="fixed inset-0 bg-black/50 z-40 md:hidden"
                />
                <motion.aside
                  initial={{ x: -300 }}
                  animate={{ x: 0 }}
                  exit={{ x: -300 }}
                  transition={{ type: 'spring', damping: 25 }}
                  className="fixed left-0 top-0 bottom-0 w-64 bg-white dark:bg-gray-800 z-50 md:hidden"
                >
                  <div className="p-6 border-b border-gray-200 dark:border-gray-700">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                          <BarChart3 className="w-6 h-6 text-white" />
                        </div>
                        <div>
                          <h1 className="text-lg font-bold text-gray-900 dark:text-white">
                            Forecast API
                          </h1>
                          <p className="text-xs text-gray-500">Dashboard v1.0</p>
                        </div>
                      </div>
                      <button
                        onClick={() => setSidebarOpen(false)}
                        className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg"
                      >
                        <X className="w-5 h-5 text-gray-500" />
                      </button>
                    </div>
                  </div>

                  <nav className="flex-1 p-4 space-y-2">
                    {menuItems.map((item) => {
                      const Icon = item.icon;
                      const isActive = currentView === item.id;
                      
                      return (
                        <button
                          key={item.id}
                          onClick={() => {
                            setCurrentView(item.id);
                            setSidebarOpen(false);
                          }}
                          className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all ${
                            isActive
                              ? 'bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400'
                              : 'text-gray-600 dark:text-gray-400 hover:bg-gray-50 dark:hover:bg-gray-700/50'
                          }`}
                        >
                          <Icon className="w-5 h-5" />
                          <span className="font-medium">{item.label}</span>
                        </button>
                      );
                    })}
                  </nav>
                </motion.aside>
              </>
            )}
          </AnimatePresence>

          {/* Main Content */}
          <main className="flex-1 overflow-y-auto">
            {/* Mobile Header */}
            <div className="md:hidden sticky top-0 z-30 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 px-4 py-3">
              <div className="flex items-center justify-between">
                <button
                  onClick={() => setSidebarOpen(true)}
                  className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg"
                >
                  <Menu className="w-6 h-6 text-gray-600 dark:text-gray-400" />
                </button>
                <h1 className="text-lg font-bold text-gray-900 dark:text-white">
                  Forecast Dashboard
                </h1>
                <button
                  onClick={toggleDarkMode}
                  className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg"
                >
                  {darkMode ? (
                    <Sun className="w-5 h-5 text-yellow-500" />
                  ) : (
                    <Moon className="w-5 h-5 text-gray-600" />
                  )}
                </button>
              </div>
            </div>

            {/* Content Area */}
            <AnimatePresence mode="wait">
              <motion.div
                key={currentView}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.2 }}
              >
                {currentView === 'dashboard' && <Dashboard />}
                {currentView === 'forecast' && (
                  <div className="p-8 text-center">
                    <BarChart3 className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                    <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
                      Vista de Pronóstico
                    </h2>
                    <p className="text-gray-600 dark:text-gray-400">
                      Próximamente: Análisis detallado de pronóstico por dimensiones
                    </p>
                  </div>
                )}
                {currentView === 'anomalies' && (
                  <div className="p-8 text-center">
                    <AlertCircle className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                    <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
                      Detección de Anomalías
                    </h2>
                    <p className="text-gray-600 dark:text-gray-400">
                      Próximamente: Dashboard interactivo de anomalías
                    </p>
                  </div>
                )}
                {currentView === 'recommendations' && (
                  <div className="p-8 text-center">
                    <Lightbulb className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                    <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
                      Recomendaciones Estratégicas
                    </h2>
                    <p className="text-gray-600 dark:text-gray-400">
                      Próximamente: Sistema de recomendaciones inteligentes
                    </p>
                  </div>
                )}
                {currentView === 'settings' && (
                  <div className="p-8 text-center">
                    <Settings className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                    <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
                      Configuración
                    </h2>
                    <p className="text-gray-600 dark:text-gray-400">
                      Próximamente: Panel de configuración
                    </p>
                  </div>
                )}
              </motion.div>
            </AnimatePresence>
          </main>
        </div>
      </div>
    </QueryClientProvider>
  );
}

export default App;