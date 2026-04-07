import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import RunList from './pages/RunList'
import RunDetail from './pages/RunDetail'
import StepDetail from './pages/StepDetail'

export default function App() {
  return (
    <BrowserRouter>
      <div className="container">
        <Routes>
          <Route path="/" element={<Navigate to="/app" replace />} />
          <Route path="/app" element={<RunList />} />
          <Route path="/app/runs/:runId" element={<RunDetail />} />
          <Route path="/app/runs/:runId/steps/:step" element={<StepDetail />} />
          <Route path="*" element={<Navigate to="/app" replace />} />
        </Routes>
      </div>
    </BrowserRouter>
  )
}
