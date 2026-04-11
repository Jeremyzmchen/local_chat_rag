import { BrowserRouter, Routes, Route } from 'react-router-dom'
import AppLayout from './components/layout/AppLayout'
import Dashboard from './pages/Dashboard'
import Documents from './pages/Documents'
import Upload    from './pages/Upload'
import Report    from './pages/Report'
import Review    from './pages/Review'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<AppLayout />}>
          <Route index           element={<Dashboard />} />
          <Route path="documents"  element={<Documents />} />
          <Route path="upload"     element={<Upload />} />
          <Route path="report"     element={<Report />} />
          <Route path="report/:id" element={<Report />} />
          <Route path="review/:id" element={<Review />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
