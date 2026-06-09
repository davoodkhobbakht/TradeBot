import { ConfigEditor } from '../components/Config/ConfigEditor'
export function ConfigPage() {
  return (
    <div className="space-y-6">
      <div><h1 className="text-2xl font-bold text-white">Configuration</h1><p className="text-dark-400 mt-1">Manage bot settings and parameters</p></div>
      <ConfigEditor />
    </div>
  )
}
