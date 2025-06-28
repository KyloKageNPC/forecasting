import { useState } from 'react';
import Chart from '../components/Chart';

export default function Home() {
  const [forecast, setForecast] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    const formData = new FormData(e.target);
    const response = await fetch('http://localhost:8000/api/forecast', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        country: formData.get('country'),
        operator: formData.get('operator'),
        horizon: parseInt(formData.get('horizon')),
      }),
    });
    const data = await response.json();
    setForecast(data);
    setLoading(false);
  };

  return (
    <div>
      <h1>Mobile Subscriber Forecast</h1>
      <form onSubmit={handleSubmit}>
        <input name="country" placeholder="Country (e.g., Ghana)" required />
        <input name="operator" placeholder="Operator (e.g., MTN)" required />
        <input
          name="horizon"
          type="number"
          placeholder="Quarters to forecast (e.g., 8)"
          min="4"
          max="20"
        />
        <button type="submit" disabled={loading}>
          {loading ? 'Generating...' : 'Forecast'}
        </button>
      </form>
      {forecast && <Chart data={forecast} />}
    </div>
  );
}
