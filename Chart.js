import { Line } from 'react-chartjs-2';
import { Chart as ChartJS } from 'chart.js/auto';

export default function Chart({ data }) {
  const chartData = {
    labels: [...Object.keys(data.historical), ...data.dates],
    datasets: [
      {
        label: 'Historical',
        data: [...Object.values(data.historical), ...Array(data.forecast.length).fill(null)],
        borderColor: 'rgb(53, 162, 235)',
      },
      {
        label: 'Forecast',
        data: [...Array(Object.keys(data.historical).length).fill(null), ...data.forecast],
        borderColor: 'rgb(255, 99, 132)',
        borderDash: [5, 5],
      },
    ],
  };

  return <Line data={chartData} />;
}
