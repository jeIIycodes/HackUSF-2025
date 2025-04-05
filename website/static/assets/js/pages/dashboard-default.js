'use strict';

document.addEventListener('DOMContentLoaded', function() {
  // Helper function to create a Chart.js line chart with threshold-based styling and hatching.
  function createChart(canvasId, data, threshold) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    const labels = data.map((_, index) => index + 1); // Generate day labels

    const config = {
      type: 'line',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Radius (mm)',
            data: data,
            borderColor: 'rgba(75, 192, 192, 1)',
            fill: false,
            tension: 0.1,
            // Use scriptable options for point radius and color:
            pointRadius: function(context) {
              const value = context.raw;
              return (value > threshold) ? 8 : 3;
            },
            pointBackgroundColor: function(context) {
              const value = context.raw;
              return (value > threshold) ? 'orange' : 'rgba(75, 192, 192, 1)';
            }
          },
          {
            label: 'Threshold',
            data: Array(data.length).fill(threshold),
            borderColor: 'rgba(255, 99, 132, 1)',
            borderDash: [5, 5],
            fill: false,
            pointRadius: 0,
            tension: 0.1
          }
        ]
      },
      options: {
        scales: {
          x: {
            title: {
              display: true,
              text: 'Days'
            }
          },
          y: {
            title: {
              display: true,
              text: 'Radius (mm)'
            }
          }
        }
      },
      plugins: [] // Custom plugin(s) will be added below.
    };

    // Custom plugin: Draw a hatched pattern in the area above the threshold line.
    config.plugins.push({
      id: 'hatchedThreshold',
      afterDatasetsDraw: function(chart) {
        const ctx = chart.ctx;
        const chartArea = chart.chartArea;
        const yScale = chart.scales.y;
        const thresholdY = yScale.getPixelForValue(threshold);
        // Ensure the threshold line lies within the chart area.
        if (thresholdY > chartArea.top && thresholdY < chartArea.bottom) {
          // Create an offscreen canvas to define a diagonal hatch pattern.
          const patternCanvas = document.createElement('canvas');
          patternCanvas.width = 10;
          patternCanvas.height = 10;
          const pctx = patternCanvas.getContext('2d');
          pctx.strokeStyle = 'rgba(200,200,200,0.5)';
          pctx.lineWidth = 1;
          pctx.beginPath();
          pctx.moveTo(0, 10);
          pctx.lineTo(10, 0);
          pctx.stroke();
          const pattern = ctx.createPattern(patternCanvas, 'repeat');

          ctx.save();
          ctx.fillStyle = pattern;
          // Fill the area above the threshold line.
          ctx.fillRect(chartArea.left, chartArea.top, chartArea.right - chartArea.left, thresholdY - chartArea.top);
          ctx.restore();
        }
      }
    });

    new Chart(ctx, config);
  }

  // Example dummy data arrays and thresholds for demonstration.
  const leftElbowData = [5, 6, 7, 8, 7, 6, 5];
  const leftThighData = [6, 7, 6.5, 7.2, 7.5, 7.3, 7];
  const rightShoulderData = [4, 4.5, 5,4.8, 4.5, 5.5, 5 ];

  const leftElbowThreshold = 7;
  const leftThighThreshold = 7.3;
  const rightShoulderThreshold = 5;

  // Create the charts for each tab.
  createChart('chart-left-elbow', leftElbowData, leftElbowThreshold);
  createChart('chart-left-thigh', leftThighData, leftThighThreshold);
  createChart('chart-right-shoulder', rightShoulderData, rightShoulderThreshold);
});
