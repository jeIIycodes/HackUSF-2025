'use strict';

document.addEventListener('DOMContentLoaded', function() {
  // Helper function to create a Chart.js line chart with delayed animations and conditional red filling.
  function createChart(canvasId, data, threshold) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    // Generate day labels from the data length
    const labels = data.map((_, index) => index + 1);

    const config = {
      type: 'line',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Radius (mm)',
            data: data,
            borderColor: 'rgba(75, 192, 192, 1)', // Blue line
            fill: false,
            tension: 0.1,
            // Animate blue line first.
            animations: {
              y: {
                duration: 1500,
                easing: 'easeInOutElastic',
                delay: 0
              }
            },
            // Use scriptable options to adjust point styling:
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
            borderColor: 'rgba(255, 99, 132, 1)', // Red dashed line
            borderDash: [5, 5],
            fill: false,
            pointRadius: 0,
            tension: 0.1,
            // Animate red dashed line with a delay.
            animations: {
              y: {
                duration: 1500,
                easing: 'easeInOutElastic',
                delay: 1500
              }
            }
          }
        ]
      },
      options: {
        scales: {
          x: {
            title: {
              display: true,
              text: 'Weeks'
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
      plugins: []
    };

    // Custom plugin: Fill with red the area where the blue line is above the threshold.
    config.plugins.push({
      id: 'fillRedPlugin',
      afterDatasetsDraw: function(chart) {
        const ctx = chart.ctx;
        const chartArea = chart.chartArea;
        const yScale = chart.scales.y;
        // Get the pixel coordinate for the threshold value.
        const thresholdY = yScale.getPixelForValue(threshold);

        // Get the blue line meta data.
        const blueMeta = chart.getDatasetMeta(0);
        if (!blueMeta || !blueMeta.data) return;

        ctx.save();
        // Set a semi-transparent red fill.
        ctx.fillStyle = 'rgba(255, 0, 0, 0.3)';

        // Iterate through each consecutive pair of points on the blue line.
        for (let i = 0; i < blueMeta.data.length - 1; i++) {
          const p0 = blueMeta.data[i];
          const p1 = blueMeta.data[i+1];
          const x0 = p0.x, y0 = p0.y;
          const x1 = p1.x, y1 = p1.y;

          // Determine if each point is above the threshold.
          const p0Above = y0 < thresholdY;
          const p1Above = y1 < thresholdY;

          if (p0Above && p1Above) {
            // Both points are above the threshold.
            ctx.beginPath();
            ctx.moveTo(x0, y0);
            ctx.lineTo(x1, y1);
            ctx.lineTo(x1, thresholdY);
            ctx.lineTo(x0, thresholdY);
            ctx.closePath();
            ctx.fill();
          } else if (p0Above && !p1Above) {
            // p0 is above, p1 is below the threshold: compute intersection.
            const t = (thresholdY - y0) / (y1 - y0);
            const xi = x0 + t * (x1 - x0);
            ctx.beginPath();
            ctx.moveTo(x0, y0);
            ctx.lineTo(xi, thresholdY);
            ctx.lineTo(x0, thresholdY);
            ctx.closePath();
            ctx.fill();
          } else if (!p0Above && p1Above) {
            // p0 is below, p1 is above: compute intersection.
            const t = (thresholdY - y0) / (y1 - y0);
            const xi = x0 + t * (x1 - x0);
            ctx.beginPath();
            ctx.moveTo(xi, thresholdY);
            ctx.lineTo(x1, y1);
            ctx.lineTo(x1, thresholdY);
            ctx.closePath();
            ctx.fill();
          }
          // If both points are below the threshold, no fill is needed.
        }
        ctx.restore();
      }
    });

    new Chart(ctx, config);
  }

  // Example dummy data arrays and thresholds for demonstration.
  const leftElbowData = [5, 5.1, 5.3, 5.9, 6.8, 7.1, 7.5];
  const leftThighData = [6, 6.3, 6.7, 7.1, 7.5, 7.3, 7];
  const rightShoulderData = [4, 4.3, 4.8, 4.7, 4.5, 4.6, 4.0];

  const leftElbowThreshold = 7;
  const leftThighThreshold = 7.3;
  const rightShoulderThreshold = 5;

  // Create the charts for each tab.
  createChart('chart-left-elbow', leftElbowData, leftElbowThreshold);
  createChart('chart-left-thigh', leftThighData, leftThighThreshold);
  createChart('chart-right-shoulder', rightShoulderData, rightShoulderThreshold);
});
