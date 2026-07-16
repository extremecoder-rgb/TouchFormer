import { Client } from "https://cdn.jsdelivr.net/npm/@gradio/client@1.3.0/dist/index.js";

document.addEventListener("DOMContentLoaded", () => {
    const btn = document.getElementById("predict-btn");
    
    if(btn) {
        btn.addEventListener("click", async () => {
            btn.innerText = "Processing...";
            btn.disabled = true;

            try {
                // Connect to HuggingFace Gradio Space API
                const client = await Client.connect("subhra509/touchformer");
                
                // Call the prediction endpoint
                const result = await client.predict("/predict");
                
                // Extract JSON response (it resides in result.data[0])
                const data = result.data[0];

                drawHeatmap(data.heatmap);
                drawSkeleton(data.skeleton);
                
                // Randomly trigger a bedsore alert simulation for the hospital page
                const alertBox = document.getElementById("alert-box");
                if(alertBox) {
                    if(Math.random() > 0.7) {
                        alertBox.className = "alert danger";
                        alertBox.innerText = "⚠ Bedsore Risk: Right Hip Immobile for 2+ Hours";
                    } else {
                        alertBox.className = "alert safe";
                        alertBox.innerText = "✓ Patient Posture Safe";
                    }
                }

            } catch (err) {
                console.error("Gradio API error:", err);
                alert("Error connecting to TouchGPT Brain: " + err.message);
            }

            btn.innerText = btn.classList.contains("robot-btn") ? "Inject Sensory Data" : "Simulate Next Data Feed";
            btn.disabled = false;
        });
    }
});

function drawHeatmap(zData) {
    const data = [{
        z: zData,
        type: 'heatmap',
        colorscale: 'Viridis',
        showscale: false
    }];
    const layout = {
        margin: { t: 10, b: 10, l: 10, r: 10 },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        xaxis: { showgrid: false, zeroline: false, visible: false },
        yaxis: { showgrid: false, zeroline: false, visible: false, autorange: 'reversed' }
    };
    Plotly.newPlot('heatmap-container', data, layout, {displayModeBar: false});
}

function drawSkeleton(joints) {
    // Joints is a 24x3 array
    const x = joints.map(j => j[0]);
    const y = joints.map(j => j[1]);
    const z = joints.map(j => j[2]);

    const data = [{
        x: x, y: y, z: z,
        mode: 'markers',
        type: 'scatter3d',
        marker: {
            color: '#3b82f6',
            size: 6,
            symbol: 'circle',
            line: { color: '#60a5fa', width: 1 }
        }
    }];

    const layout = {
        margin: { t: 0, b: 0, l: 0, r: 0 },
        paper_bgcolor: 'transparent',
        scene: {
            xaxis: { visible: false },
            yaxis: { visible: false },
            zaxis: { visible: false },
            bgcolor: 'transparent',
            camera: {
                eye: {x: 1.5, y: 1.5, z: 1.5}
            }
        }
    };
    Plotly.newPlot('skeleton-container', data, layout, {displayModeBar: false});
}
