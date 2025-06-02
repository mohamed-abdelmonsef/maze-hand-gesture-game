async function getPredictedLabel(processed_t) {
  // Convert landmarks to a flat array of [x1, y1, z1, ..., x21, y21, z21]
  const flatLandmarks = processed_t.flatMap(p => [p.x, p.y, p.z]);

  try {
    const response = await fetch("https://skyfsyeogvqx.us-east-1.clawcloudrun.com/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      // Backend expects { data: [[...]] }
      body: JSON.stringify({ data: [flatLandmarks] }),
    });

    if (!response.ok) {
      throw new Error("Network response was not ok");
    }

    const data = await response.json();
    console.log("Prediction response:", data);
    return data.command;


  } catch (error) {
    console.error("Error calling prediction API:", error);
    return null;
  }
}
