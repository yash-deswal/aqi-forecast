const AQI_CATEGORIES = [
  {
    max: 50,
    label: "Good",
    desc: "Air quality is satisfactory. Little to no risk.",
    cls: "aqi-good",
    clr: "#22c55e",
  },
  {
    max: 100,
    label: "Satisfactory",
    desc: "Acceptable air quality. Sensitive individuals may experience minor effects.",
    cls: "aqi-moderate",
    clr: "#facc15",
  },
  {
    max: 200,
    label: "Moderately Polluted",
    desc: "People with respiratory or heart disease may experience discomfort.",
    cls: "aqi-sensitive",
    clr: "#f97316",
  },
  {
    max: 300,
    label: "Poor",
    desc: "Breathing discomfort for most people. Avoid prolonged outdoor activity.",
    cls: "aqi-unhealthy",
    clr: "#ef4444",
  },
  {
    max: 400,
    label: "Very Poor",
    desc: "Serious breathing difficulties. Stay indoors if possible.",
    cls: "aqi-very-bad",
    clr: "#a21caf",
  },
  {
    max: Infinity,
    label: "Severe / Hazardous",
    desc: "Health emergency. Everyone is likely to be affected significantly.",
    cls: "aqi-hazardous",
    clr: "#7f1d1d",
  },
];

/**
 * Determine category metadata given a numeric AQI value.
 * @param {number} aqi
 * @returns {Object} matching category object
 */
function getAqiCategory(aqi) {
  return AQI_CATEGORIES.find((cat) => aqi <= cat.max);
}

const form          = document.getElementById("aqi-form");
const predictBtn    = document.getElementById("predict-btn");
const resultSection = document.getElementById("result-section");
const aqiBadge      = document.getElementById("aqi-badge");
const aqiValue      = document.getElementById("aqi-value");
const aqiCategory   = document.getElementById("aqi-category");
const aqiDesc       = document.getElementById("aqi-desc");
const errorBox      = document.getElementById("error-box");

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  resultSection.hidden = true;
  errorBox.hidden      = true;

  const day   = parseInt(document.getElementById("day").value, 10);
  const month = parseInt(document.getElementById("month").value, 10);
  const year  = parseInt(document.getElementById("year").value, 10);

  const jsDay = new Date(year, month - 1, day).getDay(); // 0=Sun
  const day_of_week = (jsDay + 6) % 7;                  // 0=Mon … 6=Sun

  const payload = {
    city:        document.getElementById("city").value,
    day,
    month,
    year,
    day_of_week,
    AQI_lag1:    parseFloat(document.getElementById("AQI_lag1").value),
    AQI_lag2:    parseFloat(document.getElementById("AQI_lag2").value),
    AQI_lag3:    parseFloat(document.getElementById("AQI_lag3").value),
  };

  if (!payload.city) {
    showError("Please select a city.");
    return;
  }

  setLoading(true);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await response.json();

    if (!response.ok) {
      showError(data.error || "Prediction failed. Please try again.");
      return;
    }

    displayResult(data.aqi);

  } catch (err) {
    showError("Network error: Could not reach the server. Is Flask running?");
    console.error(err);
  } finally {
    setLoading(false);
  }
});

function displayResult(aqi) {
  const cat = getAqiCategory(aqi);

  aqiValue.textContent    = aqi.toFixed(1);
  aqiCategory.textContent = cat.label;
  aqiDesc.textContent     = cat.desc;

  aqiBadge.className = `aqi-badge ${cat.cls}`;
  aqiValue.style.color    = cat.clr;
  aqiCategory.style.color = cat.clr;
  aqiBadge.style.borderColor  = cat.clr;
  aqiBadge.style.background   = hexToRgba(cat.clr, 0.08);
  aqiBadge.style.boxShadow    = `0 0 32px ${hexToRgba(cat.clr, 0.22)}`;

  resultSection.hidden = false;

  resultSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function showError(msg) {
  errorBox.textContent = msg;
  errorBox.hidden      = false;
}

function setLoading(on) {
  predictBtn.disabled = on;
  predictBtn.classList.toggle("loading", on);
}

function hexToRgba(hex, alpha) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}
