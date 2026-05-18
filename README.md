# SEIR Epidemic Simulator

A Python web application built with Streamlit for simulating infectious disease dynamics using the SEIR model.

## Features
- Interactive parameter control (β, σ, γ, Population, Days).
- Real-time simulation using `scipy.odeint`.
- Visualizations powered by `Plotly`.
- Peak statistics (Day and Count).

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the App

Start the application using:
```bash
streamlit run app.py
```

## Model Logic
The SEIR model divides the population into four compartments:
- **S**: Susceptible
- **E**: Exposed (Infected but not yet infectious)
- **I**: Infectious
- **R**: Recovered (with immunity)

The model is solved using a system of ordinary differential equations (ODEs).
