from fastapi.middleware.cors import CORSMiddleware
from .routes import health, home, auth
from fastapi import FastAPI

app = FastAPI(
    title="Nvidia LSTM Forecast API",
    description="Public REST API for accessing Nvidia LSTM Forecast services.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "Nvidia LSTM Forecast",
        "url": "https://github.com/LucasTechAI/nvidia-lstm-forecast",
        "email": "lucas.mendestech@gmail.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(home.router)
app.include_router(health.router)
app.include_router(auth.router)