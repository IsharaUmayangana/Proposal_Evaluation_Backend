from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from constants import ALLOWED_ORIGINS, APP_TITLE
from routers.employment import router as employment_router
from routers.investment import router as investment_router

app = FastAPI(title=APP_TITLE)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(investment_router)
app.include_router(employment_router)
