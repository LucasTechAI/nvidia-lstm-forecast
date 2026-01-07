from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    message: str

    class Config:
        title = "HealthResponse"
        json_schema_extra = {
            "example": {
                "status": "ok",
                "message": "API is healthy and database is connected.",
            }
        }


class Health:
    docs = {
        "summary": "Health check endpoint",
        "response_model": HealthResponse,
        "responses": {
            200: {
                "description": "Service is healthy.",
                "content": {
                    "application/json": {
                        "example": {
                            "status": "ok",
                            "message": "API is healthy and database is connected.",
                        }
                    }
                },
            },
            503: {
                "description": "Service is not healthy.",
                "content": {
                    "application/json": {"example": {"detail": "Service is down"}}
                },
            },
            404: {
                "description": "No matching books found",
                "content": {
                    "application/json": {
                        "example": {"detail": "No matching books found"}
                    }
                },
            },
        },
    }