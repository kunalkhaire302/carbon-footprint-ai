from backend.app import app

if __name__ == "__main__":
    settings = app.config["SETTINGS"]
    app.run(host=settings.host, port=settings.port, debug=settings.environment == "development")
