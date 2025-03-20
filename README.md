# Setup

Download MongoDB Compass if you don't have it (Windows), or start your local MongoDB instance on your computer.

Run the following command to install the required modules:
```bash
pip install -r requirements.txt
```

# Run

To run, simply use:
```bash
python main.py
```

## There are many args:
- `--init-db`: Initialize default values for the database (default: false).
- `--prod`: Run in production mode (default: false).
- `--utc <int>`: Set UTC offset from -12 to 14 (default: 7).
- `--port <int>`: Set the port (default: 8000).
- `--host <str>`: Set the host (default: 0.0.0.0).

## Environments:
- **MONGO_URI**: The URI for the database.
- **PROD**: The production state (DEV or PROD).

# TODO
- [x] Create user
- [x] Query UUID
- [x] Query range
- [x] Submit attendence
- [ ] Exception handler

> [!WARNING]  
> This API is currently in the development stage. Expect bugs.