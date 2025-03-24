'''
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)

You are free to:
- Share: Copy and redistribute the material in any medium or format.
- Adapt: Remix, transform, and build upon the material.

Under the following terms:
- Attribution: You must give appropriate credit, provide a link to the license, and indicate if changes were made.
- Non-Commercial: You may not use the material for commercial purposes.
- ShareAlike: If you remix, transform, or build upon the material, you must distribute your contributions under the same license.

Full License: https://creativecommons.org/licenses/by-nc-sa/4.0/
'''

# -------------------------------------MODULES-------------------------------------
from fastapi import (
    FastAPI,
    File, 
    UploadFile, 
    HTTPException, 
    Depends,
    APIRouter,
    Response,
    Body,
    Request,
)
from fastapi.exceptions import (
    RequestValidationError,
    ResponseValidationError
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import (
    BaseModel,
    field_validator,
    Field
)
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from typing_extensions import (
    Literal,
    Union,
    List
)
import re
import uuid
import os
import json
from datetime import (
    datetime, 
    timedelta
)
import logging
from dotenv import load_dotenv
import uvicorn
import argparse
from pytz import FixedOffset
from base64 import b64encode
import secrets
import jwt

# Default timezone is Asia/Ho_Chi_Minh

# -------------------------------------UTILS-------------------------------------

def generate_uuid():
    return str(uuid.uuid4())

def key_gen(length: int = 32):
    return secrets.token_bytes(length).hex()

# -------------------------------------CONSTANTS-------------------------------------

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB limit
JWT_SECRET = key_gen()
JWT_ALGORITHM = "HS256"
SCOPES = {
    "attendance": "Access attendance-related APIs",
    "create": "Access user creation APIs",
    "user": "Access user-related APIs",
    "general": "Access general APIs (No routers)"
}

# -------------------------------------LOGGING-------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# -------------------------------------DATABASE-------------------------------------

client = MongoClient(
    os.environ.get("MONGODB_URI"),
    connect=False,
)

# -------------------------------------ARG PARSING-------------------------------------

# Default prod
os.environ.setdefault("PROD", "DEV")

if os.path.exists(".env"):
    load_dotenv(".env")

parser = argparse.ArgumentParser(
    description="Face Recognition API parser",
    prog="Face Recognition - Fastapi backend",
)

parser.add_argument(
    "--prod",
    action="store_true",
    help="Production mode enabled",
)

parser.add_argument(
    "--init-db",
    action="store_true",
    default=False,
    help="Initialize the database with default values",
)

parser.add_argument(
    "--utc",
    type=int,
    default=7,
    help="UTC timezone (integer), default is 7 (Asia/Ho_Chi_Minh)"
)

parser.add_argument(
    "--port",
    type=int,
    default=8000,
    help="Port number, default is 8000"
)

parser.add_argument(
    "--host",
    type=str,
    default="0.0.0.0",
    help="Host, default is 0.0.0.0"
)

parser.add_argument(
    "--authkey",
    type=str,
    default=key_gen(),
    help="Authentication key, default is a random 32-byte key"
)

parser.add_argument(
    "--expire",
    type=int,
    default=30,
    help="Access token expiration time in minutes, default is 30"
)

args = parser.parse_args()

os.environ['PROD'] = "PROD" if args.prod else "DEV"

logging.info(f"Production mode: {os.environ['PROD']}")
logging.info(f"Secret key for authentication: {args.authkey}")

ACCESS_TOKEN_EXPIRE_MINUTES = args.expire
AUTHKEY = args.authkey

if -12 <= args.utc <= 14:
    server_timezone = FixedOffset(args.utc * 60)
else: 
    raise ValueError("Invalid UTC timezone")

# -------------------------------------DEFAULT VALUES-------------------------------------

default_value_dict = {
    'users': {
        'uuid': generate_uuid(),
        'full_name': 'John Doe',
        'created_at': datetime.now(tz=server_timezone),
        'updated_at': datetime.now(tz=server_timezone),
        "default": True
    },
    'history': {
        "default": True,
        'uuid': generate_uuid(),
        'full_name': 'John Doe',
        'added_at': datetime.now(tz=server_timezone),
        "b64_image": b"data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/4QBYRXhpZgAATU0AKgAAAAgAA1IBAAEAAAABAAAAPgIBAAEAAAABAAAARgICAAQAAAABAAABBAIDAQAA"
    }
}

def get_or_create_db(db_name: str, collection_name: str, default_values: bool) -> Database:
    db: Database = client[db_name]
    if collection_name not in db.list_collection_names():
        logging.info(f"Creating collection '{collection_name}' in database '{db_name}'...")
        db[collection_name].insert_one({"init": True})  # Creates DB + collection
        logging.info(f"Collection '{collection_name}' created successfully")
        db[collection_name].delete_one({"init": True})  # Remove placeholder
    if default_values:
        logging.info(F"Deleting default values in '{collection_name}'...")
        db[collection_name].delete_many({"default": True})
        logging.info(f"Inserting default values into '{collection_name}'...")
        db[collection_name].insert_one(default_value_dict[collection_name])
    return db

userdb = get_or_create_db("userdb", "users", args.init_db)
attendance_db = get_or_create_db("attendance_db", "history", args.init_db)
users_col = userdb['users']
history_col = attendance_db['history']
logging.info("Database initialized - Default by now")

# -------------------------------------FASTAPI-------------------------------------

app = FastAPI(
    title="Face Recognition API",
    description="Face Recognition API",
    version="0.0.1UDB1TF1",
    docs_url="/docs" if os.environ['PROD'] == 'DEV' else None,
    #redoc_url=None if os.environ['PROD'] == 'DEV' else None,
    #openapi_url=None if os.environ['PROD'] == 'DEV' else None,
    #debug=True if os.environ['PROD'] == 'DEV' else False,
)

if os.environ['PROD'] == 'DEV':
    mod_value = default_value_dict.copy()
    mod_value["users"]["created_at"] = mod_value["users"]["created_at"].isoformat()
    mod_value["users"]["updated_at"] = mod_value["users"]["updated_at"].isoformat()
    mod_value["history"]["added_at"] = mod_value["history"]["added_at"].isoformat()
    mod_value["history"]["b64_image"] = mod_value["history"]["b64_image"].decode("utf-8")
    del mod_value["users"]["_id"]
    del mod_value["history"]["_id"]

# -------------------------------------DEPENDENCIES-------------------------------------

def can_add_document(uuid):
    """
    Check if we can add a new document with the given UUID.
    Condition: All existing documents with the same UUID must be added before yesterday.
    """
    # Get the most recent document for this UUID
    latest_doc = history_col.find_one(
        {"uuid": uuid},
        sort=[("added_at", -1)]  # Sort by added_at descending (latest first)
    )

    if not latest_doc:
        return True  # No previous document, allow insertion

    latest_added_at = latest_doc["added_at"].replace(tzinfo=server_timezone)
    print(latest_added_at)

    # Calculate the start of today
    today = datetime.now(server_timezone).replace(hour=0, minute=0, second=0, microsecond=0)

    return latest_added_at < today  # Allow if last insert was yesterday

# uuid struct: 8-4-4-4-12
def check_uuid(uuid: str):
    if not re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', uuid):
        raise HTTPException(status_code=400, detail="Invalid UUID")
    if users_col.find_one({"uuid": uuid}) is None:
        raise HTTPException(status_code=404, detail="User not found")
    return uuid

# Dependency to check file size
def check_file(file: UploadFile = File(...)):
    file_size = 0
    for chunk in file.file:
        file_size += len(chunk)
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File size exceeds 10MB limit")
    file.file.seek(0)  # Reset file pointer after reading
    file.filename = secure_filename(file.filename)
    return file

def check_full_name(full_name: str):
    """
    Ensures the full name contains only:
    - Alphabetic characters (A-Z, a-z)
    - Vietnamese Unicode characters
    - Spaces
    
    Raises a ValueError if the name contains invalid characters.
    """
    vietnamese_pattern = r"^[AÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬB-ZĐÉÈẺẼẸÊẾỀỂỄỆGHIÍÌỈĨỊKLMNOÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢPQRSTUÚÙỦŨỤƯỨỪỬỮỰVXYÝỲỶỸỴaáàảãạăắằẳẵặâấầẩẫậb-zđéèẻẽẹêếềểễệghiíìỉĩịklmnoóòỏõọôốồổỗộơớờởỡợpqrstuúùủũụưứừửữựvxyýỳỷỹỵ\s]+$"

    if not re.fullmatch(vietnamese_pattern, full_name):
        raise ValueError(f"Invalid name: '{full_name}'. Name can only contain letters and spaces.")

    return full_name
# -------------------------------------MODELS-------------------------------------

class UserinDB(BaseModel):
    uuid: str
    full_name: str
    created_at: datetime
    updated_at: datetime

class HistoryinDB(BaseModel):
    uuid: str
    full_name: str
    added_at: datetime
    b64_image: str | None

class TokenDict(BaseModel):
    sub: Literal["authenticated"]
    scopes: List[str] = Field(..., description="List of allowed scopes")

    @field_validator("scopes", mode="before")
    @classmethod
    def check_scopes(cls, values):
        if not isinstance(values, list):
            raise ValueError("Scopes must be a list")
        
        invalid_scopes = [scope for scope in values if scope not in SCOPES.keys()]
        if invalid_scopes:
            raise ValueError(f"Invalid scopes: {invalid_scopes}. Allowed scopes: {list(SCOPES.keys())}")
        
        return values
# -------------------------------------AUTH-------------------------------------

def create_access_token(data: TokenDict):
    raw_data = data.model_dump()
    expire = datetime.now(server_timezone) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    raw_data["exp"] = expire
    encoded_jwt = jwt.encode(raw_data, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt

# -------------------------------------ROUTES-------------------------------------

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

attendance_api = APIRouter(
    prefix="/attendance",
    tags=["attendance"],
)

create_api = APIRouter(
    prefix="/create",
    tags=["create"],
)

user_api = APIRouter(
    prefix="/user",
    tags=["user"],
)

auth_api = APIRouter(
    prefix="/auth",
    tags=["auth"],
)

# -------------------------------------EXCEPTION HANDLER-------------------------------------

def value_error_handler(request: Request, exc: ValueError):
    return Response(
        status_code=400,
        content=json.dumps({"detail": str(exc), "type": "ValueError"}),
        media_type="application/json"
    )

def http_error_handler(request: Request, exc: HTTPException):
    return Response(
        status_code=exc.status_code,
        content=json.dumps({"detail": exc.detail, "type": "HTTPException"}),
        media_type="application/json"
    )

def general_error_handler(request: Request, exc: Exception):
    return Response(
        status_code=500,
        content=json.dumps({"detail": str(exc), "type": "Exception"}),
        media_type="application/json"
    )

def request_validation_error_handler(request: Request, exc: RequestValidationError):
    return Response(
        status_code=400,
        content=json.dumps({"detail": exc.errors(), "type": "RequestValidationError"}),
        media_type="application/json"
    )

def response_validation_error_handler(request: Request, exc: ResponseValidationError):
    return Response(
        status_code=400,
        content=json.dumps({"detail": exc.errors(), "type": "ResponseValidationError"}),
        media_type="application/json"
    )


app.add_exception_handler(RequestValidationError, request_validation_error_handler)
app.add_exception_handler(ResponseValidationError, response_validation_error_handler)
app.add_exception_handler(Exception, general_error_handler)
app.add_exception_handler(HTTPException, http_error_handler)
app.add_exception_handler(ValueError, value_error_handler)

# -------------------------------------ATTENDANCE-------------------------------------

@attendance_api.post("/")
def attendance(
    file: UploadFile = Depends(check_file),
    uuid: str = Depends(check_uuid),
):
    if not can_add_document(uuid):
        raise HTTPException(status_code=400, detail="Exceeded daily limit for this UUID")
    # Check if file is an image (and also if it's a valid image & base64)
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image")
    # Auth check
    ... # TODO: Add auth check
    # Save image to history col
    history_col.insert_one(
        {
            "uuid": uuid,
            "full_name": users_col.find_one({"uuid": uuid})["full_name"],
            "added_at": datetime.now(tz=server_timezone),
            "b64_image": file.file.read(),
            "default": False
        }
    )
    return Response(
        status_code=200,
        content=json.dumps({"message": "Attendance recorded"}),
        media_type="application/json"
    )

    

@attendance_api.get("/", response_model=Union[UserinDB, list[HistoryinDB]])
def get_attendance(
    uuid: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    return_image: bool = False
):
    # Dependency check
    if uuid:
        uuid = check_uuid(uuid)
        if not start_date and not end_date:
            return [
                HistoryinDB(
                    **{
                        **doc,
                        "b64_image": b64encode(doc["b64_image"]).decode("utf-8")
                        if return_image else None
                    }
                )
                for doc in history_col.find({"uuid": uuid}, {"_id": 0})
            ]
        elif start_date and end_date:
            query = {
                "uuid": uuid,
                "added_at": {
                    "$gte": datetime.fromisoformat(start_date),
                    "$lte": datetime.fromisoformat(end_date)
                }
            }
            return [
                HistoryinDB(
                    **{
                        **doc,
                        "b64_image": b64encode(doc["b64_image"]).decode("utf-8")
                        if return_image else None
                    }
                )
                for doc in history_col.find(query, {"_id": 0})
            ]
    else:
        if start_date and end_date:
            query = {
                "added_at": {
                    "$gte": datetime.fromisoformat(start_date),
                    "$lte": datetime.fromisoformat(end_date)
                }
            }
            return [
                HistoryinDB(
                    **{
                        **doc,
                        "b64_image": b64encode(doc["b64_image"]).decode("utf-8")
                        if return_image else None
                    }
                )
                for doc in history_col.find(query, {"_id": 0})
            ]
        else:
            raise HTTPException(status_code=400, detail="UUID or date range must be provided")
    raise HTTPException(status_code=400, detail="Invalid query")

# -------------------------------------CREATE-------------------------------------

@create_api.post("/user")
def create_user(
    full_name: str = Depends(check_full_name),
):
    uuid = generate_uuid()
    while users_col.find_one({"uuid": uuid}):
        uuid = generate_uuid() 
    users_col.insert_one(
        {
            "uuid": uuid,
            "full_name": full_name,
            "created_at": datetime.now(tz=server_timezone),
            "updated_at": datetime.now(tz=server_timezone),
            "default": False
        }
    )
    return Response(
        status_code=200,
        content=json.dumps({"uuid": uuid}),
        media_type="application/json"
    )
    
# -------------------------------------USER-------------------------------------

@user_api.get("/all", response_model=list[UserinDB])
def get_all_users():
    return [
        UserinDB(**doc)
        for doc in users_col.find({}, {"_id": 0})
    ]

@user_api.get("/{uuid}", response_model=UserinDB)
def get_user(uuid: str = Depends(check_uuid)):
    return UserinDB(**users_col.find_one({"uuid": uuid}, {"_id": 0}))

# -------------------------------------AUTH-------------------------------------

@auth_api.post("/token", summary="Generate JWT Token", description="Authenticate using only the password and receive a JWT token.")
def login(password: str = Body(..., embed=True)):
    """
    Authenticate using only the password and return a JWT token.
    """
    if password != AUTHKEY:
        raise HTTPException(status_code=401, detail="Invalid password")
    access_token = create_access_token(
        data=TokenDict(
            sub="authenticated",
            scopes=list(SCOPES.keys())
        )
    )
    return {"access_token": access_token, "token_type": "bearer"}

# -------------------------------------ROUTES-------------------------------------

if os.environ['PROD'] == 'DEV':
    @app.post("/values")
    def create_api_default_values():
        return Response(
            status_code=200,
            content=json.dumps(mod_value),
            media_type="application/json"
        )

app.include_router(attendance_api)
app.include_router(create_api)
app.include_router(user_api)
app.include_router(auth_api)

# -------------------------------------MAIN-------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=True if os.environ['PROD'] == 'DEV' else False,
    )