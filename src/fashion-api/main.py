# main.py
import os
import time
import logging
import random
import smtplib
import re # For password validation
from email.mime.text import MIMEText
from datetime import datetime, timedelta, timezone # timezone imported
from typing import List, Dict, Any, Optional, Tuple
from contextlib import asynccontextmanager
from io import BytesIO
import enum

from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Form, Request, BackgroundTasks, APIRouter
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field, validator
from pydantic_settings import BaseSettings
from sqlalchemy import create_engine, text, Column, Integer, String, Float, DateTime, ForeignKey, Table, MetaData, inspect, Enum as SAEnum, CheckConstraint, Boolean
from sqlalchemy.orm import sessionmaker, relationship, Mapped, mapped_column
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
# from sqlalchemy_utils import EmailType, PasswordType # Not directly used by auth, can be kept if other parts need it
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack, csr_matrix
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch # Added for CLIP feature extraction
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
import numpy as np
import jwt
from passlib.context import CryptContext
import razorpay
from jinja2 import Environment, FileSystemLoader, select_autoescape
from email_validator import validate_email, EmailNotValidError

# For Collaborative Filtering
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

# --- Configuration ---
class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite:///./database/fashion.db"
    STATIC_DIR: str = "static"
    CHROMA_DB_PATH: str = "./database/chroma_fashion"
    CORS_ORIGINS: List[str] = ["http://localhost:5173", "http://localhost:5174", "http://localhost:3000"]

    SECRET_KEY: str = "your_super_secret_key_for_jwt_replace_me" # CHANGE THIS!
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 # 1 day

    RAZORPAY_KEY_ID: Optional[str] = None
    RAZORPAY_KEY_SECRET: Optional[str] = None

    MAIL_USERNAME: Optional[EmailStr] = None
    MAIL_PASSWORD: Optional[str] = None
    MAIL_FROM: Optional[EmailStr] = None
    MAIL_SERVER: str = "smtp.gmail.com"
    MAIL_PORT: int = 587
    OTP_EXPIRE_MINUTES: int = 10

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- SQLAlchemy Setup ---
engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
metadata = MetaData()

# --- Jinja2 for Email Templates ---
template_env = Environment(
    loader=FileSystemLoader("templates"),
    autoescape=select_autoescape(['html', 'xml'])
)

# --- Password Hashing ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")


# --- SQLAlchemy Models ---
class ClothingItemDB(Base):
    __tablename__ = "clothing_items"
    id = Column(Integer, primary_key=True, index=True)
    gender = Column(String)
    masterCategory = Column(String)
    subCategory = Column(String)
    articleType = Column(String)
    baseColour = Column(String)
    season = Column(String)
    usage = Column(String)
    productDisplayName = Column(String)
    price = Column(Float, CheckConstraint('price >= 0'))
    image_filename = Column(String)

    reviews: Mapped[List["ReviewDB"]] = relationship("ReviewDB", back_populates="product")

class UserDB(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String, nullable=False)
    full_name = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    reviews: Mapped[List["ReviewDB"]] = relationship("ReviewDB", back_populates="user")
    orders: Mapped[List["OrderDB"]] = relationship("OrderDB", back_populates="user")
    cart: Mapped[Optional["CartDB"]] = relationship("CartDB", back_populates="user", uselist=False)

class OtpDB(Base):
    __tablename__ = "otps"
    id = Column(Integer, primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String, index=True)
    otp_code = Column(String)
    expires_at = Column(DateTime)
    used = Column(Boolean, default=False)
    # created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc)) # Optional

class CartDB(Base):
    __tablename__ = "carts"
    id = Column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), unique=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    user: Mapped["UserDB"] = relationship("UserDB", back_populates="cart")
    items: Mapped[List["CartItemDB"]] = relationship("CartItemDB", back_populates="cart", cascade="all, delete-orphan")

class CartItemDB(Base):
    __tablename__ = "cart_items"
    id = Column(Integer, primary_key=True, index=True)
    cart_id: Mapped[int] = mapped_column(ForeignKey("carts.id"))
    product_id: Mapped[int] = mapped_column(ForeignKey("clothing_items.id"))
    quantity = Column(Integer, CheckConstraint('quantity > 0'), default=1)
    added_at = Column(DateTime, default=lambda: datetime.now(timezone.utc)) # Updated

    cart: Mapped["CartDB"] = relationship("CartDB", back_populates="items")
    product: Mapped["ClothingItemDB"] = relationship("ClothingItemDB")

class OrderStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    PAYMENT_FAILED = "payment_failed"

class OrderDB(Base):
    __tablename__ = "orders"
    id = Column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    total_amount = Column(Float, CheckConstraint('total_amount >= 0'), nullable=False)
    status = Column(SAEnum(OrderStatus), default=OrderStatus.PENDING)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    razorpay_order_id = Column(String, nullable=True, unique=True)
    shipping_address = Column(String, nullable=True)

    user: Mapped["UserDB"] = relationship("UserDB", back_populates="orders")
    items: Mapped[List["OrderItemDB"]] = relationship("OrderItemDB", back_populates="order", cascade="all, delete-orphan")

class OrderItemDB(Base):
    __tablename__ = "order_items"
    id = Column(Integer, primary_key=True, index=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id"))
    product_id: Mapped[int] = mapped_column(ForeignKey("clothing_items.id"))
    quantity = Column(Integer, CheckConstraint('quantity > 0'), nullable=False)
    price_at_purchase = Column(Float, CheckConstraint('price_at_purchase >= 0'), nullable=False)

    order: Mapped["OrderDB"] = relationship("OrderDB", back_populates="items")
    product: Mapped["ClothingItemDB"] = relationship("ClothingItemDB")

class ReviewDB(Base):
    __tablename__ = "reviews"
    id = Column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    product_id: Mapped[int] = mapped_column(ForeignKey("clothing_items.id"))
    rating = Column(Integer, CheckConstraint('rating >= 1 AND rating <= 5'), nullable=False)
    comment = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc)) # Updated

    user: Mapped["UserDB"] = relationship("UserDB", back_populates="reviews")
    product: Mapped["ClothingItemDB"] = relationship("ClothingItemDB", back_populates="reviews")


def init_db():
    logger.info("Ensuring all database tables are created...")
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables ensured to exist.")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")


# --- Pydantic Models (Schemas) ---

# Item Schemas
class ItemBase(BaseModel):
    gender: str
    masterCategory: str
    subCategory: str
    articleType: str
    baseColour: Optional[str] = None
    season: str
    usage: str
    productDisplayName: str
    price: Optional[float] = None

class ItemCreate(ItemBase):
    pass

class Item(ItemBase):
    id: int
    image_url: Optional[str] = None
    average_rating: Optional[float] = None
    total_reviews: Optional[int] = None

    class Config:
        from_attributes = True

class ProductsResponse(BaseModel):
    products: List[Item]

# User Schemas
def validate_password_complexity(password: str) -> str:
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters long.")
    if not re.search(r"[A-Z]", password):
        raise ValueError("Password must contain at least one uppercase letter.")
    if not re.search(r"[a-z]", password):
        raise ValueError("Password must contain at least one lowercase letter.")
    if not re.search(r"[0-9]", password):
        raise ValueError("Password must contain at least one digit.")
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        raise ValueError("Password must contain at least one special character.")
    return password

class UserBase(BaseModel):
    email: EmailStr
    full_name: Optional[str] = None

class UserCreate(UserBase):
    password: str

    @validator("password")
    def password_complexity_check(cls, v):
        return validate_password_complexity(v)

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class User(UserBase):
    id: int
    is_active: bool
    is_verified: bool
    created_at: datetime

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str
    user: User

class TokenData(BaseModel):
    email: EmailStr

class OTPRequest(BaseModel):
    email: EmailStr

class OTPVerify(BaseModel):
    email: EmailStr
    otp: str

class PasswordChangeRequest(BaseModel):
    old_password: str
    new_password: str

    @validator("new_password")
    def new_password_complexity_check(cls, v):
        return validate_password_complexity(v)

class PasswordResetRequest(BaseModel):
    email: EmailStr
    otp: str
    new_password: str

    @validator("new_password")
    def new_password_complexity_check(cls, v):
        return validate_password_complexity(v)


# Cart Schemas
class CartItemBase(BaseModel):
    product_id: int
    quantity: int = Field(1, gt=0)

class CartItemCreate(CartItemBase):
    pass

class CartItem(CartItemBase):
    id: int
    product: Item
    added_at: datetime

    class Config:
        from_attributes = True

class Cart(BaseModel):
    id: int
    user_id: int
    items: List[CartItem] = []
    total_cart_price: float = 0.0
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Order Schemas
class OrderItemBase(BaseModel):
    product_id: int
    quantity: int
    price_at_purchase: float

class OrderItem(OrderItemBase):
    id: int
    product: Item

    class Config:
        from_attributes = True

class OrderBase(BaseModel):
    shipping_address: Optional[str] = None

class OrderCreate(OrderBase):
    pass

class Order(OrderBase):
    id: int
    user_id: int
    total_amount: float
    status: OrderStatus
    created_at: datetime
    updated_at: datetime
    razorpay_order_id: Optional[str] = None
    items: List[OrderItem]

    class Config:
        from_attributes = True

# Review Schemas
class ReviewBase(BaseModel):
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = None

class ReviewCreate(ReviewBase):
    product_id: int

class Review(ReviewBase):
    id: int
    user_id: int
    product_id: int
    user_full_name: Optional[str] = "Anonymous"
    created_at: datetime

    class Config:
        from_attributes = True

# Recommendation Schemas
class RecommendationResponse(BaseModel):
    content_based: List[Item] = []
    collaborative_filtering: Optional[List[Item]] = None
    # other_recommendations: Optional[List[Item]] = None

class ProductPageResponse(BaseModel):
    product: Item
    reviews: List[Review]
    recommendations: RecommendationResponse

# Search Schemas
class SearchResult(BaseModel):
    images: List[Dict[str, Any]]


# --- ML Model Global Store ---
class MLModelStore:
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.clip_model: Optional[CLIPModel] = None
        self.clip_processor: Optional[CLIPProcessor] = None
        self.chroma_client: Optional[chromadb.Client] = None
        self.fashion_collection: Optional[chromadb.API.Collection] = None
        self.cf_model_svd: Optional[SVD] = None
        self.cf_trainset = None

ml_store = MLModelStore()

# --- Helper Functions & Utilities ---

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Authentication Utilities
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

def send_email(to_email: str, subject: str, html_content: str):
    if not all([settings.MAIL_USERNAME, settings.MAIL_PASSWORD, settings.MAIL_FROM]):
        logger.error("Mail settings not configured. Cannot send email.")
        return False
    msg = MIMEText(html_content, 'html')
    msg['Subject'] = subject
    msg['From'] = settings.MAIL_FROM
    msg['To'] = to_email
    try:
        with smtplib.SMTP(settings.MAIL_SERVER, settings.MAIL_PORT) as server:
            server.starttls()
            server.login(settings.MAIL_USERNAME, settings.MAIL_PASSWORD)
            server.sendmail(settings.MAIL_FROM, [to_email], msg.as_string())
        logger.info(f"Email sent to {to_email} with subject: {subject}")
        return True
    except Exception as e:
        logger.error(f"Failed to send email to {to_email}: {e}")
        return False

def generate_otp_code(length: int = 6) -> str:
    return "".join([str(random.randint(0, 9)) for _ in range(length)])

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> UserDB:
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        email: Optional[str] = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired", headers={"WWW-Authenticate": "Bearer"})
    except jwt.PyJWTError:
        raise credentials_exception
    user = db.query(UserDB).filter(UserDB.email == token_data.email).first()
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: UserDB = Depends(get_current_user)) -> UserDB:
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user. Please contact support.")
    return current_user

async def get_current_verified_user(current_user: UserDB = Depends(get_current_active_user)) -> UserDB:
    if not current_user.is_verified:
        raise HTTPException(status_code=403, detail="User not verified. Please verify your email address.")
    return current_user


# Razorpay Client
razorpay_client = None
if settings.RAZORPAY_KEY_ID and settings.RAZORPAY_KEY_SECRET:
    razorpay_client = razorpay.Client(auth=(settings.RAZORPAY_KEY_ID, settings.RAZORPAY_KEY_SECRET))
else:
    logger.warning("Razorpay credentials not found. Payment gateway will not function.")


# --- Database Interaction / CRUD Operations ---
def load_products_df(db: Session) -> pd.DataFrame:
    logger.info("Loading product data from database into DataFrame...")
    start_time = time.time()
    items_db = db.query(ClothingItemDB).all()
    if not items_db:
        logger.warning("No items found in clothing_items table.")
        return pd.DataFrame()
    df = pd.DataFrame([item.__dict__ for item in items_db])
    df["id"] = df["id"].astype(str)
    df["image_url"] = df.apply(lambda row: f"/static/images/{row['id']}.jpg", axis=1) # Assumes image name is id.jpg
    logger.info(f"Product data loaded in {time.time() - start_time:.2f} seconds. Shape: {df.shape}")
    return df

def get_product_db(db: Session, item_id: int) -> Optional[ClothingItemDB]:
    return db.query(ClothingItemDB).filter(ClothingItemDB.id == item_id).first()

def get_product_with_average_rating(db: Session, item_id: int) -> Optional[Item]:
    product_db = get_product_db(db, item_id)
    if not product_db:
        return None
    reviews = db.query(ReviewDB.rating).filter(ReviewDB.product_id == item_id).all()
    avg_rating = None
    total_reviews = len(reviews)
    if total_reviews > 0:
        avg_rating = sum(r.rating for r in reviews) / total_reviews
    product_data = product_db.__dict__
    product_data["image_url"] = f"/static/images/{product_db.id}.jpg"
    product_data["average_rating"] = avg_rating
    product_data["total_reviews"] = total_reviews
    return Item.from_orm(product_data)


# --- Image Processing ---
def predict_attributes_from_image(image: Image.Image) -> dict:
    if not ml_store.clip_model or not ml_store.clip_processor or ml_store.df is None:
        logger.error("CLIP model/processor or DataFrame not initialized for prediction.")
        return {}
    logger.info("Predicting attributes from image (simplified)...")
    try:
        article_types = ml_store.df["articleType"].unique().tolist()
        if not article_types: article_types = ["Unknown"]
        inputs = ml_store.clip_processor(text=article_types, images=image, return_tensors="pt", padding=True, truncation=True)
        outputs = ml_store.clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        predicted_index = probs.argmax().item()
        return {"articleType": article_types[predicted_index], "gender": "Unisex", "usage": "Casual", "season": "All", "baseColour": "Unknown"}
    except Exception as e:
        logger.error(f"Error in predict_attributes_from_image: {e}")
        return {}


# --- Recommendation Engines ---
def get_content_based_recommendations(item_id: int, top_n: int = 5) -> List[Item]:
    if not ml_store.fashion_collection or ml_store.df is None:
        logger.warning("ChromaDB collection or product DataFrame not loaded. Cannot get content-based recs.")
        return []
    target_item_results = ml_store.fashion_collection.get(ids=[str(item_id)], include=["embeddings"])
    embeddings_list = target_item_results.get("embeddings")
    if embeddings_list is None or len(embeddings_list) == 0:
        logger.warning(f"No embeddings list found or it's empty for item {item_id} in ChromaDB.")
        return []
    first_embedding = embeddings_list[0]
    if first_embedding is None or (hasattr(first_embedding, '__len__') and len(first_embedding) == 0):
        logger.warning(f"First embedding for item {item_id} in ChromaDB is None or empty.")
        return []
    target_embedding = first_embedding
    try:
        similar_items_results = ml_store.fashion_collection.query(
            query_embeddings=[target_embedding],
            n_results=top_n + 5,
            include=["metadatas", "distances"]
        )
    except Exception as e:
        logger.error(f"Error querying ChromaDB for content-based recs: {e}")
        return []
    recommendations = []
    if similar_items_results and similar_items_results.get("ids") and similar_items_results["ids"][0]:
        db = next(get_db())
        try:
            for res_id_str, distance in zip(similar_items_results["ids"][0], similar_items_results["distances"][0]):
                if not res_id_str: continue
                try:
                    res_id = int(res_id_str)
                except ValueError:
                    logger.warning(f"Skipping non-integer ID from ChromaDB: {res_id_str}")
                    continue
                if res_id == item_id: continue
                product_item = get_product_with_average_rating(db, res_id)
                if product_item:
                    recommendations.append(product_item)
                if len(recommendations) >= top_n:
                    break
        finally:
            db.close()
    logger.info(f"Content-based recommendations for item {item_id}: found {len(recommendations)} items.")
    return recommendations

def train_cf_model(db: Session):
    logger.info("Training Collaborative Filtering model (SVD)...")
    reviews_data = db.query(ReviewDB.user_id, ReviewDB.product_id, ReviewDB.rating).all()
    if not reviews_data:
        logger.warning("No review data available to train CF model.")
        ml_store.cf_model_svd = None
        ml_store.cf_trainset = None
        return
    df_reviews = pd.DataFrame(reviews_data, columns=['userID', 'itemID', 'rating'])
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df_reviews[['userID', 'itemID', 'rating']], reader)
    ml_store.cf_trainset = data.build_full_trainset()
    ml_store.cf_model_svd = SVD(n_factors=50, n_epochs=20, random_state=42, lr_all=0.005, reg_all=0.02)
    ml_store.cf_model_svd.fit(ml_store.cf_trainset)
    logger.info("Collaborative Filtering model trained successfully.")

def get_collaborative_filtering_recommendations(user_id: int, top_n: int = 5) -> List[Item]:
    if not ml_store.cf_model_svd or not ml_store.cf_trainset or ml_store.df is None:
        logger.warning("CF model not trained or product df not available. Cannot get CF recs.")
        return []
    try:
        user_inner_id = ml_store.cf_trainset.to_inner_uid(user_id)
        items_rated_by_user_inner_ids = [iid for (iid, _) in ml_store.cf_trainset.ur[user_inner_id]]
    except ValueError:
        logger.info(f"User {user_id} not in CF trainset. Cannot generate personalized CF recs for now.")
        return []
    all_item_ids_str = ml_store.df['id'].unique().tolist()
    predictions = []
    for item_id_str in all_item_ids_str:
        try:
            item_id = int(item_id_str)
            item_inner_id = ml_store.cf_trainset.to_inner_iid(item_id)
            if item_inner_id not in items_rated_by_user_inner_ids:
                prediction = ml_store.cf_model_svd.predict(user_id, item_id)
                predictions.append((item_id, prediction.est))
        except ValueError: continue
        except Exception as e:
            logger.error(f"Error predicting for user {user_id}, item {item_id_str}: {e}")
            continue
    predictions.sort(key=lambda x: x[1], reverse=True)
    recommended_item_ids = [item_id for item_id, _ in predictions[:top_n + 10]]
    recommendations = []
    db = next(get_db())
    try:
        for rec_id in recommended_item_ids:
            product_item = get_product_with_average_rating(db, rec_id)
            if product_item:
                recommendations.append(product_item)
            if len(recommendations) >= top_n:
                break
    finally:
        db.close()
    logger.info(f"Collaborative filtering recommendations for user {user_id}: found {len(recommendations)} items.")
    return recommendations


# --- Lifespan Management (Startup & Shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup...")
    init_db()
    db_session = SessionLocal()
    try:
        ml_store.df = load_products_df(db_session)
    finally:
        db_session.close()
    logger.info("Initializing CLIP model...")
    start_time = time.time()
    try:
        ml_store.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        ml_store.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        logger.info(f"CLIP model initialized in {time.time() - start_time:.2f} seconds.")
    except Exception as e:
        logger.error(f"Failed to initialize CLIP model: {e}")
    logger.info(f"Initializing ChromaDB client from path: {settings.CHROMA_DB_PATH}")
    try:
        ml_store.chroma_client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
        try:
            ml_store.fashion_collection = ml_store.chroma_client.get_collection(
                name="fashion",
                embedding_function=OpenCLIPEmbeddingFunction()
            )
            logger.info(f"ChromaDB 'fashion' collection loaded. Count: {ml_store.fashion_collection.count()}")
        except Exception as e:
             logger.warning(f"ChromaDB 'fashion' collection not found or error: {e}. Search/Content-based recs might fail.")
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB client: {e}")
    db_session_for_cf = SessionLocal()
    try:
        train_cf_model(db_session_for_cf)
    finally:
        db_session_for_cf.close()
    logger.info("Application startup complete.")
    yield
    logger.info("Application shutdown.")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Fashion Recommendation API",
    description="E-commerce API with ML recommendations, auth, and payments.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=settings.STATIC_DIR), name="static")


# --- API Routers ---

# Authentication Router
auth_router = APIRouter(prefix="/api/auth", tags=["Authentication"])

@auth_router.post("/register", response_model=User)
async def register_user(user_in: UserCreate, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    existing_user = db.query(UserDB).filter(UserDB.email == user_in.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered.")
    try:
        valid_email = validate_email(user_in.email, check_deliverability=False)
        email = valid_email.normalized
    except EmailNotValidError:
        raise HTTPException(status_code=400, detail="Invalid email format.")
    hashed_password = get_password_hash(user_in.password)
    db_user = UserDB(
        email=email,
        hashed_password=hashed_password,
        full_name=user_in.full_name,
        is_active=True,
        is_verified=False
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    db.query(OtpDB).filter(OtpDB.email == db_user.email, OtpDB.used == False).update(
        {"used": True, "expires_at": datetime.now(timezone.utc)}
    )
    otp_code = generate_otp_code()
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=settings.OTP_EXPIRE_MINUTES)
    db_otp = OtpDB(email=db_user.email, otp_code=otp_code, expires_at=expires_at)
    db.add(db_otp)
    db.commit()
    try:
        email_template = template_env.get_template("email_otp_verification.html")
        html_content = email_template.render(
            otp_code=otp_code,
            user_name=db_user.full_name or db_user.email,
            otp_expire_minutes=settings.OTP_EXPIRE_MINUTES
        )
        background_tasks.add_task(send_email, db_user.email, "Verify Your Email Address", html_content)
    except Exception as e:
        logger.error(f"Failed to render or send OTP email template during registration: {e}")
    return db_user

@auth_router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(UserDB).filter(UserDB.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Account is inactive. Please contact support.")
    if not user.is_verified:
        raise HTTPException(status_code=403, detail="Account not verified. Please check your email for OTP or request a new one.")
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    user_data = User.from_orm(user)
    return {"access_token": access_token, "token_type": "bearer", "user": user_data}

@auth_router.post("/verify-otp", response_model=Dict[str, str])
async def verify_otp(otp_data: OTPVerify, db: Session = Depends(get_db)):
    user = db.query(UserDB).filter(UserDB.email == otp_data.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    if user.is_verified:
        raise HTTPException(status_code=400, detail="User already verified.")
    db_otp = db.query(OtpDB).filter(
        OtpDB.email == otp_data.email,
        OtpDB.otp_code == otp_data.otp,
        OtpDB.used == False,
        OtpDB.expires_at > datetime.now(timezone.utc)
    ).order_by(OtpDB.id.desc()).first()
    if not db_otp:
        raise HTTPException(status_code=400, detail="Invalid or expired OTP.")
    user.is_verified = True
    db_otp.used = True
    db.query(OtpDB).filter(
        OtpDB.email == user.email,
        OtpDB.id != db_otp.id,
        OtpDB.used == False
    ).update({"used": True, "expires_at": datetime.now(timezone.utc)})
    db.commit()
    db.refresh(user)
    return {"message": "Email verified successfully. You can now login."}

@auth_router.post("/resend-otp", response_model=Dict[str, str])
async def resend_verification_otp(otp_request: OTPRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    user = db.query(UserDB).filter(UserDB.email == otp_request.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    if user.is_verified:
        raise HTTPException(status_code=400, detail="User already verified.")
    db.query(OtpDB).filter(
        OtpDB.email == user.email,
        OtpDB.used == False
    ).update({"used": True, "expires_at": datetime.now(timezone.utc)})
    otp_code = generate_otp_code()
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=settings.OTP_EXPIRE_MINUTES)
    db_otp = OtpDB(email=user.email, otp_code=otp_code, expires_at=expires_at)
    db.add(db_otp)
    db.commit()
    try:
        email_template = template_env.get_template("email_otp_verification.html")
        html_content = email_template.render(
            otp_code=otp_code,
            user_name=user.full_name or user.email,
            otp_expire_minutes=settings.OTP_EXPIRE_MINUTES
        )
        background_tasks.add_task(send_email, user.email, "Verify Your Email Address", html_content)
    except Exception as e:
        logger.error(f"Failed to render or send OTP email template for resend: {e}")
    return {"message": "New OTP sent to your email address."}

@auth_router.post("/request-password-reset-otp", response_model=Dict[str, str])
async def request_password_reset_otp(otp_request: OTPRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    user = db.query(UserDB).filter(UserDB.email == otp_request.email).first()
    if user and user.is_active:
        db.query(OtpDB).filter(
            OtpDB.email == user.email,
            OtpDB.used == False
        ).update({"used": True, "expires_at": datetime.now(timezone.utc)})
        otp_code = generate_otp_code()
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=settings.OTP_EXPIRE_MINUTES)
        db_otp = OtpDB(email=user.email, otp_code=otp_code, expires_at=expires_at)
        db.add(db_otp)
        db.commit()
        try:
            email_template = template_env.get_template("email_password_reset.html")
            html_content = email_template.render(
                otp_code=otp_code,
                user_name=user.full_name or user.email,
                otp_expire_minutes=settings.OTP_EXPIRE_MINUTES
            )
            background_tasks.add_task(send_email, user.email, "Password Reset Request", html_content)
        except Exception as e:
            logger.error(f"Failed to render or send password reset OTP email template: {e}")
    else:
        logger.info(f"Password reset OTP request for non-existent or inactive email: {otp_request.email}")
    return {"message": "If an account with this email exists and is active, an OTP for password reset has been sent."}

@auth_router.post("/reset-password", response_model=Dict[str, str])
async def reset_password_with_otp(reset_request: PasswordResetRequest, db: Session = Depends(get_db)):
    user = db.query(UserDB).filter(UserDB.email == reset_request.email).first()
    if not user:
        raise HTTPException(status_code=400, detail="Invalid OTP or email.")
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Account is inactive. Cannot reset password.")
    db_otp = db.query(OtpDB).filter(
        OtpDB.email == reset_request.email,
        OtpDB.otp_code == reset_request.otp,
        OtpDB.used == False,
        OtpDB.expires_at > datetime.now(timezone.utc)
    ).order_by(OtpDB.id.desc()).first()
    if not db_otp:
        raise HTTPException(status_code=400, detail="Invalid or expired OTP.")
    user.hashed_password = get_password_hash(reset_request.new_password)
    db_otp.used = True
    db.query(OtpDB).filter(
        OtpDB.email == user.email,
        OtpDB.id != db_otp.id,
        OtpDB.used == False
    ).update({"used": True, "expires_at": datetime.now(timezone.utc)})
    db.commit()
    return {"message": "Password has been reset successfully."}

@auth_router.post("/change-password", response_model=Dict[str, str])
async def change_password(
    password_data: PasswordChangeRequest,
    current_user: UserDB = Depends(get_current_verified_user),
    db: Session = Depends(get_db)
):
    if not verify_password(password_data.old_password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect old password.")
    if password_data.old_password == password_data.new_password:
        raise HTTPException(status_code=400, detail="New password cannot be the same as the old password.")
    current_user.hashed_password = get_password_hash(password_data.new_password)
    current_user.updated_at = datetime.now(timezone.utc)
    db.commit()
    return {"message": "Password changed successfully."}

@auth_router.get("/me", response_model=User)
async def read_users_me(current_user: UserDB = Depends(get_current_user)):
    return current_user

app.include_router(auth_router)

# Product & Recommendation Router
product_router = APIRouter(prefix="/api/products", tags=["Products & Recommendations"])

@product_router.get("/{item_id}", response_model=ProductPageResponse)
async def get_product_details_and_recommendations(
    item_id: int,
    request: Request,
    db: Session = Depends(get_db)
):
    product = get_product_with_average_rating(db, item_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    reviews_db = db.query(ReviewDB).filter(ReviewDB.product_id == item_id).order_by(ReviewDB.created_at.desc()).limit(10).all()
    reviews_list = []
    for r_db in reviews_db:
        user_full_name = r_db.user.full_name if r_db.user else "Anonymous"
        reviews_list.append(Review(
            id=r_db.id, user_id=r_db.user_id, product_id=r_db.product_id, rating=r_db.rating,
            comment=r_db.comment, created_at=r_db.created_at, user_full_name=user_full_name
        ))
    content_based_recs = get_content_based_recommendations(item_id, top_n=5)
    cf_recs = None
    current_user: Optional[UserDB] = None
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        try:
            current_user = await get_current_user(token=token, db=db)
        except HTTPException:
            current_user = None
    if current_user:
        cf_recs = get_collaborative_filtering_recommendations(current_user.id, top_n=5)
    recommendations = RecommendationResponse(content_based=content_based_recs, collaborative_filtering=cf_recs)
    return ProductPageResponse(product=product, reviews=reviews_list, recommendations=recommendations)

@product_router.post("/recommend-from-image", response_model=RecommendationResponse)
async def recommend_from_image(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    if not ml_store.clip_model or not ml_store.clip_processor or ml_store.df is None:
         raise HTTPException(status_code=500, detail="Image processing models not initialized.")
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")
    if not ml_store.clip_processor or not ml_store.clip_model:
        raise HTTPException(status_code=503, detail="CLIP model not available")
    inputs = ml_store.clip_processor(images=image, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        image_features = ml_store.clip_model.get_image_features(**inputs)
    query_embedding = image_features.cpu().numpy().tolist()[0]
    if not ml_store.fashion_collection:
        raise HTTPException(status_code=503, detail="Search collection not available")
    similar_items_results = ml_store.fashion_collection.query(
        query_embeddings=[query_embedding], n_results=1, include=["ids"]
    )
    if not similar_items_results or not similar_items_results.get("ids") or not similar_items_results["ids"][0] or not similar_items_results["ids"][0][0]:
        raise HTTPException(status_code=404, detail="No similar product found for the uploaded image.")
    most_similar_item_id_str = similar_items_results["ids"][0][0]
    try:
        most_similar_item_id = int(most_similar_item_id_str)
    except ValueError:
        raise HTTPException(status_code=500, detail="Invalid item ID format from search.")
    content_based_recs = get_content_based_recommendations(most_similar_item_id, top_n=5)
    cf_recs = None
    current_user: Optional[UserDB] = None
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        try:
            current_user = await get_current_user(token=token, db=db)
        except HTTPException:
            current_user = None
    if current_user:
        cf_recs = get_collaborative_filtering_recommendations(current_user.id, top_n=5)
    return RecommendationResponse(content_based=content_based_recs, collaborative_filtering=cf_recs)

@product_router.post("/search", response_model=SearchResult)
async def search_products(query: str = Form(...), db: Session = Depends(get_db)):
    if ml_store.fashion_collection is None:
        raise HTTPException(status_code=503, detail="Search service not available.")
    if not query or query.isspace():
         raise HTTPException(status_code=400, detail="Search query cannot be empty.")
    start_time = time.time()
    try:
        results_chroma = ml_store.fashion_collection.query(
            query_texts=[query], n_results=10, include=["metadatas", "distances"]
        )
        logger.info(f"ChromaDB query took {time.time() - start_time:.2f}s")
        image_data = []
        if results_chroma and results_chroma.get("ids") and results_chroma["ids"][0]:
            ids = results_chroma["ids"][0]
            distances = results_chroma["distances"][0]
            metadatas = results_chroma["metadatas"][0] if results_chroma.get("metadatas") else [{}] * len(ids)
            for i, item_id_str in enumerate(ids):
                if not item_id_str: continue
                try:
                    item_id_int = int(item_id_str)
                    image_url = f"/static/images/{item_id_int}.jpg"
                    product_info = get_product_with_average_rating(db, item_id_int)
                    image_data.append({
                        "id": item_id_int, "distance": distances[i], "image_url": image_url,
                        "metadata": metadatas[i],
                        "product_name": product_info.productDisplayName if product_info else "N/A",
                        "price": product_info.price if product_info else "N/A"
                    })
                except ValueError:
                     logger.warning(f"Chroma result ID {item_id_str} is not numeric.")
                except Exception as e:
                    logger.error(f"Error processing Chroma search result {item_id_str}: {e}")
        return SearchResult(images=image_data[:10])
    except Exception as e:
        logger.error(f"Error during search query '{query}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred during search: {str(e)}")

app.include_router(product_router)


# Cart Router
cart_router = APIRouter(prefix="/api/cart", tags=["Shopping Cart"])

def _get_or_create_cart(db: Session, user_id: int) -> CartDB:
    cart = db.query(CartDB).filter(CartDB.user_id == user_id).first()
    if not cart:
        cart = CartDB(user_id=user_id)
        db.add(cart)
        db.commit()
        db.refresh(cart)
    return cart

def _calculate_cart_total(cart: CartDB) -> float:
    total = 0.0
    for item in cart.items:
        if item.product and item.product.price is not None:
            total += item.product.price * item.quantity
    return total

@cart_router.get("/", response_model=Cart)
async def get_user_cart(
    current_user: UserDB = Depends(get_current_verified_user),
    db: Session = Depends(get_db)
):
    cart_db = _get_or_create_cart(db, current_user.id)
    cart_data = Cart.from_orm(cart_db)
    cart_data.total_cart_price = _calculate_cart_total(cart_db)
    return cart_data

@cart_router.post("/items", response_model=CartItem)
async def add_item_to_cart(
    item_in: CartItemCreate,
    current_user: UserDB = Depends(get_current_verified_user),
    db: Session = Depends(get_db)
):
    cart_db = _get_or_create_cart(db, current_user.id)
    product_db = get_product_db(db, item_in.product_id)
    if not product_db:
        raise HTTPException(status_code=404, detail="Product not found")
    if product_db.price is None:
        raise HTTPException(status_code=400, detail="Product does not have a price and cannot be added to cart.")
    cart_item_db = db.query(CartItemDB).filter(
        CartItemDB.cart_id == cart_db.id, CartItemDB.product_id == item_in.product_id
    ).first()
    if cart_item_db:
        cart_item_db.quantity += item_in.quantity
    else:
        cart_item_db = CartItemDB(
            cart_id=cart_db.id, product_id=item_in.product_id, quantity=item_in.quantity
        )
        db.add(cart_item_db)
    db.commit()
    db.refresh(cart_item_db)
    db.refresh(cart_item_db.product)
    return cart_item_db

@cart_router.put("/items/{cart_item_id}", response_model=CartItem)
async def update_cart_item_quantity(
    cart_item_id: int,
    quantity: int = Form(..., gt=0),
    current_user: UserDB = Depends(get_current_verified_user),
    db: Session = Depends(get_db)
):
    cart_db = _get_or_create_cart(db, current_user.id)
    cart_item_db = db.query(CartItemDB).filter(
        CartItemDB.id == cart_item_id, CartItemDB.cart_id == cart_db.id
    ).first()
    if not cart_item_db:
        raise HTTPException(status_code=404, detail="Cart item not found")
    cart_item_db.quantity = quantity
    db.commit()
    db.refresh(cart_item_db)
    db.refresh(cart_item_db.product)
    return cart_item_db

@cart_router.delete("/items/{cart_item_id}", status_code=204)
async def remove_item_from_cart(
    cart_item_id: int,
    current_user: UserDB = Depends(get_current_verified_user),
    db: Session = Depends(get_db)
):
    cart_db = _get_or_create_cart(db, current_user.id)
    cart_item_db = db.query(CartItemDB).filter(
        CartItemDB.id == cart_item_id, CartItemDB.cart_id == cart_db.id
    ).first()
    if not cart_item_db:
        raise HTTPException(status_code=404, detail="Cart item not found")
    db.delete(cart_item_db)
    db.commit()
    return None

@cart_router.delete("/", status_code=204)
async def clear_cart(
    current_user: UserDB = Depends(get_current_verified_user),
    db: Session = Depends(get_db)
):
    cart_db = db.query(CartDB).filter(CartDB.user_id == current_user.id).first()
    if cart_db:
        db.query(CartItemDB).filter(CartItemDB.cart_id == cart_db.id).delete()
        db.commit()
    return None

app.include_router(cart_router)


# Order & Payment Router
order_router = APIRouter(prefix="/api/orders", tags=["Orders & Payments"])

@order_router.post("/", response_model=Order)
async def create_order_from_cart(
    order_in: OrderCreate,
    current_user: UserDB = Depends(get_current_verified_user),
    db: Session = Depends(get_db)
):
    cart = db.query(CartDB).filter(CartDB.user_id == current_user.id).first()
    if not cart or not cart.items:
        raise HTTPException(status_code=400, detail="Cart is empty. Cannot create order.")
    total_amount = 0
    order_items_to_create = []
    for cart_item in cart.items:
        if not cart_item.product or cart_item.product.price is None:
            product_name = cart_item.product.productDisplayName if cart_item.product else f'ID: {cart_item.product_id}'
            raise HTTPException(status_code=400, detail=f"Product '{product_name}' in cart has no price.")
        total_amount += cart_item.product.price * cart_item.quantity
        order_items_to_create.append(
            OrderItemDB(
                product_id=cart_item.product_id,
                quantity=cart_item.quantity,
                price_at_purchase=cart_item.product.price
            )
        )
    if total_amount <= 0:
         raise HTTPException(status_code=400, detail="Order total must be greater than zero.")
    razorpay_order_id = None
    if razorpay_client:
        try:
            razorpay_order_data = {
                "amount": int(total_amount * 100), "currency": "INR",
                "receipt": f"receipt_user_{current_user.id}_{int(time.time())}",
                "notes": {"user_id": str(current_user.id), "email": current_user.email}
            }
            rp_order = razorpay_client.order.create(data=razorpay_order_data)
            razorpay_order_id = rp_order['id']
            logger.info(f"Razorpay order created: {razorpay_order_id} for amount {total_amount}")
        except Exception as e:
            logger.error(f"Failed to create Razorpay order: {e}")
            raise HTTPException(status_code=500, detail="Payment gateway error. Could not create order.")
    else:
        logger.warning("Razorpay client not configured. Proceeding without Razorpay order ID.")
    db_order = OrderDB(
        user_id=current_user.id, total_amount=total_amount, status=OrderStatus.PENDING,
        shipping_address=order_in.shipping_address, razorpay_order_id=razorpay_order_id
    )
    db_order.items.extend(order_items_to_create)
    db.add(db_order)
    db.query(CartItemDB).filter(CartItemDB.cart_id == cart.id).delete()
    try:
        db.commit()
        db.refresh(db_order)
        for item in db_order.items: db.refresh(item.product)
        return db_order
    except IntegrityError as e:
        db.rollback()
        logger.error(f"Integrity error creating order: {e}")
        raise HTTPException(status_code=500, detail="Could not create order due to a database issue.")
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating order: {e}")
        raise HTTPException(status_code=500, detail="Failed to create order.")

@order_router.post("/verify-payment", response_model=Dict[str, Any])
async def verify_razorpay_payment(
    razorpay_order_id: str = Form(...),
    razorpay_payment_id: str = Form(...),
    razorpay_signature: str = Form(...),
    current_user: UserDB = Depends(get_current_verified_user),
    db: Session = Depends(get_db)
):
    if not razorpay_client:
        raise HTTPException(status_code=503, detail="Payment gateway not configured.")
    order_db = db.query(OrderDB).filter(
        OrderDB.razorpay_order_id == razorpay_order_id, OrderDB.user_id == current_user.id
    ).first()
    if not order_db:
        raise HTTPException(status_code=404, detail="Order not found or does not belong to user.")
    if order_db.status not in [OrderStatus.PENDING, OrderStatus.PAYMENT_FAILED]:
        raise HTTPException(status_code=400, detail=f"Order is already in status: {order_db.status}")
    params_dict = {
        'razorpay_order_id': razorpay_order_id,
        'razorpay_payment_id': razorpay_payment_id,
        'razorpay_signature': razorpay_signature
    }
    try:
        razorpay_client.utility.verify_payment_signature(params_dict)
        order_db.status = OrderStatus.PROCESSING
        db.commit()
        logger.info(f"Payment verified for Razorpay order ID: {razorpay_order_id}")
        return {"status": "success", "message": "Payment verified successfully.", "order_id": order_db.id, "order_status": order_db.status}
    except razorpay.errors.SignatureVerificationError as e:
        logger.error(f"Razorpay signature verification failed: {e}")
        order_db.status = OrderStatus.PAYMENT_FAILED
        db.commit()
        raise HTTPException(status_code=400, detail="Payment verification failed. Invalid signature.")
    except Exception as e:
        logger.error(f"Error during payment verification: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during payment verification.")

@order_router.get("/", response_model=List[Order])
async def get_user_order_history(
    current_user: UserDB = Depends(get_current_verified_user),
    db: Session = Depends(get_db), skip: int = 0, limit: int = 10
):
    orders_db = db.query(OrderDB).filter(OrderDB.user_id == current_user.id)\
        .order_by(OrderDB.created_at.desc()).offset(skip).limit(limit).all()
    for order in orders_db:
        for item in order.items: db.refresh(item.product)
    return orders_db

@order_router.get("/{order_id}", response_model=Order)
async def get_order_details(
    order_id: int,
    current_user: UserDB = Depends(get_current_verified_user),
    db: Session = Depends(get_db)
):
    order_db = db.query(OrderDB).filter(
        OrderDB.id == order_id, OrderDB.user_id == current_user.id
    ).first()
    if not order_db:
        raise HTTPException(status_code=404, detail="Order not found or does not belong to user.")
    for item in order_db.items: db.refresh(item.product)
    return order_db

app.include_router(order_router)

# Review Router
review_router = APIRouter(prefix="/api/reviews", tags=["Product Reviews"])

@review_router.post("/", response_model=Review, status_code=201)
async def create_product_review(
    review_in: ReviewCreate,
    current_user: UserDB = Depends(get_current_verified_user),
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    product = get_product_db(db, review_in.product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    existing_review = db.query(ReviewDB).filter(
        ReviewDB.user_id == current_user.id, ReviewDB.product_id == review_in.product_id
    ).first()
    if existing_review:
        raise HTTPException(status_code=400, detail="You have already reviewed this product.")
    db_review = ReviewDB(**review_in.model_dump(), user_id=current_user.id)
    db.add(db_review)
    db.commit()
    db.refresh(db_review)
    db.refresh(db_review.user)
    background_tasks.add_task(train_cf_model, db)
    return Review(
        id=db_review.id, user_id=db_review.user_id, product_id=db_review.product_id,
        rating=db_review.rating, comment=db_review.comment, created_at=db_review.created_at,
        user_full_name=db_review.user.full_name or "Anonymous"
    )

@review_router.get("/product/{product_id}", response_model=List[Review])
async def get_reviews_for_product(
    product_id: int, db: Session = Depends(get_db), skip: int = 0, limit: int = 10
):
    product = get_product_db(db, product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    reviews_db = db.query(ReviewDB).filter(ReviewDB.product_id == product_id)\
        .order_by(ReviewDB.created_at.desc()).offset(skip).limit(limit).all()
    reviews_list = []
    for r_db in reviews_db:
        user_full_name = r_db.user.full_name if r_db.user else "Anonymous"
        reviews_list.append(Review(
            id=r_db.id, user_id=r_db.user_id, product_id=r_db.product_id, rating=r_db.rating,
            comment=r_db.comment, created_at=r_db.created_at, user_full_name=user_full_name
        ))
    return reviews_list

@review_router.get("/user", response_model=List[Review])
async def get_reviews_by_user(
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db), skip: int = 0, limit: int = 10
):
    reviews_db = db.query(ReviewDB).filter(ReviewDB.user_id == current_user.id)\
        .order_by(ReviewDB.created_at.desc()).offset(skip).limit(limit).all()
    reviews_list = []
    for r_db in reviews_db:
        user_full_name = r_db.user.full_name if r_db.user else "Anonymous"
        reviews_list.append(Review(
            id=r_db.id, user_id=r_db.user_id, product_id=r_db.product_id, rating=r_db.rating,
            comment=r_db.comment, created_at=r_db.created_at, user_full_name=user_full_name
        ))
    return reviews_list

app.include_router(review_router)


# --- Root and Health Check ---
@app.get("/")
async def read_root():
    return {"message": "Fashion Recommendation API V2 is running."}

@app.get("/health")
async def health_check():
    db_ok = False
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception as e:
        logger.error(f"Health check DB error: {e}")
    finally:
        if 'db' in locals() and db: db.close()
    chroma_ok = ml_store.chroma_client is not None and ml_store.fashion_collection is not None
    clip_ok = ml_store.clip_model is not None and ml_store.clip_processor is not None
    cf_model_ok = ml_store.cf_model_svd is not None
    status = "ok"
    details = {
        "database": "ok" if db_ok else "error",
        "chromadb": "ok" if chroma_ok else "error/not_loaded",
        "clip_model": "ok" if clip_ok else "error/not_loaded",
        "cf_model": "ok" if cf_model_ok else "error/not_trained"
    }
    if not all([db_ok, chroma_ok, clip_ok, cf_model_ok]): status = "error"
    return {"status": status, "details": details}

if __name__ == "__main__":
    import uvicorn
    os.makedirs("templates", exist_ok=True)
    if not os.path.exists("templates/email_otp_verification.html"):
        with open("templates/email_otp_verification.html", "w") as f:
            f.write("<h1>Email Verification</h1><p>Hi {{ user_name }},</p><p>Your OTP is: <strong>{{ otp_code }}</strong></p><p>This OTP is valid for {{ otp_expire_minutes }} minutes.</p>")
    if not os.path.exists("templates/email_password_reset.html"):
        with open("templates/email_password_reset.html", "w") as f:
            f.write("<h1>Password Reset</h1><p>Hi {{ user_name }},</p><p>Your OTP for password reset is: <strong>{{ otp_code }}</strong></p><p>This OTP is valid for {{ otp_expire_minutes }} minutes.</p>")
    uvicorn.run(app, host="localhost", port=8000)
