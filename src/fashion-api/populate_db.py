import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from PIL import Image, ImageDraw, ImageFont

DATABASE_URL = "sqlite:///./database/fashion.db"
STATIC_DIR = "static"
IMAGES_DIR = os.path.join(STATIC_DIR, "images")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ClothingItemDB(Base):
    """SQLAlchemy model for clothing_items table."""
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
    price = Column(Float)

def create_db_and_tables():
    """Create the SQLite database and tables if they don't exist."""
    if not inspect(engine).has_table("clothing_items"):
        Base.metadata.create_all(bind=engine)
        print("Created database and tables.")
    else:
        print("Database and tables already exist.")

def populate_database_from_csv(product_details_csv_path, product_prices_csv_path):
    """Populate the database with data from CSV files."""
    print("Reading product details CSV...")
    product_details_df = pd.read_csv(product_details_csv_path)
    print("Reading product prices CSV...")
    product_prices_df = pd.read_csv(product_prices_csv_path)

    print("Merging dataframes...")
    merged_df = pd.merge(product_details_df, product_prices_df[['id', 'original_price']], on='id', how='inner')
    merged_df = merged_df.rename(columns={'original_price': 'price'})

    db: SessionLocal = SessionLocal()
    try:
        print("Populating database...")
        for index, row in merged_df.iterrows():
            item = ClothingItemDB(
                id=row['id'],
                gender=row['gender'],
                masterCategory=row['masterCategory'],
                subCategory=row['subCategory'],
                articleType=row['articleType'],
                baseColour=row['baseColour'],
                season=row['season'],
                usage=row['usage'],
                productDisplayName=row['productDisplayName'],
                price=row['price']
            )
            db.add(item)
        db.commit()
        print("Database populated successfully.")
    except Exception as e:
        db.rollback()
        print(f"Error populating database: {e}")
    finally:
        db.close()

def create_placeholder_images(num_images):
    """Create placeholder images and save them in static/images directory."""
    if not os.path.exists(IMAGES_DIR):
        os.makedirs(IMAGES_DIR, exist_ok=True)

    print(f"Creating {num_images} placeholder images in {IMAGES_DIR}...")
    for i in range(1, num_images + 1):
        img = Image.new('RGB', (224, 224), color='lightgrey')
        d = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 40) # Use arial font, you might need to adjust path or font name
        except IOError:
            font = ImageFont.load_default() # Fallback to default font
        d.text((50, 100), f"Item ID {i}", fill=(0, 0, 0), font=font)
        img.save(os.path.join(IMAGES_DIR, f"{i}.jpg"))
    print("Placeholder images created.")

if __name__ == "__main__":
    product_details_csv = '../../datasets/styles_filtered.csv'
    product_prices_csv = '../../datasets/pricing.csv'

    create_db_and_tables()
    populate_database_from_csv(product_details_csv, product_prices_csv)

    print("Database population script completed.")
