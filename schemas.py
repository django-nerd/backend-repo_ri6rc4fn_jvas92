"""
Database Schemas

Define your MongoDB collection schemas here using Pydantic models.
These schemas are used for data validation in your application.

Each Pydantic model represents a collection in your database.
Model name is converted to lowercase for the collection name:
- User -> "user" collection
- Product -> "product" collection
- BlogPost -> "blogs" collection
"""

from pydantic import BaseModel, Field
from typing import Optional, List

# Example schemas (kept for reference)
class User(BaseModel):
    name: str = Field(..., description="Full name")
    email: str = Field(..., description="Email address")
    address: str = Field(..., description="Address")
    age: Optional[int] = Field(None, ge=0, le=120, description="Age in years")
    is_active: bool = Field(True, description="Whether user is active")

class Product(BaseModel):
    title: str = Field(..., description="Product title")
    description: Optional[str] = Field(None, description="Product description")
    price: float = Field(..., ge=0, description="Price in dollars")
    category: str = Field(..., description="Product category")
    in_stock: bool = Field(True, description="Whether product is in stock")

# Waste analysis schema used by the app
class Wasteanalysis(BaseModel):
    """
    Waste analysis documents
    Collection name: "wasteanalysis" (lowercase of class name)
    """
    filename: str = Field(..., description="Uploaded file name")
    mime_type: str = Field(..., description="MIME type of the image")
    category: str = Field(..., description="Detected waste category/material")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score between 0 and 1")
    instructions: List[str] = Field(..., description="Eco-friendly dismantling and disposal steps")
    notes: Optional[str] = Field(None, description="Additional notes or safety info")
