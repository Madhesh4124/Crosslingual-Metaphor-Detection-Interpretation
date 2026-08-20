"""
MongoDB database configuration and operations
"""
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
from typing import Optional, List
import logging
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# MongoDB configuration
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "metaphor_detector")

# Global database client
mongodb_client: Optional[AsyncIOMotorClient] = None
database = None


async def connect_to_mongodb():
    """Connect to MongoDB"""
    global mongodb_client, database
    
    try:
        masked_url = "MongoDB Atlas Cluster" if "mongodb+srv" in MONGODB_URL else MONGODB_URL.split("@")[-1]
        logger.info(f"Connecting to MongoDB ({masked_url})...")
        mongodb_client = AsyncIOMotorClient(MONGODB_URL, serverSelectionTimeoutMS=2000)
        
        # Test the connection
        await mongodb_client.admin.command('ping')
        database = mongodb_client[MONGODB_DB_NAME]
        logger.info(f"✓ Successfully connected to MongoDB database: {MONGODB_DB_NAME}")
        
        # Create indexes for better query performance
        await database.predictions.create_index([("timestamp", -1)])
        await database.predictions.create_index([("language", 1)])
        await database.predictions.create_index([("label", 1)])
        logger.info("✓ Database indexes created")
        
        return True
    except Exception as e:
        database = None
        mongodb_client = None
        logger.error(f"✗ Failed to connect to MongoDB: {str(e)}")
        logger.warning("History feature will be disabled")
        return False


async def close_mongodb_connection():
    """Close MongoDB connection"""
    global mongodb_client
    
    if mongodb_client:
        mongodb_client.close()
        logger.info("MongoDB connection closed")


async def save_prediction(prediction_data: dict) -> Optional[str]:
    """
    Save a prediction to the database
    
    Args:
        prediction_data: Dictionary containing prediction results
        
    Returns:
        str: ID of the saved document, or None if save failed
    """
    if database is None:
        logger.warning("Database not connected, skipping save")
        return None
    
    try:
        # Add timestamp (using local time instead of UTC)
        prediction_data["timestamp"] = datetime.now()
        
        # Insert into database
        result = await database.predictions.insert_one(prediction_data)
        logger.info(f"✓ Saved prediction to database: {result.inserted_id}")
        return str(result.inserted_id)
    except Exception as e:
        logger.error(f"Failed to save prediction: {str(e)}")
        return None


async def get_prediction_history(limit: int = 50, skip: int = 0, language: Optional[str] = None, label: Optional[str] = None, session_id: Optional[str] = None) -> List[dict]:
    """
    Get prediction history from database
    
    Args:
        limit: Maximum number of results to return
        skip: Number of results to skip (for pagination)
        language: Filter by language (optional)
        label: Filter by label (metaphor/normal) (optional)
        session_id: Filter by user session (optional)
        
    Returns:
        List of prediction documents
    """
    if database is None:
        logger.warning("Database not connected")
        return []
    
    try:
        # Build query filter
        query = {}
        if language:
            query["language"] = language
        if label:
            query["label"] = label
        
        # Get predictions sorted by timestamp (newest first)
        cursor = database.predictions.find(query).sort("timestamp", -1).skip(skip).limit(limit)
        predictions = await cursor.to_list(length=limit)
        
        # Convert ObjectId to string for JSON serialization
        for pred in predictions:
            pred["_id"] = str(pred["_id"])
            # Convert datetime to ISO format string if it's a datetime object
            if "timestamp" in pred and hasattr(pred["timestamp"], "isoformat"):
                pred["timestamp"] = pred["timestamp"].isoformat()
            elif "timestamp" in pred:
                pred["timestamp"] = str(pred["timestamp"])
        
        return predictions
    except Exception as e:
        logger.error(f"Failed to get prediction history: {str(e)}")
        return []


async def get_prediction_by_id(prediction_id: str) -> Optional[dict]:
    """
    Get a specific prediction by ID
    
    Args:
        prediction_id: MongoDB document ID
        
    Returns:
        Prediction document or None
    """
    if database is None:
        return None
    
    try:
        from bson import ObjectId
        prediction = await database.predictions.find_one({"_id": ObjectId(prediction_id)})
        
        if prediction:
            prediction["_id"] = str(prediction["_id"])
            if "timestamp" in prediction and hasattr(prediction["timestamp"], "isoformat"):
                prediction["timestamp"] = prediction["timestamp"].isoformat()
            elif "timestamp" in prediction:
                prediction["timestamp"] = str(prediction["timestamp"])
        
        return prediction
    except Exception as e:
        logger.error(f"Failed to get prediction by ID: {str(e)}")
        return None


async def delete_prediction(prediction_id: str) -> bool:
    """
    Delete a prediction from database
    
    Args:
        prediction_id: MongoDB document ID
        
    Returns:
        bool: True if deleted successfully
    """
    if database is None:
        return False
    
    try:
        from bson import ObjectId
        result = await database.predictions.delete_one({"_id": ObjectId(prediction_id)})
        return result.deleted_count > 0
    except Exception as e:
        logger.error(f"Failed to delete prediction: {str(e)}")
        return False


async def clear_all_history() -> int:
    """
    Clear all prediction history
    
    Returns:
        int: Number of documents deleted
    """
    if database is None:
        return 0
    
    try:
        result = await database.predictions.delete_many({})
        logger.info(f"Cleared {result.deleted_count} predictions from history")
        return result.deleted_count
    except Exception as e:
        logger.error(f"Failed to clear history: {str(e)}")
        return 0


async def get_statistics() -> dict:
    """
    Get statistics about predictions
    
    Returns:
        Dictionary with statistics
    """
    if database is None:
        return {
            "total_predictions": 0,
            "metaphor_count": 0,
            "normal_count": 0,
            "languages": {}
        }
    
    try:
        total = await database.predictions.count_documents({})
        metaphor_count = await database.predictions.count_documents({"label": "metaphor"})
        normal_count = await database.predictions.count_documents({"label": "normal"})
        
        # Get language distribution
        pipeline = [
            {"$group": {"_id": "$language", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        language_stats = await database.predictions.aggregate(pipeline).to_list(length=10)
        
        languages = {stat["_id"]: stat["count"] for stat in language_stats}
        
        return {
            "total_predictions": total,
            "metaphor_count": metaphor_count,
            "normal_count": normal_count,
            "languages": languages
        }
    except Exception as e:
        logger.error(f"Failed to get statistics: {str(e)}")
        return {
            "total_predictions": 0,
            "metaphor_count": 0,
            "normal_count": 0,
            "languages": {}
        }
