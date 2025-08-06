from pymongo import MongoClient
import os

MONGO_URI = os.environ.get("MONGO_URI", "mongodb+srv://Flash:You5exDrS5APDmDa@bds.7de1oam.mongodb.net/")
DB_NAME = "energy_coaching"
COLLECTION_NAME = "users"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
users_collection = db[COLLECTION_NAME]

def create_user(username, password, is_admin=False):
    if users_collection.find_one({"username": username}):
        return False, "Username already exists."
    users_collection.insert_one({"username": username, "password": password, "is_admin": is_admin})
    return True, "User registered successfully."

def authenticate_user(username, password):
    user = users_collection.find_one({"username": username, "password": password})
    return user

def get_all_users():
    return list(users_collection.find({}, {"_id": 0, "username": 1, "is_admin": 1}))

# For admin: get all users except admin

def get_non_admin_users():
    return list(users_collection.find({"is_admin": False}, {"_id": 0, "username": 1}))

# Log user activity
def log_user_activity(username, action, details=None):
    import datetime
    activity = {
        "username": username,
        "action": action,
        "details": details or {},
        "timestamp": datetime.datetime.utcnow()
    }
    db["user_activity"].insert_one(activity)

# Get all activities (for admin)
def get_all_user_activities():
    return list(db["user_activity"].find({}, {"_id": 0}))

# Get activities for a specific user
def get_user_activities(username):
    return list(db["user_activity"].find({"username": username}, {"_id": 0}))

# Delete user by username
def delete_user(username):
    result = users_collection.delete_one({"username": username})
    return result.deleted_count > 0
