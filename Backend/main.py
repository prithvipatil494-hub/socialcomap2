# ============================================
# FILE 1: main.py
# ============================================
import os
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, relationship, sessionmaker
from pydantic import BaseModel, EmailStr
from typing import List, Optional
from datetime import datetime, timedelta
from passlib.context import CryptContext
import jwt
import uuid

load_dotenv()

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:ranka;aismybestfriend@db.xkpwcchgzumymmugsrps.supabase.co:5432/postgres")
SECRET_KEY = os.getenv("SECRET_KEY", "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7")
ALGORITHM = "HS256"

# Supabase Storage (optional)
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# Database setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Security
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

app = FastAPI(title="SocialMap API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== DATABASE MODELS ==========
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    avatar_url = Column(String, nullable=True)
    bio = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.utcnow)
    is_online = Column(Boolean, default=False)
    
    posts = relationship("Post", back_populates="user")
    following = relationship("Follow", foreign_keys="Follow.follower_id", back_populates="follower")
    followers = relationship("Follow", foreign_keys="Follow.followed_id", back_populates="followed")

class Post(Base):
    __tablename__ = "posts"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    content = Column(Text)
    image_url = Column(String, nullable=True)
    latitude = Column(Float)
    longitude = Column(Float)
    location_name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="posts")
    likes = relationship("Like", back_populates="post", cascade="all, delete-orphan")
    comments = relationship("Comment", back_populates="post", cascade="all, delete-orphan")

class Like(Base):
    __tablename__ = "likes"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    post_id = Column(Integer, ForeignKey("posts.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    post = relationship("Post", back_populates="likes")

class Comment(Base):
    __tablename__ = "comments"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    post_id = Column(Integer, ForeignKey("posts.id"))
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    post = relationship("Post", back_populates="comments")

class Follow(Base):
    __tablename__ = "follows"
    id = Column(Integer, primary_key=True, index=True)
    follower_id = Column(Integer, ForeignKey("users.id"))
    followed_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    follower = relationship("User", foreign_keys=[follower_id], back_populates="following")
    followed = relationship("User", foreign_keys=[followed_id], back_populates="followers")

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    sender_id = Column(Integer, ForeignKey("users.id"))
    receiver_id = Column(Integer, ForeignKey("users.id"))
    content = Column(Text)
    is_read = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class UserLocation(Base):
    __tablename__ = "user_locations"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)
    latitude = Column(Float)
    longitude = Column(Float)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Create all tables
Base.metadata.create_all(bind=engine)

# ========== PYDANTIC SCHEMAS ==========
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    avatar_url: Optional[str] = None
    bio: Optional[str] = None
    is_online: bool
    
    class Config:
        from_attributes = True

class PostCreate(BaseModel):
    content: str
    latitude: float
    longitude: float
    location_name: Optional[str] = None
    image_url: Optional[str] = None

class PostResponse(BaseModel):
    id: int
    user_id: int
    username: str
    user_avatar: Optional[str]
    content: str
    image_url: Optional[str]
    latitude: float
    longitude: float
    location_name: Optional[str]
    created_at: datetime
    likes_count: int
    comments_count: int
    
    class Config:
        from_attributes = True

class MessageCreate(BaseModel):
    receiver_id: int
    content: str

class MessageResponse(BaseModel):
    id: int
    sender_id: int
    receiver_id: int
    content: str
    is_read: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

class LocationUpdate(BaseModel):
    latitude: float
    longitude: float

# ========== DEPENDENCIES ==========
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_user(token: str, db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = int(payload.get("sub"))
        user = db.query(User).filter(User.id == user_id).first()
        if user is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user
    except:
        raise HTTPException(status_code=401, detail="Invalid token")

# ========== WEBSOCKET MANAGER ==========
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict = {}

    async def connect(self, websocket: WebSocket, user_id: int):
        await websocket.accept()
        self.active_connections[user_id] = websocket

    def disconnect(self, user_id: int):
        if user_id in self.active_connections:
            del self.active_connections[user_id]

    async def send_personal_message(self, message: str, user_id: int):
        if user_id in self.active_connections:
            await self.active_connections[user_id].send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections.values():
            await connection.send_text(message)

manager = ConnectionManager()

# ========== AUTH HELPERS ==========
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=7)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

# ========== API ROUTES ==========

@app.get("/")
def root():
    return {
        "message": "SocialMap API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}

# ===== AUTH =====
@app.post("/api/auth/register", response_model=UserResponse)
def register(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already taken")
    
    hashed_password = get_password_hash(user.password)
    new_user = User(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@app.post("/api/auth/login")
def login(email: str, password: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_access_token({"sub": str(user.id)})
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "avatar_url": user.avatar_url,
            "bio": user.bio
        }
    }

# ===== USERS =====
@app.get("/api/users/{user_id}", response_model=UserResponse)
def get_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.get("/api/users/{user_id}/stats")
def get_user_stats(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    posts_count = db.query(Post).filter(Post.user_id == user_id).count()
    followers_count = db.query(Follow).filter(Follow.followed_id == user_id).count()
    following_count = db.query(Follow).filter(Follow.follower_id == user_id).count()
    
    return {
        "posts": posts_count,
        "followers": followers_count,
        "following": following_count
    }

# ===== POSTS =====
@app.post("/api/posts", response_model=PostResponse)
def create_post(post: PostCreate, user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    new_post = Post(
        user_id=user_id,
        content=post.content,
        latitude=post.latitude,
        longitude=post.longitude,
        location_name=post.location_name,
        image_url=post.image_url
    )
    db.add(new_post)
    db.commit()
    db.refresh(new_post)
    
    response = PostResponse(
        id=new_post.id,
        user_id=new_post.user_id,
        username=user.username,
        user_avatar=user.avatar_url,
        content=new_post.content,
        image_url=new_post.image_url,
        latitude=new_post.latitude,
        longitude=new_post.longitude,
        location_name=new_post.location_name,
        created_at=new_post.created_at,
        likes_count=0,
        comments_count=0
    )
    return response

@app.get("/api/posts/nearby", response_model=List[PostResponse])
def get_nearby_posts(lat: float, lng: float, radius: float = 10, db: Session = Depends(get_db)):
    posts = db.query(Post).all()
    nearby = []
    
    for post in posts:
        lat_diff = abs(post.latitude - lat)
        lng_diff = abs(post.longitude - lng)
        
        if lat_diff < radius/111 and lng_diff < radius/111:
            user = db.query(User).filter(User.id == post.user_id).first()
            likes_count = db.query(Like).filter(Like.post_id == post.id).count()
            comments_count = db.query(Comment).filter(Comment.post_id == post.id).count()
            
            post_response = PostResponse(
                id=post.id,
                user_id=post.user_id,
                username=user.username if user else "Unknown",
                user_avatar=user.avatar_url if user else None,
                content=post.content,
                image_url=post.image_url,
                latitude=post.latitude,
                longitude=post.longitude,
                location_name=post.location_name,
                created_at=post.created_at,
                likes_count=likes_count,
                comments_count=comments_count
            )
            nearby.append(post_response)
    
    return nearby

@app.get("/api/posts/feed", response_model=List[PostResponse])
def get_feed(user_id: int, skip: int = 0, limit: int = 20, db: Session = Depends(get_db)):
    following = db.query(Follow.followed_id).filter(Follow.follower_id == user_id).all()
    following_ids = [f[0] for f in following]
    following_ids.append(user_id)
    
    posts = db.query(Post).filter(Post.user_id.in_(following_ids)).order_by(Post.created_at.desc()).offset(skip).limit(limit).all()
    
    result = []
    for post in posts:
        user = db.query(User).filter(User.id == post.user_id).first()
        likes_count = db.query(Like).filter(Like.post_id == post.id).count()
        comments_count = db.query(Comment).filter(Comment.post_id == post.id).count()
        
        result.append(PostResponse(
            id=post.id,
            user_id=post.user_id,
            username=user.username if user else "Unknown",
            user_avatar=user.avatar_url if user else None,
            content=post.content,
            image_url=post.image_url,
            latitude=post.latitude,
            longitude=post.longitude,
            location_name=post.location_name,
            created_at=post.created_at,
            likes_count=likes_count,
            comments_count=comments_count
        ))
    
    return result

@app.post("/api/posts/{post_id}/like")
def like_post(post_id: int, user_id: int, db: Session = Depends(get_db)):
    post = db.query(Post).filter(Post.id == post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    
    existing = db.query(Like).filter(Like.post_id == post_id, Like.user_id == user_id).first()
    if existing:
        db.delete(existing)
        db.commit()
        return {"liked": False, "likes_count": db.query(Like).filter(Like.post_id == post_id).count()}
    
    like = Like(user_id=user_id, post_id=post_id)
    db.add(like)
    db.commit()
    return {"liked": True, "likes_count": db.query(Like).filter(Like.post_id == post_id).count()}

# ===== FOLLOWS =====
@app.post("/api/follows/{followed_id}")
def follow_user(followed_id: int, user_id: int, db: Session = Depends(get_db)):
    if user_id == followed_id:
        raise HTTPException(status_code=400, detail="Cannot follow yourself")
    
    existing = db.query(Follow).filter(
        Follow.follower_id == user_id,
        Follow.followed_id == followed_id
    ).first()
    
    if existing:
        db.delete(existing)
        db.commit()
        return {"following": False}
    
    follow = Follow(follower_id=user_id, followed_id=followed_id)
    db.add(follow)
    db.commit()
    return {"following": True}

@app.get("/api/users/{user_id}/following", response_model=List[UserResponse])
def get_following(user_id: int, db: Session = Depends(get_db)):
    follows = db.query(Follow).filter(Follow.follower_id == user_id).all()
    users = []
    for f in follows:
        user = db.query(User).filter(User.id == f.followed_id).first()
        if user:
            users.append(user)
    return users

@app.get("/api/users/{user_id}/followers", response_model=List[UserResponse])
def get_followers(user_id: int, db: Session = Depends(get_db)):
    follows = db.query(Follow).filter(Follow.followed_id == user_id).all()
    users = []
    for f in follows:
        user = db.query(User).filter(User.id == f.follower_id).first()
        if user:
            users.append(user)
    return users

# ===== MESSAGES =====
@app.post("/api/messages", response_model=MessageResponse)
def send_message(message: MessageCreate, user_id: int, db: Session = Depends(get_db)):
    new_message = Message(
        sender_id=user_id,
        receiver_id=message.receiver_id,
        content=message.content
    )
    db.add(new_message)
    db.commit()
    db.refresh(new_message)
    return new_message

@app.get("/api/messages/{other_user_id}", response_model=List[MessageResponse])
def get_messages(other_user_id: int, user_id: int, db: Session = Depends(get_db)):
    messages = db.query(Message).filter(
        ((Message.sender_id == user_id) & (Message.receiver_id == other_user_id)) |
        ((Message.sender_id == other_user_id) & (Message.receiver_id == user_id))
    ).order_by(Message.created_at).all()
    
    db.query(Message).filter(
        Message.sender_id == other_user_id,
        Message.receiver_id == user_id,
        Message.is_read == False
    ).update({"is_read": True})
    db.commit()
    
    return messages

# ===== LOCATION =====
@app.post("/api/location")
def update_location(location: LocationUpdate, user_id: int, db: Session = Depends(get_db)):
    user_loc = db.query(UserLocation).filter(UserLocation.user_id == user_id).first()
    if user_loc:
        user_loc.latitude = location.latitude
        user_loc.longitude = location.longitude
        user_loc.updated_at = datetime.utcnow()
    else:
        user_loc = UserLocation(
            user_id=user_id,
            latitude=location.latitude,
            longitude=location.longitude
        )
        db.add(user_loc)
    db.commit()
    return {"status": "updated"}

@app.get("/api/locations/friends")
def get_friends_locations(user_id: int, db: Session = Depends(get_db)):
    follows = db.query(Follow).filter(Follow.follower_id == user_id).all()
    locations = []
    for follow in follows:
        loc = db.query(UserLocation).filter(UserLocation.user_id == follow.followed_id).first()
        if loc:
            user = db.query(User).filter(User.id == follow.followed_id).first()
            locations.append({
                "user_id": user.id,
                "username": user.username,
                "avatar_url": user.avatar_url,
                "is_online": user.is_online,
                "latitude": loc.latitude,
                "longitude": loc.longitude,
                "updated_at": loc.updated_at
            })
    return locations

# ===== FILE UPLOAD =====
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), user_id: int = 0):
    try:
        if SUPABASE_URL and SUPABASE_KEY:
            from supabase import create_client
            supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
            
            file_name = f"{user_id}/{uuid.uuid4()}{file.filename}"
            file_bytes = await file.read()
            
            res = supabase.storage.from_('posts').upload(
                file_name,
                file_bytes,
                {'content-type': file.content_type}
            )
            
            public_url = supabase.storage.from_('posts').get_public_url(file_name)
            return {"url": public_url}
        
        return {"url": f"https://placeholder.com/{uuid.uuid4()}.jpg"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ===== WEBSOCKET =====
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: int, db: Session = Depends(get_db)):
    await manager.connect(websocket, user_id)
    
    user = db.query(User).filter(User.id == user_id).first()
    if user:
        user.is_online = True
        user.last_seen = datetime.utcnow()
        db.commit()
    
    try:
        while True:
            data = await websocket.receive_text()
            await manager.broadcast(f"User {user_id}: {data}")
    except WebSocketDisconnect:
        manager.disconnect(user_id)
        if user:
            user.is_online = False
            user.last_seen = datetime.utcnow()
            db.commit()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)