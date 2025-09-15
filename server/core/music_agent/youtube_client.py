"""
YouTube Data API v3 Client

Handles YouTube video search and recommendations.
"""

import aiohttp
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class YouTubeVideo:
    video_id: str
    title: str
    description: str
    channel_title: str
    duration: str
    view_count: int
    like_count: int
    published_at: str
    thumbnail_url: str
    video_url: str
    category: str

class YouTubeClient:
    """Client for YouTube Data API v3."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.googleapis.com/youtube/v3"
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def search_videos(self, query: str, max_results: int = 10, 
                          category: Optional[str] = None) -> List[YouTubeVideo]:
        """
        Search for videos on YouTube.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            category: Optional category filter
            
        Returns:
            List of YouTubeVideo objects
        """
        if not self.session:
            raise RuntimeError("YouTubeClient must be used as async context manager")
        
        try:
            params = {
                'part': 'snippet',
                'q': query,
                'type': 'video',
                'maxResults': min(max_results, 50),  # YouTube API limit
                'key': self.api_key,
                'order': 'relevance'
            }
            
            if category:
                params['videoCategoryId'] = self._get_category_id(category)
            
            async with self.session.get(f"{self.base_url}/search", params=params) as response:
                if response.status != 200:
                    logger.error(f"YouTube API error: {response.status}")
                    return []
                
                data = await response.json()
                videos = []
                
                for item in data.get('items', []):
                    video = await self._parse_video_item(item)
                    if video:
                        videos.append(video)
                
                return videos
                
        except Exception as e:
            logger.error(f"Error searching YouTube videos: {e}")
            return []

    async def get_video_details(self, video_id: str) -> Optional[YouTubeVideo]:
        """
        Get detailed information about a specific video.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            YouTubeVideo object or None if not found
        """
        if not self.session:
            raise RuntimeError("YouTubeClient must be used as async context manager")
        
        try:
            params = {
                'part': 'snippet,statistics,contentDetails',
                'id': video_id,
                'key': self.api_key
            }
            
            async with self.session.get(f"{self.base_url}/videos", params=params) as response:
                if response.status != 200:
                    logger.error(f"YouTube API error: {response.status}")
                    return None
                
                data = await response.json()
                items = data.get('items', [])
                
                if not items:
                    return None
                
                return await self._parse_video_details(items[0])
                
        except Exception as e:
            logger.error(f"Error getting video details: {e}")
            return None

    async def get_trending_videos(self, category: Optional[str] = None, 
                                max_results: int = 10) -> List[YouTubeVideo]:
        """
        Get trending videos.
        
        Args:
            category: Optional category filter
            max_results: Maximum number of results to return
            
        Returns:
            List of trending YouTubeVideo objects
        """
        if not self.session:
            raise RuntimeError("YouTubeClient must be used as async context manager")
        
        try:
            params = {
                'part': 'snippet',
                'chart': 'mostPopular',
                'maxResults': min(max_results, 50),
                'key': self.api_key,
                'regionCode': 'US'  # Can be made configurable
            }
            
            if category:
                params['videoCategoryId'] = self._get_category_id(category)
            
            async with self.session.get(f"{self.base_url}/videos", params=params) as response:
                if response.status != 200:
                    logger.error(f"YouTube API error: {response.status}")
                    return []
                
                data = await response.json()
                videos = []
                
                for item in data.get('items', []):
                    video = await self._parse_video_item(item)
                    if video:
                        videos.append(video)
                
                return videos
                
        except Exception as e:
            logger.error(f"Error getting trending videos: {e}")
            return []

    async def search_music_videos(self, query: str, max_results: int = 10) -> List[YouTubeVideo]:
        """
        Search specifically for music videos.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of music YouTubeVideo objects
        """
        # Add music-specific terms to the query
        music_query = f"{query} music video official"
        return await self.search_videos(music_query, max_results, category="Music")

    async def _parse_video_item(self, item: Dict[str, Any]) -> Optional[YouTubeVideo]:
        """Parse a video item from YouTube API response."""
        try:
            if not item or not isinstance(item, dict):
                return None
                
            snippet = item.get('snippet', {})
            video_id = item.get('id', {}).get('videoId', '') if item.get('id') else ''
            
            if not video_id:
                return None
            
            return YouTubeVideo(
                video_id=video_id,
                title=snippet.get('title', ''),
                description=snippet.get('description', ''),
                channel_title=snippet.get('channelTitle', ''),
                duration='',  # Not available in search results
                view_count=0,  # Not available in search results
                like_count=0,  # Not available in search results
                published_at=snippet.get('publishedAt', ''),
                thumbnail_url=snippet.get('thumbnails', {}).get('high', {}).get('url', '') if snippet.get('thumbnails') else '',
                video_url=f"https://www.youtube.com/watch?v={video_id}",
                category=snippet.get('categoryId', '')
            )
        except Exception as e:
            logger.error(f"Error parsing video item: {e}")
            return None

    async def _parse_video_details(self, item: Dict[str, Any]) -> Optional[YouTubeVideo]:
        """Parse detailed video information from YouTube API response."""
        try:
            snippet = item.get('snippet', {})
            statistics = item.get('statistics', {})
            content_details = item.get('contentDetails', {})
            video_id = item.get('id', '')
            
            if not video_id:
                return None
            
            return YouTubeVideo(
                video_id=video_id,
                title=snippet.get('title', ''),
                description=snippet.get('description', ''),
                channel_title=snippet.get('channelTitle', ''),
                duration=content_details.get('duration', ''),
                view_count=int(statistics.get('viewCount', 0)),
                like_count=int(statistics.get('likeCount', 0)),
                published_at=snippet.get('publishedAt', ''),
                thumbnail_url=snippet.get('thumbnails', {}).get('high', {}).get('url', ''),
                video_url=f"https://www.youtube.com/watch?v={video_id}",
                category=snippet.get('categoryId', '')
            )
        except Exception as e:
            logger.error(f"Error parsing video details: {e}")
            return None

    def _get_category_id(self, category: str) -> str:
        """Get YouTube category ID from category name."""
        categories = {
            'music': '10',
            'entertainment': '24',
            'comedy': '23',
            'education': '27',
            'science': '28',
            'sports': '17',
            'gaming': '20',
            'news': '25',
            'travel': '19',
            'autos': '2',
            'pets': '15',
            'people': '22',
            'howto': '26',
            'tech': '28',
            'film': '1',
            'animation': '10'
        }
        return categories.get(category.lower(), '10')  # Default to Music

    def format_duration(self, duration: str) -> str:
        """Format YouTube duration (PT4M13S) to readable format (4:13)."""
        if not duration or not duration.startswith('PT'):
            return duration
        
        duration = duration[2:]  # Remove 'PT'
        minutes = 0
        seconds = 0
        
        if 'M' in duration:
            minutes = int(duration.split('M')[0])
            duration = duration.split('M')[1]
        
        if 'S' in duration:
            seconds = int(duration.split('S')[0])
        
        return f"{minutes}:{seconds:02d}"
