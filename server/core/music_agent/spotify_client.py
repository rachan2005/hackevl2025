"""
Spotify Web API Client

Handles Spotify music search and recommendations.
"""

import aiohttp
import base64
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SpotifyTrack:
    track_id: str
    name: str
    artists: List[str]
    album: str
    duration_ms: int
    popularity: int
    preview_url: Optional[str]
    external_urls: Dict[str, str]
    genres: List[str]
    explicit: bool

@dataclass
class SpotifyPlaylist:
    playlist_id: str
    name: str
    description: str
    owner: str
    tracks_count: int
    external_urls: Dict[str, str]
    images: List[Dict[str, str]]

class SpotifyClient:
    """Client for Spotify Web API."""
    
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.base_url = "https://api.spotify.com/v1"
        self.session: Optional[aiohttp.ClientSession] = None
        self.access_token: Optional[str] = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        await self._authenticate()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def _authenticate(self):
        """Authenticate with Spotify API using Client Credentials flow."""
        try:
            # Encode client credentials
            credentials = f"{self.client_id}:{self.client_secret}"
            encoded_credentials = base64.b64encode(credentials.encode()).decode()
            
            headers = {
                'Authorization': f'Basic {encoded_credentials}',
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            data = {'grant_type': 'client_credentials'}
            
            async with self.session.post('https://accounts.spotify.com/api/token', 
                                       headers=headers, data=data) as response:
                if response.status == 200:
                    token_data = await response.json()
                    self.access_token = token_data.get('access_token')
                    logger.info("Successfully authenticated with Spotify API")
                else:
                    logger.error(f"Spotify authentication failed: {response.status}")
                    
        except Exception as e:
            logger.error(f"Error authenticating with Spotify: {e}")

    async def search_tracks(self, query: str, limit: int = 20, 
                          market: str = 'US') -> List[SpotifyTrack]:
        """
        Search for tracks on Spotify.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            market: Market code (e.g., 'US', 'GB')
            
        Returns:
            List of SpotifyTrack objects
        """
        if not self.access_token:
            logger.error("Not authenticated with Spotify API")
            return []
        
        try:
            headers = {'Authorization': f'Bearer {self.access_token}'}
            params = {
                'q': query,
                'type': 'track',
                'limit': min(limit, 50),  # Spotify API limit
                'market': market
            }
            
            async with self.session.get(f"{self.base_url}/search", 
                                      headers=headers, params=params) as response:
                if response.status != 200:
                    logger.error(f"Spotify API error: {response.status}")
                    return []
                
                data = await response.json()
                tracks = []
                
                for item in data.get('tracks', {}).get('items', []):
                    track = self._parse_track(item)
                    if track:
                        tracks.append(track)
                
                return tracks
                
        except Exception as e:
            logger.error(f"Error searching Spotify tracks: {e}")
            return []

    async def get_track(self, track_id: str) -> Optional[SpotifyTrack]:
        """
        Get detailed information about a specific track.
        
        Args:
            track_id: Spotify track ID
            
        Returns:
            SpotifyTrack object or None if not found
        """
        if not self.access_token:
            logger.error("Not authenticated with Spotify API")
            return None
        
        try:
            headers = {'Authorization': f'Bearer {self.access_token}'}
            
            async with self.session.get(f"{self.base_url}/tracks/{track_id}", 
                                      headers=headers) as response:
                if response.status != 200:
                    logger.error(f"Spotify API error: {response.status}")
                    return None
                
                data = await response.json()
                return self._parse_track(data)
                
        except Exception as e:
            logger.error(f"Error getting track details: {e}")
            return None

    async def get_recommendations(self, seed_tracks: List[str] = None,
                                seed_genres: List[str] = None,
                                seed_artists: List[str] = None,
                                limit: int = 20,
                                **kwargs) -> List[SpotifyTrack]:
        """
        Get track recommendations based on seeds.
        
        Args:
            seed_tracks: List of track IDs to use as seeds
            seed_genres: List of genre names to use as seeds
            seed_artists: List of artist IDs to use as seeds
            limit: Maximum number of recommendations
            **kwargs: Additional parameters (target_energy, target_valence, etc.)
            
        Returns:
            List of recommended SpotifyTrack objects
        """
        if not self.access_token:
            logger.error("Not authenticated with Spotify API")
            return []
        
        try:
            headers = {'Authorization': f'Bearer {self.access_token}'}
            params = {'limit': min(limit, 100)}  # Spotify API limit
            
            # Add seeds
            if seed_tracks:
                params['seed_tracks'] = ','.join(seed_tracks[:5])  # Max 5 seeds
            if seed_genres:
                params['seed_genres'] = ','.join(seed_genres[:5])
            if seed_artists:
                params['seed_artists'] = ','.join(seed_artists[:5])
            
            # Add additional parameters
            for key, value in kwargs.items():
                if key.startswith('target_') or key.startswith('min_') or key.startswith('max_'):
                    params[key] = value
            
            async with self.session.get(f"{self.base_url}/recommendations", 
                                      headers=headers, params=params) as response:
                if response.status != 200:
                    logger.error(f"Spotify API error: {response.status}")
                    return []
                
                data = await response.json()
                tracks = []
                
                for item in data.get('tracks', []):
                    track = self._parse_track(item)
                    if track:
                        tracks.append(track)
                
                return tracks
                
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return []

    async def search_playlists(self, query: str, limit: int = 20) -> List[SpotifyPlaylist]:
        """
        Search for playlists on Spotify.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            
        Returns:
            List of SpotifyPlaylist objects
        """
        if not self.access_token:
            logger.error("Not authenticated with Spotify API")
            return []
        
        try:
            headers = {'Authorization': f'Bearer {self.access_token}'}
            params = {
                'q': query,
                'type': 'playlist',
                'limit': min(limit, 50)
            }
            
            async with self.session.get(f"{self.base_url}/search", 
                                      headers=headers, params=params) as response:
                if response.status != 200:
                    logger.error(f"Spotify API error: {response.status}")
                    return []
                
                data = await response.json()
                playlists = []
                
                for item in data.get('playlists', {}).get('items', []):
                    playlist = self._parse_playlist(item)
                    if playlist:
                        playlists.append(playlist)
                
                return playlists
                
        except Exception as e:
            logger.error(f"Error searching playlists: {e}")
            return []

    async def get_available_genres(self) -> List[str]:
        """
        Get list of available genre seeds for recommendations.
        
        Returns:
            List of genre names
        """
        if not self.access_token:
            logger.error("Not authenticated with Spotify API")
            return []
        
        try:
            headers = {'Authorization': f'Bearer {self.access_token}'}
            
            async with self.session.get(f"{self.base_url}/recommendations/available-genre-seeds", 
                                      headers=headers) as response:
                if response.status != 200:
                    logger.error(f"Spotify API error: {response.status}")
                    return []
                
                data = await response.json()
                return data.get('genres', [])
                
        except Exception as e:
            logger.error(f"Error getting available genres: {e}")
            return []

    def _parse_track(self, item: Dict[str, Any]) -> Optional[SpotifyTrack]:
        """Parse a track item from Spotify API response."""
        try:
            track_id = item.get('id', '')
            if not track_id:
                return None
            
            artists = [artist.get('name', '') for artist in item.get('artists', [])]
            album = item.get('album', {}).get('name', '')
            
            return SpotifyTrack(
                track_id=track_id,
                name=item.get('name', ''),
                artists=artists,
                album=album,
                duration_ms=item.get('duration_ms', 0),
                popularity=item.get('popularity', 0),
                preview_url=item.get('preview_url'),
                external_urls=item.get('external_urls', {}),
                genres=[],  # Not available in track object
                explicit=item.get('explicit', False)
            )
        except Exception as e:
            logger.error(f"Error parsing track: {e}")
            return None

    def _parse_playlist(self, item: Dict[str, Any]) -> Optional[SpotifyPlaylist]:
        """Parse a playlist item from Spotify API response."""
        try:
            if not item or not isinstance(item, dict):
                return None
                
            playlist_id = item.get('id', '')
            if not playlist_id:
                return None
            
            return SpotifyPlaylist(
                playlist_id=playlist_id,
                name=item.get('name', ''),
                description=item.get('description', ''),
                owner=item.get('owner', {}).get('display_name', '') if item.get('owner') else '',
                tracks_count=item.get('tracks', {}).get('total', 0) if item.get('tracks') else 0,
                external_urls=item.get('external_urls', {}),
                images=[img.get('url', '') for img in item.get('images', []) if img]
            )
        except Exception as e:
            logger.error(f"Error parsing playlist: {e}")
            return None

    def format_duration(self, duration_ms: int) -> str:
        """Format duration in milliseconds to readable format (MM:SS)."""
        seconds = duration_ms // 1000
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes}:{seconds:02d}"
