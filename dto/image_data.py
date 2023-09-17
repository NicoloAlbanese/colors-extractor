from pydantic import BaseModel
from typing import List


class Color(BaseModel):
    '''
    Color representation as RGB values.
    '''
    R: int
    G: int
    B: int


class ColorInfo(BaseModel):
    '''
    Information about a color: RGB and percentage of pixels across image.
    '''
    color: Color
    percentage: str


class ColorExtractionRequest(BaseModel):
    '''
    Colors extraction request.
    '''
    url_or_path: str
    num_clusters: int = 4 # Default to 4 most predominant colors if not provided


class ColorExtractionResponse(BaseModel):
    '''
    Color extraction response from an image analysis request.
    '''
    predominant_colors: List[ColorInfo]
