"""Fixed mapped-river route and hydraulic-port-proxy definitions for Phase 2C-fix."""
from functools import lru_cache
from pathlib import Path
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import substring

ENTRY = (334051.8, 3143392.5)
OFFICIAL_PORT = (340837.9, 3129051.7)
PROXY = (340571.8257777202, 3129640.775894154)
# Existing model-entry documentation measured this vector route as 17.624 km.
ROUTE_DISTANCE_M = 17624.0
CORRIDOR_TOLERANCE_M = 320.0

@lru_cache(maxsize=1)
def route():
    root = Path(__file__).resolve().parents[3]
    shp = next((root/'data/extracted/basic_geographic_data').rglob('四级河流.shp'))
    multi = gpd.read_file(shp).to_crs(32645).geometry.iloc[0]
    a,b = multi.geoms
    start = a.project(Point(ENTRY))
    upstream_tail = substring(a, start, a.length)
    # Preserve the downloaded map's explicit inter-component gap; it is not a
    # surveyed channel segment.  The analysis distance is normalized to the
    # already documented 17.624-km entry-to-terminus mapped-route distance.
    coords = list(upstream_tail.coords) + list(b.coords)
    return LineString(coords)

def route_scale(): return ROUTE_DISTANCE_M/route().length

def nearest_s(x, y):
    p=Point(float(x),float(y)); line=route(); d=line.distance(p)
    return (line.project(p)*route_scale(), d)

def local_flow_direction():
    c=list(route().coords); v=np.array(c[-1])-np.array(c[-2]); return v/np.linalg.norm(v)
