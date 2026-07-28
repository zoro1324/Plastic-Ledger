import math
import random
import numpy as np
import datetime
import parcels
from parcels import FieldSet, ParticleSet, ScipyParticle, AdvectionRK4

# Test kernel random call without ParcelsRandom
def Diff(particle, fieldset, time):
    kh = fieldset.Kh
    if kh > 0.0:
        # Using pseudo-random generator based on particle id and time to avoid GCC compile trigger
        seed = math.sin(particle.id * 12.9898 + time * 78.233) * 43758.5453
        r1 = seed - math.floor(seed)
        seed2 = math.sin((particle.id + 1.0) * 12.9898 + (time + 1.0) * 78.233) * 43758.5453
        r2 = seed2 - math.floor(seed2)
        
        u1 = max(0.0001, min(0.9999, r1))
        u2 = max(0.0001, min(0.9999, r2))
        
        r_lon = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        r_lat = math.sqrt(-2.0 * math.log(u1)) * math.sin(2.0 * math.pi * u2)
        
        lat_dist = 111000.0
        lon_dist = 111000.0 * math.cos(particle.lat * math.pi / 180.0)
        dt_abs = math.fabs(particle.dt)
        step_scale = math.sqrt(2.0 * kh * dt_abs)
        
        particle_dlon += (r_lon * step_scale / lon_dist)
        particle_dlat += (r_lat * step_scale / lat_dist)

if __name__ == "__main__":
    fieldset = FieldSet.from_data(
        {'U': np.array([[0.1, 0.1], [0.1, 0.1]], dtype=np.float32),
         'V': np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)},
        {'lon': [0.0, 5.0], 'lat': [40.0, 45.0]}
    )
    fieldset.add_constant('Kh', 1.5)
    pset = ParticleSet.from_list(fieldset=fieldset, pclass=ScipyParticle, lon=[2.24], lat=[41.18])
    kernel = pset.Kernel(AdvectionRK4) + pset.Kernel(Diff)
    pset.execute(kernel, runtime=datetime.timedelta(hours=5), dt=-datetime.timedelta(hours=1))
    print("SUCCESS! Final particle position:", pset.lon[0], pset.lat[0])
