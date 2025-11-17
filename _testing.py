import numpy as np

intersections = np.array([
    [351.2990283,-41.2027070],
    [351.2986312,-41.2025634],
    [351.2984311,-41.2015725],
])

centers = np.array([
    [351.2994934,-41.2009422],
    [351.2994299,-41.1979980],
    [351.3014218,-41.1970784],
])

nearest_points = np.array([
    [351.2985903,-41.2025503],
    [351.2968536,-41.1992017],
    [351.2978192,-41.1971508],
])

quad_ids = np.array(['2', '3', '6'])

quad_offsets = centers - nearest_points

inter_mean = np.mean(intersections, axis=0)
pred = np.array([351.2987392,-41.2023645])
sze = np.array([351.2988, -41.2037])
xray = np.array([351.3020513,-41.1977812])

sze_offset = pred - sze
xray_offset = pred - xray

for quad_id, quad_offset in zip(quad_ids, quad_offsets):
    print(f'Quad {quad_id} offset: {np.round(np.linalg.norm(quad_offset)*3600, 3)} arcseconds')
print(f'SZE Offset: {np.round(np.linalg.norm(sze_offset)*3600, 3)} arcseconds')
print(f'X-ray Offset: {np.round(np.linalg.norm(xray_offset)*3600, 3)} arcseconds')
print(f'Intersections Range: {np.round(np.ptp(intersections, axis=0)*3600, 3)} arcseconds')