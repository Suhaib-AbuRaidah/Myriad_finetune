"""Resample original URDF visual surfaces for display, without a renderer."""
from pathlib import Path, PurePosixPath
import posixpath
import tarfile
import xml.etree.ElementTree as ET

import numpy as np


def rotation_rpy(rpy):
    roll, pitch, yaw = rpy
    cr, cp, cy = np.cos([roll, pitch, yaw])
    sr, sp, sy = np.sin([roll, pitch, yaw])
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
        [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]])


def find_archive(metadata, asset_root):
    obj = metadata['object']
    relative = PurePosixPath(obj['source_archive'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Expected a relative object archive reference')
    root = Path(asset_root).expanduser().resolve()
    category, subcategory = obj['category'], obj['subcategory']
    candidates = [root/str(relative), root/category/'sub_categories'/subcategory/'fit'/
        obj['source_split']/relative.name]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f'Original surface archive not found under {root}. '
        'Set --asset-root to the Articraft-10K folder or use --cloud-points 0 for the saved cloud.')


def read_obj(data):
    vertices, faces = [], []
    for line in data.decode('utf-8-sig').splitlines():
        fields = line.split('#',1)[0].split()
        if not fields:
            continue
        if fields[0] == 'v':
            vertices.append([float(x) for x in fields[1:4]])
        elif fields[0] == 'f':
            indices = [int(field.split('/')[0]) for field in fields[1:]]
            indices = [i-1 if i>0 else len(vertices)+i for i in indices]
            faces.extend([[indices[0],indices[j],indices[j+1]] for j in range(1,len(indices)-1)])
    vertices, faces = np.asarray(vertices,float), np.asarray(faces,int)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError('OBJ has no usable triangle surface')
    if not np.isfinite(vertices).all() or faces.min()<0 or faces.max()>=len(vertices):
        raise ValueError('OBJ contains invalid vertices or face indices')
    return vertices, faces


def load_surfaces(source, asset_root):
    metadata = source['metadata']
    archive = find_archive(metadata, asset_root)
    with tarfile.open(archive, 'r:*') as bundle:
        members = [m for m in bundle.getmembers() if m.isfile() and m.name.endswith('.urdf')]
        preferred = [m for m in members if PurePosixPath(m.name).name == 'model.urdf']
        if len(preferred or members) != 1:
            raise ValueError(f'Expected one model URDF in {archive}')
        member = (preferred or members)[0]
        robot = ET.fromstring(bundle.extractfile(member).read())
        # Read referenced meshes in memory; never extract archive paths to disk.
        asset_bytes = {m.name:bundle.extractfile(m).read() for m in bundle.getmembers()
            if m.isfile() and m.name.lower().endswith('.obj')}
        urdf_directory = PurePosixPath(member.name).parent
    scale = float(source['object_scale'])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Object scale must be finite and positive')
    links = {int(link['bullet_link_index']):link['name'] for link in metadata['links']}
    urdf_links = {link.attrib['name']:link for link in robot.findall('link')}
    verification = metadata['object'].get('canonical',{}).get('verification',{})
    baked_rotation = rotation_rpy(np.radians(verification.get('applied_object_euler_deg',[0,0,0])))
    per_link_rotation = verification.get('applied_link_euler_deg',{})
    if any(np.any(np.abs(v)>1e-8) for v in per_link_rotation.values()):
        raise ValueError('Unsupported additional baked link rotation; use --cloud-points 0.')
    surfaces = []
    for node, label in enumerate(source['link_indices']):
        name = links[int(label)]
        if name not in urdf_links:
            raise ValueError(f'Missing URDF link {name!r} in {archive}')
        for visual in urdf_links[name].findall('visual'):
            geometry = visual.find('geometry')
            if geometry is None or not len(geometry):
                continue
            shape = geometry[0]
            extra = {}
            if shape.tag == 'box':
                size = np.fromstring(shape.attrib['size'], sep=' ')
                area = 2*(size[0]*size[1]+size[0]*size[2]+size[1]*size[2])
            elif shape.tag == 'cylinder':
                size = np.array([float(shape.attrib['radius']), float(shape.attrib['length'])])
                area = 2*np.pi*size[0]*(size[0]+size[1])
            elif shape.tag == 'sphere':
                size = np.array([float(shape.attrib['radius'])])
                area = 4*np.pi*size[0]**2
            elif shape.tag == 'mesh':
                filename = shape.attrib['filename']
                reference = posixpath.normpath(str(urdf_directory/filename))
                if reference not in asset_bytes:
                    raise ValueError(f'OBJ surface {filename!r} not found in {archive}')
                vertices, faces = read_obj(asset_bytes[reference])
                mesh_scale = np.fromstring(shape.attrib.get('scale','1 1 1'),sep=' ')
                vertices = vertices*mesh_scale
                triangles = vertices[faces]
                areas = .5*np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],
                    triangles[:,2]-triangles[:,0]),axis=1)
                keep = areas>1e-15
                triangles, areas = triangles[keep], areas[keep]
                area = areas.sum()
                if not np.isfinite(area) or area <= 0:
                    raise ValueError('OBJ has no positive surface area')
                size = np.ones(1)
                extra = dict(triangles=triangles,triangle_areas=areas)
            else:
                raise ValueError(f'Unsupported visual geometry {shape.tag!r} in {archive}. '
                    'Use --cloud-points 0 for the saved cloud.')
            if not np.isfinite(size).all() or np.any(size <= 0):
                raise ValueError(f'Invalid {shape.tag} dimensions in {archive}')
            origin = visual.find('origin')
            attributes = {} if origin is None else origin.attrib
            offset = baked_rotation@np.fromstring(attributes.get('xyz','0 0 0'), sep=' ')
            rotation = baked_rotation@rotation_rpy(np.fromstring(attributes.get('rpy','0 0 0'), sep=' '))
            surfaces.append(dict(kind=shape.tag, size=size, area=float(area)*scale**2,
                offset=offset, rotation=rotation, transform=source['link_transforms_world'][0,node].astype(float),
                scale=scale, label=int(label),**extra))
    if not surfaces:
        raise ValueError(f'No visual surfaces found in {archive}')
    return surfaces, archive


def sample_surface(surface, count, rng):
    kind, size = surface['kind'], surface['size']
    if kind == 'box':
        areas = np.array([size[1]*size[2],size[0]*size[2],size[0]*size[1]])
        axes = rng.choice(3,size=count,p=areas/areas.sum())
        points = rng.uniform(-.5,.5,(count,3))*size
        points[np.arange(count),axes] = rng.choice([-1.,1.],size=count)*size[axes]/2
    elif kind == 'cylinder':
        radius, height = size
        angle = rng.uniform(0,2*np.pi,count)
        lateral = rng.random(count) < height/(height+radius)
        radial = np.where(lateral,radius,radius*np.sqrt(rng.random(count)))
        z = np.where(lateral,rng.uniform(-height/2,height/2,count),rng.choice([-1.,1.],size=count)*height/2)
        points = np.column_stack((radial*np.cos(angle),radial*np.sin(angle),z))
    elif kind == 'sphere':
        direction = rng.normal(size=(count,3))
        points = size[0]*direction/np.linalg.norm(direction,axis=1,keepdims=True)
    else:
        areas = surface['triangle_areas']
        triangles = surface['triangles'][rng.choice(len(areas),size=count,p=areas/areas.sum())]
        u = np.sqrt(rng.random(count))
        v = rng.random(count)
        points = ((1-u)[:,None]*triangles[:,0]+(u*(1-v))[:,None]*triangles[:,1]+(u*v)[:,None]*triangles[:,2])
    local = (points@surface['rotation'].T+surface['offset'])*surface['scale']
    t = surface['transform']
    return local@t[:3,:3].T+t[:3,3]


def distance_to_surface(points, surface):
    """Analytic distance to the closed primitive; used to check frame alignment."""
    t = surface['transform']
    local = ((points-t[:3,3])@np.linalg.inv(t[:3,:3]).T)/surface['scale']
    p = (local-surface['offset'])@surface['rotation']
    size, kind = surface['size'], surface['kind']
    if kind == 'box':
        q = np.abs(p)-size/2
        distance = np.linalg.norm(np.maximum(q,0),axis=1)+np.minimum(np.max(q,axis=1),0)
    elif kind == 'cylinder':
        q = np.column_stack((np.linalg.norm(p[:,:2],axis=1)-size[0],np.abs(p[:,2])-size[1]/2))
        distance = np.linalg.norm(np.maximum(q,0),axis=1)+np.minimum(np.max(q,axis=1),0)
    elif kind == 'sphere':
        distance = np.linalg.norm(p,axis=1)-size[0]
    else:
        # CPU-only raycasting for an independent distance-to-triangle check.
        import open3d as o3d
        if '_raycast' not in surface:
            triangles = surface['triangles']
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(o3d.core.Tensor(triangles.reshape(-1,3).astype(np.float32)),
                o3d.core.Tensor(np.arange(triangles.size//3,dtype=np.uint32).reshape(-1,3)))
            surface['_raycast'] = scene
        distance = surface['_raycast'].compute_distance(o3d.core.Tensor(p.astype(np.float32))).numpy()
    return np.abs(distance)*surface['scale']


def dense_points(source, asset_root, count=10000, seed=42):
    surfaces, archive = load_surfaces(source,asset_root)
    # Verify the archive and link/scale conventions against the already saved
    # mesh samples before drawing a denser surface in the wrong coordinate frame.
    errors = []
    for label in np.unique(source['mesh_point_link_labels']):
        saved = source['mesh_points'][source['mesh_point_link_labels']==label]
        saved = saved[np.isfinite(saved).all(-1)]
        choices = [s for s in surfaces if s['label']==int(label)]
        if len(saved) and not choices:
            raise ValueError(f'No original visual surfaces for saved part {label}')
        if len(saved):
            errors.extend(np.min(np.stack([distance_to_surface(saved,s) for s in choices]),axis=0))
    maximum = float(max(errors)) if errors else float('inf')
    extent = float(np.linalg.norm(np.ptp(source['mesh_points'],axis=0)))
    # Archived cylinder/sphere samples may lie on tessellated facets, slightly
    # inside the analytic surface. Allow 0.2% of the object diagonal for this.
    if maximum > max(2e-5,extent*2e-3):
        raise ValueError(f'Original surface alignment check failed (max distance {maximum:.6g}). '
            'Use the matching compiled assets or --cloud-points 0.')
    rng = np.random.default_rng(seed)
    areas = np.array([s['area'] for s in surfaces])
    counts = rng.multinomial(count,areas/areas.sum())
    points = np.concatenate([sample_surface(s,int(n),rng) for s,n in zip(surfaces,counts) if n])
    labels = np.concatenate([np.full(n,s['label'],dtype=int) for s,n in zip(surfaces,counts) if n])
    return points, labels, dict(archive=str(archive), points=len(points),
        original_to_surface_max_distance=maximum, geometry='URDF visual surfaces at initial saved link poses')
