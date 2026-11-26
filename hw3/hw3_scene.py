import json
import numpy as np
import buffer
import math
import os
import slangpy as spy
import trimesh

SPHERE = 0  # Must match Slang's ShapeType in hw3_common.slang
TRIANGLE_MESH = 1

def parse_triangle_mesh(filename):
    if filename.lower().endswith('.ply'):
        mesh = trimesh.load_mesh(filename, process=False)

        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.uint32)
        vertex_normals = np.asarray(
            mesh.vertex_normals, dtype=np.float32)

    else:
        with open(filename) as f:
            mesh = json.load(f)

        vertices = np.array(mesh['vertices'], dtype=np.float32)
        faces = np.array(mesh['faces'], dtype=np.uint32)

        if 'vertex_normals' in mesh:
            vertex_normals = np.array(
                mesh['vertex_normals'], dtype=np.float32)
        else:
            # Generate smooth vertex normals.
            tri_mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                process=False)
            vertex_normals = np.asarray(
                tri_mesh.vertex_normals, dtype=np.float32)

    return vertices, faces, vertex_normals

def compose_transformations(transformations):
    """
        Given a list of transformations, compose them into a matrix.
    """
    F = np.eye(4, dtype=np.float32)

    for transform in transformations:
        matrix = np.eye(4, dtype=np.float32)

        if "scale" in transform:
            values = transform["scale"]
            matrix = np.array([[values[0],        0 ,        0 , 0],
                                [       0 , values[1],        0 , 0],
                                [       0 ,        0 , values[2], 0],
                                [       0 ,        0 ,        0 , 1]])
    
        elif "translate" in transform:
            values = transform["translate"]
            matrix = np.array([[       1 ,        0 ,        0 , values[0]],
                                [       0 ,        1 ,        0 , values[1]],
                                [       0 ,        0 ,        1 , values[2]],
                                [       0 ,        0 ,        0 ,        1 ]])

        elif "rotate_x" in transform:
            values = transform["rotate_x"]
            theta = np.radians(values[0])
            matrix = np.array([[       1 ,             0 ,              0 ,        0 ],
                                [       0 , np.cos(theta) , -np.sin(theta) ,        0 ],
                                [       0 , np.sin(theta) ,  np.cos(theta) ,        0 ],
                                [       0 ,             0 ,              0 ,        1 ]])

        elif "rotate_y" in transform:
            values = transform["rotate_y"]
            theta = np.radians(values[0])
            matrix = np.array([[  np.cos(theta) ,         0 , np.sin(theta) ,        0 ],
                                [              0 ,         1 ,             0 ,        0 ],
                                [ -np.sin(theta) ,         0 , np.cos(theta) ,        0 ],
                                [              0 ,         0 ,             0 ,        1 ]])

        elif "rotate_z" in transform:
            values = transform["rotate_z"]
            theta = np.radians(values[0])
            matrix = np.array([[  np.cos(theta) , -np.sin(theta) ,             0 ,        0 ],
                                [  np.sin(theta) ,  np.cos(theta) ,             0 ,        0 ],
                                [              0 ,              0 ,             1 ,        0 ],
                                [              0 ,              0 ,             0 ,        1 ]])

        elif "lookat" in transform:
            values = transform["lookat"]
            p = np.array(values["position"])
            t = np.array(values["target"])
            u = np.array(values["up"])

            d = (t-p) / np.linalg.norm(t-p)
            r = np.cross(d, u) / np.linalg.norm(np.cross(d, u))
            u_prime = np.cross(r, d)

            matrix = np.array([[r[0], u_prime[0], -d[0], p[0]],
                                [r[1], u_prime[1], -d[1], p[1]],
                                [r[2], u_prime[2], -d[2], p[2]],
                                [  0 ,         0 ,    0 ,   1 ]])
        
        F = matrix @ F
        

    return F

def interpolate_transformation(transform_keyframes, t):
    # Find the two keyframes surrounding t and linearly
    # interpolate the corresponding transformation parameters.
    #
    # Return a list of transformations in the same format
    # as the "transform" field in the scene JSON.
    if t <= transform_keyframes[0]["time"]:
            return transform_keyframes[0]["transform"]
    
    if t >= transform_keyframes[-1]["time"]:
        return transform_keyframes[-1]["transform"]
        
    for i in range(len(transform_keyframes) - 1):
        t0 = transform_keyframes[i]["time"]
        t1 = transform_keyframes[i + 1]["time"]
    
        if t >= t0 and t <= t1:
            w = (t - t0)/(t1 - t0)
    
            keyframe0 = transform_keyframes[i]["transform"]
            keyframe1 = transform_keyframes[i + 1]["transform"]
    
            interpolated_transforms = []
    
            for j in range(len(keyframe0)):
                transform0 = keyframe0[j]
                transform1 = keyframe1[j]
    
                transform_type = next(iter(transform0))
    
                values0 = transform0[transform_type]
                values1 = transform1[transform_type]

                if transform_type == "lookat":
                    interpolated_values = {}

                    for key in values0:
                        v0 = np.array(values0[key])
                        v1 = np.array(values1[key])

                        interpolated_values[key] = ( (1-w) * v0 + w * v1).tolist()
                else:
                    interpolated_values = []

                    for k in range(len(values0)):
                        value = (1 - w) * values0[k] + w * values1[k]
                        interpolated_values.append(value)
    
                interpolated = {transform_type: interpolated_values}
    
                interpolated_transforms.append(interpolated)
    
            return interpolated_transforms

def get_transformation(obj, t):
    transforms = obj.get('transform', [])
    if 'transform_keyframes' in obj:
        transforms = interpolate_transformation(
            obj['transform_keyframes'], t)
    return compose_transformations(transforms)

def build_bvh(vertices, faces, triangle_offset, bvh_nodes):
    # A recursive BVH builder that split the longest axis
    # using the median of the centroids. The leaf nodes
    # have maximum of 4 triangles.

    assert(len(faces) > 0)
    # Needs a copy because we will sort them during the recursion
    faces = faces.copy()
    centers = vertices[faces].mean(axis=1)

    def build(begin, end):
        node_id = len(bvh_nodes)
        # Important: append a node first
        # before the recursion, so that the order
        # is "preorder" 
        bvh_nodes.append(None)

        # Bounding box of all triangles in this node
        points = vertices[faces[begin:end]].reshape(-1, 3)
        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)

        # Stop recursion if the node has fewer than 4 triangles
        is_leaf = end - begin <= 4
        if not is_leaf:
            # Find the longest axis
            c = centers[begin:end]
            axis = np.argmax(c.max(axis=0) - c.min(axis=0))

            # Sort triangles by their centroids
            order = np.argsort(c[:, axis])
            faces[begin:end] = faces[begin:end][order]
            centers[begin:end] = c[order]

            # Split
            mid = (begin + end) // 2
            build(begin, mid)
            build(mid, end)

        bvh_nodes[node_id] = {
            'bbox_min': bbox_min,
            'bbox_max': bbox_max,
            'next_index': len(bvh_nodes),
            'triangle_begin': triangle_offset + begin,
            'triangle_end': triangle_offset + end,
            'is_leaf': is_leaf
        }

        return node_id

    root = build(0, len(faces))
    return root, faces

def upload_scene(filename, module, slang_device, use_bvh = False):
    with open(filename) as f:
        scene = json.load(f)

    background = scene.get('background', (0.5, 0.5, 0.5))

    camera = {'cam_to_world': np.eye(4, dtype=np.float32),
              'fov_x': 90.0}

    if 'camera' in scene:
        camera['cam_to_world'] = \
            get_transformation(scene['camera'], 0.0)
        camera['fov_x'] = scene['camera'].get('fov_x', 90.0)

    # Triangle meshes loading + BVH building
    mesh_descs = []
    vertices_list = []
    normals_list = []
    triangles_list = []
    bvh_nodes = []

    vertex_offset = 0
    triangle_offset = 0

    scene_dir = os.path.dirname(filename)

    for mesh_data in scene.get('meshes', []):
        mesh_filename = os.path.join(
            scene_dir, mesh_data['filename'])
        vertices, faces, vertex_normals = \
            parse_triangle_mesh(mesh_filename)

        # Construct the BVH and reorder triangles
        # The returned faces is a copy of our
        # original faces.
        if use_bvh:
            bvh_root_index, faces = build_bvh(
                vertices, faces, triangle_offset, bvh_nodes)
        else:
            bvh_root_index = 0

        mesh_descs.append({
            'vertex_offset': vertex_offset,
            'triangle_begin': triangle_offset,
            'triangle_end': triangle_offset + len(faces),
            'bvh_root_index': bvh_root_index
        })

        vertices_list.append(vertices)
        normals_list.append(vertex_normals)
        triangles_list.append(faces)

        vertex_offset += len(vertices)
        triangle_offset += len(faces)

    # Merge into global arrays
    if len(mesh_descs) > 0:
        vertices = np.concatenate(vertices_list).astype(np.float32)
        normals = np.concatenate(normals_list).astype(np.float32)
        triangles = np.concatenate(triangles_list).astype(np.uint32)
    else:
        # Use dummy entries if the scene has no meshes
        vertices = np.zeros((1, 3), dtype=np.float32)
        normals = np.zeros((1, 3), dtype=np.float32)
        triangles = np.zeros((1, 3), dtype=np.uint32)

    # Parse object instances.
    spheres = []
    shapes = []

    for obj in scene['objects']:
        if obj['type'] == 'sphere':
            shape_type = SPHERE
            shape_index = len(spheres)
            spheres.append({
                'center': obj['center'],
                'radius': obj['radius']
            })
        elif obj['type'] == 'mesh':
            shape_type = TRIANGLE_MESH
            shape_index = obj['index']
        else:
            raise ValueError(f"Unknown shape type: {obj['type']}")

        obj_to_world = get_transformation(obj, 0.0)
        world_to_obj = np.linalg.inv(obj_to_world)

        shapes.append({
            'type': shape_type,
            'index': shape_index,
            'Kd': obj['Kd'],
            'Kr': obj['Kr'],
            'world_to_obj': world_to_obj,
            'obj_to_world': obj_to_world
        })

    lights = scene['lights']

    max_depth = scene.get('max_depth', 20)

    bindings = {
        'background': spy.float3(background),
        'camera': camera,
        'shapes': buffer.create_structured_buffer(
            slang_device, module.Shape, shapes),
        'num_shapes': len(shapes),
        'spheres': buffer.create_structured_buffer(
            slang_device, module.Sphere, spheres),
        'meshes': buffer.create_structured_buffer(
            slang_device, module.TriangleMesh, mesh_descs),
        'vertices': buffer.structured_typed_buffer_from_numpy(
            slang_device, vertices, 'float3'),
        'vertex_normals': buffer.structured_typed_buffer_from_numpy(
            slang_device, normals, 'float3'),
        'triangles': buffer.structured_typed_buffer_from_numpy(
            slang_device, triangles, 'uint3'),
        'bvh_nodes': buffer.create_structured_buffer(
            slang_device, module.BVHNode, bvh_nodes),
        'lights': buffer.create_structured_buffer(
            slang_device, module.PointLight, lights),
        'num_lights': len(lights),
        'max_depth': max_depth
    }

    return bindings, scene, shapes
