from mpi4py import MPI
import numpy as np
import gmsh
gmsh.initialize()

import dolfinx.plot as plot
from dolfinx import fem, mesh, io, default_scalar_type
from dolfinx.fem import Function, functionspace, Constant
from dolfinx.mesh import (CellType, compute_midpoints, create_unit_cube,
                          create_unit_square, meshtags, create_box, create_mesh, refine)
from dolfinx.io.gmshio import model_to_mesh
from dolfinx.io import gmshio

normal_skin_length = 1
dermis_thickness = 0.1
scar_width = 0.1
epidermis_thickness = 0.01
subcutis_thickness = 0.3
scar_thickness = dermis_thickness + epidermis_thickness + subcutis_thickness
N = 10
lc = normal_skin_length / N

force_left_marker, force_right_marker, free_marker, wound_marker = 1, 2, 3, 4

def linearScarMesh(normal_skin_length=normal_skin_length, \
            epidermis_thickness=epidermis_thickness,
            dermis_thickness=dermis_thickness,
            subcutis_thickness=subcutis_thickness,
            scar_width=scar_width,
            scar_height=normal_skin_length,
            scar_thickness=scar_thickness, \
            lc=lc):
    scar_bottom = -dermis_thickness - epidermis_thickness - subcutis_thickness
    skin_thickness = -scar_bottom
    skin_left = np.array([[-normal_skin_length, -normal_skin_length, scar_bottom], [0, normal_skin_length, 0]])
    skin_right = np.array([[scar_width, -normal_skin_length, scar_bottom], [scar_width + normal_skin_length, normal_skin_length, 0]])
    gmsh.clear()
    gmsh.option.setNumber("General.Terminal", 0)
    model = gmsh.model()
    model.add("DFG 3D")
    model.occ.addBox(*skin_left[0], *(skin_left[1] - skin_left[0]))
    model.occ.addBox(*skin_right[0], *(skin_right[1] - skin_right[0]))
    model.occ.addBox(0, -scar_height, scar_bottom, scar_width, 2 * scar_height, scar_thickness)
    model.occ.synchronize()
    
    volumes = model.getEntities(dim=3)
    model.occ.fuse([volumes[0]], volumes[1:])
    gmsh.model.occ.synchronize()
    
    surfaces = model.occ.getEntities(dim=2)
    walls = []
    for surface in surfaces:
        com = model.occ.getCenterOfMass(surface[0], surface[1])
        if np.allclose(com, [-normal_skin_length, 0, -skin_thickness / 2]):
            model.addPhysicalGroup(surface[0], [surface[1]], force_left_marker)
            left_force = surface[1]
            model.setPhysicalName(surface[0], force_left_marker, "Left force")
        elif np.allclose(com, [scar_width + normal_skin_length, 0, -skin_thickness / 2]):
            model.addPhysicalGroup(surface[0], [surface[1]], force_right_marker)
            right_force = surface[1]
            model.setPhysicalName(surface[0], force_right_marker, "Right force")
        elif np.allclose(com, [0, 0, -skin_thickness / 2]) or np.allclose(com, [scar_width, 0, -skin_thickness / 2]):
            continue
        else:
            walls.append(surface[1])

    model.addPhysicalGroup(2, walls, free_marker)
    model.setPhysicalName(2, free_marker, "Free wall")
    
    model.addPhysicalGroup(3, [i[1] for i in volumes], 4)
    model.setPhysicalName(3, 4, 'Volumes')

    gmsh.model.mesh.field.add("Box", 6)
    gmsh.model.mesh.field.setNumber(6, "VIn", lc)
    gmsh.model.mesh.field.setNumber(6, "VOut", lc)
    gmsh.model.mesh.field.setNumber(6, "XMin", 0.3)
    gmsh.model.mesh.field.setNumber(6, "XMax", 0.6)
    gmsh.model.mesh.field.setNumber(6, "YMin", 0.3)
    gmsh.model.mesh.field.setNumber(6, "YMax", 0.6)
    gmsh.model.mesh.field.setNumber(6, "Thickness", 0.3)
    gmsh.model.mesh.field.add("Min", 7)
    gmsh.model.mesh.field.setNumbers(7, "FieldsList", [2, 3, 5, 6])
    gmsh.model.mesh.field.setAsBackgroundMesh(7)
    

    model.occ.synchronize()
    model.mesh.generate(3)
    model_rank = 0
    mesh, cell_tags, facet_tags = model_to_mesh(model, MPI.COMM_WORLD, model_rank)
    return mesh, cell_tags, facet_tags

def cylindricalLinearScarMesh(normal_skin_length=normal_skin_length, \
            epidermis_thickness=epidermis_thickness,
            dermis_thickness=dermis_thickness,
            subcutis_thickness=subcutis_thickness,
            scar_width=scar_width,
            scar_height=normal_skin_length,
            scar_thickness=scar_thickness, height_offset=0, \
            lc=lc):
    scar_bottom = -dermis_thickness - epidermis_thickness - subcutis_thickness
    skin_thickness = -scar_bottom
    skin_left = np.array([[-normal_skin_length, -normal_skin_length, -skin_thickness], [0, normal_skin_length, 0]])
    skin_right = np.array([[scar_width, -normal_skin_length, -skin_thickness], [scar_width + normal_skin_length, normal_skin_length, 0]])
    gmsh.clear()
    cylinderCenter = [scar_width / 2, -normal_skin_length, -height_offset]
    cylinderAxis = [0, 2 * normal_skin_length, 0]
    cylinderRadius = np.sqrt(np.square(scar_width / 2) + np.square(height_offset))
    
    gmsh.option.setNumber("General.Terminal", 0)
    model = gmsh.model()
    model.add("DFG 3D")
    occ = model.occ
    occ.addBox(*skin_left[0], *(skin_left[1] - skin_left[0]))
    occ.addBox(*skin_right[0], *(skin_right[1] - skin_right[0]))
    occ.addBox(0, -scar_height, scar_bottom, scar_width, 2 * scar_height, scar_thickness)
    scar_tag = occ.addCylinder(*cylinderCenter, *cylinderAxis, cylinderRadius)
    model.occ.synchronize()

    box_tag = occ.addBox(-normal_skin_length - cylinderRadius, -normal_skin_length - cylinderRadius, -skin_thickness, 2 * (normal_skin_length + cylinderRadius) + scar_width,
                         2 * (normal_skin_length + cylinderRadius), -1.1 * cylinderRadius)
    
    occ.cut([(3,scar_tag)], [(3,box_tag)])
    occ.synchronize()
    volumes = model.getEntities(dim=3)
    occ.fuse([volumes[0]], volumes[1:])
    occ.synchronize()
    
    surfaces = model.occ.getEntities(dim=2)
    walls = []
    for surface in surfaces:
        com = model.occ.getCenterOfMass(surface[0], surface[1])
        if np.allclose(com, [-normal_skin_length, 0, -skin_thickness / 2]):
            model.addPhysicalGroup(surface[0], [surface[1]], force_left_marker)
            left_force = surface[1]
            model.setPhysicalName(surface[0], force_left_marker, "Left force")
        elif np.allclose(com, [scar_width + normal_skin_length, 0, -skin_thickness / 2]):
            model.addPhysicalGroup(surface[0], [surface[1]], force_right_marker)
            right_force = surface[1]
            model.setPhysicalName(surface[0], force_right_marker, "Right force")
        elif np.allclose(com, [0, 0, -skin_thickness / 2]) or np.allclose(com, [scar_width, 0, -skin_thickness / 2]):
            continue
        else:
            walls.append(surface[1])

    model.addPhysicalGroup(2, walls, free_marker)
    model.setPhysicalName(2, free_marker, "Free wall")
    
    model.addPhysicalGroup(3, [i[1] for i in volumes], 4)
    model.setPhysicalName(3, 4, 'Volumes')

    field = gmsh.model.mesh.field
    field.add("Box", 6)
    field.setNumber(6, "VIn", lc)
    field.setNumber(6, "VOut", lc)
    field.setNumber(6, "XMin", 0.3)
    field.setNumber(6, "XMax", 0.6)
    field.setNumber(6, "YMin", 0.3)
    field.setNumber(6, "YMax", 0.6)
    field.setNumber(6, "Thickness", 0.3)
    field.add("Min", 7)
    field.setNumbers(7, "FieldsList", [2, 3, 5, 6])
    field.setAsBackgroundMesh(7)
    

    occ.synchronize()
    model.mesh.generate(3)
    model_rank = 0
    mesh, cell_tags, facet_tags = model_to_mesh(model, MPI.COMM_WORLD, model_rank)
    return mesh, cell_tags, facet_tags

ncycles = 1
npts = 7
amplitude = 0.04
scar_width=0.3
def curvedScarMesh(normal_skin_length=normal_skin_length,
            epidermis_thickness=epidermis_thickness,
            dermis_thickness=dermis_thickness,
            subcutis_thickness=subcutis_thickness,
            scar_width=scar_width,
            scar_height=normal_skin_length,
            scar_thickness=scar_thickness,
            lc=lc, height_offset=0, ncycles=ncycles, npts=npts, amplitude=amplitude, start=-normal_skin_length, stop=normal_skin_length):
    scar_bottom = -dermis_thickness - epidermis_thickness - subcutis_thickness
    skin_thickness = -scar_bottom
    skin_left = np.array([[-normal_skin_length, -normal_skin_length, -skin_thickness], [0, normal_skin_length, 0]])
    skin_right = np.array([[scar_width, -normal_skin_length, -skin_thickness], [scar_width + normal_skin_length, normal_skin_length, 0]])
    gmsh.clear()
    gmsh.option.setNumber("General.Terminal", 0)
    model = gmsh.model()
    model.add("DFG 3D")
    occ = model.occ
    occ.addBox(*skin_left[0], *(skin_left[1] - skin_left[0]))
    occ.addBox(*skin_right[0], *(skin_right[1] - skin_right[0]))
    occ.addBox(0, -scar_height, scar_bottom, scar_width, 2 * scar_height, scar_thickness)
    occ.synchronize()

    gmsh.option.setNumber("Geometry.OCCUnionUnify", 1)
    scar_radius = np.sqrt(np.square(scar_width / 2) + np.square(height_offset))

    scarLength = stop - start
    omega = 2 * np.pi / scarLength * ncycles
    p = []
    for i, theta in enumerate(np.linspace(start, stop, npts)):
        gmsh.model.occ.addPoint(amplitude * np.sin(omega * theta - start), theta, -height_offset, tag=1000 + i)
        p.append(1000 + i)
    occ.addSpline(p, 1000)
    occ.addWire([1000], 1000)

    occ.addDisk(0, 0, 0, scar_radius, scar_radius, 1000)
    occ.rotate([(2, 1000)], 0, 0, 0, 1, 0, 0, np.pi / 2)
    scar_tag = occ.addPipe([(2, 1000)], 1000, 'DiscreteTrihedron')
    occ.remove([(2, 1000)])
    occ.synchronize()
    volumes = model.getEntities(dim=3)

    com = -np.array(occ.getCenterOfMass(*volumes[-1]))
    com[-1] = 0
    com[0] += scar_width / 2
    occ.translate(volumes[-1:], *com)
    occ.synchronize()

    box_tag = occ.addBox(-normal_skin_length - scar_radius, -normal_skin_length - scar_radius, -skin_thickness, 2 * (normal_skin_length + scar_radius) + scar_width,
                         2 * (normal_skin_length + scar_radius), -1.1 * scar_radius)
    
    occ.cut(scar_tag, [(3,box_tag)])
    occ.synchronize()
    volumes = model.getEntities(dim=3)
    occ.fuse([volumes[0]], volumes[1:])
    occ.synchronize()
    
    surfaces = occ.getEntities(dim=2)
    walls = []
    for surface in surfaces:
        com = occ.getCenterOfMass(surface[0], surface[1])
        if np.allclose(com, [-normal_skin_length, 0, -skin_thickness / 2]):
            model.addPhysicalGroup(surface[0], [surface[1]], force_left_marker)
            left_force = surface[1]
            model.setPhysicalName(surface[0], force_left_marker, "Left force")
        elif np.allclose(com, [scar_width + normal_skin_length, 0, -skin_thickness / 2]):
            model.addPhysicalGroup(surface[0], [surface[1]], force_right_marker)
            right_force = surface[1]
            model.setPhysicalName(surface[0], force_right_marker, "Right force")
        elif np.allclose(com, [0, 0, -skin_thickness / 2]) or np.allclose(com, [scar_width, 0, -skin_thickness / 2]):
            continue
        else:
            walls.append(surface[1])

    model.addPhysicalGroup(2, walls, free_marker)
    model.setPhysicalName(2, free_marker, "Free wall")
    volumes = model.getEntities(dim=3)
    assert len(volumes) == 1
    model.addPhysicalGroup(3, [i[1] for i in volumes], 4)
    model.setPhysicalName(3, 4, 'Volumes')

    field = model.mesh.field
    field.add("Box", 6)
    field.setNumber(6, "VIn", lc)
    field.setNumber(6, "VOut", lc)
    field.setNumber(6, "XMin", 0.3)
    field.setNumber(6, "XMax", 0.6)
    field.setNumber(6, "YMin", 0.3)
    field.setNumber(6, "YMax", 0.6)
    field.setNumber(6, "Thickness", 0.3)
    field.add("Min", 7)
    field.setNumbers(7, "FieldsList", [2, 3, 5, 6])
    field.setAsBackgroundMesh(7)
    

    occ.synchronize()
    model.mesh.generate(3)
    #gmsh.write("t19.msh")
    model_rank = 0
    mesh, cell_tags, facet_tags = model_to_mesh(model, MPI.COMM_WORLD, model_rank)
    return mesh, cell_tags, facet_tags


def curvedWoundMesh(
    normal_skin_length=normal_skin_length,
    epidermis_thickness=epidermis_thickness,
    dermis_thickness=dermis_thickness,
    subcutis_thickness=subcutis_thickness,
    scar_width=scar_width,
    scar_height=normal_skin_length,
    scar_thickness=scar_thickness,
    lc=lc,
    height_offset=0,
    ncycles=ncycles,
    npts=npts, 
    amplitude=amplitude, 
    start=-normal_skin_length,
    stop=normal_skin_length,
    meshSizeCurvature=20,
    meshSizeMin=0.1,
    meshSizeMax=3
):    
    scar_bottom = -dermis_thickness - epidermis_thickness - subcutis_thickness
    skin_thickness = -scar_bottom
    skin_left = np.array([[-normal_skin_length, -normal_skin_length, -skin_thickness], [0, normal_skin_length, 0]])
    skin_right = np.array([[scar_width, -normal_skin_length, -skin_thickness], [scar_width + normal_skin_length, normal_skin_length, 0]])
    scar_radius = np.sqrt(np.square(scar_width / 2) + np.square(height_offset))
    gmsh.clear()
    gmsh.option.setNumber("General.Terminal", 0)
    model = gmsh.model()
    model.add("DFG 3D")
    occ = model.occ
    occ.addBox(*skin_left[0], *(skin_right[1] - skin_left[0]))
    occ.synchronize()

    gmsh.option.setNumber("Mesh.MeshSizeMin", meshSizeMin * lc)
    gmsh.option.setNumber("Mesh.MeshSizeMax", meshSizeMax * lc)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", meshSizeCurvature)
    
    gmsh.option.setNumber("Geometry.OCCUnionUnify", 1)

    scarLength = stop - start
    omega = 2 * np.pi / scarLength * ncycles
    p = []

    for i, theta in enumerate(np.linspace(start - 0.2 * normal_skin_length, stop + 0.2 * normal_skin_length, npts)):
        gmsh.model.occ.addPoint(amplitude * np.sin(omega * theta - start), theta, height_offset, tag=1000 + i)
        p.append(1000 + i)
    occ.addSpline(p, 1000)
    occ.addWire([1000], 1000)

    occ.addDisk(0, 0, 0, scar_radius, scar_radius, 1000)
    occ.rotate([(2, 1000)], 0, 0, 0, 1, 0, 0, np.pi / 2)
    scar_tag = occ.addPipe([(2, 1000)], 1000, 'DiscreteTrihedron')
    occ.remove([(2, 1000)])
    occ.synchronize()
    volumes = model.getEntities(dim=3)
    print(volumes)

    com = -np.array(occ.getCenterOfMass(*volumes[-1]))
    com[-1] = 0
    com[0] += scar_width / 2
    occ.translate(volumes[-1:], *com)
    occ.synchronize()
    
    volumes = model.getEntities(dim=3)
    occ.cut([volumes[0]], [volumes[1]])
    occ.synchronize()
    print(volumes)
    
    surfaces = occ.getEntities(dim=2)
    walls = []
    wound_surfaces = []
    for surface in surfaces:
        com = occ.getCenterOfMass(surface[0], surface[1])
        if np.allclose(com, [-normal_skin_length, 0, -skin_thickness / 2]):
            model.addPhysicalGroup(surface[0], [surface[1]], force_left_marker)
            left_force = surface[1]
            model.setPhysicalName(surface[0], force_left_marker, "Left force")
        elif np.allclose(com, [scar_width + normal_skin_length, 0, -skin_thickness / 2]):
            model.addPhysicalGroup(surface[0], [surface[1]], force_right_marker)
            right_force = surface[1]
            model.setPhysicalName(surface[0], force_right_marker, "Right force")
        elif not (#np.allclose(com[0], -normal_skin_length) | np.allclose(com[0], scar_width + normal_skin_length) | 
                  np.allclose(com[1], -normal_skin_length) | np.allclose(com[1], normal_skin_length) |
                  np.allclose(com[2], scar_bottom) | np.allclose(com[2], 0)):
            wound_surfaces.append(surface[1])
        else:
            if np.allclose(com[2], scar_bottom):
                bottom_surface = surface[1]
            walls.append(surface[1])
    model.addPhysicalGroup(surface[0], wound_surfaces, wound_marker)
    model.setPhysicalName(surface[0], wound_marker, 'Wound')
    model.addPhysicalGroup(2, walls, free_marker)
    model.setPhysicalName(2, free_marker, "Free wall")
    volumes = model.getEntities(dim=3)
    #assert len(volumes) == 1
    print(volumes)
    model.addPhysicalGroup(3, [i[1] for i in volumes], 4)
    model.setPhysicalName(3, 4, 'Volumes')

    
    field = model.mesh.field
    field.add("MathEval", 1)
    field.setString(1, "F", f'(1 + floor(-z / ({2 * (dermis_thickness + epidermis_thickness)}))) * {lc}')
    field.setAsBackgroundMesh(1)
    
    occ.synchronize()
    
    model.mesh.generate(3)
    model_rank = 0
    mesh, cell_tags, facet_tags = model_to_mesh(model, MPI.COMM_WORLD, model_rank)
    return mesh, cell_tags, facet_tags

def wedgeWoundMesh(
    normal_skin_length=normal_skin_length,
    epidermis_thickness=epidermis_thickness,
    dermis_thickness=dermis_thickness,
    subcutis_thickness=subcutis_thickness,
    scar_width=scar_width,
    scar_height=normal_skin_length,
    scar_thickness=scar_thickness,
    lc=lc,
    height_offset=0,
    meshSizeCurvature=40,
    meshSizeMin=0.1,
    meshSizeMax=3,
    woundDepth=0.5 * (dermis_thickness + epidermis_thickness + subcutis_thickness),
):    
    scar_bottom = -dermis_thickness - epidermis_thickness - subcutis_thickness
    skin_thickness = -scar_bottom
    skin_left = np.array([[-normal_skin_length, -normal_skin_length, -skin_thickness], [0, normal_skin_length, 0]])
    skin_right = np.array([[scar_width, -normal_skin_length, -skin_thickness], [scar_width + normal_skin_length, normal_skin_length, 0]])
    scar_radius = np.sqrt(np.square(scar_width / 2) + np.square(height_offset))
    gmsh.clear()
    gmsh.option.setNumber("General.Terminal", 0)
    model = gmsh.model()
    model.add("DFG 3D")
    occ = model.occ
    m0 = occ.addBox(*skin_left[0], *(skin_right[1] - skin_left[0]))
    occ.synchronize()

    gmsh.option.setNumber("Mesh.MeshSizeMin", meshSizeMin * lc)
    gmsh.option.setNumber("Mesh.MeshSizeMax", meshSizeMax * lc)
    gmsh.option.setNumber("Mesh.ElementOrder", 2)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", meshSizeCurvature)
    gmsh.option.setNumber("Geometry.OCCUnionUnify", 1)

    a = model.occ.addWedge(0, scar_width / 2, 0, 
                   woundDepth, scar_width / 2, 4 * normal_skin_length, zAxis=[0.0, 1.0, 0.0])
    b = model.occ.addWedge(0, scar_width / 2, 0, 
                       woundDepth, scar_width / 2, 4 * normal_skin_length, zAxis=[0.0, -1.0, 0.0])
    model.occ.rotate([(3, b)], 0, 0, 0, 0, 0, 1, np.pi)
    model.occ.rotate([(3, b)], 0, 0, 0, 1, 0, 0, np.pi)
    model.occ.translate([(3, b)], scar_width / 2, 2 * normal_skin_length, 0)
    model.occ.translate([(3, a)], scar_width / 2, -2 * normal_skin_length, 0)
    model.occ.rotate([(3, a), (3, b)], scar_width / 2, 0, 0, 1, 0, 0, np.pi)
    
    occ.synchronize()
    volumes = model.getEntities(dim=3)
    m0 = occ.cut([(3, m0)], [(3, a), (3, b)])
    occ.synchronize()
    
    surfaces = occ.getEntities(dim=2)
    walls = []
    wound_surfaces = []
    for surface in surfaces:
        com = occ.getCenterOfMass(surface[0], surface[1])
        if np.allclose(com, [-normal_skin_length, 0, -skin_thickness / 2]):
            model.addPhysicalGroup(surface[0], [surface[1]], force_left_marker)
            left_force = surface[1]
            model.setPhysicalName(surface[0], force_left_marker, "Left force")
        elif np.allclose(com, [scar_width + normal_skin_length, 0, -skin_thickness / 2]):
            model.addPhysicalGroup(surface[0], [surface[1]], force_right_marker)
            right_force = surface[1]
            model.setPhysicalName(surface[0], force_right_marker, "Right force")
        elif not (#np.allclose(com[0], -normal_skin_length) | np.allclose(com[0], scar_width + normal_skin_length) | 
                  np.allclose(com[1], -normal_skin_length) | np.allclose(com[1], normal_skin_length) |
                  np.allclose(com[2], scar_bottom) | np.allclose(com[2], 0)):
            wound_surfaces.append(surface[1])
        else:
            if np.allclose(com[2], scar_bottom):
                bottom_surface = surface[1]
            walls.append(surface[1])
    model.addPhysicalGroup(surface[0], wound_surfaces, wound_marker)
    model.setPhysicalName(surface[0], wound_marker, 'Wound')
    model.addPhysicalGroup(2, walls, free_marker)
    model.setPhysicalName(2, free_marker, "Free wall")
    volumes = model.getEntities(dim=3)
    assert len(volumes) == 1
    model.addPhysicalGroup(3, [i[1] for i in volumes], 4)
    model.setPhysicalName(3, 4, 'Volumes')

    
    field = model.mesh.field
    field.add("MathEval", 1)
    field.setString(1, "F", f'(1 + floor((-z+1e-8) / ({(dermis_thickness + epidermis_thickness)}))) * {lc}')
    field.setAsBackgroundMesh(1)
    
    occ.synchronize()
    
    model.mesh.generate(3)
    model_rank = 0
    mesh, cell_tags, facet_tags = model_to_mesh(model, MPI.COMM_WORLD, model_rank)
    return mesh, cell_tags, facet_tags