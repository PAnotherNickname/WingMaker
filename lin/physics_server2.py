from fastapi import FastAPI, Request
import aerosandbox as asb
import aerosandbox.numpy as np
import pyvista as pv
import subprocess
import re
import os
import shutil

app = FastAPI()
FLUIDX3D_DIR = "FluidX3D"

def build_single_wing_mesh(span, chord, sweep_angle, twist, taper, x_offset, z_offset, airfoil_name="mh45"):
    af = asb.Airfoil(airfoil_name)
    if af.coordinates is None: af = asb.Airfoil("naca0012")
    coords = af.coordinates
    N = len(coords)
    
    # Root coordinates
    root_x = coords[:, 0] * chord
    root_z = coords[:, 1] * chord
    root_pts = np.column_stack((root_x, np.zeros(N), root_z))
    
    tip_chord = max(0.01, chord * taper) 
    sweep_x = (span / 2.0) * np.tan(np.radians(sweep_angle))
    twist_rad = np.radians(twist)
    
    tip_x_base = coords[:, 0] * tip_chord
    tip_z_base = coords[:, 1] * tip_chord
    
    # Twist around quarter-chord
    pivot_x = 0.25 * tip_chord
    tip_x_shift = tip_x_base - pivot_x
    tip_x_rot = tip_x_shift * np.cos(twist_rad) - tip_z_base * np.sin(twist_rad)
    tip_z_rot = tip_x_shift * np.sin(twist_rad) + tip_z_base * np.cos(twist_rad)
    
    tip_x_final = tip_x_rot + pivot_x + sweep_x
    
    # Apply Span
    right_pts = np.column_stack((tip_x_final, np.full(N, span / 2.0), tip_z_rot))
    left_pts = np.column_stack((tip_x_final, np.full(N, -span / 2.0), tip_z_rot))
    
    # Apply X and Z Offsets for Canards, Tails, or Biplanes
    offset_vector = np.array([x_offset, 0.0, z_offset])
    root_pts += offset_vector
    right_pts += offset_vector
    left_pts += offset_vector
    
    vertices = np.vstack((left_pts, root_pts, right_pts))
    faces = []
    
    def add_quad(p1, p2, p3, p4):
        faces.extend([4, p1, p2, p3, p4])
        
    for i in range(N - 1): add_quad(i, i+1, i+1+N, i+N)
    add_quad(N-1, 0, N, 2*N-1) 
    
    for i in range(N, 2*N - 1): add_quad(i, i+1, i+1+N, i+N)
    add_quad(2*N-1, N, 2*N, 3*N-1) 
    
    # Cap the wingtips
    faces.extend([N] + list(range(N-1, -1, -1)))
    faces.extend([N] + list(range(2*N, 3*N)))
    
    return pv.PolyData(vertices, faces)

def build_universal_airplane_stl(surfaces, filename="THICK_CHAMPION.stl"):
    import math
    combined_mesh = pv.PolyData()
    valid_surfaces = []
    
    for surf in surfaces:
        # If the AI shrinks the span to near-zero, it means it wants to delete this wing
        if float(surf["span"]) < 0.05:
            continue
            
        wing_mesh = build_single_wing_mesh(
            span=float(surf["span"]),
            chord=float(surf["chord"]),
            sweep_angle=float(surf["sweep_angle"]),
            twist=float(surf["twist"]),
            taper=float(surf["taper"]),
            x_offset=float(surf["x"]),
            z_offset=float(surf["z"]),
            airfoil_name=str(surf.get("airfoil_name", "naca0012"))
        )
        combined_mesh = combined_mesh.merge(wing_mesh)
        valid_surfaces.append(surf)
        
    # --- NEW: THE PHYSICAL FUSELAGE BUILDER ---
    # If the AI successfully built two separate wings, we MUST connect them with a physical boom!
    if len(valid_surfaces) == 2:
        x1, z1 = float(valid_surfaces[0]["x"]), float(valid_surfaces[0]["z"])
        x2, z2 = float(valid_surfaces[1]["x"]), float(valid_surfaces[1]["z"])
        
        # Calculate the distance between the two wing roots
        dist = math.sqrt((x2 - x1)**2 + (z2 - z1)**2)
        
        # If they are separated by more than a few centimeters, spawn a fuselage boom
        if dist > 0.05:
            center_x = (x1 + x2) / 2.0
            center_z = (z1 + z2) / 2.0
            
            # Draw a 6cm thick carbon fiber boom connecting the exact center points
            boom = pv.Cylinder(
                center=(center_x, 0.0, center_z),
                direction=(x2 - x1, 0.0, z2 - z1),
                radius=0.03, 
                height=dist
            )
            # Melt the boom into the wings to make a single solid drone
            combined_mesh = combined_mesh.merge(boom)
            
    # Pitch the entire unified aircraft up 5 degrees so the CFD tunnel catches the air
    combined_mesh.rotate_y(-5.0, inplace=True)
    combined_mesh.save(filename)
    return filename

def run_fluidx3d_cfd(stl_filename="THICK_CHAMPION.stl"):
    bin_dir = os.path.join(FLUIDX3D_DIR, "bin")
    os.makedirs(bin_dir, exist_ok=True)
    shutil.copy(stl_filename, os.path.join(FLUIDX3D_DIR, stl_filename))
    shutil.copy(stl_filename, os.path.join(bin_dir, stl_filename))

    subprocess.run("git reset --hard", shell=True, cwd=FLUIDX3D_DIR, stdout=subprocess.DEVNULL)
    defines_path = os.path.join(FLUIDX3D_DIR, "src", "defines.hpp")
    with open(defines_path, "r") as f: defines_content = f.read()
    defines_content = defines_content.replace("#define BENCHMARK", "//#define BENCHMARK")
    defines_content = defines_content.replace("//#define FORCE_FIELD", "#define FORCE_FIELD")
    defines_content = defines_content.replace("//#define EQUILIBRIUM_BOUNDARIES", "#define EQUILIBRIUM_BOUNDARIES")
    with open(defines_path, "w") as f: f.write(defines_content)

    setup_cpp = f"""
#include "lbm.hpp"
#include <iostream>
#include <stdio.h> 

void main_setup() {{
    const uint Nx = 384, Ny = 384, Nz = 128;
    LBM lbm(Nx, Ny, Nz, 0.02f);
    
    parallel_for(lbm.get_N(), [&](ulong n) {{
        uint x=0u, y=0u, z=0u; 
        lbm.coordinates(n, x, y, z);
        lbm.rho[n] = 1.0f;
        lbm.u.x[n] = 0.05f; 
        if(y==0u || y==Ny-1u || z==0u || z==Nz-1u) {{ lbm.flags[n] = TYPE_S; lbm.u.x[n] = 0.0f; }}
        if(x==0u) {{ lbm.flags[n] = TYPE_E; lbm.u.x[n] = 0.05f; }}
        if(x==Nx-1u) {{ lbm.flags[n] = TYPE_E; lbm.rho[n] = 1.0f; }}
    }});
    
    lbm.run(0u);
    
    Mesh* mesh = read_stl("{stl_filename}");
    mesh->translate((mesh->pmax + mesh->pmin) * -0.5f);
    
    float3 size = mesh->pmax - mesh->pmin;
    float max_dim = size.x;
    if(size.y > max_dim) max_dim = size.y;
    if(size.z > max_dim) max_dim = size.z;
    
    float scale = 250.0f / max_dim;
    mesh->scale(scale);
    float3 target_center = float3(Nx / 3.0f, Ny / 2.0f, Nz / 2.0f);
    mesh->translate(target_center);
    
    lbm.voxelize_mesh_on_device(mesh, TYPE_S|TYPE_X);
    lbm.run(5000);
    
    const float3 native_force = lbm.object_force(TYPE_S|TYPE_X);
    printf("\\n===FINAL_CFD_RESULTS==\\n");
    printf("RAW_LIFT: %f\\n", native_force.z);
    printf("RAW_DRAG: %f\\n", native_force.x);
    printf("======================\\n");
}}
    """
    with open(os.path.join(FLUIDX3D_DIR, "src", "setup.cpp"), "w") as f:
        f.write(setup_cpp)

    process = subprocess.run("bash make.sh", shell=True, cwd=FLUIDX3D_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = process.stdout.decode("utf-8")

    lift, drag, real_ld = 0.0, 0.0, 0.0
    for line in output.split("\n"):
        if "RAW_LIFT:" in line:
            match = re.findall(r"[-+]?(?:\d*\.\d+(?:[eE][-+]?\d+)?|\d+|nan|inf|NAN|INF)", line.split(":")[1].strip().lower())
            if match: lift = float(match[0])
        elif "RAW_DRAG:" in line:
            match = re.findall(r"[-+]?(?:\d*\.\d+(?:[eE][-+]?\d+)?|\d+|nan|inf|NAN|INF)", line.split(":")[1].strip().lower())
            if match: drag = float(match[0])
            
    if drag > 0:
        real_ld = lift / drag
        
    return lift, drag, real_ld, "CFD Evaluated"


@app.post("/simulate")
async def simulate_drone(request: Request):
    data = await request.json()
    
    run_cfd = bool(data.get("run_cfd", False))
    export_final_stl = bool(data.get("export_final_stl", False))
    surfaces = data.get("surfaces", [])

    if not surfaces:
        return {"status": "rejected", "lift_to_drag_ratio": 0.0, "pitch_moment": 99.0}

    # Dynamically build the AeroSandbox Airplane
    asb_wings = []
    for i, surf in enumerate(surfaces):
        span = float(surf.get("span", 0.0))
        if span < 0.05: 
            continue # AI decided to delete this wing
            
        chord = float(surf.get("chord", 1.0))
        sweep = float(surf.get("sweep_angle", 0.0))
        twist = float(surf.get("twist", 0.0))
        taper = float(surf.get("taper", 0.5))
        x_off = float(surf.get("x", 0.0))
        z_off = float(surf.get("z", 0.0))
        af_name = str(surf.get("airfoil_name", "naca0012"))
        
        af = asb.Airfoil(af_name)
        if af.coordinates is None: af = asb.Airfoil("naca0012")
        tip_chord = max(0.01, chord * taper)

        asb_wings.append(asb.Wing(
            name=f"Surface_{i}", symmetric=True,
            xsecs=[
                asb.WingXSec(xyz_le=[x_off, 0, z_off], chord=chord, twist=0.0, airfoil=af),
                asb.WingXSec(xyz_le=[x_off + (span/2)*np.tand(sweep), span/2, z_off], chord=tip_chord, twist=twist, airfoil=af)
            ]
        ))

    airplane = asb.Airplane(name="Universal_AI_Craft", wings=asb_wings)

    # 1. Run the fast VLM Physics Check
    try:
        vlm = asb.VortexLatticeMethod(airplane=airplane, op_point=asb.OperatingPoint(velocity=15.0, alpha=5.0)).run()
        vlm_ld = float(vlm["CL"] / vlm["CD"]) if vlm["CD"] > 0 else 0.0
        cm = float(vlm["Cm"])
    except:
        vlm_ld, cm = 0.0, 99.0

    # 2. Run the Heavy CFD Tunnel (If requested by Mac)
    if run_cfd:
        stl_name = "THICK_CHAMPION.stl"
        if export_final_stl:
            stl_name = "ULTIMATE_CFD_CHAMPION.stl"
            print(f"\n🎉 EXPORTING FINAL OPTIMIZED STL: {stl_name}")
            
        stl_path = build_universal_airplane_stl(surfaces, filename=stl_name)
        lift, drag, cfd_ld, msg = run_fluidx3d_cfd(stl_path)
        
        return {
            "status": "success",
            "lift_to_drag_ratio": cfd_ld,
            "pitch_moment": cm,
            "raw_cfd_lift": lift,
            "raw_cfd_drag": drag,
            "message": msg
        }

    return {
        "status": "success",
        "lift_to_drag_ratio": vlm_ld,
        "pitch_moment": cm
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)