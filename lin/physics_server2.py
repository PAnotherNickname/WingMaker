from fastapi import FastAPI, Request
import aerosandbox as asb
import aerosandbox.numpy as np
import pyvista as pv
import subprocess, re, os, shutil

app = FastAPI()
FLUIDX3D_DIR = "FluidX3D"

def build_single_wing_mesh(span, chord, sweep_angle, twist, taper, x_off, z_off, af_name):
    af = asb.Airfoil(af_name)
    coords = af.coordinates if af.coordinates is not None else asb.Airfoil("naca0012").coordinates
    N = len(coords)
    
    root_pts = np.column_stack((coords[:, 0]*chord + x_off, np.zeros(N), coords[:, 1]*chord + z_off))
    tc = max(0.01, chord * taper)
    sw_x = (span/2)*np.tan(np.radians(sweep_angle))
    tr = np.radians(twist)
    tip_x = (coords[:, 0]-0.25)*tc*np.cos(tr) - (coords[:, 1]*tc*np.sin(tr)) + 0.25*tc + sw_x + x_off
    tip_z = (coords[:, 0]-0.25)*tc*np.sin(tr) + (coords[:, 1]*tc*np.cos(tr)) + z_off
    r_pts = np.column_stack((tip_x, np.full(N, span/2), tip_z))
    l_pts = np.column_stack((tip_x, np.full(N, -span/2), tip_z))
    
    vertices = np.vstack((l_pts, root_pts, r_pts))
    faces = []
    
    for i in range(N-1): faces.extend([4, i, i+1, i+1+N, i+N]) # Left to Root
    faces.extend([4, N-1, 0, N, 2*N-1])
    for i in range(N, 2*N-1): faces.extend([4, i, i+1, i+1+N, i+N]) # Root to Right
    faces.extend([4, 2*N-1, N, 2*N, 3*N-1])
    
    # Cap the wingtips so FluidX3D doesn't crash
    faces.extend([N] + list(range(N-1, -1, -1)))
    faces.extend([N] + list(range(2*N, 3*N)))
    
    return pv.PolyData(vertices, faces)

def build_universal_airplane_stl(surfaces, filename="THICK_CHAMPION.stl"):
    combined_mesh = pv.PolyData()
    for s in surfaces:
        if float(s["span"]) < 0.05: continue
        wing = build_single_wing_mesh(float(s["span"]), float(s["chord"]), float(s["sweep_angle"]), 
                                      float(s["twist"]), float(s["taper"]), float(s["x"]), float(s["z"]), s["airfoil_name"])
        combined_mesh = combined_mesh.merge(wing)
        
    combined_mesh = combined_mesh.extract_surface().clean()
    # 🛑 THE NORMAL FIXER: Forces all faces outward to prevent FluidX3D raycasting errors
    combined_mesh.compute_normals(consistent_normals=True, inplace=True)
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
            
    if drag > 0: real_ld = lift / drag
    return lift, drag, real_ld, "CFD Evaluated"

@app.post("/simulate")
async def simulate(request: Request):
    data = await request.json()
    run_cfd = bool(data.get("run_cfd", False))
    export_final_stl = bool(data.get("export_final_stl", False))
    surfaces = data.get("surfaces", [])

    if not surfaces: return {"status": "rejected", "lift_to_drag_ratio": 0.0, "pitch_moment": 99.0}

    asb_wings = []
    for i, s in enumerate(surfaces):
        span = float(s.get("span", 0.0))
        if span < 0.05: continue 
        chord = float(s.get("chord", 1.0))
        sweep = float(s.get("sweep_angle", 0.0))
        twist = float(s.get("twist", 0.0))
        taper = float(s.get("taper", 0.5))
        x_off, z_off = float(s.get("x", 0.0)), float(s.get("z", 0.0))
        af_name = str(s.get("airfoil_name", "naca0012"))
        
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

    airplane = asb.Airplane(name="AI_Assembly", wings=asb_wings)

    try:
        vlm = asb.VortexLatticeMethod(airplane=airplane, op_point=asb.OperatingPoint(velocity=15.0, alpha=5.0)).run()
        # THE FRICTION TAX: Forces AI to build realistic wings
        cd_total = float(vlm["CD"]) + 0.015 
        vlm_ld = float(vlm["CL"] / cd_total) if cd_total > 0 else 0.0
        cm = float(vlm["Cm"])
    except:
        vlm_ld, cm = 0.0, 99.0

    if run_cfd:
        stl_name = "ULTIMATE_CFD_CHAMPION.stl" if export_final_stl else "THICK_CHAMPION.stl"
        stl_path = build_universal_airplane_stl(surfaces, filename=stl_name)
        lift, drag, cfd_ld, msg = run_fluidx3d_cfd(stl_path)
        return {"status": "success", "lift_to_drag_ratio": cfd_ld, "pitch_moment": cm, "raw_cfd_lift": lift, "raw_cfd_drag": drag, "message": msg}

    return {"status": "success", "lift_to_drag_ratio": vlm_ld, "pitch_moment": cm}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)