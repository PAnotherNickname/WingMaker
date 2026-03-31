import json
import requests
import os
import sys
import time
import math
import re
import warnings
from scipy.optimize import differential_evolution

warnings.filterwarnings("ignore")

try:
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
except ImportError:
    print("❌ Missing Libraries. Run: pip install mlx-lm requests scipy")
    sys.exit(1)

DELL_LINUX_URL = "http://172.22.228.131:8000/simulate" 

print("\n⚙️ Aero-Optimizer v15.7: Deep Debug Edition")
model, tokenizer = load("mlx-community/Qwen2.5-14B-Instruct-4bit")

user_prompt = input("\n🎯 ENTER DESIGN GOAL: ")
if not user_prompt.strip():
    user_prompt = "Design a highly efficient aircraft with a tail."

print("🧠 14B Routing Agent Analyzing Request...")
extract_sys = """You are an aerospace routing agent. Map the user's request to EXACTLY ONE of the following keywords:
- CONVENTIONAL : Route here if the user asks for a tail, rear stabilizer, or a tail in the back.
- CANARD       : Route here if the user asks for a front tail, front wing, or canard.
- FLYING_WING  : Route here if the user asks for a tailless design, a pure wing, or a flying wing.
- TANDEM       : Route here if the user asks for two main wings, a biplane, or tandem.
- UNIVERSAL    : Route here if the user DOES NOT specify a physical layout.
CRITICAL INSTRUCTION: If the user explicitly asks for a "tail", you MUST output CONVENTIONAL.
Output ONLY the keyword."""

extract_prompt = tokenizer.apply_chat_template([{"role": "system", "content": extract_sys}, {"role": "user", "content": user_prompt}], tokenize=False, add_generation_prompt=True)
llm_route = generate(model, tokenizer, prompt=extract_prompt, max_tokens=20, sampler=make_sampler(temp=0.1)).strip().upper()

target_config = "UNIVERSAL"
for route in ["CONVENTIONAL", "CANARD", "FLYING_WING", "TANDEM", "UNIVERSAL"]:
    if route in llm_route: target_config = route

print(f"✅ ACTION ROUTE: {target_config}")

dynamic_bounds = [
    (0.1, 2.5), (0.1, 0.4), (-20.0, 45.0), (-10.0, 5.0), (0.1, 1.0), (-0.5, 0.5), (-0.2, 0.2), (0.0, 3.99), # Wing A
    (0.1, 2.5), (0.05, 0.4), (-20.0, 45.0), (-15.0, 10.0), (0.1, 1.0), (-1.5, 1.5), (-0.2, 0.2), (0.0, 3.99), # Wing B
    (0.05, 0.2), (0.5, 2.0), (-2.0, 2.0), (0.0, 0.0), (0.2, 1.0), (-1.0, 1.0), (-0.1, 0.1), (0.0, 3.99) # Fuselage
]

AIRFOIL_ROSTER = ["mh45", "mh60", "pw75", "naca0012"]
best_ld_seen = 0.0
vlm_history = []

def get_current_config(x_a, x_b, s_b):
    if s_b < 0.25: return "FLYING_WING"
    if x_b > x_a + 0.25: return "CONVENTIONAL"
    if x_b < x_a - 0.25: return "CANARD"
    return "TANDEM"

def evaluate_design(x):
    global best_ld_seen
    s_a, c_a, sw_a, tw_a, ta_a, x_a, z_a, af_idx_a = x[0:8]
    s_b, c_b, sw_b, tw_b, ta_b, x_b, z_b, af_idx_b = x[8:16]
    s_c, c_c, sw_c, tw_c, ta_c, x_c, z_c, af_idx_c = x[16:24]
    
    current_layout = get_current_config(x_a, x_b, s_b)
    layout_penalty = 500000.0 if target_config != "UNIVERSAL" and current_layout != target_config else 0.0
    connection_penalty = 0.0

    if target_config == "CONVENTIONAL":
        if x_a < x_c: connection_penalty += ((x_c - x_a)**2) * 1e6
        if x_a > x_c + (c_c * 0.5): connection_penalty += ((x_a - (x_c + c_c * 0.5))**2) * 1e6
        target_tail_x = x_c + (c_c * 0.7)
        if x_b < target_tail_x: connection_penalty += ((target_tail_x - x_b)**2) * 1e6
        if (x_b + c_b) > (x_c + c_c): connection_penalty += (((x_b + c_b) - (x_c + c_c))**2) * 1e6
    else:
        if x_a < x_c: connection_penalty += ((x_c - x_a)**2) * 1e6
        if (x_a + c_a) > (x_c + c_c): connection_penalty += (((x_a + c_a) - (x_c + c_c))**2) * 1e6
        if x_b < x_c: connection_penalty += ((x_c - x_b)**2) * 1e6
        if (x_b + c_b) > (x_c + c_c): connection_penalty += (((x_b + c_b) - (x_c + c_c))**2) * 1e6

    if connection_penalty > 1.0 or layout_penalty > 0:
        return 99999.0 + connection_penalty + layout_penalty

    payload = {
        "run_cfd": False,
        "surfaces": [
            {"span": float(s_a), "chord": float(c_a), "sweep_angle": float(sw_a), "twist": float(tw_a), "taper": float(ta_a), "x": float(x_a), "z": float(z_c), "airfoil_name": AIRFOIL_ROSTER[int(af_idx_a)]},
            {"span": float(s_b), "chord": float(c_b), "sweep_angle": float(sw_b), "twist": float(tw_b), "taper": float(ta_b), "x": float(x_b), "z": float(z_c), "airfoil_name": AIRFOIL_ROSTER[int(af_idx_b)]},
            {"span": float(s_c), "chord": float(c_c), "sweep_angle": float(sw_c), "twist": float(tw_c), "taper": float(ta_c), "x": float(x_c), "z": float(z_c), "airfoil_name": AIRFOIL_ROSTER[int(af_idx_c)]}
        ]
    }

    try:
        r = requests.post(DELL_LINUX_URL, json=payload, timeout=40)
        data = r.json()
        ld = float(data.get("lift_to_drag_ratio", 0.0))
        cm = float(data.get("pitch_moment", 99.0))
    except: return 99999.0 

    fitness = -ld + (abs(cm)**2)*5000 
    if ld > best_ld_seen and abs(cm) < 0.05: best_ld_seen = ld
    return fitness

def print_progress(xk, convergence):
    global vlm_history
    print(f"🧬 Route: {target_config} | Best L/D: {best_ld_seen:.1f}")
    sys.stdout.flush()
    vlm_history.append(best_ld_seen)
    if len(vlm_history) > 20: vlm_history.pop(0)
    
    if best_ld_seen > 0.1 and len(vlm_history) == 20:
        if (max(vlm_history) - min(vlm_history)) < 0.1: 
            print("\n🛑 VLM PLATEAU REACHED.")
            sys.stdout.flush()
            return True
    return False

def run_cfd_refinement_loop(champion_surfaces):
    print("\n" + "🔥"*25 + "\nPHASE 2: INFINITE CFD REFINEMENT LOOP\n" + "🔥"*25)
    sys.stdout.flush()
    
    state = {
        "wing_span": champion_surfaces[0]["span"], "wing_chord": champion_surfaces[0]["chord"], "wing_x": champion_surfaces[0]["x"],
        "tail_span": champion_surfaces[1]["span"], "tail_chord": champion_surfaces[1]["chord"], "tail_x": champion_surfaces[1]["x"],
        "fuse_span": champion_surfaces[2]["span"], "fuse_chord": champion_surfaces[2]["chord"], "fuse_x": champion_surfaces[2]["x"]
    }
    
    cfd_history = []
    best_cfd_ld = -1.0
    best_payload = None
    step, skip_cfd, ld = 1, False, 0.0
    
    while step <= 10:
        if not skip_cfd:
            payload = {"run_cfd": True, "export_final_stl": False, "surfaces": champion_surfaces}
            payload["surfaces"][0]["span"], payload["surfaces"][0]["chord"], payload["surfaces"][0]["x"] = state["wing_span"], state["wing_chord"], state["wing_x"]
            payload["surfaces"][1]["span"], payload["surfaces"][1]["chord"], payload["surfaces"][1]["x"] = state["tail_span"], state["tail_chord"], state["tail_x"]
            payload["surfaces"][2]["span"], payload["surfaces"][2]["chord"], payload["surfaces"][2]["x"] = state["fuse_span"], state["fuse_chord"], state["fuse_x"]
            
            try:
                print(f"⏳ Waiting for FluidX3D Simulation on Linux (This takes a few minutes)...")
                sys.stdout.flush()
                r = requests.post(DELL_LINUX_URL, json=payload, timeout=400)
                data = r.json()
                
                ld = float(data.get('lift_to_drag_ratio', 0.0))
                # DEEP DEBUG: Extract Raw Lift and Drag
                raw_lift = float(data.get('raw_cfd_lift', 0.0))
                raw_drag = float(data.get('raw_cfd_drag', 0.0))
                
                if ld > best_cfd_ld:
                    best_cfd_ld = ld
                    best_payload = payload.copy()
                cfd_history.append(ld)
                
                print(f"📊 Step {step} CFD | L/D: {ld:.2f} | Raw Lift: {raw_lift:.2f} | Raw Drag: {raw_drag:.2f}")
                sys.stdout.flush()
                
                if len(cfd_history) >= 3 and (max(cfd_history[-3:]) - min(cfd_history[-3:])) < 0.01: 
                    print("\n🛑 CFD PLATEAU REACHED.")
                    break
            except Exception as e: 
                print(f"❌ CFD Run Failed: {e}")
                break

        ref_sys = f"""You are an aerospace engineer. Improve the Lift-to-Drag ratio. Maintain {target_config}.
CRITICAL: You MUST change the numerical values slightly to explore new geometries. Do NOT output identical numbers.
Output ONLY valid JSON matching this exact structure:
{{
  "wing_span": float, "wing_chord": float, "wing_x": float,
  "tail_span": float, "tail_chord": float, "tail_x": float,
  "fuse_span": float, "fuse_chord": float, "fuse_x": float
}}
Return ONLY JSON."""
        
        prompt = tokenizer.apply_chat_template([
            {"role": "system", "content": ref_sys}, 
            {"role": "user", "content": f"Current State: {json.dumps(state)}. L/D is {ld}. Modify the values to improve aerodynamic efficiency."}
        ], tokenize=False, add_generation_prompt=True)

        response = generate(model, tokenizer, prompt=prompt, max_tokens=1024, sampler=make_sampler(temp=0.3)).strip()
        
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                new_state = json.loads(json_match.group(0))
                if new_state == state:
                    print("⚠️ LLM returned identical values. Forcing retry...")
                    skip_cfd = True
                    continue
                
                for k in state.keys(): state[k] = float(new_state.get(k, state[k]))
                step += 1 
                skip_cfd = False
            else: raise ValueError("No JSON block found.")
        except Exception as e:
            print("⚠️ 14B Model generated invalid JSON. Retrying...")
            skip_cfd = True

    if best_payload:
        print("\n🎉 OPTIMIZATION COMPLETE! Generating Final 3D STL...")
        best_payload["export_final_stl"] = True
        requests.post(DELL_LINUX_URL, json=best_payload)

if __name__ == "__main__":
    result = differential_evolution(evaluate_design, dynamic_bounds, popsize=6, workers=1, updating='deferred', callback=print_progress)
    x = result.x
    vlm_champion = [
        {"span": float(x[0]), "chord": float(x[1]), "sweep_angle": float(x[2]), "twist": float(x[3]), "taper": float(x[4]), "x": float(x[5]), "z": float(x[16]), "airfoil_name": AIRFOIL_ROSTER[int(x[7])]},
        {"span": float(x[8]), "chord": float(x[9]), "sweep_angle": float(x[10]), "twist": float(x[11]), "taper": float(x[12]), "x": float(x[13]), "z": float(x[16]), "airfoil_name": AIRFOIL_ROSTER[int(x[15])]},
        {"span": float(x[16]), "chord": float(x[17]), "sweep_angle": float(x[18]), "twist": float(x[19]), "taper": float(x[20]), "x": float(x[21]), "z": float(x[16]), "airfoil_name": AIRFOIL_ROSTER[int(x[23])]}
    ]
    run_cfd_refinement_loop(vlm_champion)