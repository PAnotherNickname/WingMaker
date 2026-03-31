import json
import requests
import os
import sys
import time
import math
import warnings
from scipy.optimize import differential_evolution

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
except ImportError:
    print("❌ Missing Libraries. Run: pip install mlx-lm requests langchain-huggingface chromadb scipy")
    sys.exit(1)

DELL_LINUX_URL = "http://172.22.228.131:8000/simulate"
CHROMA_DB_DIR = os.path.expanduser("~/mlx_env/chroma_db")

print("\n⚙️ Launching Aero-Optimizer v12.0 (UNIVERSAL MDO + STRUCTURAL INTEGRITY)...")
model, tokenizer = load("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit")

# ====================================================================
# PRE-FLIGHT
# ====================================================================
user_prompt = input("\n🎯 ENTER DESIGN GOAL (e.g., 'Design an ultra-efficient aircraft'): ")
if not user_prompt.strip():
    user_prompt = "Design a highly efficient aircraft optimized for high lift-to-drag ratio."

print("📚 Querying RAG Database for targeted context...")
try:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(user_prompt)
    raw_context = "\n\n".join([d.page_content for d in docs])[:4000]
except Exception as e:
    raw_context = ""

# THE UNIVERSAL SEARCH SPACE (16 VARIABLES)
dynamic_bounds = [
    # --- SURFACE A (The Main Wing) ---
    (0.1, 2.5),    # 0: Span A 
    (0.05, 0.4),   # 1: Chord A 
    (-20.0, 45.0), # 2: Sweep A 
    (-10.0, 5.0),  # 3: Twist A
    (0.1, 1.0),    # 4: Taper A 
    (-0.5, 0.5),   # 5: X Position A 
    (-0.2, 0.2),   # 6: Z Position A 
    (0.0, 3.99),   # 7: Airfoil A Index
    
    # --- SURFACE B (The Secondary Wing / Tail / Canard) ---
    (0.1, 2.5),    # 8: Span B
    (0.05, 0.4),   # 9: Chord B
    (-20.0, 45.0), # 10: Sweep B
    (-15.0, 10.0), # 11: Twist B 
    (0.1, 1.0),    # 12: Taper B
    (-1.5, 1.5),   # 13: X Position B 
    (-0.2, 0.2),   # 14: Z Position B 
    (0.0, 3.99)    # 15: Airfoil B Index
]

iteration_count = 0
best_ld_seen = 0.0
AIRFOIL_ROSTER = ["mh45", "mh60", "pw75", "naca0012"]
vlm_history = []

# ====================================================================
# PHASE 1: VLM FITNESS FUNCTION (UNIVERSAL PHYSICS)
# ====================================================================
def evaluate_design(x):
    global iteration_count, best_ld_seen
    
    # Unpack all 16 variables
    s_a, c_a, sw_a, tw_a, ta_a, x_a, z_a, af_idx_a = x[0:8]
    s_b, c_b, sw_b, tw_b, ta_b, x_b, z_b, af_idx_b = x[8:16]
    
    af_a = AIRFOIL_ROSTER[int(af_idx_a)]
    af_b = AIRFOIL_ROSTER[int(af_idx_b)]
    
    payload = {
        "run_cfd": False,
        "surfaces": [
            {"span": float(s_a), "chord": float(c_a), "sweep_angle": float(sw_a), "twist": float(tw_a), "taper": float(ta_a), "x": float(x_a), "z": float(z_a), "airfoil_name": af_a},
            {"span": float(s_b), "chord": float(c_b), "sweep_angle": float(sw_b), "twist": float(tw_b), "taper": float(ta_b), "x": float(x_b), "z": float(z_b), "airfoil_name": af_b}
        ]
    }

    try:
        r = requests.post(DELL_LINUX_URL, json=payload, timeout=40)
        data = r.json()
        ld = float(data.get("lift_to_drag_ratio", 0.0))
        cm = float(data.get("pitch_moment", 99.0))
    except Exception:
        return 99999.0 

    # --- THE UNIVERSAL PHYSICS BALANCER ---
    base_score = -ld 
    cm_penalty = (abs(cm) ** 2) * 5000 
    
    # Volume calculation for both surfaces
    tip_c_a = max(0.01, c_a * ta_a)
    vol_a = (s_a * (c_a + tip_c_a) / 2.0) * ((c_a + tip_c_a) / 2.0 * 0.10)
    
    tip_c_b = max(0.01, c_b * ta_b)
    vol_b = (s_b * (c_b + tip_c_b) / 2.0) * ((c_b + tip_c_b) / 2.0 * 0.10)
    
    total_volume = vol_a + vol_b
    
    volume_penalty = 0.0
    target_volume = 0.0015 # Still need 1.5 Liters of total space
    if total_volume < target_volume:
        volume_penalty = ((target_volume - total_volume) ** 2) * 1000000

    # Structural Mass (Spars for both wings)
    spar_weight_a = (s_a ** 2) / (c_a * 10.0)
    spar_weight_b = (s_b ** 2) / (c_b * 10.0)
    
    # THE INVISIBLE PIPE PENALTY: Fuselage Boom Weight
    distance_between_wings = math.sqrt((x_a - x_b)**2 + (z_a - z_b)**2)
    fuselage_weight = distance_between_wings * 0.5 
    
    structural_penalty = (spar_weight_a + spar_weight_b + fuselage_weight) * 0.5 

    # --- THE NEW WINGTIP JOINT PENALTY ---
    # Calculate the exact 3D coordinates of both wingtips
    tip_x_a = x_a + (s_a / 2.0) * math.tan(math.radians(sw_a))
    tip_x_b = x_b + (s_b / 2.0) * math.tan(math.radians(sw_b))
    
    # Calculate the physical distance between the two wingtips
    tip_gap = math.sqrt((tip_x_a - tip_x_b)**2 + (z_a - z_b)**2 + ((s_a/2.0) - (s_b/2.0))**2)

    joint_penalty = 0.0
    # If the AI tries to join the wings (tips are within 15cm of each other)...
    if tip_gap < 0.15:
        # ...the joint MUST be thick enough to hold a structural bracket/spar!
        joint_thickness = tip_c_a + tip_c_b
        if joint_thickness < 0.15: # Demands at least 15cm of combined physical material
            joint_penalty = ((0.15 - joint_thickness) ** 2) * 5000000
    
    # Final fitness includes the new joint penalty
    fitness = base_score + cm_penalty + volume_penalty + structural_penalty + joint_penalty
    
    if ld > best_ld_seen and abs(cm) < 0.05 and total_volume >= target_volume:
        best_ld_seen = ld
        
    iteration_count += 1
    return fitness

def print_progress(xk, convergence):
    global vlm_history
    s_a, c_a, sw_a, tw_a, ta_a, x_a, z_a, af_idx_a = xk[0:8]
    s_b, c_b, sw_b, tw_b, ta_b, x_b, z_b, af_idx_b = xk[8:16]
    
    config = "Flying Wing"
    if s_b < 0.3: config = "Flying Wing (Surface B Deleted)"
    elif x_b > x_a + 0.3: config = "Conventional (Tail in Back)"
    elif x_b < x_a - 0.3: config = "Canard (Tail in Front)"
    elif abs(x_b - x_a) <= 0.3 and abs(z_b - z_a) > 0.1: config = "Biplane / Tandem"
    elif abs(x_b - x_a) <= 0.3 and abs(z_b - z_a) <= 0.1: config = "Joined Wing"
    
    print(f"🧬 VLM Iteration | Guessing Config: [{config}] | Est L/D: {best_ld_seen:.1f}")

    vlm_history.append(best_ld_seen)
    if len(vlm_history) > 20:
        vlm_history.pop(0)
    
    if len(vlm_history) == 20:
        if (max(vlm_history) - min(vlm_history)) < 0.1:
            print("\n🛑 VLM PLATEAU REACHED: No significant improvement in 20 generations. Moving to CFD.")
            return True 
    return False

# ====================================================================
# PHASE 2 & 3: THE INFINITE CFD REFINEMENT LOOP
# ====================================================================
def run_cfd_refinement_loop(champion_params):
    print("\n" + "🔥"*25)
    print("PHASE 2: INITIATING INFINITE CFD REFINEMENT LOOP")
    print("🔥"*25)
    
    current_design = champion_params.copy()
    cfd_history = []
    
    best_cfd_ld = -999.0
    best_cfd_design = None
    
    step = 1
    while True:
        print(f"\n🔄 --- CFD REFINEMENT GENERATION {step} ---")
        
        cfd_payload = current_design.copy()
        cfd_payload["run_cfd"] = True
        cfd_payload["export_final_stl"] = False 
        
        try:
            cfd_r = requests.post(DELL_LINUX_URL, json=cfd_payload, timeout=400)
            cfd_data = cfd_r.json()
            cfd_ld = float(cfd_data.get('lift_to_drag_ratio', 0.0))
            cfd_lift = cfd_data.get('raw_cfd_lift', 0.0)
            cfd_drag = cfd_data.get('raw_cfd_drag', 0.0)
            
            print(f"📊 FluidX3D Results -> Lift: {cfd_lift:.2f} | Drag: {cfd_drag:.2f} | True L/D: {cfd_ld:.2f}")
            
            if cfd_ld > best_cfd_ld:
                best_cfd_ld = cfd_ld
                best_cfd_design = current_design.copy()
                
            cfd_history.append(cfd_ld)
            
            if len(cfd_history) >= 3:
                last_3 = cfd_history[-3:]
                variance = max(last_3) - min(last_3)
                if variance <= 0.01:
                    print(f"\n🛑 CFD PLATEAU REACHED: Last 3 scores fluctuated by only {variance:.3f}.")
                    break
            
        except Exception as e:
            print(f"❌ CFD Run Failed: {e}. Aborting refinement.")
            break
            
        print("🧠 LLM Analyzing Multi-Surface CFD results...")
        refinement_sys = """You are an AI aerodynamicist fine-tuning a multi-surface aircraft.
Make SLIGHT adjustments to the spans, chords, sweeps, twists, and positions to improve Lift-to-Drag.
Output ONLY valid JSON matching the exact structure of the input 'surfaces' array.
Do NOT output markdown (```json)."""

        user_msg = f"""CURRENT DESIGN:
{json.dumps(current_design, indent=2)}

CFD WIND TUNNEL RESULTS: L/D: {cfd_ld}

INSTRUCTIONS: Output the updated JSON payload tweaking variables to increase L/D."""

        prompt = tokenizer.apply_chat_template([
            {"role": "system", "content": refinement_sys}, 
            {"role": "user", "content": user_msg}
        ], tokenize=False, add_generation_prompt=True)

        response = generate(model, tokenizer, prompt=prompt, max_tokens=300, sampler=make_sampler(temp=0.3)).strip()
        
        try:
            clean_json = response.replace("```json", "").replace("```", "").strip()
            new_params = json.loads(clean_json)
            current_design = new_params
        except Exception as e:
            print(f"⚠️ LLM Output parsing failed. Keeping current design for next loop.")
            
        step += 1

    if best_cfd_design:
        print("\n🎉 OPTIMIZATION COMPLETE! Generating Final 3D STL...")
        final_payload = best_cfd_design.copy()
        final_payload["run_cfd"] = True
        final_payload["export_final_stl"] = True
        
        try:
            requests.post(DELL_LINUX_URL, json=final_payload, timeout=400)
            print("✅ ULTIMATE_CFD_CHAMPION.stl successfully saved on the Linux server!")
            print(f"FINAL STATS -> True L/D: {best_cfd_ld:.2f}")
        except Exception as e:
            print(f"⚠️ Failed to export final STL: {e}")

if __name__ == "__main__":
    print(f"🚀 PHASE 1: GENETIC ALGORITHM (Universal MDO Exploration)\n")
    start_time = time.time()
    
    result = differential_evolution(
        evaluate_design, dynamic_bounds, strategy='best1bin', 
        maxiter=1000, popsize=15, tol=0.01, callback=print_progress, disp=False
    )
    
    x = result.x
    vlm_champion = {
        "run_cfd": False,
        "surfaces": [
            {"span": float(x[0]), "chord": float(x[1]), "sweep_angle": float(x[2]), "twist": float(x[3]), "taper": float(x[4]), "x": float(x[5]), "z": float(x[6]), "airfoil_name": AIRFOIL_ROSTER[int(x[7])]},
            {"span": float(x[8]), "chord": float(x[9]), "sweep_angle": float(x[10]), "twist": float(x[11]), "taper": float(x[12]), "x": float(x[13]), "z": float(x[14]), "airfoil_name": AIRFOIL_ROSTER[int(x[15])]}
        ]
    }
    
    print("\n" + "="*50)
    print(f"🎯 VLM CONVERGENCE ACHIEVED in {time.time() - start_time:.1f} seconds!")
    print("="*50)
    
    run_cfd_refinement_loop(vlm_champion)