# predict_hackathon.py
import argparse
import json
import os
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any, List, Optional

import yaml
from hackathon_api import Datapoint, Protein, SmallMolecule

import csv
import math
from copy import deepcopy


DEFAULT_OUT_DIR = Path("predictions")
DEFAULT_SUBMISSION_DIR = Path("submission")
DEFAULT_INPUTS_DIR = Path("inputs")

P2RANK_HOME = "/opt/p2rank/"
TOP_K_POCKETS = int(os.environ.get("TOP_K_POCKETS", "5"))
POCKET_RES_MAX = int(os.environ.get("POCKET_RES_MAX", "24"))

# How many pockets to generate YAMLs for (default uses your old TOP_K_POCKETS)
TOP_N_POCKETS = int(os.environ.get("TOP_N_POCKETS", str(TOP_K_POCKETS)))
POCKET_MAX_DISTANCE = float(os.environ.get("POCKET_MAX_DISTANCE", "12"))  # Å
POCKET_FORCE = os.environ.get("POCKET_FORCE", "false").lower() in ("1", "true", "yes")

# Ranking weights: composite = w_iptm * iptm + w_bind * bind_prob - w_aff * affinity_pred_value
RANK_W_IPTM = float(os.environ.get("RANK_W_IPTM", "0.6"))
RANK_W_BINDPROB = float(os.environ.get("RANK_W_BINDPROB", "0.4"))
RANK_W_AFFINITY = float(os.environ.get("RANK_W_AFFINITY", "0.1"))

# Sampling knobs
DIFFUSION_SAMPLES = int(os.environ.get("DIFFUSION_SAMPLES", "1"))
DIFFUSION_SAMPLES_AFFINITY = int(os.environ.get("DIFFUSION_SAMPLES_AFFINITY", "1"))
SAMPLING_STEPS_AFFINITY = int(os.environ.get("SAMPLING_STEPS_AFFINITY", "200"))
USE_POTENTIALS = os.environ.get("USE_POTENTIALS", "true").lower() in ("1", "true", "yes")
AFFINITY_MW_CORRECTION = os.environ.get("AFFINITY_MW_CORRECTION", "true").lower() in ("1","true","yes")

# ---------------------------------------------------------------------------
# ---- Participants should modify these four functions ----------------------
# ---------------------------------------------------------------------------


def prepare_protein_complex(datapoint_id: str, proteins: List[Protein], input_dict: dict, msa_dir: Optional[Path] = None) -> List[tuple[dict, List[str]]]:
    """
    Prepare input dict and CLI args for a protein complex prediction.
    You can return multiple configurations to run by returning a list of (input_dict, cli_args) tuples.
    Args:
        datapoint_id: The unique identifier for this datapoint
        proteins: List of protein sequences to predict as a complex
        input_dict: Prefilled input dict
        msa_dir: Directory containing MSA files (for computing relative paths)
    Returns:
        List of tuples of (final input dict that will get exported as YAML, list of CLI args). Each tuple represents a separate configuration to run.
    """
    # Please note:
    # `proteins`` will contain 3 chains
    # H,L: heavy and light chain of the Fv or Fab region
    # A: the antigen
    #
    # you can modify input_dict to change the input yaml file going into the prediction, e.g.
    # ```
    # input_dict["constraints"] = [{
    #   "contact": {
    #       "token1" : [CHAIN_ID, RES_IDX/ATOM_NAME], 
    #       "token1" : [CHAIN_ID, RES_IDX/ATOM_NAME]
    #   }
    # }]
    # ```
    #
    # will add contact constraints to the input_dict

    # Example: predict 5 structures
    cli_args = ["--diffusion_samples", "5"]
    return [(input_dict, cli_args)]

def prepare_protein_ligand(datapoint_id: str, protein: Protein, ligands: list[SmallMolecule], input_dict: dict, msa_dir: Optional[Path] = None) -> List[tuple[dict, List[str]]]:
    """
    For a single-chain protein + single small-molecule ligand:
      1) run apo -> p2rank
      2) create one YAML per top-N pockets with pocket constraint
      3) enable affinity property
      4) return corresponding CLI args for Boltz-2
    """
    if len(ligands) != 1:
        raise ValueError("Expected a single ligand for protein_ligand task.")
    ligand = ligands[0]
    ligand_chain = ligand.id

    # 1) quick apo to get PDB for p2rank
    apo_pdb = _run_boltz_protein_only_apo(datapoint_id, protein, msa_dir, args.intermediate_dir)

    # 2) p2rank on apo, pick top-N pockets
    p2_out = args.intermediate_dir / "p2rank" / datapoint_id
    pockets_csv = _run_p2rank(apo_pdb, p2_out)
    pockets = _load_p2rank_pockets(pockets_csv)
    if not pockets:
        raise RuntimeError("No pockets found by p2rank.")

    chosen = pockets[:TOP_N_POCKETS]
    print(f"Using top {len(chosen)} pocket(s): ranks {[p['rank'] for p in chosen]}")

    # Base CLI args
    cli_base = [
        "--diffusion_samples", str(DIFFUSION_SAMPLES),
        "--diffusion_samples_affinity", str(DIFFUSION_SAMPLES_AFFINITY),
        "--sampling_steps_affinity", str(SAMPLING_STEPS_AFFINITY),
        "--output_format", "pdb",
        # "--no_kernels",
    ]
    if USE_POTENTIALS:
        cli_base.append("--use_potentials")
    if AFFINITY_MW_CORRECTION:
        cli_base.append("--affinity_mw_correction")

    configs: List[tuple[dict, List[str]]] = []

    for pk_idx, pk in enumerate(chosen):
        cfg = deepcopy(input_dict)

        # ensure properties + affinity
        props = cfg.setdefault("properties", [])
        # remove any previous affinity entries to avoid duplicates
        props = [p for p in props if "affinity" not in p]
        props.append({"affinity": {"binder": ligand_chain}})
        cfg["properties"] = props
        if pk_idx == 0: # add it once without constrains
            print(cfg)
            configs.append((cfg, list(cli_base)))
            cfg = deepcopy(cfg)

        # ensure constraints list
        if "constraints" not in cfg or cfg["constraints"] is None:
            cfg["constraints"] = []

        # add pocket
        cfg["constraints"].append(_make_pocket_constraint(ligand_chain, pk["residues"]))
        
        configs.append((cfg, list(cli_base)))  # independent list

    return configs


def post_process_protein_ligand(datapoint: Datapoint, input_dicts: List[dict[str, Any]], cli_args_list: List[list[str]], prediction_dirs: List[Path]) -> List[Path]:
    """
    Rank all produced models with:
      score = RANK_W_IPTM * iptm + RANK_W_BINDPROB * affinity_probability_binary - RANK_W_AFFINITY * affinity_pred_value
    Higher is better. Returns model paths sorted by score desc.
    If fewer than 5 models are found, duplicate the best ones to return at least 5.
    """
    scored: list[tuple[float, Path]] = []

    for config_idx, prediction_dir in enumerate(prediction_dirs):
        base = f"{datapoint.datapoint_id}_config_{config_idx}"
        # per-config folder like .../predictions/<base>/
        pred_glob = sorted(prediction_dir.glob(f"{base}_model_*.pdb"))
        if not pred_glob:
            # also support mmcif in case user changed output_format
            pred_glob = sorted(prediction_dir.glob(f"{base}_model_*.cif"))

        # affinity JSON is per-input (no model index)
        affinity_json = prediction_dir / f"affinity_{base}.json"
        bind_prob = 0.0
        aff_value = float("inf")
        if affinity_json.exists():
            try:
                with open(affinity_json) as f:
                    aff = json.load(f)
                bind_prob = float(aff.get("affinity_probability_binary", 0.0))
                aff_value = float(aff.get("affinity_pred_value", float("inf")))
                print(f"Bind_prob {bind_prob}")
            except Exception as e:
                print(f"WARNING: failed reading {affinity_json}: {e}")

        for model_path in pred_glob:
            # confidence JSON is per-model
            conf_json = prediction_dir / f"confidence_{model_path.stem}.json"  # stem includes base_model_k
            iptm = 0.0
            if conf_json.exists():
                try:
                    with open(conf_json) as f:
                        cj = json.load(f)
                    iptm = float(cj.get("iptm"))
                    print(f"iPTm {iptm}")
                except Exception as e:
                    print(f"WARNING: failed reading {conf_json}: {e}")

            score = RANK_W_IPTM * iptm + RANK_W_BINDPROB * bind_prob
            scored.append((score, model_path))
            print(f"[rank] cfg {config_idx} | {model_path.name} -> iptm={iptm:.3f} bindP={bind_prob:.3f} affVal={aff_value:.3f} score={score:.3f}")

    if not scored:
        raise FileNotFoundError(f"No model files found for {datapoint.datapoint_id}")

    scored.sort(key=lambda x: x[0], reverse=True)
    ranked_paths = [p for _, p in scored]

    # Pad to at least 5 by duplicating from the top in order
    if len(ranked_paths) < 5:
        orig = ranked_paths[:]  # preserve original ranking
        k = 0
        while len(ranked_paths) < 5 and orig:
            ranked_paths.append(orig[k % len(orig)])
            k += 1

    # Return sorted paths (could be >5; caller already slices to top 5)
    return ranked_paths[:5]

# -----------------------------------------------------------------------------
# ---- End of participant section ---------------------------------------------
# -----------------------------------------------------------------------------




ap = argparse.ArgumentParser(
    description="Hackathon scaffold for Boltz predictions",
    epilog="Examples:\n"
            "  Single datapoint: python predict_hackathon.py --input-json examples/specs/example_protein_ligand.json --msa-dir ./msa --submission-dir submission --intermediate-dir intermediate\n"
            "  Multiple datapoints: python predict_hackathon.py --input-jsonl examples/test_dataset.jsonl --msa-dir ./msa --submission-dir submission --intermediate-dir intermediate",
    formatter_class=argparse.RawDescriptionHelpFormatter
)

input_group = ap.add_mutually_exclusive_group(required=True)
input_group.add_argument("--input-json", type=str,
                        help="Path to JSON datapoint for a single datapoint")
input_group.add_argument("--input-jsonl", type=str,
                        help="Path to JSONL file with multiple datapoint definitions")

ap.add_argument("--msa-dir", type=Path,
                help="Directory containing MSA files (for computing relative paths in YAML)")
ap.add_argument("--submission-dir", type=Path, required=False, default=DEFAULT_SUBMISSION_DIR,
                help="Directory to place final submissions")
ap.add_argument("--intermediate-dir", type=Path, required=False, default=Path("hackathon_intermediate"),
                help="Directory to place generated input YAML files and predictions")
ap.add_argument("--group-id", type=str, required=False, default=None,
                help="Group ID to set for submission directory (sets group rw access if specified)")
ap.add_argument("--result-folder", type=Path, required=False, default=None,
                help="Directory to save evaluation results. If set, will automatically run evaluation after predictions.")

args = ap.parse_args()

def _prefill_input_dict(datapoint_id: str, proteins: Iterable[Protein], ligands: Optional[list[SmallMolecule]] = None, msa_dir: Optional[Path] = None) -> dict:
    """
    Prepare input dict for Boltz YAML.
    """
    seqs = []
    for p in proteins:
        if msa_dir and p.msa:
            if Path(p.msa).is_absolute():
                msa_full_path = Path(p.msa)
            else:
                msa_full_path = msa_dir / p.msa
            try:
                msa_relative_path = os.path.relpath(msa_full_path, Path.cwd())
            except ValueError:
                msa_relative_path = str(msa_full_path)
        else:
            msa_relative_path = p.msa
        entry = {
            "protein": {
                "id": p.id,
                "sequence": p.sequence,
                "msa": msa_relative_path
            }
        }
        seqs.append(entry)
    if ligands:
        def _format_ligand(ligand: SmallMolecule) -> dict:
            output =  {
                "ligand": {
                    "id": ligand.id,
                    "smiles": ligand.smiles
                }
            }
            return output
        
        for ligand in ligands:
            seqs.append(_format_ligand(ligand))
    doc = {
        "version": 1,
        "sequences": seqs,
    }
    return doc

def _run_boltz_protein_only_apo(datapoint_id: str, protein: Protein, msa_dir: Optional[Path], intermediate_dir: Path) -> Path:
    """
    Runs a quick protein-only Boltz prediction to obtain an apo PDB.
    Returns path to the best (or first) apo model PDB.
    """
    input_dir = intermediate_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    out_dir = intermediate_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Minimal YAML: one protein, no ligands
    apo_yaml = input_dir / f"{datapoint_id}_apo.yaml"
    apo_doc = _prefill_input_dict(datapoint_id, [protein], ligands=None, msa_dir=msa_dir)
    with open(apo_yaml, "w") as f:
        yaml.safe_dump(apo_doc, f, sort_keys=False)

    cache = os.environ.get("BOLTZ_CACHE", str(Path.home() / ".boltz"))
    cmd = [
        "boltz", "predict", str(apo_yaml),
        "--devices", "1",
        "--out_dir", str(out_dir),
        "--cache", cache,
        # "--no_kernels",
        "--use_potentials",
        "--output_format", "pdb",
        "--diffusion_samples", "1",          # fast apo
    ]
    print("Running apo prediction:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    # Find first produced apo model
    # Matches scaffold’s output folder layout:
    apo_dir_glob = list((out_dir).glob(f"boltz_results_{datapoint_id}_apo*/predictions/*"))
    if not apo_dir_glob:
        # try standard name we used
        apo_dir_glob = list((out_dir).glob(f"boltz_results_{datapoint_id}_apo/predictions/*"))
    if not apo_dir_glob:
        # fallback: any predictions subfolder for this datapoint
        apo_dir_glob = list((out_dir).glob(f"boltz_results_{datapoint_id}*/predictions/*"))

    pdbs = []
    for d in apo_dir_glob:
        pdbs.extend(sorted(d.glob("*.pdb")))
    if not pdbs:
        raise FileNotFoundError(f"No apo PDBs found for {datapoint_id}")

    return pdbs[0]

def _run_p2rank(pdb_path: Path, out_dir: Path) -> Path:
    """
    Runs p2rank and returns path to its main pockets CSV.
    """
    if P2RANK_HOME is None:
        raise EnvironmentError("P2RANK_HOME is not set. Please export P2RANK_HOME to your p2rank installation.")

    out_dir.mkdir(parents=True, exist_ok=True)
    p2rank_script = "prank"
    cmd = [str(Path(P2RANK_HOME) / p2rank_script), "predict", "-f", str(pdb_path), "-o", str(out_dir)]
    print("Running p2rank:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    csv_candidates = list(out_dir.glob("*_predictions.csv"))
    if not csv_candidates:
        raise FileNotFoundError("p2rank pockets CSV not found.")
    return csv_candidates[0]

def _load_p2rank_pockets(csv_path: Path):
    """
    Parse p2rank *_predictions.csv and return list of pocket dicts:
    [{'rank': int, 'score': float, 'prob': float, 'center': (x,y,z),
      'residues': [(chain, resi-int), ...]}, ...] (sorted by rank asc)
    """
    def _parse_res_list(s: str):
        out = []
        if not s:
            return out
        for tok in s.replace(",", " ").split():
            tok = tok.strip()
            # Accept A_133 or A:133 or A-133
            for sep in ("_", ":", "-"):
                if sep in tok:
                    ch, idx = tok.split(sep, 1)
                    try:
                        out.append((ch.strip(), int(idx)))
                    except ValueError:
                        pass
                    break
        # de-dup while preserving order
        seen = set()
        uniq = []
        for ch, i in out:
            key = (ch, i)
            if key not in seen:
                seen.add(key)
                uniq.append(key)
        return uniq

    pockets = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        for row in reader:
            row = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
            try:
                rk = int(row.get("rank"))
            except Exception:
                continue
            score = float(row.get("score", "nan")) if row.get("score") else float("nan")
            prob = float(row.get("probability", "nan")) if row.get("probability") else float("nan")
            cx = float(row.get("center_x", "nan")) if row.get("center_x") else float("nan")
            cy = float(row.get("center_y", "nan")) if row.get("center_y") else float("nan")
            cz = float(row.get("center_z", "nan")) if row.get("center_z") else float("nan")
            residues = _parse_res_list(row.get("residue_ids", ""))
            # cap number of residues per pocket
            residues = residues[:POCKET_RES_MAX]
            pockets.append({
                "rank": rk,
                "score": score,
                "prob": prob,
                "center": (cx, cy, cz),
                "residues": residues,
            })
    pockets.sort(key=lambda x: x["rank"])
    return pockets

def _make_pocket_constraint(ligand_chain_id: str, residues: list[tuple[str, int]]) -> dict:
    """
    Build a Boltz pocket constraint from residues [(chain, idx), ...].
    """
    contacts = [[ch, int(idx)] for ch, idx in residues]
    return {
        "pocket": {
            "binder": ligand_chain_id,
            "contacts": contacts,
            "max_distance": POCKET_MAX_DISTANCE,
            "force": POCKET_FORCE,
        }
    }

def _make_pocket_noconstraint(ligand_chain_id: str) -> dict:
    """
    Build a Boltz pocket constraint from residues [(chain, idx), ...].
    """
    return {
        "pocket": {
            "binder": ligand_chain_id,
        }
    }
    
def _run_boltz_and_collect(datapoint) -> None:
    """
    New flow: prepare input dict, write yaml, run boltz, post-process, copy submissions.
    """
    out_dir = args.intermediate_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    subdir = args.submission_dir / datapoint.datapoint_id
    subdir.mkdir(parents=True, exist_ok=True)

    # Prepare input dict and CLI args
    base_input_dict = _prefill_input_dict(datapoint.datapoint_id, datapoint.proteins, datapoint.ligands, args.msa_dir)

    if datapoint.task_type == "protein_complex":
        configs = prepare_protein_complex(datapoint.datapoint_id, datapoint.proteins, base_input_dict, args.msa_dir)
    elif datapoint.task_type == "protein_ligand":
        configs = prepare_protein_ligand(datapoint.datapoint_id, datapoint.proteins[0], datapoint.ligands, base_input_dict, args.msa_dir)
    else:
        raise ValueError(f"Unknown task_type: {datapoint.task_type}")

    # Run boltz for each configuration
    all_input_dicts = []
    all_cli_args = []
    all_pred_subfolders = []
    
    input_dir = args.intermediate_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    
    for config_idx, (input_dict, cli_args) in enumerate(configs):
        # Write input YAML with config index suffix
        yaml_path = input_dir / f"{datapoint.datapoint_id}_config_{config_idx}.yaml"
        with open(yaml_path, "w") as f:
            yaml.safe_dump(input_dict, f, sort_keys=False, default_flow_style=True)

        # Run boltz
        cache = os.environ.get("BOLTZ_CACHE", str(Path.home() / ".boltz"))
        fixed = [
            "boltz", "predict", str(yaml_path),
            "--devices", "1",
            "--out_dir", str(out_dir),
            "--cache", cache,
            # "--no_kernels",
            "--output_format", "pdb",
        ]
        cmd = fixed + cli_args
        print(f"Running config {config_idx}:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

        # Compute prediction subfolder for this config
        pred_subfolder = out_dir / f"boltz_results_{datapoint.datapoint_id}_config_{config_idx}" / "predictions" / f"{datapoint.datapoint_id}_config_{config_idx}"
        
        all_input_dicts.append(input_dict)
        all_cli_args.append(cli_args)
        all_pred_subfolders.append(pred_subfolder)

    # Post-process and copy submissions
    if datapoint.task_type == "protein_complex":
        ranked_files = post_process_protein_complex(datapoint, all_input_dicts, all_cli_args, all_pred_subfolders)
    elif datapoint.task_type == "protein_ligand":
        ranked_files = post_process_protein_ligand(datapoint, all_input_dicts, all_cli_args, all_pred_subfolders)
    else:
        raise ValueError(f"Unknown task_type: {datapoint.task_type}")

    if not ranked_files:
        raise FileNotFoundError(f"No model files found for {datapoint.datapoint_id}")

    for i, file_path in enumerate(ranked_files[:5]):
        target = subdir / (f"model_{i}.pdb" if file_path.suffix == ".pdb" else f"model_{i}{file_path.suffix}")
        shutil.copy2(file_path, target)
        print(f"Saved: {target}")

    if args.group_id:
        try:
            subprocess.run(["chgrp", "-R", args.group_id, str(subdir)], check=True)
            subprocess.run(["chmod", "-R", "g+rw", str(subdir)], check=True)
        except Exception as e:
            print(f"WARNING: Failed to set group ownership or permissions: {e}")

def _load_datapoint(path: Path):
    """Load JSON datapoint file."""
    with open(path) as f:
        return Datapoint.from_json(f.read())

def _run_evaluation(input_file: str, task_type: str, submission_dir: Path, result_folder: Path):
    """
    Run the appropriate evaluation script based on task type.
    
    Args:
        input_file: Path to the input JSON or JSONL file
        task_type: Either "protein_complex" or "protein_ligand"
        submission_dir: Directory containing prediction submissions
        result_folder: Directory to save evaluation results
    """
    script_dir = Path(__file__).parent
    
    if task_type == "protein_complex":
        eval_script = script_dir / "evaluate_abag.py"
        cmd = [
            "python", str(eval_script),
            "--dataset-file", input_file,
            "--submission-folder", str(submission_dir),
            "--result-folder", str(result_folder)
        ]
    elif task_type == "protein_ligand":
        eval_script = script_dir / "evaluate_asos.py"
        cmd = [
            "python", str(eval_script),
            "--dataset-file", input_file,
            "--submission-folder", str(submission_dir),
            "--result-folder", str(result_folder)
        ]
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
    
    print(f"\n{'=' * 80}")
    print(f"Running evaluation for {task_type}...")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 80}\n")
    
    subprocess.run(cmd, check=True)
    print(f"\nEvaluation complete. Results saved to {result_folder}")

def _process_jsonl(jsonl_path: str, msa_dir: Optional[Path] = None):
    """Process multiple datapoints from a JSONL file."""
    print(f"Processing JSONL file: {jsonl_path}")

    for line_num, line in enumerate(Path(jsonl_path).read_text().splitlines(), 1):
        if not line.strip():
            continue

        print(f"\n--- Processing line {line_num} ---")

        try:
            datapoint = Datapoint.from_json(line)
            _run_boltz_and_collect(datapoint)

        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid JSON on line {line_num}: {e}")
            continue
        except Exception as e:
            print(f"ERROR: Failed to process datapoint on line {line_num}: {e}")
            raise e
            continue

def _process_json(json_path: str, msa_dir: Optional[Path] = None):
    """Process a single datapoint from a JSON file."""
    print(f"Processing JSON file: {json_path}")

    try:
        datapoint = _load_datapoint(Path(json_path))
        _run_boltz_and_collect(datapoint)
    except Exception as e:
        print(f"ERROR: Failed to process datapoint: {e}")
        raise

def main():
    """Main entry point for the hackathon scaffold."""
    # Determine task type from first datapoint for evaluation
    task_type = None
    input_file = None
    
    if args.input_json:
        input_file = args.input_json
        _process_json(args.input_json, args.msa_dir)
        # Get task type from the single datapoint
        try:
            datapoint = _load_datapoint(Path(args.input_json))
            task_type = datapoint.task_type
        except Exception as e:
            print(f"WARNING: Could not determine task type: {e}")
    elif args.input_jsonl:
        input_file = args.input_jsonl
        _process_jsonl(args.input_jsonl, args.msa_dir)
        # Get task type from first datapoint in JSONL
        try:
            with open(args.input_jsonl) as f:
                first_line = f.readline().strip()
                if first_line:
                    first_datapoint = Datapoint.from_json(first_line)
                    task_type = first_datapoint.task_type
        except Exception as e:
            print(f"WARNING: Could not determine task type: {e}")
    
    # Run evaluation if result folder is specified and task type was determined
    if args.result_folder and task_type and input_file:
        try:
            _run_evaluation(input_file, task_type, args.submission_dir, args.result_folder)
        except Exception as e:
            print(f"WARNING: Evaluation failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
