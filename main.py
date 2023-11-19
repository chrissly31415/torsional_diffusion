# Standard library imports
from argparse import ArgumentParser
from typing import Dict, List, Union, Optional

# Third-party imports
import torch
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rdkit.Chem import Mol, AllChem, MolToMolBlock

# Local application/library specific imports
from diffusion.likelihood import populate_likelihood
from diffusion.sampling import get_seed, embed_seeds, perturb_seeds, pyg_to_mol, sample
from utils.utils import get_model


# If your conformer generation logic is in a different module, import it here
# from your_script_name import generate_conformers

app = FastAPI()

class ConformerRequest(BaseModel):
    smiles: list[str]
    energy_calculation: str = "xtb"  # 'xtb' or 'mmff'
    num_conformers: int = 10


parser = ArgumentParser()
args = parser.parse_args()
with open(f'workdir/drugs_default/model_parameters.yml') as f:
    args.__dict__.update(yaml.full_load(f))


model = get_model(args)
state_dict = torch.load(f'workdir/drugs_default/best_model.pt', map_location=torch.device('cpu'))
args.xtb = "/home/loschen/calc/xtb-6.6.1/bin/xtb"
model.load_state_dict(state_dict, strict=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()


@app.post("/generate_conformers/")
async def generate_conformers_endpoints(request: ConformerRequest):
    if request.energy_calculation not in ['xtb', 'mmff']:
        raise HTTPException(status_code=400, detail="Energy calculation method must be 'xtb' or 'mmff'")

    result = generate_conformers(request.smiles, request.energy_calculation, request.num_conformers)
    print(result)
    return {"message": "Conformer generation completed", "result": result}




def embed_func(mol: Mol, numConfs: int, numThreads = 4) -> Mol:
    """
    Embed multiple conformations for a given molecule.

    This function uses RDKit's EmbedMultipleConfs method to generate multiple
    conformations for a given RDKit molecule object.

    Args:
    - mol (Mol): An RDKit molecule object.
    - numConfs (int): The number of conformations to generate.
    - numThreads (int): The number of threads.

    Returns:
    - Mol: The RDKit molecule object with embedded conformations.
    """
    AllChem.EmbedMultipleConfs(mol, numConfs=numConfs, numThreads=numThreads)
    return mol


def generate_conformers(smiles_list: List[str], energy_type: str, num_conformers: int = 10, return_sdf: bool = True) -> Dict[str, Union[List[str], List[Mol]]]:
    """
    Generate conformers for a list of SMILES strings.

    This function generates conformers for each SMILES string in the given list.
    It supports energy calculation using either 'xtb' or 'mmff' methods. The function
    can return either a list of RDKit molecule objects or their SDF string representations.

    Args:
    - smiles_list (List[str]): A list of SMILES strings representing the molecules.
    - energy_type (str): The type of energy calculation to perform ('xtb' or 'mmff').
    - num_conformers (int, optional): The number of conformers to generate for each molecule. Defaults to 10.
    - return_sdf (bool, optional): Whether to return SDF strings instead of RDKit molecule objects. Defaults to True.

    Returns:
    - Dict[str, Union[List[str], List[Mol]]]: A dictionary with SMILES as keys and either a list of SDF strings
      or a list of RDKit molecule objects as values, depending on the 'return_sdf' argument.
    """
    # Energy calculations based on the specified type
    if energy_type == 'xtb':
        energy_type = "/home/loschen/calc/xtb-6.6.1/bin/xtb"
    
    results = {}
    for smi in smiles_list:
        mols = sample_confs(smi, num_conformers, energy_type)
        if not mols:
            continue

        # Store the results
        conformer_dict = {}
        conformer_dict["SMILES"] = AllChem.MolToSmiles(mols[0])
        conformer_dict["inchikeys"] = AllChem.MolToInchiKey(mols[0])
        #conformer_dict["conformers"] = mols
        energies = [m.xtb_energy for m in mols]
        
        mols_with_energy = list(zip(mols, energies))
        sorted_mols_with_energy = sorted(mols_with_energy, key=lambda x: x[1])
        sorted_mols, sorted_energies = zip(*sorted_mols_with_energy)
        conformer_dict["energies"] = sorted_energies
        
        sdf_strings = []
        for mol in sorted_mols:
                sdf_string = MolToMolBlock(mol)
                sdf_strings.append(sdf_string)
        conformer_dict["SDF"] = sdf_strings
        
        results[smi] = conformer_dict
        
    return results


def sample_confs(smi: str, n_confs: int, energy_type: str) -> Optional[List[Mol]]:
    """
    Generate conformers for a given SMILES string.

    This function generates conformers for a molecule represented by a SMILES string.
    It involves several steps including seed generation, embedding of seeds, optional
    perturbation, and sampling of conformers. The energy calculation can be done using 
    either 'mmff' or 'xtb', determined by the 'energy_type' parameter.

    Args:
    - smi (str): The SMILES string representing the molecule.
    - n_confs (int): The number of conformers to generate.
    - energy_type (str): The type of energy calculation to perform ('xtb' or 'mmff').

    Returns:
    - Optional[List[Mol]]: A list of RDKit molecule objects with generated conformers.
      Returns None if the seed generation or embedding fails.
    """
    mol, data = get_seed(smi, dataset=args.dataset)
    if not mol:
        print('Failed to get seed', smi)
        return None

    n_rotable_bonds = int(data.edge_mask.sum())

    conformers, pdb = embed_seeds(mol, data, n_confs, single_conf=False,
                                  pdb=False, embed_func=embed_func, mmff=False)
    if not conformers:
        print("Failed to embed", smi)
        return None

    if n_rotable_bonds > 0.5:
        conformers = perturb_seeds(conformers, pdb)

    if n_rotable_bonds > 0.5:
        conformers = sample(conformers, model, args.sigma_max, args.sigma_min, n_confs,
                            args.batch_size, False, None, pdb, mol=mol)

    mols = [pyg_to_mol(mol, conf, (energy_type == "mmff"), rmsd=True) for conf in conformers]

    for mol, data in zip(mols, conformers):
        populate_likelihood(mol, data, water=True, xtb=energy_type)

    if "xtb" in energy_type:
        mols = [mol for mol in mols if mol.xtb_energy]
    return mols


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8090)
    #result = generate_conformers(["CCCCO","CCCCNCC"],"/home/loschen/calc/xtb-6.6.1/bin/xtb",10)
    #print(result)
    