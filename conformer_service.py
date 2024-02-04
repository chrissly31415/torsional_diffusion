import io
from typing import List
import gradio as gr
import pandas as pd


from rdkit.Chem import AllChem as Chem
from rdkit.Chem import SDWriter

from main import generate_conformers

def process_file(input_file: gr.File, smiles: List[str], energy_type: List[str], num_conformers: int, keep: int):
    smiles_list = []
    if smiles:
        smiles_list = smiles.split()
    
    if input_file is not None:
        # Determine the type of the file based on its extension
        if input_file.name.endswith('.smi'):
            # Assuming a .smi file with one SMILES per line
            content = str(input_file.read(), 'utf-8')
            smiles_list.extend(content.strip().split('\n'))
        elif input_file.name.endswith('.sdf'):
            # Parse the .sdf file to extract SMILES
            suppl = Chem.SDMolSupplier(input_file)
            smiles_list.extend([Chem.MolToSmiles(mol) for mol in suppl if mol is not None])
        
    result = generate_conformers(smiles_list,energy_type,num_conformers,True)

    sdf_file_path = "conformers.sdf"
    writer = SDWriter(sdf_file_path)
    
    data = []
    for i,smi in enumerate(result.keys()):
        sdf_strings = result[smi]['SDF']
        energies = result[smi]['energies']
        smiles = result[smi]["SMILES"]
        
        for j,(sdf,e) in enumerate(zip(sdf_strings,energies)):
            if j>=keep: break
            mol = Chem.MolFromMolBlock(sdf,removeHs=False)
            mol.SetProp("energy_xtb", str(e))
            smiles = Chem.MolToSmiles(Chem.MolFromMolBlock(sdf))
            mol.SetProp("SMILES", smiles)
            writer.write(mol)
            data.append({'system': i,'energy_xtb(kcal/mol)':e,  'SMILES':smiles, 'molBlock': sdf})

    writer.close()
    df = pd.DataFrame(data)
    return sdf_file_path, df

# Define the Gradio interface
inputs = [gr.File(label=".smi or .sdf File",file_types=['.smi','.sdf']),gr.TextArea(label="SMILES"), gr.Dropdown(["xtb","mmff"],value="xtb"), gr.Slider(value=10, label="conformers to generate"),gr.Slider(value=1, label="conformers to keep")]
demo = gr.Interface(fn=process_file,
                     inputs=inputs,
                     outputs=[gr.File(label="SDF file"),gr.DataFrame(label="energies")],
                     title="Conformer Generation",
                     description="using torsional diffusion model (Jing et al. 2022)")

# Launch the app
demo.launch()