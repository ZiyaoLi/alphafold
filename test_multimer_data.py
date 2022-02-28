from alphafold.inference.multimer_data import UFMPipeline

if __name__ == "__main__":
  # test UFMPipeline
  pipl = UFMPipeline(
    "./local_data/multi_chain_map.json",
    "./local_data/unifold_features",
    "./local_data/uniprot_msas")

  feats_hetero = pipl.process(
      "./local_data/mmcif/4z95.cif",
      "./local_dump/4z95.multimer.features.pkl",
      is_prokaryote=False)
  
  feats_homo = pipl.process(
      "./local_data/mmcif/4xnw.cif",
      "./local_dump/4xnw.multimer.features.pkl",
      is_prokaryote=False)

  pass
