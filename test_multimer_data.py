from alphafold.inference.multimer_data import UFMPipeline

if __name__ == "__main__":
  # test UFMPipeline
  pipl = UFMPipeline(
    "./example_data/multi_chain_map.json",
    "./example_data/unifold_features",
    "./example_data/uniprot_msas")

  feats_hetero = pipl.process(
      "./example_data/mmcif/4z95.cif",
      "./dump/4z95.multimer.features.pkl",
      is_prokaryote=False)
  
  feats_homo = pipl.process(
      "./example_data/mmcif/4xnw.cif",
      "./dump/4xnw.multimer.features.pkl",
      is_prokaryote=False)

  pass
