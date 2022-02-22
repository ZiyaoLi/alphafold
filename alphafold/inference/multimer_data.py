'''converting old features.pkl to new features.pkl'''

from importlib.resources import path
import os
from absl import logging
from Bio.PDB import PDBParser
import copy
import json
import numpy as np
import pathlib
import pickle
from typing import Dict, List, Mapping, Optional, Sequence

from alphafold.common.protein import from_pdb_string
from alphafold.common import residue_constants as rc
from alphafold.data import feature_processing
from alphafold.data.feature_processing import _is_homomer_or_monomer, pair_and_merge
from alphafold.data.mmcif_parsing import MmcifObject, parse as parse_mmcif_string
from alphafold.data.parsers import parse_stockholm
from alphafold.data.pipeline import FeatureDict, make_msa_features
from alphafold.data.pipeline_multimer import add_assembly_features, convert_monomer_features
from alphafold.train.utils import get_atom_positions

NR = 256
NM = 512
NT = 4

# single sequence features in multimer chain data (unaltered).
SEQ_FEATS = {
  'aatype': (NR, 21),           # one-hot encoding of sequence.
  'between_segment_residues': (NR,),
  'domain_name': (1,),          # a copy of fasta description.
  'residue_index': (NR,),       # := np.array(range(NR), dtype=np.int32)
  'seq_length': (NR,),          # := np.array([NR] * NR, dtype=np.int32)
  'sequence': (1,)              # the sequence string.
}

# msa features in multimer chain data (new features added).
MSA_FEATS = {
  'deletion_matrix_int': (NM, NR),    # the deletion matrix in int32.
  'msa': (NM, NR),              # the index of msas.
  'num_alignments': (NR,),      # := np.array([NM] * NR, dtype=np.int32)
  'msa_uniprot_accession_identifiers': (NM,),
                                # the extracted uniprot accession identifiers
                                # in string format. default ''.encode('utf-8').
  'msa_species_identifiers': (NM,)
                                # the extracted uniprot species identifiers
                                # in string format. default ''.encode('utf-8').
}

# template features in multimer chain data (redundant features removed).
TEMPLATE_FEATS = {
  'template_aatype': (NT, NR, 22),    # one-hot encodings of templates (+gap).
  'template_all_atom_masks': (NT, NR, 37),
  'template_all_atom_positions': (NT, NR, 37, 3),
  'template_domain_names': (NT,),     # domain names. default ''.encode().
  'template_sequence': (NT,),         # template seqs. default ''.encode().
  'template_sum_probs': (NT,)         # in float32.
}

MULTIMER_CHAIN_FEATS = {**SEQ_FEATS, **MSA_FEATS, **TEMPLATE_FEATS}

def convert_unifold_feats(feats: FeatureDict):
  feats = {k: v for k, v in feats
           if k in MULTIMER_CHAIN_FEATS.keys()}
  num_align = feats['num_alignments'][0]    # get NM.
  if 'msa_uniprot_accession_identifiers' not in feats.keys():
    feats['msa_uniprot_accession_identifiers'] = np.array(
        [''.encode('utf-8')] * num_align, dtype=np.object_)
  if 'msa_species_identifiers' not in feats.keys():
    feats['msa_species_identifiers'] = np.array(
        [''.encode('utf-8')] * num_align, dtype=np.object_)
  return feats

def load_unifold_feats(unifold_feats_path: str):
  with open(unifold_feats_path, 'rb') as fp:
    feats = pickle.load(fp)
  feats = convert_unifold_feats(feats)
  return feats

def load_uniprot_feats(
    uniprot_msa_path: str,
    max_uniprot_hits: int = 50000) -> FeatureDict:
  """load uniprot features from a given stockholm msa output."""
  if uniprot_msa_path.endswith('sto'):
    # assert uniprot_msa_out_path.endswith('uniprot_hits.sto')
    with open(uniprot_msa_path, 'r') as fp:
      uniprot_result = fp.read()
    uniprot_msa = parse_stockholm(uniprot_result)
    uniprot_msa = uniprot_msa.truncate(max_seqs=max_uniprot_hits)
    uniprot_feats = make_msa_features((uniprot_msa,))
  elif uniprot_msa_path.endswith('pkl'):
    with open(uniprot_msa_path, 'rb') as fp:
      uniprot_feats = pickle.load(fp)
  else:
    raise ValueError(f"the format of uniprot msa path {uniprot_msa_path} " \
                     f"is not supported.")
  all_seq_key = lambda k: f'{k}_all_seq' if not k.endswith('_all_seq') else k
  uniprot_feats = {all_seq_key(k): v for k, v in uniprot_feats.items()}
  return uniprot_feats


class UFMPipeline:
  def __init__(
      self,
      multi_chain_map_path: str,
      unifold_feats_dir: str,
      uniprot_msa_dir: str,
      uniprot_msa_format: str = 'sto',
      max_uniprot_hits: int = 50000):
    with open(multi_chain_map_path, 'r') as fp:
      multi_chain_map = json.load(fp)
    self.canon_chain_map = UFMPipeline._inverse_map(multi_chain_map)
    self._max_uniprot_hits = max_uniprot_hits

    self.unifold_feats_dir = unifold_feats_dir
    self.uniprot_feats_dir = uniprot_msa_dir

    # specify here how the input paths are composed:
    self.make_unifold_feats_path = lambda chain_id: \
        os.path.join(self.unifold_feats_dir, chain_id, 'features.pkl')
    if uniprot_msa_format == 'sto':
      self.make_uniprot_msa_path = lambda chain_id: \
          os.path.join(self.uniprot_feats_dir, chain_id, 'uniprot_hits.sto')
    elif uniprot_msa_format == 'pkl':
      self.make_uniprot_msa_path = lambda chain_id: \
          os.path.join(self.uniprot_feats_dir, chain_id, 'uniprot_feats.pkl')
  
  @staticmethod
  def _inverse_map(ent_ref_map: Mapping[str, List[str]]):
    ref_ent_map = {}
    for ent, refs in ent_ref_map.items():
      for ref in refs:
        if ref in ref_ent_map:
          ent_2 = ref_ent_map[ref]   # another ent exists for this ref.
          assert ent == ent_2, \
              f"multiple entities ({ent_2}, {ent}) exist for reference {ref}."
        ref_ent_map[ref] = ent
    return ref_ent_map

  def get_single_chain_feats(
      self,
      canon_chain_id: str,
      is_homomer_or_monomer: bool) -> FeatureDict:
    unifold_feats_path = self.make_unifold_feats_path(canon_chain_id)
    chain_feats = load_unifold_feats(unifold_feats_path)
    if not is_homomer_or_monomer:   # add all_seq_feats (uniprot)
      uniprot_msa_path = self.make_uniprot_msa_path(canon_chain_id)
      all_seq_feats = load_uniprot_feats(
          uniprot_msa_path, self._max_uniprot_hits)
      chain_feats.update(all_seq_feats)
    return chain_feats
  
  def get_single_chain_labs(
      self,
      short_chain_id: str,    # single letter chain id
      mmcif_object: MmcifObject) -> FeatureDict:
    chain_labs = {}
    all_atom_positions, all_atom_mask = get_atom_positions(
        mmcif_object, short_chain_id, max_ca_ca_distance=np.inf)
    chain_labs['all_atom_positions'] = all_atom_positions
    chain_labs['all_atom_mask'] = all_atom_mask
    return chain_labs
    
  def process(
      self,
      input_mmcif_path: str,
      dump_path: Optional[str] = None,) -> FeatureDict:
    # auto detect pdb_id from mmcif path.
    pdb_id = pathlib.Path(input_mmcif_path).stem
    # load mmcif object.
    with open(input_mmcif_path, 'r') as fp:
      mmcif_string = fp.read()
    mmcif_obj = parse_mmcif_string(
        file_id=pdb_id, mmcif_string=mmcif_string).mmcif_object

    chain_seqs = mmcif_obj.chain_to_seqres
    is_homomer_or_monomer = (len(set(chain_seqs.values())) == 1)

    all_chain_feats = {}
    seen_chain_feats = {}

    # var `cid` used for single letter chain id (auth_chain_id, e.g. 'A')
    # var `chain_id` used for full chain id (e.g. '4xnw_A')
    for cid, seq in chain_seqs.items():
      # get (canonical) chain ids.
      chain_id = f'{pdb_id}_{cid}'
      try:
        canon_chain_id = self.canon_chain_map[chain_id]
      except KeyError:
        raise ValueError(f"cannot find chain {chain_id} in the canonical map.")

      # try to reuse processed chain features by deep copying them.
      if canon_chain_id in seen_chain_feats:
        all_chain_feats[chain_id] = copy.deepcopy(
            seen_chain_feats[canon_chain_id])
        continue
      
      # get chain features and labels.
      chain_feats = self.get_single_chain_feats(
          canon_chain_id, is_homomer_or_monomer)
      chain_labs = self.get_single_chain_labs(
          short_chain_id=cid, mmcif_object=mmcif_obj)
      
      # check for conflicts in chain sequences between feats and labs.
      assert chain_feats['sequence'] == seq, \
          f"sequence conflicts detected for {chain_id} (canonical id " \
          f"{canon_chain_id}). sequence in mmcif: {seq}; sequence in " \
          f"unifold features: {chain_feats['sequence']}."
      
      # converting the chain feats and add the labels.
      chain_feats = convert_monomer_features(chain_feats, chain_id)
      chain_feats.update(chain_labs)

      # add to all chain feats
      all_chain_feats[chain_id] = chain_feats
      seen_chain_feats[canon_chain_id] = chain_feats
    
    all_chain_feats = add_assembly_features(all_chain_feats)
    np_example = pair_and_merge(all_chain_feats)
    if dump_path is not None:
      try:
        with open(dump_path, 'wb'):
          pickle.dump(np_example, fp)
      except:
        logging.warning(f"dumping multimer features to {dump_path} failed.")
    
    return np_example




