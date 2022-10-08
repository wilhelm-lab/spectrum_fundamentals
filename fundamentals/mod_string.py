from typing import List, Dict, Optional, Union, Tuple
from .constants import SPECTRONAUT_MODS, MAXQUANT_VAR_MODS, MOD_MASSES, MAXQUANT_NC_TERM, MOD_NAMES
import numpy as np
import re
import difflib


def internal_to_spectronaut(sequences: List[str]) -> List[str]:
    """
    Function to translate a modstring from the interal format to the spectronaut format
    :param sequences: List[str] of sequences
    :return: List[str] of modified sequences
    """
    regex = re.compile("(%s)" % "|".join(map(re.escape, SPECTRONAUT_MODS.keys())))
    return [regex.sub(lambda mo: SPECTRONAUT_MODS[mo.string[mo.start():mo.end()]], seq) for seq in sequences]


def maxquant_to_internal(
        sequences: List[str],
        fixed_mods: Optional[Dict[str, str]] = {'C': 'C[UNIMOD:4]'}
) -> List[str]:
    """
    Function to translate a MaxQuant modstring to the Prosit format
    :param sequences: List[str] of sequences
    :param fixed_mods: Optional dictionary of modifications with key aa and value mod, e.g. 'M': 'M(UNIMOD:35)'.
    Fixed modifications must be included in the variable modificatons dictionary throws Assertion error otherwise.
    :return: List[str] of modified sequences.
    """
    err_msg = f"Provided illegal fixed mod, supported modifications are {set(MAXQUANT_VAR_MODS.values())}."
    assert all(x in MAXQUANT_VAR_MODS.values() for x in fixed_mods.values()), err_msg

    replacements = {**MAXQUANT_VAR_MODS, **fixed_mods}

    def custom_regex_escape(key: str) -> str:
        """
        Subfunction to escape only normal brackets in the modstring
        :param key: The match to escape.
        :return match with escaped special characters.
        """
        for k, v in {"(": "\(", ")": "\)"}.items():
            key = key.replace(k, v)
        return key

    regex = re.compile("|".join(map(custom_regex_escape, replacements.keys())))

    def find_replacement(match: re) -> str:
        """
        Subfunction to find the corresponding substitution for a match.
        :param match: an re.Match object found by re.sub
        :return substitution string for the given match
        """
        key = match.string[match.start():match.end()]
        if "_" in key:  # If _ is in the match we need to differentiate n and c term
            if match.start() == 0:
                key = f"^{key}"
            else:
                key = f"{key}$"

        return replacements[key]

    return [regex.sub(find_replacement, seq).replace('_','') for seq in sequences]


def internal_without_mods(
        sequences: List[str]
) -> List[str]:
    """
    Function to remove any mod identifiers and return the plain AA sequence.
    :param sequences: List[str] of sequences
    :return: List[str] of modified sequences.
    """
    regex = "\[.*?\]|\-"
    return [re.sub(regex, "", seq) for seq in sequences]


def internal_to_mod_mass(
        sequences: List[str],
) -> List[str]:
    """
    Function to exchange the internal mod identifiers with the masses of the specific modifiction.
    :param sequences: List[str] of sequences
    :return List[str] of modified sequences.
    """
    regex = re.compile("(%s)" % "|".join(map(re.escape, MOD_MASSES.keys())))
    replacement_func = lambda match: f"[+{MOD_MASSES[match.string[match.start():match.end()]]}]"
    return [regex.sub(replacement_func, seq) for seq in sequences]


def internal_to_mod_names(
        sequences: List[str],
) -> List[Tuple[str, str]]:
    """
    Function to translate an internal modstring to MSP format
    :param sequences: List[str] of sequences
    :return: List[Tuple[str, str] of mod summary and mod sequences.
    """
    match_list = []
    pos = [0]
    offset = [0]

    def msp_string_mapper(seq: str):
        """
        Internal function to create the mod summary and mod_string from given sequence and match_list
        :param seq: The sequence to modify.
        :return Tuple with mod summary and mod_string
        """
        seq = regex.sub(replace_and_store, seq)
        mod_string = f"{seq}//{'; '.join([f'{name}@{seq[pos]}{pos}' for name, pos in match_list])}"
        mod = f"{len(match_list)}"
        if len(match_list) > 0:
            mod += f"/{'/'.join([f'{pos},{seq[pos]},{name}' for name, pos in match_list])}"
        pos[0] = 0
        offset[0] = 0
        match_list.clear()
        return mod, mod_string

    def replace_and_store(match: re):
        """
        Internal function that removes matched internal mods and stores there position in the sequence.
        :param match: an re.Match object found by re.sub
        """
        pos[0] = match.start() - 1 - offset[0]
        offset[0] += match.end() - match.start()
        match_list.append((MOD_NAMES[match.string[match.start():match.end()]], pos[0]))
        return ""

    regex = re.compile("(%s)" % "|".join(map(re.escape, MOD_NAMES.keys())))
    return [msp_string_mapper(seq) for seq in sequences]


def parse_modstrings(sequences, alphabet, translate=False, filter=False):
    """
    :param sequences: List of strings
    :param ALPHABET: dictionary where the keys correspond to all possible 'Elements' that can occur in the string
    :param translate: boolean to determine if the Elements should be translated to the corresponding values of ALPHABET
    :return: generator that yields a list of sequence 'Elements' or the translated sequence "Elements"
    """
    import re
    from itertools import repeat

    def split_modstring(sequence, r_pattern):
        # Ugly and fast fix for reading modifications as is from maxquant we should reconsider how to fix it.
        # sequence = sequence.replace('M(ox)','M(U:35)')
        # sequence = sequence.replace('C','C(U:4)')
        split_seq = r_pattern.findall(sequence)
        if "".join(split_seq) == sequence:
            if translate:
                return [alphabet[aa] for aa in split_seq]
            elif not translate:
                return split_seq
        elif filter:
            return [0]
        else:
            not_parsable_elements = "".join(
                [li[2] for li in difflib.ndiff(sequence, "".join(split_seq)) if li[0] == '-'])
            raise ValueError(f"The element(s) [{not_parsable_elements}] "
                             f"in the sequence [{sequence}] could not be parsed")

    pattern = sorted(alphabet, key=len, reverse=True)
    
    pattern = [re.escape(i) for i in pattern]
    regex_pattern = re.compile("|".join(pattern))
    return map(split_modstring, sequences, repeat(regex_pattern))
    
    
def proteomicsDB_to_internal(sequence, mods_variable, mods_fixed):
    """
    Function to create a sequence with UNIMOD modifications from given sequence and it's varaible and fixed modifications
    :param 
        sequence: The sequence to modify.
        mods_variable: the variable modifacations (e.g. "Oxidation@M45")
        mods_fixed: the fixed modifacations (e.g. "Carbamidomethyl@C")
    :return sequence with unimods (e.g."AAC[UNIMOD:4]GHK")
    """
    
    if mods_variable == "" and mods_fixed == "":
        return "[]-" + sequence + "-[]" 
    elif mods_variable == "" and not mods_fixed == "":
        mods_list = mods_fixed.split(';')
    elif not mods_variable == "" and mods_fixed == "":
        mods_list = mods_variable.split(';')
    else:
        mods_list = mods_variable.split(';') + mods_fixed.split(';')
        
    # Splitting "modificationtyp@acid_and_position" into [typ, acid_and_pos]
    for count in range(len(mods_list)):
        mods_list[count] = mods_list[count].strip()
        mods_list[count] = mods_list[count].split('@')
        
    mods_dict = {"M": "m", "C": "c", "K": "k", "S": "s", "T": "t", "Y": "y"}        
    mod_at_start = False
    
    # Tag all the modified AAs to replace them with UNIMOD later
    for count in range(len(mods_list)):
        mod_and_position = mods_list[count]
        amino_acid = mod_and_position[1][0]
            
        if len(mod_and_position[1]) > 1:
            position = int(mod_and_position[1][1:])
            if not position == 0 and not position > len(sequence):
                sequence = sequence[:position-1] + mods_dict[amino_acid] + sequence[position:]
            elif position == 0:
                mod_at_start = True
            
        elif len(mod_and_position[1]) == 1:
            sequence = sequence.replace(amino_acid, mods_dict[amino_acid])
          
    sequence = sequence.replace("m", "M[UNIMOD:35]")
    sequence = sequence.replace("c", "C[UNIMOD:4]")
    sequence = sequence.replace("k", "K[UNIMOD:737]")
    sequence = sequence.replace("s", "S[UNIMOD:21]")
    sequence = sequence.replace("t", "T[UNIMOD:21]")
    sequence = sequence.replace("y", "Y[UNIMOD:21]")

    sequence =  sequence + "-[]"
    if mod_at_start:
        sequence = "[UNIMOD:1]-" + sequence
    else:
        sequence = "[]-" + sequence
         
    return sequence
    

