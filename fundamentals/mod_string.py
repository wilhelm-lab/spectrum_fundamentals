from typing import List, Dict, Optional, Union, Tuple
from .constants import SPECTRONAUT_MODS, MAXQUANT_VAR_MODS, MOD_MASSES, MOD_NAMES
import numpy as np
import re


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
    fixed_mods: Optional[Dict[str, str]] = {'C': '(U:4)'}
) -> List[str]:
    """
    Function to translate a MaxQuant modstring to the Prosit format
    :param sequences: List[str] of sequences
    :param fixed_mods: Optional dictionary of modifications with key aa and value mod, e.g. 'M': '(U:35)'.
    :return: List[str] of modified sequences.
    """
    err_msg = f"Provided illegal fixed mod, supported modifications are {set(MAXQUANT_VAR_MODS.values())}."
    assert all(x in MAXQUANT_VAR_MODS.values() for x in fixed_mods.values()), err_msg

    modified_sequences = []
    def transform_fixed_mod(key, value) -> str:
        """
        Subfunction to prepend a given value with its key. Used to insert a modification (value) after a given match
        (key) in a mod_string. Otherwise the match would substituted (aa would be lost). This method is needed as
        fixed mods are usually given in (key, value) or (aa, mod) pairs instead of (aa, aamod).
        :param key: the key to prepend
        :param value: the value to be prepended by the key
        :return substitution string prepended by aa.
        """
        if key == '^_': # N terminal modification. Don't add the '^' identifier to value
            return f'_{value}'
        elif key == '_$':   # C terminal modification. Don't add the '$' identifier to value
            return f'_{value}'
        else:
            return f'{key}{value}'
    fixed_mods = {key: transform_fixed_mod(key, value) for key, value in fixed_mods.items()}
    replacements = {**MAXQUANT_VAR_MODS, **fixed_mods}

    def custom_regex_escape(key: str) -> str:
        """
        Subfunction that wraps re.escape. Used to only escape subsets of special characters.
        :param key: The match to escape.
        :return match with escaped special characters.
        """
        if key in ['^_', '_$']: # Don't escape beginning and end identifier in order to match N and C terminal mods
            return key
        return re.escape(key)

    regex = re.compile("(%s)" % "|".join(map(custom_regex_escape, replacements.keys())))

    def find_replacement(match: re.Match) -> str:
        """
        Subfunction to find the corresponding substitution for a match.
        :param match: an re.Match object found by re.sub
        :return substitution string for the given match
        """
        key = match.string[match.start():match.end()]
        if key == '_':  # in this special case, ^ and $ are not part of the match so they need to be readded to the key
            if match.start() == 0:
                key = "^_"
            elif match.start() == len(match.string) - 1:
                key = "_$"
            else:
                assert False, "'_' is not allowed amidst aa sequence. Please check."
        return replacements[key]

    return [regex.sub(find_replacement, seq) for seq in sequences]

def internal_without_mods(
    sequences: List[str],
    remove_underscores: Optional[bool] = False
) -> List[str]:
    """
    Function to remove any mod identifiers and return the plain AA sequence.
    :param sequences: List[str] of sequences
    :return: List[str] of modified sequences.
    """
    regex = "\(U:.+?\)|[.+?]"
    if remove_underscores:
        regex += '|_'
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
        mod = f"{len(match_list)}/{'/'.join([f'{pos},{seq[pos]},{name}' for name, pos in match_list])}"
        pos[0] = 0
        offset[0] = 0
        match_list.clear()
        return mod, mod_string

    def replace_and_store(match: re.Match):
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
