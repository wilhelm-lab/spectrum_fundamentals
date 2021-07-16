from typing import List, Dict, Optional, Union
from .constants import SPECTRONAUT_MODS, MAXQUANT_VAR_MODS
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
    regex = "\(U:.+?\)"
    if remove_underscores:
        regex += '|_'
    return [re.sub(regex, "", seq) for seq in sequences]
