import difflib
import re
from itertools import combinations, repeat
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from .constants import MOD_MASSES, MOD_NAMES, OPENMS_VAR_MODS, SPECTRONAUT_MODS, XISEARCH_VAR_MODS


def sage_to_internal(sequences: List[str], mods: Dict[str, str]) -> List[str]:
    """
    Convert mod string from sage to the internal format.

    This function converts sequences using the mass change of a modification in
    square brackets as done by Sage to the internal format by replacing the mass
    shift with the corresponding UNIMOD identifier of known and supported
    modifications defined in the constants.

    :param sequences: A list of sequences with values inside square brackets.
    :param mods: Dict with all Sage-specific and custom modifications
    :raises AssertionError: if modifications in custom or internal format were provided in the wrong type.
    :return: A list of modified sequences with values converted to internal format.
    """
    # Define a regular expression pattern to match values within square brackets, like [+1.0] or [-2.0].
    pattern = r"[A-Z]?\[([\+\-]\d+\.\d+)\]-?"

    # Define a function 'replace' that takes a regex match object.
    # Define a function 'replace' that takes a regex match object.
    def replace(match):
        # Extract the value inside the square brackets as a float.
        value = str(float(match.group(1)))
        key = match.string[match.start() : match.end()]
        if key.endswith("-"):
            unimod_expression = f"{mods.get(value, match.group(0))}-"
        # custom mods can be either the entire key or just the numeric value as string
        elif key in mods.keys() or value in mods.keys():
            key_pref = ""
            if key[0].isalpha():
                key_pref = key[0]
            unimod_expression = f"{key_pref}{mods.get(value, match.group(0))}"
        elif key.startswith("C"):
            unimod_expression = f"C{mods.get(value, match.group(0))}"
        elif key.startswith("K"):
            unimod_expression = f"K{mods.get(value, match.group(0))}"
        elif key.startswith("M"):
            unimod_expression = f"M{mods.get(value, match.group(0))}"
        else:
            unimod_expression = match.group(0)
        # Check if the 'mods' dictionary has a replacement value for the extracted value.
        # If it does, use the replacement value; otherwise, use the original value from the match.
        return unimod_expression

    # Create an empty list 'modified_strings' to store the modified sequences.
    modified_strings = []

    if not all(isinstance(val, str) for val in mods.values()) or not all(
        isinstance(key, (str, float)) for key in mods.keys()
    ):
        raise AssertionError("All custom modifications entries must have keys of type str and values of type str.")

    # Iterate through the input 'sequences'.
    for string in sequences:

        # Use 're.sub' to search and replace values within square brackets in the 'string' using the 'replace' function.
        modified_string = re.sub(pattern, replace, string)

        # Append the modified string to the 'modified_strings' list.
        modified_strings.append(modified_string)

    # Return the list of modified sequences.
    return modified_strings


def xisearch_to_internal(
    xl: str,
    seq: str,
    mod: str,
    crosslinker_position: int,
    mod_positions: str,
):
    """
    Function to translate a xisearch modstring to the XL-Prosit format.

    :param xl: type of crosslinker used. Can be 'DSSO' or 'DSBU'.
    :param seq: unmodified peptide sequence
    :param mod: all modifications of pep
    :param crosslinker_position: crosslinker position of peptide
    :param mod_positions: position of all modifications of peptide
    :raises ValueError: if suplied type of crosslinker is unknown

    :return: modified sequence
    """

    def add_mod_sequence(split_seq: List[str], mods: str, mod_positions: str):
        """
        Apply modifications.

        :param split_seq: List containing the sequence characters
        :param mods: String containing modifications
        :param mod_positions: String containing positions of modifications
        """
        if mod_positions.lower() in ["", "nan", "null"]:
            return

        split_mod = mods.split(";")
        split_mod_positions = mod_positions.split(";")

        for mod, pos in zip(split_mod, split_mod_positions):
            modification = XISEARCH_VAR_MODS.get(mod)
            pos_mod = int(pos)
            if modification:
                split_seq[pos_mod - 1] += modification
            else:
                split_seq[pos_mod - 1] += f"({mod})"

    # Check the crosslinker type and apply modification accordingly
    modification = XISEARCH_VAR_MODS.get(xl.lower())
    if modification is None:
        raise ValueError(f"Unknown crosslinker type provided: {xl}. Only 'DSSO' and 'DSBU' are supported.")

    split_seq = [x for x in seq]
    add_mod_sequence(split_seq, mod, mod_positions)
    split_seq[crosslinker_position - 1] += modification
    return "".join(split_seq)


def internal_to_spectronaut(sequences: Union[np.ndarray, pd.Series, List[str]]) -> List[str]:
    """
    Function to translate a modstring from the internal format to the spectronaut format.

    :param sequences: List[str] of sequences
    :return: List[str] of modified sequences
    """
    regex = re.compile("(%s)" % "|".join(map(re.escape, SPECTRONAUT_MODS.keys())))
    return [regex.sub(lambda mo: SPECTRONAUT_MODS[mo.string[mo.start() : mo.end()]], seq) for seq in sequences]


def maxquant_to_internal(sequences: Union[np.ndarray, pd.Series, List[str]], mods: Dict[str, str]) -> List[str]:
    """
    Function to translate a MaxQuant modstring to the Prosit format.

    :param sequences: List[str] of sequences
    :param mods: Dictionary of modifications with optional fixed mods (key aa and value mod, e.g. 'M[147]': '[UNIMOD:35]').
        custom variable modifications and standard MAXQUANT var mods. Custom static mods are not visible in the mod string,
        therefore input needs to change to key = aa and value aa and unimod identifier.
    :raises AssertionError: if illegal modification was provided in the fixed_mods dictionary or custom mods in illegal type format.
    :return: a list of modified sequences
    """
    if not all(isinstance(val, str) for val in mods.values()) or not all(
        isinstance(key, (str, float)) for key in mods.keys()
    ):
        raise AssertionError("All custom modifications entries must have keys of type str and values of type str.")

    regex = re.compile("|".join(map(custom_regex_escape, mods.keys())))

    def find_replacement(match: re.Match) -> str:
        """
        Subfunction to find the corresponding substitution for a match.

        :param match: an re.Match object found by re.sub
        :return: substitution string for the given match
        """
        key = match.string[match.start() : match.end()]
        if "_" in key:  # If _ is in the match we need to differentiate n and c term
            if match.start() == 0:
                key = f"^{key}"
            else:
                key = f"{key}$"

        value = mods[key]
        if key[0].isalpha() and not value[0].isalpha():
            value = f"{key[0]}{value}"
        return value

    return [regex.sub(lambda match: find_replacement(match), seq).replace("_", "") for seq in sequences]


def msfragger_to_internal(sequences: Union[np.ndarray, pd.Series, List[str]], mods: Dict[str, str]) -> List[str]:
    """
    Function to translate a MSFragger modstring to the Prosit format.

    :param sequences: List[str] of sequences
    :param mods: Dictionary of modifications with optional fixed mods (key aa and value mod, e.g. 'M[147]': '[UNIMOD:35]').
        custom static and variable modifications and in case of MSFragger also standard static mods
    :return: a list of modified sequences
    """
    return _to_internal(sequences=sequences, mods=mods)


def openms_to_internal(sequences: List[str], fixed_mods: Optional[Dict[str, str]] = None) -> List[str]:
    """
    Function to translate a OpenMS modstring to the Prosit format.

    :param sequences: List[str] of sequences
    :param fixed_mods: Optional dictionary of modifications with key aa and value mod, e.g. 'M(Oxidation)': 'M(UNIMOD:35)'.
        Fixed modifications must be included in the variable modificatons dictionary.
        By default, i.e. if nothing is supplied to fixed_mods, carbamidomethylation on cystein will be included
        in the fixed modifications. If you want to have no fixed modifictions at all, supply fixed_mods={}
    :raises AssertionError: if illegal modification was provided in the fixed_mods dictionary.
    :return: a list of modified sequences
    """
    if fixed_mods is None:
        fixed_mods = {"C": "C[UNIMOD:4]"}
    err_msg = f"Provided illegal fixed mod, supported modifications are {set(OPENMS_VAR_MODS.values())}."
    assert all(x in OPENMS_VAR_MODS.values() for x in fixed_mods.values()), err_msg

    replacements = {**OPENMS_VAR_MODS, **fixed_mods}

    def custom_regex_escape(key: str) -> str:
        """
        Subfunction to escape only normal brackets in the modstring.

        :param key: The match to escape
        :return: match with escaped special characters
        """
        for k, v in {"(": r"\(", ")": r"\)"}.items():
            key = key.replace(k, v)
        return key

    regex = re.compile("|".join(map(custom_regex_escape, replacements.keys())))

    def find_replacement(match: re.Match) -> str:
        """
        Subfunction to find the corresponding substitution for a match.

        :param match: an re.Match object found by re.sub
        :return: substitution string for the given match
        """
        key = match.string[match.start() : match.end()]

        return replacements[key]

    return [regex.sub(find_replacement, seq) for seq in sequences]


def internal_without_mods(sequences: List[str]) -> List[str]:
    """
    Function to remove any mod identifiers and return the plain AA sequence.

    :param sequences: List[str] of sequences
    :return: List[str] of modified sequences
    """
    regex = r"\[.*?\]|\-"
    return [re.sub(regex, "", seq) for seq in sequences]


def internal_to_mod_mass(sequences: List[str], custom_mods: Optional[Dict[str, float]] = None) -> List[str]:
    """
    Function to exchange the internal mod identifiers with the masses of the specific modifiction.

    :param sequences: List[str] of sequences
    :param custom_mods: custom mods with the identifier (=key), respespective unimod identifier and mass (value)
    :return: List[str] of modified sequences
    """
    mod_masses = MOD_MASSES | (custom_mods or {})

    regex = re.compile("(%s)" % "|".join(map(re.escape, mod_masses.keys())))
    replacement_func = lambda match: f"[+{mod_masses[match.string[match.start():match.end()]]}]"
    return [regex.sub(replacement_func, seq) for seq in sequences]


def internal_to_msp(
    sequences: Union[List[str], pd.Series],
    mods: Dict[str, str],
) -> List[Tuple[str, str]]:
    """
    Function to translate an internal modstring to modstring and Mods for MSP format.

    :param sequences: sequences to translate
    :param mods: dictionary mapping from internal unimod format (keys) to MSP format (values).
    :return: a tuple for each sequence, containing (Mods, mod_string) for the MSP format
    """
    ret_vals = []
    p = re.compile("|".join(mods.keys()))
    for seq in sequences:
        offset = 0
        mod_list = []
        matches = p.finditer(seq)
        for match in matches:
            replacement = mods[re.escape(match.group())]
            start, end = match.span()
            actual_start = start - offset
            mod_list.append((actual_start, replacement))

            offset += end - start - 1

        mod_string = "; ".join([f"{mod[2:]}@{mod[0]}{pos}" for pos, mod in mod_list])
        n_mods = len(mod_list)
        if n_mods > 0:
            mods_field = f"{n_mods}/{'/'.join([f'{pos},{mod}' for pos, mod in mod_list])}"
        else:
            mods_field = "0"
        ret_vals.append((mods_field, mod_string))
    return ret_vals


def internal_to_mod_names(
    sequences: List[str],
) -> List[Tuple[str, str]]:
    """
    Function to translate an internal modstring to MSP format.

    :param sequences: List[str] of sequences
    :return: List[Tuple[str, str] of mod summary and mod sequences
    """
    match_list = []
    pos = [0]
    offset = [0]

    def msp_string_mapper(seq: str):
        """
        Internal function to create the mod summary and mod_string from given sequence and match_list.

        :param seq: The sequence to modify
        :return: Tuple with mod summary and mod_string
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

    def replace_and_store(match: re.Match) -> str:
        """
        Internal function that removes matched internal mods and stores there position in the sequence.

        :param match: an re.Match object found by re.sub
        :return: empty string
        """
        pos[0] = match.start() - 1 - offset[0]
        offset[0] += match.end() - match.start()
        match_list.append((MOD_NAMES[match.string[match.start() : match.end()]], pos[0]))
        return ""

    regex = re.compile("(%s)" % "|".join(map(re.escape, MOD_NAMES.keys())))
    return [msp_string_mapper(seq) for seq in sequences]


def parse_modstrings(sequences: List[str], alphabet: Dict[str, int], translate: bool = False, filter: bool = False):
    """
    Parse modstrings.

    :param sequences: List of strings
    :param alphabet: dictionary where the keys correspond to all possible 'Elements' that can occur in the string
    :param translate: boolean to determine if the Elements should be translated to the corresponding values of ALPHABET
    :param filter: boolean to determine if non-parsable sequences should be filtered out
    :return: generator that yields a list of sequence 'Elements' or the translated sequence "Elements"
    """

    def split_modstring(sequence: str, r_pattern):
        # Ugly and fast fix for reading modifications as is from maxquant we should reconsider how to fix it.
        # sequence = sequence.replace('M(ox)','M(U:35)')
        # sequence = sequence.replace('C','C(U:4)')
        val = max(alphabet.values()) + 1
        split_seq = r_pattern.findall(sequence)
        if "".join(split_seq) == sequence:
            if translate:
                results = []
                for aa in split_seq:
                    if aa not in alphabet:  # does not exist
                        alphabet[aa] = val
                        val += 1
                    results.append(alphabet[aa])
                return results
            else:
                return split_seq
        elif filter:
            return [0]
        else:
            not_parsable_elements = "".join(
                [li[2] for li in difflib.ndiff(sequence, "".join(split_seq)) if li[0] == "-"]
            )
            raise ValueError(
                f"The element(s) [{not_parsable_elements}] " f"in the sequence [{sequence}] could not be parsed"
            )

    unimod_pattern = r"[A-Z]\[UNIMOD:\d+\]"
    alphabet_pattern = [re.escape(i) for i in sorted(alphabet, key=len, reverse=True)]

    pattern = [unimod_pattern] + alphabet_pattern
    regex_pattern = re.compile("|".join(pattern))
    return map(split_modstring, sequences, repeat(regex_pattern))


def get_all_tokens(sequences: List[str]) -> Set[str]:
    """Parse given sequences in UNIMOD ProForma standard into a set of all tokens."""
    pattern = r"[ACDEFGHIKLMNPQRSTVWY](\[UNIMOD:\d+\])?"
    tokens = set()
    for seq in sequences:
        tokens |= {match.group() for match in re.finditer(pattern, seq)}
    return tokens


def add_permutations(
    modified_sequence: str, unimod_id: int, residues: List[str], allow_one_less_modification: bool = False
):
    """
    Generate different peptide sequences with moving the modification to all possible residues.

    :param modified_sequence: Peptide sequence
    :param unimod_id: modification unimod id to be used for generating different permutations.
    :param residues: possible amino acids where this mod can exist
    :param allow_one_less_modification: Flag to indicate if permutations with one less modification should be generated to check
        whether the modification mass was mistakenly picked as the monoisotopic peak. Mainly used for Citrullination.
    :return: list of possible sequence permutations
    """
    modified_sequence = modified_sequence.replace("UNIMOD", "unimod")
    sequence = modified_sequence.replace("[unimod:" + str(unimod_id) + "]", "")
    modifications = len(re.findall("unimod:" + str(unimod_id), modified_sequence))
    if modifications == 0:
        modified_sequence = modified_sequence.replace("unimod", "UNIMOD")
        return [modified_sequence]
    possible_positions = [i for i, ltr in enumerate(sequence) if ltr in residues]
    possible_positions.sort(reverse=True)
    all_combinations = [list(each_permutation) for each_permutation in combinations(possible_positions, modifications)]

    if allow_one_less_modification:
        all_combinations_1 = [
            list(each_permutation) for each_permutation in combinations(possible_positions, modifications - 1)
        ]
        all_combinations = all_combinations + all_combinations_1

    modified_sequences_comb = []
    for comb in all_combinations:
        modified_sequence = sequence
        for index in comb:
            modified_sequence = (
                modified_sequence[: index + 1] + "[unimod:" + str(unimod_id) + "]" + modified_sequence[index + 1 :]
            )
        modified_sequence = modified_sequence.replace("unimod", "UNIMOD")
        modified_sequences_comb.append(modified_sequence)
    return modified_sequences_comb


def proteomicsdb_to_internal(sequence: str, mods_variable: str = "", mods_fixed: str = "") -> str:
    """
    Function to create a sequence with UNIMOD modifications from given sequence and it's varaible and fixed modifications.

    :param sequence: The sequence to modify
    :param mods_variable: the variable modifacations (e.g. "Oxidation@M45")
    :param mods_fixed: the fixed modifacations (e.g. "Carbamidomethyl@C")
    :return: sequence with unimods (e.g."AAC[UNIMOD:4]GHK")
    """
    if mods_variable == "" and mods_fixed == "":
        return "[]-" + sequence + "-[]"
    else:
        mods_list = get_mods_list(mods_variable, mods_fixed)

    # Splitting "modificationtyp@acid_and_position" into [typ, acid_and_pos]
    for count in range(len(mods_list)):
        mods_list[count] = mods_list[count].strip()
        mods_list[count] = mods_list[count].split("@")

    mods_dict = {"M": "m", "C": "c", "K": "k", "S": "s", "T": "t", "Y": "y"}
    mod_at_start = False

    # Tag all the modified AAs to replace them with UNIMOD later
    for count in range(len(mods_list)):
        mod_and_position = mods_list[count]
        amino_acid = mod_and_position[1][0]

        if len(mod_and_position[1]) > 1:
            position = int(mod_and_position[1][1:])
            if position == 0:
                mod_at_start = True
            else:
                if position <= len(sequence):
                    sequence = sequence[: position - 1] + mods_dict[amino_acid] + sequence[position:]

        elif len(mod_and_position[1]) == 1:
            sequence = sequence.replace(amino_acid, mods_dict[amino_acid])

    sequence = sequence.replace("m", "M[UNIMOD:35]")
    sequence = sequence.replace("c", "C[UNIMOD:4]")
    sequence = sequence.replace("k", "K[UNIMOD:737]")
    sequence = sequence.replace("s", "S[UNIMOD:21]")
    sequence = sequence.replace("t", "T[UNIMOD:21]")
    sequence = sequence.replace("y", "Y[UNIMOD:21]")

    sequence = sequence + "-[]"
    if mod_at_start:
        sequence = "[UNIMOD:1]-" + sequence
    else:
        sequence = "[]-" + sequence

    return sequence


def get_mods_list(mods_variable: str, mods_fixed: str):
    """Helper function to get mods list."""
    if mods_variable == "" and not mods_fixed == "":
        return mods_fixed.split(";")
    elif not mods_variable == "" and mods_fixed == "":
        return mods_variable.split(";")
    else:
        return mods_variable.split(";") + mods_fixed.split(";")


def custom_regex_escape(key: str) -> str:
    """
    Subfunction to escape normal, square brackets and the plus-sign in the modstring.

    :param key: The match to escape
    :return: match with escaped special characters
    """
    for k, v in {"(": r"\(", ")": r"\)", "[": r"\[", "]": r"\]", "+": r"\+", "-": r"\-"}.items():
        key = key.replace(k, v)
    return key


def custom_to_internal(sequences: Union[np.ndarray, pd.Series, List[str]], mods: Dict[str, str]) -> List[str]:
    """
    Function to translate custom modstrings to the Prosit format.

    :param sequences: List[str] of sequences
    :param mods: Dictionary of modifications with optional fixed mods (key aa and value mod, e.g. 'M[147]': '[UNIMOD:35]').
        custom static and variable modifications and in case of MSFragger also standard static mods
    :return: a list of modified sequences
    """
    return _to_internal(sequences=sequences, mods=mods)


def _to_internal(sequences: Union[np.ndarray, pd.Series, List[str]], mods: Dict[str, str]) -> List[str]:
    """
    Function to translate a modstring to the internal Prosit format.

    :param sequences: List[str] of sequences
    :param mods: Dictionary of modifications with optional fixed mods (key aa and value mod, e.g. 'M[147]': '[UNIMOD:35]').
        custom static and variable modifications and in case of MSFragger also standard static mods
    :return: a list of modified sequences
    """

    def find_replacement(match: re.Match) -> str:
        """
        Subfunction to find the corresponding substitution for a match.

        :param match: an re.Match object found by re.sub
        :return: substitution string for the given match
        """
        key = match.string[match.start() : match.end()]
        value = mods[key]
        if key[0].isalpha() and key[0].isupper() and not value[0].isalpha():
            value = f"{key[0]}{value}"
        return value

    regex = re.compile("|".join(map(custom_regex_escape, mods.keys())))

    return [regex.sub(lambda match: find_replacement(match), seq) for seq in sequences]
