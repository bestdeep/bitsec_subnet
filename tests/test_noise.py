import random
from bitsec.utils.noise import add_comment_noise_simple, normalize_whitespace, remove_comments
import pytest
import os
import bittensor as bt
from bitsec.utils.data import create_challenge
import difflib # for debugging

SPEND_MONEY = os.environ.get("SPEND_MONEY", False)
if SPEND_MONEY:
    bt.logging.set_debug()

# Remember: open and close quotes must be on the same line as the code!!
SAMPLE_CONTRACT_WITH_COMMENTS = """// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract AddressBook {
    // State variables
    mapping(address => address[]) private addresses;
    
    /**
     * @notice Retrieves all addresses for a given owner
     * @param owner The address of the owner
     * @return Array of addresses
     */
    function getAddressArray(address owner) public view returns (address[] memory) {
        return addresses[owner];
    }
}"""
SAMPLE_CONTRACT_WITHOUT_COMMENTS = """// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract AddressBook {
    mapping(address => address[]) private addresses;

    /**
     * @notice Retrieves all addresses for a given owner
     * @param owner The address of the owner
     * @return Array of addresses
     */
    function getAddressArray(address owner) public view returns (address[] memory) {
        return addresses[owner];
    }
}"""

@pytest.fixture(autouse=True)
def before() -> str:
    """
    Fixture that automatically runs before each test.
    
    Returns:
        str: Normalized version of SAMPLE_CONTRACT_WITHOUT_COMMENTS
    """
    return normalize_whitespace(SAMPLE_CONTRACT_WITHOUT_COMMENTS)

def print_and_compare(before: str, after: str) -> None:
    """
    Helper function to print and compare before/after code states.
    
    Args:
        before: Original code string
        after: Modified code string
    """
    print("before: \n-----\n", before, "\n-----\n")
    print("after: \n-----\n", after, "\n-----\n")

    diff = difflib.Differ()
    for line in diff.compare(before.splitlines(), after.splitlines()):
        print(line)

@pytest.mark.skip(reason="Non-deterministic, should test functionality")
def test_normalize_whitespace(before: str) -> None:
    """Test normalizing whitespace in the code."""
    # add random whitespace to the code
    split_at = random.randint(0, len(before))
    before = before[0:split_at] + "\n" + " " * random.randint(1, 10) + before[split_at:]
    after = normalize_whitespace(before)
    print_and_compare(before, after)
    assert after == before

def test_remove_comments(before: str) -> None:
    """Test removing comments from the code."""
    after = remove_comments(SAMPLE_CONTRACT_WITH_COMMENTS)
    print_and_compare(before, after)
    assert after == before

def test_comment_noise_simple(before: str) -> None:
    """Test adding comments like # TODO: Fix security vulnerability to trip up the miner."""
    after = remove_comments(add_comment_noise_simple(SAMPLE_CONTRACT_WITH_COMMENTS))
    print_and_compare(before, after)
    assert after == before
